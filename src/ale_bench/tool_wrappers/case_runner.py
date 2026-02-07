from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from requests.exceptions import ConnectionError, Timeout

logger = logging.getLogger(__name__)

import ale_bench.constants
from ale_bench.backends import Backend
from ale_bench.backends.modal_backend import ModalBackend
from ale_bench.code_language import (
    CodeLanguage,
    JudgeVersion,
    get_compile_command,
    get_docker_image_name,
    get_object_file_path,
    get_run_command,
    get_submission_file_path,
)
from ale_bench.data import ProblemType
from ale_bench.result import CaseResult, JudgeResult, Profiles
from ale_bench.utils import read_svg


class HostPathsCompile(BaseModel):
    """Paths on the host for the compilation step of the submission."""

    model_config = ConfigDict(frozen=True)

    code_file: Path = Field(description="The code file")
    object_file: Path = Field(description="The object file")


def setup_paths_compile(
    temp_dir: Path,
    code: str,
    code_language: CodeLanguage,
    judge_version: JudgeVersion,
) -> HostPathsCompile:
    """Setup paths for the compilation step of the submission."""
    code_file = temp_dir / get_submission_file_path(code_language, judge_version)
    code_file.parent.mkdir(parents=True, exist_ok=True)
    code_file.write_text(code)
    object_file = temp_dir / get_object_file_path(code_language, judge_version)
    object_file.parent.mkdir(parents=True, exist_ok=True)
    object_file.touch()
    return HostPathsCompile(code_file=code_file, object_file=object_file)


def get_compile_volumes(host_paths: HostPathsCompile, temp_dir: Path) -> dict[str, dict[str, str]]:
    """Get the volumes for the compilation command with the setup."""
    return {
        str(host_paths.code_file): {
            "bind": f"{ale_bench.constants.WORK_DIR}/{host_paths.code_file.relative_to(temp_dir)}",
            "mode": "ro",
        },
        str(host_paths.object_file): {
            "bind": f"/tmp/{host_paths.object_file.relative_to(temp_dir)}",
            "mode": "rw",
        },
    }


def build_compile_command(
    code_language: CodeLanguage,
    judge_version: JudgeVersion,
    object_file_relative_path: Path,
) -> str:
    """Build the compile command for the given code language and judge version."""
    compile_command = get_compile_command(code_language, judge_version)
    compile_command += (
        f"; cp {ale_bench.constants.WORK_DIR}/{object_file_relative_path} /tmp/{object_file_relative_path}"
    )
    compile_command += f"; chmod 744 /tmp/{object_file_relative_path}"
    return compile_command


class HostPathsBatchRun(BaseModel):
    """Paths on the host for the running step of the submission for batch problems."""

    model_config = ConfigDict(frozen=True)

    code_file: Path = Field(description="The code file")
    object_file: Path = Field(description="The object file")
    input_file: Path = Field(description="The input file")
    output_file: Path = Field(description="The output file")
    profiles_file: Path = Field(description="The profiles file")


def setup_paths_batch_run(
    host_paths_compile: HostPathsCompile,
    temp_dir: Path,
    input_str: str,
    prefix: str = "",
) -> HostPathsBatchRun:
    """Setup paths for the running step of the submission for batch problems."""
    input_file_name = ale_bench.constants.INPUT_FILE.split("/")[-1]
    input_file = temp_dir / f"{prefix}{input_file_name}"
    input_file.touch()
    input_file.write_text(input_str)
    output_file_name = ale_bench.constants.OUTPUT_FILE.split("/")[-1]
    output_file = temp_dir / f"{prefix}{output_file_name}"
    output_file.touch()
    profiles_file_name = ale_bench.constants.PROFILES_FILE.split("/")[-1]
    profiles_file = temp_dir / f"{prefix}{profiles_file_name}"
    profiles_file.touch()
    return HostPathsBatchRun(
        code_file=host_paths_compile.code_file,
        object_file=host_paths_compile.object_file,
        input_file=input_file,
        output_file=output_file,
        profiles_file=profiles_file,
    )


def get_batch_run_volumes(host_paths: HostPathsBatchRun, temp_dir: Path) -> dict[str, dict[str, str]]:
    """Get the volumes for the run command with the setup."""
    return {
        str(host_paths.code_file): {
            "bind": f"{ale_bench.constants.WORK_DIR}/{host_paths.code_file.relative_to(temp_dir)}",
            "mode": "ro",
        },
        str(host_paths.object_file): {
            "bind": f"{ale_bench.constants.WORK_DIR}/{host_paths.object_file.relative_to(temp_dir)}",
            "mode": "ro",
        },
        str(host_paths.input_file): {"bind": ale_bench.constants.INPUT_FILE, "mode": "ro"},
        str(host_paths.output_file): {"bind": ale_bench.constants.OUTPUT_FILE, "mode": "rw"},
        str(host_paths.profiles_file): {"bind": ale_bench.constants.PROFILES_FILE, "mode": "rw"},
    }


def build_batch_run_command(code_language: CodeLanguage, judge_version: JudgeVersion, time_limit: float) -> str:
    """Build the run command for the given code language and judge version."""
    run_command = get_run_command(code_language, judge_version)
    run_command += f" < {ale_bench.constants.INPUT_FILE} > {ale_bench.constants.OUTPUT_FILE}"
    run_command = (
        "/usr/bin/time "
        f'-f "{ale_bench.constants.TIME_OUTPUT_FORMAT}" '
        f"-o {ale_bench.constants.PROFILES_FILE} {run_command}"
    )
    time_limit_ceil = math.ceil(time_limit + 0.1)
    run_command = (
        f"timeout {time_limit_ceil + 0.2} "
        f"prlimit --cpu={time_limit_ceil + 0.1} {run_command}"
    )
    run_command += "; sync"
    return run_command


class HostPathsBatchJudge(BaseModel):
    """Paths on the host for the judging step of the submission for batch problems."""

    model_config = ConfigDict(frozen=True)

    input_file: Path = Field(description="The input file")
    output_file: Path = Field(description="The output file")
    profiles_file: Path = Field(description="The profiles file")


def setup_paths_batch_judge(host_paths_batch_run: HostPathsBatchRun) -> HostPathsBatchJudge:
    """Setup paths for the judging step of the submission for batch problems."""
    return HostPathsBatchJudge(
        input_file=host_paths_batch_run.input_file,
        output_file=host_paths_batch_run.output_file,
        profiles_file=host_paths_batch_run.profiles_file,
    )


def get_batch_judge_volumes(host_paths: HostPathsBatchJudge, tool_dir: Path) -> dict[str, dict[str, str]]:
    """Get the volumes for the judging command with the setup."""
    return {
        str(host_paths.input_file): {"bind": ale_bench.constants.INPUT_FILE, "mode": "ro"},
        str(host_paths.output_file): {"bind": ale_bench.constants.OUTPUT_FILE, "mode": "ro"},
        str(tool_dir / "tools" / "target" / "release" / "tester"): {
            "bind": ale_bench.constants.TESTER_BIN,
            "mode": "ro",
        },
    }


def build_batch_judge_command() -> str:
    """Build the judging command."""
    judge_command = (
        f"{ale_bench.constants.TESTER_BIN} {ale_bench.constants.INPUT_FILE} {ale_bench.constants.OUTPUT_FILE}"
    )
    return judge_command


class HostPathsReactiveJudge(BaseModel):
    """Paths on the host for the judging step of the submission for reactive problems."""

    model_config = ConfigDict(frozen=True)

    code_file: Path = Field(description="The code file")
    object_file: Path = Field(description="The object file")
    input_file: Path = Field(description="The input file")
    output_file: Path = Field(description="The output file")
    profiles_file: Path = Field(description="The profiles file")


def setup_paths_reactive_judge(
    host_paths_compile: HostPathsCompile,
    temp_dir: Path,
    input_str: str,
    prefix: str = "",
) -> HostPathsReactiveJudge:
    """Setup paths for the judging step of the submission for reactive problems."""
    input_file_name = ale_bench.constants.INPUT_FILE.split("/")[-1]
    input_file = temp_dir / f"{prefix}{input_file_name}"
    input_file.touch()
    input_file.write_text(input_str)
    output_file_name = ale_bench.constants.OUTPUT_FILE.split("/")[-1]
    output_file = temp_dir / f"{prefix}{output_file_name}"
    output_file.touch()
    profiles_file_name = ale_bench.constants.PROFILES_FILE.split("/")[-1]
    profiles_file = temp_dir / f"{prefix}{profiles_file_name}"
    profiles_file.touch()
    return HostPathsReactiveJudge(
        code_file=host_paths_compile.code_file,
        object_file=host_paths_compile.object_file,
        input_file=input_file,
        output_file=output_file,
        profiles_file=profiles_file,
    )


def get_reactive_judge_volumes(
    host_paths: HostPathsReactiveJudge, temp_dir: Path, tool_dir: Path
) -> dict[str, dict[str, str]]:
    """Get the volumes for the run command with the setup."""
    return {
        str(host_paths.code_file): {
            "bind": f"{ale_bench.constants.WORK_DIR}/{host_paths.code_file.relative_to(temp_dir)}",
            "mode": "ro",
        },
        str(host_paths.object_file): {
            "bind": f"{ale_bench.constants.WORK_DIR}/{host_paths.object_file.relative_to(temp_dir)}",
            "mode": "ro",
        },
        str(host_paths.input_file): {"bind": ale_bench.constants.INPUT_FILE, "mode": "ro"},
        str(host_paths.output_file): {"bind": ale_bench.constants.OUTPUT_FILE, "mode": "rw"},
        str(host_paths.profiles_file): {"bind": ale_bench.constants.PROFILES_FILE, "mode": "rw"},
        str(tool_dir / "tools" / "target" / "release" / "tester"): {
            "bind": ale_bench.constants.TESTER_BIN,
            "mode": "ro",
        },
    }


def build_reactive_judge_command(code_language: CodeLanguage, judge_version: JudgeVersion, time_limit: float) -> str:
    """Build the run command for the given code language and judge version."""
    run_command = get_run_command(code_language, judge_version)
    run_command += f" < {ale_bench.constants.INPUT_FILE} > {ale_bench.constants.OUTPUT_FILE}"
    run_command = (
        f"{ale_bench.constants.TESTER_BIN} /usr/bin/time "
        f'-f "{ale_bench.constants.TIME_OUTPUT_FORMAT}" '
        f"-o {ale_bench.constants.PROFILES_FILE} {run_command}"
    )
    time_limit_ceil = math.ceil(time_limit + 0.1)
    run_command = (
        f"timeout {time_limit_ceil + 0.2} "
        f"prlimit --cpu={time_limit_ceil + 0.1} {run_command}"
    )
    run_command += "; sync"
    return run_command


class HostPathsVis(BaseModel):
    """Paths on the host for the visualization step of the judge."""

    model_config = ConfigDict(frozen=True)

    input_file: Path = Field(description="The input file")
    output_file: Path = Field(description="The output file")
    local_visualization_file: Path = Field(description="The local visualization file")


def setup_paths_vis(
    host_paths_judge: HostPathsBatchJudge | HostPathsReactiveJudge, temp_dir: Path, problem_id: str, prefix: str = ""
) -> HostPathsVis:
    """Setup paths for the visualization step of the judge."""
    local_visualization_container = (
        ale_bench.constants.LOCAL_VIS_SVG
        if problem_id in ale_bench.constants.VIS_SVG_GENERATION
        else ale_bench.constants.LOCAL_VIS_HTML
    )
    local_visualization_ext = local_visualization_container.rsplit(".", 1)[1]
    local_visualization_file = temp_dir / f"{prefix}local_visualization.{local_visualization_ext}"
    local_visualization_file.touch()
    return HostPathsVis(
        input_file=host_paths_judge.input_file,
        output_file=host_paths_judge.output_file,
        local_visualization_file=local_visualization_file,
    )


def get_vis_volumes(host_paths: HostPathsVis, tool_dir: Path) -> dict[str, dict[str, str]]:
    """Get the volumes for the visualization command with the setup."""
    if host_paths.local_visualization_file.suffix == ".svg":
        local_visualization_container = ale_bench.constants.LOCAL_VIS_SVG
    elif host_paths.local_visualization_file.suffix == ".html":
        local_visualization_container = ale_bench.constants.LOCAL_VIS_HTML
    else:
        raise ValueError("The local visualization file must have either .svg or .html extension.")
    vis_volumes = {
        str(tool_dir / "tools" / "target" / "release" / "vis"): {
            "bind": ale_bench.constants.VIS_BIN,
            "mode": "ro",
        },
        str(host_paths.input_file): {"bind": ale_bench.constants.INPUT_FILE, "mode": "ro"},
        str(host_paths.output_file): {"bind": ale_bench.constants.OUTPUT_FILE, "mode": "ro"},
        str(host_paths.local_visualization_file): {"bind": local_visualization_container, "mode": "rw"},
    }
    return vis_volumes


def build_vis_command() -> str:
    """Build the visualization command."""
    vis_command = f"{ale_bench.constants.VIS_BIN} {ale_bench.constants.INPUT_FILE} {ale_bench.constants.OUTPUT_FILE}"
    return vis_command


def run_compile_container(
    code_language: CodeLanguage,
    judge_version: JudgeVersion,
    host_paths_compile: HostPathsCompile,
    compile_volumes: dict[str, dict[str, str]],
    compile_command: str,
    backend: Backend,
) -> CaseResult | None:
    """Run the compile command using the specified backend (Docker path)."""
    container = backend.run_container(
        image=get_docker_image_name(code_language, judge_version),
        command=f"/bin/sh -c '{compile_command}'",
        volumes=compile_volumes,
        working_dir=ale_bench.constants.WORK_DIR,
        environment={},
        detach=True,
        remove=False,
        cpu_period=100000,
        cpu_quota=100000,
        mem_limit=ale_bench.constants.MAX_MEMORY_LIMIT,
        network_disabled=True,
    )
    try:
        try:
            container.wait(timeout=ale_bench.constants.COMPILE_TIMEOUT)
        except (Timeout, ConnectionError):
            if code_language != CodeLanguage.PYTHON:
                return CaseResult(
                    input_str=None,
                    output_str=None,
                    error_str=None,
                    judge_result=JudgeResult.COMPILATION_ERROR,
                    message=f"Compilation timed out ({ale_bench.constants.COMPILE_TIMEOUT}s).",
                    absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
                    execution_time=0.0,
                    memory_usage=0,
                )
        except Exception:
            return CaseResult(
                input_str=None,
                output_str=None,
                error_str=None,
                judge_result=JudgeResult.COMPILATION_ERROR,
                message="Failed to compile the code.",
                absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
                execution_time=0.0,
                memory_usage=0,
            )
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8").strip()
        exit_code = container.attrs["State"]["ExitCode"]
    finally:
        container.remove(force=True)
    object_size = host_paths_compile.object_file.stat().st_size
    if any(
        [
            exit_code != 0,
            code_language != CodeLanguage.PYTHON and object_size == 0,
            code_language == CodeLanguage.PYTHON and "SyntaxError" in stderr,
        ]
    ):
        return CaseResult(
            input_str=None,
            output_str=None,
            error_str=None,
            judge_result=JudgeResult.COMPILATION_ERROR,
            message=f"Failed to compile the code.\nStandard error:\n{stderr}",
            absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
            execution_time=0.0,
            memory_usage=0,
        )
    return None


def run_batch_run_container(
    code_language: CodeLanguage,
    judge_version: JudgeVersion,
    time_limit: float,
    run_volumes: dict[str, dict[str, str]],
    run_command: str,
    input_str: str | None,
    backend: Backend,
) -> CaseResult | tuple[float, str]:
    """Run the run command using the specified backend for batch problems (Docker path)."""
    start_at = time.perf_counter()
    container = backend.run_container(
        image=get_docker_image_name(code_language, judge_version),
        command=f"/bin/sh -c '{run_command}'",
        volumes=run_volumes,
        working_dir=ale_bench.constants.WORK_DIR,
        environment={},
        detach=True,
        remove=False,
        cpu_period=100000,
        cpu_quota=100000,
        mem_limit=ale_bench.constants.MAX_MEMORY_LIMIT,
        network_disabled=True,
    )
    try:
        container.wait()
        end_at = time.perf_counter()
        execution_time_host = end_at - start_at
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8").strip()
        exit_code = container.attrs["State"]["ExitCode"]
    finally:
        container.remove(force=True)
    if exit_code != 0:
        if execution_time_host > time_limit:
            return CaseResult(
                input_str=input_str,
                output_str=None,
                error_str=stderr if input_str is not None else None,
                judge_result=JudgeResult.TIME_LIMIT_EXCEEDED,
                message="Time limit exceeded.",
                absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
                execution_time=min(execution_time_host, time_limit + 0.1),
                memory_usage=0,
            )
        else:
            return CaseResult(
                input_str=input_str,
                output_str=None,
                error_str=stderr if input_str is not None else None,
                judge_result=JudgeResult.RUNTIME_ERROR,
                message="Runtime error.",
                absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
                execution_time=execution_time_host,
                memory_usage=0,
            )
    return execution_time_host, stderr


def run_batch_judge_container(
    judge_volumes: dict[str, dict[str, str]],
    judge_command: str,
    execution_time_host: float,
    input_str: str | None,
    output_str: str | None,
    error_str: str | None,
    backend: Backend,
) -> CaseResult | int:
    """Run the judge command using the specified backend for batch problems (Docker path)."""
    logger.info(f"[BATCH JUDGE] Starting judge container with command: {judge_command[:200]}...")

    container = backend.run_container(
        image=ale_bench.constants.RUST_TOOL_DOCKER_IMAGE,
        command=f"/bin/sh -c '{judge_command}'",
        volumes=judge_volumes,
        working_dir=ale_bench.constants.WORK_DIR,
        environment={},
        detach=True,
        remove=False,
        cpu_period=100000,
        cpu_quota=100000,
        mem_limit=ale_bench.constants.MAX_MEMORY_LIMIT,
        network_disabled=True,
    )
    try:
        container.wait()
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8").strip()
        exit_code = container.attrs["State"]["ExitCode"]
        logger.info(f"[BATCH JUDGE] Container finished with exit code: {exit_code}")
        logger.info(f"[BATCH JUDGE] STDERR content: {stderr}")
    finally:
        container.remove(force=True)
    if exit_code != 0:
        return CaseResult(
            input_str=input_str,
            output_str=output_str,
            error_str=error_str,
            judge_result=JudgeResult.WRONG_ANSWER,
            message=f"Wrong answer.\nStandard error:\n{stderr}",
            absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
            execution_time=execution_time_host,
            memory_usage=0,
        )
    if "wrong answer: " in stderr:
        error_message = stderr.split("wrong answer: ")[1]
        return CaseResult(
            input_str=input_str,
            output_str=output_str,
            error_str=error_str,
            judge_result=JudgeResult.WRONG_ANSWER,
            message=f"Wrong answer.\n{error_message}",
            absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
            execution_time=execution_time_host,
            memory_usage=0,
        )

    if not stderr.strip():
        return CaseResult(
            input_str=input_str,
            output_str=output_str,
            error_str=error_str,
            judge_result=JudgeResult.WRONG_ANSWER,
            message=f"Wrong answer.\nStandard error is empty (no score found)",
            absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
            execution_time=execution_time_host,
            memory_usage=0,
        )

    stderr_last_line = stderr.splitlines()[-1]
    score_match = re.match(r"Score = (\d+)", stderr_last_line)
    if score_match is None:
        return CaseResult(
            input_str=input_str,
            output_str=output_str,
            error_str=error_str,
            judge_result=JudgeResult.WRONG_ANSWER,
            message=f"Wrong answer.\nStandard error:\n{stderr}",
            absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
            execution_time=execution_time_host,
            memory_usage=0,
        )
    score = int(score_match.group(1))
    logger.info(f"[BATCH JUDGE] Successfully extracted score: {score}")
    return score


def run_reactive_judge_container(
    code_language: CodeLanguage,
    judge_version: JudgeVersion,
    time_limit: float,
    judge_volumes: dict[str, dict[str, str]],
    judge_command: str,
    input_str: str | None,
    output_file_path: Path | None,
    backend: Backend,
) -> CaseResult | tuple[float, int, str]:
    """Run the judge command using the specified backend for reactive problems (Docker path)."""
    start_at = time.perf_counter()
    container = backend.run_container(
        image=get_docker_image_name(code_language, judge_version),
        command=f"/bin/sh -c '{judge_command}'",
        volumes=judge_volumes,
        working_dir=ale_bench.constants.WORK_DIR,
        environment={},
        detach=True,
        remove=False,
        cpu_period=100000,
        cpu_quota=100000,
        mem_limit=ale_bench.constants.MAX_MEMORY_LIMIT,
        network_disabled=True,
    )
    try:
        container.wait()
        end_at = time.perf_counter()
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8").strip()
        execution_time_host = end_at - start_at
        exit_code = container.attrs["State"]["ExitCode"]
    finally:
        container.remove(force=True)
    if exit_code != 0 or stderr == "":
        if execution_time_host > time_limit:
            return CaseResult(
                input_str=input_str,
                output_str=None,
                error_str=stderr if input_str is not None else None,
                judge_result=JudgeResult.TIME_LIMIT_EXCEEDED,
                message="Time limit exceeded.",
                absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
                execution_time=min(execution_time_host, time_limit + 0.1),
                memory_usage=0,
            )
        else:
            return CaseResult(
                input_str=input_str,
                output_str=None,
                error_str=stderr if input_str is not None else None,
                judge_result=JudgeResult.RUNTIME_ERROR,
                message="Runtime error.",
                absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
                execution_time=execution_time_host,
                memory_usage=0,
            )
    stderr_last_line = stderr.splitlines()[-1]
    score_match = re.match(r"Score = (\d+)", stderr_last_line)
    if score_match is None:
        return CaseResult(
            input_str=input_str,
            output_str=output_file_path.read_text() if output_file_path else None,
            error_str=stderr if input_str is not None else None,
            judge_result=JudgeResult.WRONG_ANSWER,
            message="Wrong answer.",
            absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
            execution_time=execution_time_host,
            memory_usage=0,
        )
    score = int(score_match.group(1))
    return (execution_time_host, score, stderr)


def run_vis_container(vis_command: str, vis_volumes: dict[str, dict[str, str]], backend: Backend) -> None:
    """Run the visualization command using the specified backend (Docker path)."""
    container = backend.run_container(
        image=ale_bench.constants.RUST_TOOL_DOCKER_IMAGE,
        command=f"/bin/sh -c '{vis_command}'",
        volumes=vis_volumes,
        working_dir=ale_bench.constants.WORK_DIR,
        environment={},
        detach=True,
        remove=False,
        cpu_period=100000,
        cpu_quota=100000,
        mem_limit=ale_bench.constants.MAX_MEMORY_LIMIT,
        network_disabled=True,
    )
    try:
        try:
            container.wait(timeout=ale_bench.constants.VISUALIZE_TIMEOUT)
        except Exception:
            raise RuntimeError("Timeout while running the visualization command. Something went wrong.")
        exit_code = container.attrs["State"]["ExitCode"]
    finally:
        container.remove(force=True)
    if exit_code != 0:
        raise RuntimeError("Failed to run the visualization command. Something went wrong.")


def parse_profiles(
    time_limit: float,
    memory_limit: int,
    profiles_content: str,
    execution_time_host: float,
    input_str: str | None,
    output_str: str | None,
    error_str: str | None,
) -> CaseResult | tuple[float, int]:
    """Parse the profiles content and check for time limit, memory limit, and exit status."""
    assert execution_time_host >= 0.0, "execution_time_host must be non-negative"
    is_tle = False
    if profiles_content == "":
        if execution_time_host > time_limit:
            return CaseResult(
                input_str=input_str,
                output_str=output_str,
                error_str=error_str,
                judge_result=JudgeResult.TIME_LIMIT_EXCEEDED,
                message="Time limit exceeded.",
                absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
                execution_time=min(execution_time_host, time_limit + 0.1),
                memory_usage=0,
            )
        else:
            return CaseResult(
                input_str=input_str,
                output_str=output_str,
                error_str=error_str,
                judge_result=JudgeResult.RUNTIME_ERROR,
                message="Runtime error.",
                absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
                execution_time=execution_time_host,
                memory_usage=0,
            )
    elif profiles_content.startswith("Command terminated by signal 9"):
        profiles_content = profiles_content.split("\n", 1)[1]
        is_tle = True
    elif profiles_content.startswith("Command exited with non-zero status"):
        profiles_content = profiles_content.split("\n", 1)[1]
    profiles_content = profiles_content.strip()
    try:
        profiles_dict = json.loads(profiles_content)
    except json.JSONDecodeError:
        return CaseResult(
            input_str=input_str,
            output_str=output_str,
            error_str=error_str,
            judge_result=JudgeResult.WRONG_ANSWER,
            message="Wrong answer.",
            absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
            execution_time=min(execution_time_host, time_limit + 0.1),
            memory_usage=0,
        )
    try:
        profiles = Profiles(**profiles_dict)
    except ValueError:
        return CaseResult(
            input_str=input_str,
            output_str=output_str,
            error_str=error_str,
            judge_result=JudgeResult.INTERNAL_ERROR,
            message="Internal Error: Invalid profiles format.",
            absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
            execution_time=min(execution_time_host, time_limit + 0.1),
            memory_usage=0,
        )
    exit_status = profiles.exit_status
    execution_time = max(profiles.elapsed_time_seconds, profiles.user_cpu_seconds + profiles.system_cpu_seconds)
    memory_usage = profiles.max_resident_set_size_kbytes * 1024
    if exit_status != 0:
        return CaseResult(
            input_str=input_str,
            output_str=output_str,
            error_str=error_str,
            judge_result=JudgeResult.RUNTIME_ERROR,
            message="Runtime error.",
            absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
            execution_time=min(execution_time, time_limit + 0.1),
            memory_usage=memory_usage,
        )
    elif execution_time > time_limit or is_tle:
        return CaseResult(
            input_str=input_str,
            output_str=output_str,
            error_str=error_str,
            judge_result=JudgeResult.TIME_LIMIT_EXCEEDED,
            message="Time limit exceeded.",
            absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
            execution_time=min(execution_time, time_limit + 0.1),
            memory_usage=memory_usage,
        )
    elif memory_usage > memory_limit:
        return CaseResult(
            input_str=input_str,
            output_str=output_str,
            error_str=error_str,
            judge_result=JudgeResult.MEMORY_LIMIT_EXCEEDED,
            message="Memory limit exceeded.",
            absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
            execution_time=execution_time,
            memory_usage=memory_usage,
        )
    return execution_time, memory_usage


def _parse_judge_stderr(stderr: str, execution_time_host: float, input_str, output_str, error_str) -> CaseResult | int:
    """Parse judge stderr to extract score (shared by Docker and Modal paths)."""
    if not stderr.strip():
        return CaseResult(
            input_str=input_str,
            output_str=output_str,
            error_str=error_str,
            judge_result=JudgeResult.WRONG_ANSWER,
            message="Wrong answer.\nStandard error is empty (no score found)",
            absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
            execution_time=execution_time_host,
            memory_usage=0,
        )
    if "wrong answer: " in stderr:
        error_message = stderr.split("wrong answer: ")[1]
        return CaseResult(
            input_str=input_str,
            output_str=output_str,
            error_str=error_str,
            judge_result=JudgeResult.WRONG_ANSWER,
            message=f"Wrong answer.\n{error_message}",
            absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
            execution_time=execution_time_host,
            memory_usage=0,
        )
    stderr_last_line = stderr.splitlines()[-1]
    score_match = re.match(r"Score = (\d+)", stderr_last_line)
    if score_match is None:
        return CaseResult(
            input_str=input_str,
            output_str=output_str,
            error_str=error_str,
            judge_result=JudgeResult.WRONG_ANSWER,
            message=f"Wrong answer.\nStandard error:\n{stderr}",
            absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
            execution_time=execution_time_host,
            memory_usage=0,
        )
    return int(score_match.group(1))


# ===== Modal-specific implementations =====

def _compile_modal(
    code: str,
    code_language: CodeLanguage,
    judge_version: JudgeVersion,
    backend: ModalBackend,
) -> CaseResult | None:
    """Compile code in Modal sandbox. Returns CaseResult on failure, None on success."""
    submission_rel = get_submission_file_path(code_language, judge_version)
    submission_path = f"{ale_bench.constants.WORK_DIR}/{submission_rel}"
    backend.write_file(submission_path, code)

    object_rel = get_object_file_path(code_language, judge_version)
    object_path = f"/tmp/{object_rel}"
    backend.write_file(object_path, "")

    compile_cmd = get_compile_command(code_language, judge_version)
    compile_cmd += f"; cp {ale_bench.constants.WORK_DIR}/{object_rel} /tmp/{object_rel}"
    compile_cmd += f"; chmod 744 /tmp/{object_rel}"

    exit_code, _, stderr = backend.exec_command(
        compile_cmd, workdir=ale_bench.constants.WORK_DIR, timeout=ale_bench.constants.COMPILE_TIMEOUT
    )

    object_size = backend.file_size(object_path)
    if any([
        exit_code != 0,
        code_language != CodeLanguage.PYTHON and object_size == 0,
        code_language == CodeLanguage.PYTHON and "SyntaxError" in stderr,
    ]):
        return CaseResult(
            input_str=None,
            output_str=None,
            error_str=None,
            judge_result=JudgeResult.COMPILATION_ERROR,
            message=f"Failed to compile the code.\nStandard error:\n{stderr.strip()}",
            absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
            execution_time=0.0,
            memory_usage=0,
        )
    return None


def _case_iter_func_modal(
    problem_id: str,
    time_limit: float,
    memory_limit: int,
    problem_type: ProblemType,
    case_idx: int,
    input_str: str,
    code_language: CodeLanguage,
    judge_version: JudgeVersion,
    tool_dir: Path,
    return_details: bool,
    skip_local_visualization: bool,
    batch_run_command: str,
    batch_judge_command: str,
    reactive_judge_command: str,
    vis_command: str,
    backend: ModalBackend,
) -> CaseResult:
    """Run a single case using Modal backend primitives."""
    result_input_str = input_str if return_details else None
    execution_time_host = -1.0

    if problem_type == ProblemType.BATCH:
        # Write input + output/profiles placeholders in single round-trip
        backend.write_files({
            ale_bench.constants.INPUT_FILE: input_str,
            ale_bench.constants.OUTPUT_FILE: "",
            ale_bench.constants.PROFILES_FILE: "",
        })

        # Run
        start_at = time.perf_counter()
        exit_code, _, run_stderr = backend.exec_command(
            batch_run_command, workdir=ale_bench.constants.WORK_DIR
        )
        end_at = time.perf_counter()
        execution_time_host = end_at - start_at
        run_stderr = run_stderr.strip()

        if exit_code != 0:
            if execution_time_host > time_limit:
                return CaseResult(
                    input_str=result_input_str,
                    output_str=None,
                    error_str=run_stderr if result_input_str is not None else None,
                    judge_result=JudgeResult.TIME_LIMIT_EXCEEDED,
                    message="Time limit exceeded.",
                    absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
                    execution_time=min(execution_time_host, time_limit + 0.1),
                    memory_usage=0,
                )
            else:
                return CaseResult(
                    input_str=result_input_str,
                    output_str=None,
                    error_str=run_stderr if result_input_str is not None else None,
                    judge_result=JudgeResult.RUNTIME_ERROR,
                    message="Runtime error.",
                    absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
                    execution_time=execution_time_host,
                    memory_usage=0,
                )

        # Batch read output + profiles in single round-trip
        if return_details:
            output_and_profiles = backend.read_files([ale_bench.constants.OUTPUT_FILE, ale_bench.constants.PROFILES_FILE])
            result_output_str = output_and_profiles[0]
            profiles_content = output_and_profiles[1]
        else:
            result_output_str = None
            profiles_content = backend.read_file(ale_bench.constants.PROFILES_FILE)
        result_error_str = run_stderr if return_details else None
        profiles_result = parse_profiles(
            time_limit, memory_limit, profiles_content,
            execution_time_host, result_input_str, result_output_str, result_error_str,
        )
        if isinstance(profiles_result, CaseResult):
            return profiles_result
        execution_time, memory_usage = profiles_result

        # Judge - tool links already set up, just run
        judge_exit, _, judge_stderr = backend.exec_command(
            batch_judge_command, workdir=ale_bench.constants.WORK_DIR
        )
        judge_stderr = judge_stderr.strip()

        if judge_exit != 0:
            return CaseResult(
                input_str=result_input_str,
                output_str=result_output_str,
                error_str=result_error_str,
                judge_result=JudgeResult.WRONG_ANSWER,
                message=f"Wrong answer.\nStandard error:\n{judge_stderr}",
                absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
                execution_time=execution_time_host,
                memory_usage=0,
            )

        judge_result = _parse_judge_stderr(judge_stderr, execution_time_host, result_input_str, result_output_str, result_error_str)
        if isinstance(judge_result, CaseResult):
            return judge_result
        absolute_score = judge_result

    elif problem_type == ProblemType.REACTIVE:
        # Write input + output/profiles placeholders in single round-trip
        backend.write_files({
            ale_bench.constants.INPUT_FILE: input_str,
            ale_bench.constants.OUTPUT_FILE: "",
            ale_bench.constants.PROFILES_FILE: "",
        })

        start_at = time.perf_counter()
        exit_code, _, judge_stderr = backend.exec_command(
            reactive_judge_command, workdir=ale_bench.constants.WORK_DIR
        )
        end_at = time.perf_counter()
        execution_time_host = end_at - start_at
        judge_stderr = judge_stderr.strip()

        wo_profile_result = None
        if exit_code != 0 or judge_stderr == "":
            if execution_time_host > time_limit:
                wo_profile_result = CaseResult(
                    input_str=result_input_str,
                    output_str=None,
                    error_str=judge_stderr if result_input_str is not None else None,
                    judge_result=JudgeResult.TIME_LIMIT_EXCEEDED,
                    message="Time limit exceeded.",
                    absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
                    execution_time=min(execution_time_host, time_limit + 0.1),
                    memory_usage=0,
                )
            else:
                wo_profile_result = CaseResult(
                    input_str=result_input_str,
                    output_str=None,
                    error_str=judge_stderr if result_input_str is not None else None,
                    judge_result=JudgeResult.RUNTIME_ERROR,
                    message="Runtime error.",
                    absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
                    execution_time=execution_time_host,
                    memory_usage=0,
                )
            result_output_str = wo_profile_result.output_str
            result_error_str = wo_profile_result.error_str
        else:
            stderr_last_line = judge_stderr.splitlines()[-1]
            score_match = re.match(r"Score = (\d+)", stderr_last_line)
            if score_match is None:
                output_content = backend.read_file(ale_bench.constants.OUTPUT_FILE) if return_details else None
                return CaseResult(
                    input_str=result_input_str,
                    output_str=output_content,
                    error_str=judge_stderr if result_input_str is not None else None,
                    judge_result=JudgeResult.WRONG_ANSWER,
                    message="Wrong answer.",
                    absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
                    execution_time=execution_time_host,
                    memory_usage=0,
                )
            absolute_score = int(score_match.group(1))
            result_output_str = backend.read_file(ale_bench.constants.OUTPUT_FILE) if return_details else None
            result_error_str = judge_stderr if return_details else None

        # Parse profiles
        profiles_content = backend.read_file(ale_bench.constants.PROFILES_FILE)
        profiles_result = parse_profiles(
            time_limit, memory_limit, profiles_content,
            execution_time_host, result_input_str, result_output_str, result_error_str,
        )
        if isinstance(profiles_result, CaseResult):
            return profiles_result
        execution_time, memory_usage = profiles_result

        if wo_profile_result is not None:
            return CaseResult(
                input_str=wo_profile_result.input_str,
                output_str=wo_profile_result.output_str,
                error_str=wo_profile_result.error_str,
                judge_result=wo_profile_result.judge_result,
                message=wo_profile_result.message,
                absolute_score=wo_profile_result.absolute_score,
                execution_time=execution_time,
                memory_usage=memory_usage,
            )
    else:
        raise ValueError(f"Invalid problem type: {problem_type}")

    # Visualization
    local_visualization = None
    if not skip_local_visualization and problem_id not in ale_bench.constants.NO_LOCAL_VIS:
        vis_exit, _, vis_stderr = backend.exec_command(
            vis_command, workdir=ale_bench.constants.WORK_DIR, timeout=ale_bench.constants.VISUALIZE_TIMEOUT
        )
        if vis_exit != 0:
            raise RuntimeError("Failed to run the visualization command. Something went wrong.")

        # Determine which vis output path to read
        vis_container_path = (
            ale_bench.constants.LOCAL_VIS_SVG
            if problem_id in ale_bench.constants.VIS_SVG_GENERATION
            else ale_bench.constants.LOCAL_VIS_HTML
        )
        svg_text = backend.read_file(vis_container_path)
        svg_text = svg_text.replace("\n", "").removeprefix("<html><body>").removesuffix("</body></html>")
        if svg_text == "":
            raise RuntimeError("The local visualization file is empty. Something went wrong.")
        local_visualization = read_svg(svg_text)

    return CaseResult(
        input_str=result_input_str,
        output_str=result_output_str,
        error_str=result_error_str,
        judge_result=JudgeResult.ACCEPTED,
        message="",
        absolute_score=absolute_score,
        local_visualization=local_visualization,
        execution_time=execution_time,
        memory_usage=memory_usage,
    )


# ===== Docker path (unchanged) =====

def case_iter_func(
    problem_id: str,
    time_limit: float,
    memory_limit: int,
    problem_type: ProblemType,
    case_idx: int,
    input_str: str,
    code_language: CodeLanguage,
    judge_version: JudgeVersion,
    temp_dir: Path,
    tool_dir: Path,
    return_details: bool,
    skip_local_visualization: bool,
    host_paths_compile: HostPathsCompile,
    batch_run_command: str,
    batch_judge_command: str,
    reactive_judge_command: str,
    vis_command: str,
    backend=None,
) -> CaseResult:
    result_input_str = input_str if return_details else None
    host_paths_judge: HostPathsBatchJudge | HostPathsReactiveJudge
    execution_time_host = -1.0

    if problem_type == ProblemType.BATCH:
        host_paths_run = setup_paths_batch_run(host_paths_compile, temp_dir, input_str, f"{problem_id}_{case_idx:06d}_")
        run_volumes = get_batch_run_volumes(host_paths_run, temp_dir)
        run_result = run_batch_run_container(
            code_language, judge_version, time_limit, run_volumes, batch_run_command, result_input_str, backend
        )
        if isinstance(run_result, CaseResult):
            return run_result
        assert isinstance(run_result, tuple), "Run result must be a tuple"
        execution_time_host, stderr = run_result
        result_output_str = host_paths_run.output_file.read_text() if return_details else None
        result_error_str = stderr if return_details else None
        profiles_content = host_paths_run.profiles_file.read_text()
        profiles_result = parse_profiles(
            time_limit, memory_limit, profiles_content,
            execution_time_host, result_input_str, result_output_str, result_error_str,
        )
        if isinstance(profiles_result, CaseResult):
            return profiles_result
        assert isinstance(profiles_result, tuple), "Profiles result must be a tuple"
        execution_time, memory_usage = profiles_result
        host_paths_judge = setup_paths_batch_judge(host_paths_run)
        judge_volumes = get_batch_judge_volumes(host_paths_judge, tool_dir)
        batch_judge_result = run_batch_judge_container(
            judge_volumes, batch_judge_command, execution_time_host,
            result_input_str, result_output_str, result_error_str, backend,
        )
        if isinstance(batch_judge_result, CaseResult):
            return batch_judge_result
        assert isinstance(batch_judge_result, int), "Judge result must be an integer"
        absolute_score = batch_judge_result
    elif problem_type == ProblemType.REACTIVE:
        host_paths_judge = setup_paths_reactive_judge(
            host_paths_compile, temp_dir, input_str, f"{problem_id}_{case_idx:06d}_",
        )
        judge_volumes = get_reactive_judge_volumes(host_paths_judge, temp_dir, tool_dir)
        reactive_judge_result = run_reactive_judge_container(
            code_language, judge_version, time_limit, judge_volumes,
            reactive_judge_command, result_input_str,
            host_paths_judge.output_file if return_details else None, backend,
        )
        wo_profile_result = None
        if isinstance(reactive_judge_result, CaseResult):
            wo_profile_result = reactive_judge_result
            execution_time_host = reactive_judge_result.execution_time
            result_output_str = reactive_judge_result.output_str
            result_error_str = reactive_judge_result.error_str
        else:
            assert isinstance(reactive_judge_result, tuple), "Judge result must be a tuple"
            execution_time_host, absolute_score, stderr = reactive_judge_result
            result_output_str = host_paths_judge.output_file.read_text() if return_details else None
            result_error_str = stderr if return_details else None
        profiles_content = host_paths_judge.profiles_file.read_text()
        profiles_result = parse_profiles(
            time_limit, memory_limit, profiles_content,
            execution_time_host, result_input_str, result_output_str, result_error_str,
        )
        if isinstance(profiles_result, CaseResult):
            return profiles_result
        assert isinstance(profiles_result, tuple), "Profiles result must be a tuple"
        execution_time, memory_usage = profiles_result
        if wo_profile_result is not None:
            return CaseResult(
                input_str=wo_profile_result.input_str,
                output_str=wo_profile_result.output_str,
                error_str=wo_profile_result.error_str,
                judge_result=wo_profile_result.judge_result,
                message=wo_profile_result.message,
                absolute_score=wo_profile_result.absolute_score,
                execution_time=execution_time,
                memory_usage=memory_usage,
            )
    else:
        raise ValueError(f"Invalid problem type: {problem_type}")

    local_visualization = None
    if not skip_local_visualization and problem_id not in ale_bench.constants.NO_LOCAL_VIS:
        host_paths_vis = setup_paths_vis(host_paths_judge, temp_dir, problem_id, f"{problem_id}_{case_idx:06d}_")
        vis_volumes = get_vis_volumes(host_paths_vis, tool_dir)
        run_vis_container(vis_command, vis_volumes, backend)
        svg_text = host_paths_vis.local_visualization_file.read_text()
        svg_text = svg_text.replace("\n", "").removeprefix("<html><body>").removesuffix("</body></html>")
        if svg_text == "":
            raise RuntimeError("The local visualization file is empty. Something went wrong.")
        local_visualization = read_svg(svg_text)

    return CaseResult(
        input_str=result_input_str,
        output_str=result_output_str,
        error_str=result_error_str,
        judge_result=JudgeResult.ACCEPTED,
        message="",
        absolute_score=absolute_score,
        local_visualization=local_visualization,
        execution_time=execution_time,
        memory_usage=memory_usage,
    )


def _run_all_cases_single_exec(
    inputs: list[str],
    run_command: str,
    judge_command: str,
    time_limit: float,
    memory_limit: int,
    return_details: bool,
    backend: ModalBackend,
) -> list[dict]:
    """Run all BATCH cases in a single sandbox exec (1 round-trip for N cases).

    Creates per-case directories, writes all inputs, runs all cases + judges
    inside a single shell script, and returns JSON results.
    """
    import json as _json

    n = len(inputs)

    # Write all input files in one round-trip using per-case dirs
    files_to_write = {}
    for i, inp in enumerate(inputs):
        case_dir = f"/tmp/cases/{i:06d}"
        files_to_write[f"{case_dir}/input.txt"] = inp
        files_to_write[f"{case_dir}/output.txt"] = ""
        files_to_write[f"{case_dir}/profiles.json"] = ""
    backend.write_files(files_to_write)

    # Build a shell script that runs all cases sequentially inside the sandbox
    # For each case: substitute paths, run, judge, collect results
    # We use per-case dirs to avoid file conflicts
    inp_path = ale_bench.constants.INPUT_FILE    # /tmp/input.txt
    out_path = ale_bench.constants.OUTPUT_FILE   # /tmp/output.txt
    prof_path = ale_bench.constants.PROFILES_FILE  # /tmp/profiles.json

    # The run_command and judge_command reference the constant paths.
    # We'll copy per-case input into the constant path, run, then copy results back.
    script_lines = [
        '#!/bin/sh',
        'echo "["',
    ]
    for i in range(n):
        case_dir = f"/tmp/cases/{i:06d}"
        comma = ',' if i > 0 else ''
        script_lines.append(f'# Case {i}')
        script_lines.append(f'cp {case_dir}/input.txt {inp_path}')
        script_lines.append(f': > {out_path}')
        script_lines.append(f': > {prof_path}')
        # Run the program
        script_lines.append(f'START_NS=$(date +%s%N 2>/dev/null || echo 0)')
        script_lines.append(f'cd {ale_bench.constants.WORK_DIR} && {run_command}')
        script_lines.append(f'RUN_EXIT=$?')
        script_lines.append(f'END_NS=$(date +%s%N 2>/dev/null || echo 0)')
        # Judge
        script_lines.append(f'JUDGE_STDERR=$({judge_command} 2>&1 1>/dev/null)')
        script_lines.append(f'JUDGE_EXIT=$?')
        # Copy results back to per-case dir
        script_lines.append(f'cp {out_path} {case_dir}/output.txt')
        script_lines.append(f'cp {prof_path} {case_dir}/profiles.json')
        # Output JSON for this case
        script_lines.append(
            f'echo "{comma}{{' +
            f'\\"run_exit\\": $RUN_EXIT, ' +
            f'\\"judge_exit\\": $JUDGE_EXIT, ' +
            f'\\"start_ns\\": $START_NS, ' +
            f'\\"end_ns\\": $END_NS, ' +
            f'\\"judge_stderr\\": \\"$(echo "$JUDGE_STDERR" | tail -1 | sed \'s/"/\\\\\\\\"/g\')\\"' +
            f'}}"'
        )
    script_lines.append('echo "]"')
    script = '\n'.join(script_lines)

    # Write and execute the script in one exec
    backend.write_file("/tmp/_run_all.sh", script)
    exit_code, stdout, stderr = backend.exec_command(
        "sh /tmp/_run_all.sh",
        workdir=ale_bench.constants.WORK_DIR,
        timeout=int((time_limit + 5) * n + 60),
    )

    # Parse the JSON output
    try:
        raw_results = _json.loads(stdout)
    except _json.JSONDecodeError:
        logger.error(f"Failed to parse batch results JSON. stdout={stdout[:500]}, stderr={stderr[:500]}")
        raise RuntimeError(f"Failed to parse batch results: {stdout[:200]}")

    # Batch read all profiles and (optionally) outputs
    profiles_paths = [f"/tmp/cases/{i:06d}/profiles.json" for i in range(n)]
    if return_details:
        output_paths = [f"/tmp/cases/{i:06d}/output.txt" for i in range(n)]
        all_reads = backend.read_files(profiles_paths + output_paths)
        profiles_contents = all_reads[:n]
        output_contents = all_reads[n:]
    else:
        profiles_contents = backend.read_files(profiles_paths)
        output_contents = [None] * n

    # Assemble results
    results = []
    for i in range(n):
        results.append({
            "run_exit": raw_results[i]["run_exit"],
            "judge_exit": raw_results[i]["judge_exit"],
            "judge_stderr": raw_results[i]["judge_stderr"],
            "start_ns": raw_results[i].get("start_ns", 0),
            "end_ns": raw_results[i].get("end_ns", 0),
            "profiles": profiles_contents[i],
            "output": output_contents[i],
        })
    return results


def _run_cases_modal(
    inputs: list[str],
    code: str,
    code_language: CodeLanguage,
    judge_version: JudgeVersion,
    time_limit: float,
    memory_limit: int,
    problem_id: str,
    problem_type: ProblemType,
    tool_dir: Path,
    return_details: bool,
    skip_local_visualization: bool,
    backend: ModalBackend,
) -> list[CaseResult]:
    """Run cases using Modal backend primitives (no local temp files)."""
    # Setup tool links
    backend.setup_tool_links(str(tool_dir))

    # Compile
    compile_result = _compile_modal(code, code_language, judge_version, backend)
    if compile_result is not None:
        return [compile_result for _ in inputs]

    # Build commands
    batch_run_command = build_batch_run_command(code_language, judge_version, time_limit)
    batch_judge_command = build_batch_judge_command()
    reactive_judge_command = build_reactive_judge_command(code_language, judge_version, time_limit)
    vis_command = build_vis_command()

    # For BATCH problems with multiple cases and no vis, use the optimized single-exec path
    if problem_type == ProblemType.BATCH and len(inputs) > 1 and skip_local_visualization:
        return _run_cases_modal_batch_optimized(
            inputs, time_limit, memory_limit, problem_id,
            return_details, batch_run_command, batch_judge_command, backend,
        )

    # Fallback: sequential per-case execution (for REACTIVE, single case, or vis)
    case_results = []
    for case_idx, input_str in enumerate(inputs):
        case_result = _case_iter_func_modal(
            problem_id, time_limit, memory_limit, problem_type,
            case_idx, input_str, code_language, judge_version,
            tool_dir, return_details, skip_local_visualization,
            batch_run_command, batch_judge_command, reactive_judge_command,
            vis_command, backend,
        )
        case_results.append(case_result)

    return case_results


def _run_cases_modal_batch_optimized(
    inputs: list[str],
    time_limit: float,
    memory_limit: int,
    problem_id: str,
    return_details: bool,
    run_command: str,
    judge_command: str,
    backend: ModalBackend,
) -> list[CaseResult]:
    """Optimized batch case runner: all cases in ~3 round-trips instead of N×4."""
    raw_results = _run_all_cases_single_exec(
        inputs, run_command, judge_command,
        time_limit, memory_limit, return_details, backend,
    )

    case_results = []
    for i, (input_str, raw) in enumerate(zip(inputs, raw_results)):
        result_input_str = input_str if return_details else None
        result_output_str = raw["output"] if return_details else None
        result_error_str = None

        # Calculate execution time from nanosecond timestamps
        try:
            start_ns = int(raw["start_ns"])
            end_ns = int(raw["end_ns"])
            execution_time_host = (end_ns - start_ns) / 1e9 if end_ns > start_ns else 0.0
        except (ValueError, TypeError):
            execution_time_host = 0.0

        run_exit = raw["run_exit"]
        if run_exit != 0:
            if execution_time_host > time_limit:
                case_results.append(CaseResult(
                    input_str=result_input_str, output_str=None,
                    error_str=None,
                    judge_result=JudgeResult.TIME_LIMIT_EXCEEDED,
                    message="Time limit exceeded.",
                    absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
                    execution_time=min(execution_time_host, time_limit + 0.1),
                    memory_usage=0,
                ))
                continue
            else:
                case_results.append(CaseResult(
                    input_str=result_input_str, output_str=None,
                    error_str=None,
                    judge_result=JudgeResult.RUNTIME_ERROR,
                    message="Runtime error.",
                    absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
                    execution_time=execution_time_host,
                    memory_usage=0,
                ))
                continue

        # Parse profiles
        profiles_result = parse_profiles(
            time_limit, memory_limit, raw["profiles"],
            execution_time_host, result_input_str, result_output_str, result_error_str,
        )
        if isinstance(profiles_result, CaseResult):
            case_results.append(profiles_result)
            continue
        execution_time, memory_usage = profiles_result

        # Parse judge result
        judge_stderr = raw["judge_stderr"]
        if raw["judge_exit"] != 0:
            case_results.append(CaseResult(
                input_str=result_input_str, output_str=result_output_str,
                error_str=result_error_str,
                judge_result=JudgeResult.WRONG_ANSWER,
                message=f"Wrong answer.\nStandard error:\n{judge_stderr}",
                absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
                execution_time=execution_time_host, memory_usage=0,
            ))
            continue

        judge_result = _parse_judge_stderr(
            judge_stderr, execution_time_host,
            result_input_str, result_output_str, result_error_str,
        )
        if isinstance(judge_result, CaseResult):
            case_results.append(judge_result)
            continue
        absolute_score = judge_result

        case_results.append(CaseResult(
            input_str=result_input_str,
            output_str=result_output_str,
            error_str=result_error_str,
            judge_result=JudgeResult.ACCEPTED,
            message="",
            absolute_score=absolute_score,
            execution_time=execution_time,
            memory_usage=memory_usage,
        ))

    return case_results


def _run_cases_docker(
    inputs: list[str],
    code: str,
    code_language: CodeLanguage,
    judge_version: JudgeVersion,
    time_limit: float,
    memory_limit: int,
    problem_id: str,
    problem_type: ProblemType,
    tool_dir: Path,
    return_details: bool,
    skip_local_visualization: bool,
    num_workers: int,
    backend: Backend,
) -> list[CaseResult]:
    """Run cases using Docker backend (existing local temp file approach)."""
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        host_paths_compile = setup_paths_compile(temp_dir, code, code_language, judge_version)
        compile_volumes = get_compile_volumes(host_paths_compile, temp_dir)
        compile_command = build_compile_command(
            code_language, judge_version, host_paths_compile.object_file.relative_to(temp_dir)
        )
        batch_run_command = build_batch_run_command(code_language, judge_version, time_limit)
        batch_judge_command = build_batch_judge_command()
        reactive_judge_command = build_reactive_judge_command(code_language, judge_version, time_limit)
        vis_command = build_vis_command()

        compile_result = run_compile_container(
            code_language, judge_version, host_paths_compile,
            compile_volumes, compile_command, backend,
        )
        if compile_result is not None:
            return [compile_result for _ in inputs]

        case_results: list[CaseResult] = []
        if len(inputs) == 1 or num_workers == 1:
            for case_idx, input_str in enumerate(inputs):
                case_result = case_iter_func(
                    problem_id, time_limit, memory_limit, problem_type,
                    case_idx, input_str, code_language, judge_version,
                    temp_dir, tool_dir, return_details, skip_local_visualization,
                    host_paths_compile, batch_run_command, batch_judge_command,
                    reactive_judge_command, vis_command, backend,
                )
                case_results.append(case_result)
        else:
            case_results = [
                CaseResult(
                    input_str=input_str if return_details else None,
                    output_str=None,
                    error_str=None,
                    judge_result=JudgeResult.INTERNAL_ERROR,
                    message="Internal Error: Unexpected error occurred.",
                    absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
                    execution_time=0.0,
                    memory_usage=0,
                )
                for input_str in inputs
            ]
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_case_idx = {}
                for case_idx, input_str in enumerate(inputs):
                    future = executor.submit(
                        case_iter_func,
                        problem_id, time_limit, memory_limit, problem_type,
                        case_idx, input_str, code_language, judge_version,
                        temp_dir, tool_dir, return_details, skip_local_visualization,
                        host_paths_compile, batch_run_command, batch_judge_command,
                        reactive_judge_command, vis_command,
                    )
                    future_to_case_idx[future] = case_idx
                for future in as_completed(future_to_case_idx):
                    case_idx = future_to_case_idx[future]
                    try:
                        case_result = future.result()
                    except Exception as e:
                        case_result = CaseResult(
                            input_str=inputs[case_idx] if return_details else None,
                            output_str=None,
                            error_str=None,
                            judge_result=JudgeResult.INTERNAL_ERROR,
                            message=f"Internal Error: {e}",
                            absolute_score=ale_bench.constants.REJECTED_ABSOLUTE_SCORE,
                            execution_time=0.0,
                            memory_usage=0,
                        )
                    case_results[case_idx] = case_result

    return case_results


def run_cases(
    inputs: list[str],
    code: str,
    code_language: CodeLanguage,
    judge_version: JudgeVersion,
    time_limit: float,
    memory_limit: int,
    problem_id: str,
    problem_type: ProblemType,
    tool_dir: Path,
    return_details: bool,
    skip_local_visualization: bool,
    num_workers: int,
    backend: Backend,
) -> list[CaseResult]:
    """Run the cases for the given inputs and code.

    Args:
        inputs (list[str]): The list of inputs.
        code (str): The code to run.
        code_language (CodeLanguage): The code language.
        judge_version (JudgeVersion): The judge version.
        time_limit (float): The time limit in seconds.
        memory_limit (int): The memory limit in bytes.
        problem_id (str): The problem ID.
        problem_type (ProblemType): The problem type.
        tool_dir (Path): The directory of the tools.
        return_details (bool): Whether to return detailed results.
        skip_local_visualization (bool): Whether to skip local visualization.
        num_workers (int): The number of workers for running cases.
        backend (Backend): Execution backend to use.

    Returns:
        list[CaseResult]: The list of case results.
    """
    if isinstance(backend, ModalBackend):
        return _run_cases_modal(
            inputs, code, code_language, judge_version,
            time_limit, memory_limit, problem_id, problem_type,
            tool_dir, return_details, skip_local_visualization, backend,
        )
    else:
        return _run_cases_docker(
            inputs, code, code_language, judge_version,
            time_limit, memory_limit, problem_id, problem_type,
            tool_dir, return_details, skip_local_visualization,
            num_workers, backend,
        )
