from __future__ import annotations

import enum
import json
import os
import tempfile
import time
from pathlib import Path

from requests.exceptions import ConnectionError, Timeout

import ale_bench.constants
from ale_bench.backends import Backend
from ale_bench.backends.local_backend import LocalBackend
from ale_bench.backends.modal_backend import ModalBackend
from ale_bench.code_language import (
    CodeLanguage,
    JudgeVersion,
    get_docker_image_name,
    get_object_file_path,
    get_submission_file_path,
)
from ale_bench.result import CodeRunResult, Profiles
from ale_bench.tool_wrappers.case_runner import (
    HostPathsCompile,
    build_batch_run_command,
    build_compile_command,
    get_batch_run_volumes,
    get_compile_volumes,
    setup_paths_batch_run,
    setup_paths_compile,
)


class ExitStatus(enum.IntEnum):
    """Exit status codes."""

    SUCCESS = 0
    COMPILE_ERROR = -1
    RUNTIME_ERROR = 1
    TIME_LIMIT_EXCEEDED = 9
    MEMORY_LIMIT_EXCEEDED = 9


def run_compile_container(
    code_language: CodeLanguage,
    judge_version: JudgeVersion,
    host_paths_compile: HostPathsCompile,
    compile_volumes: dict[str, dict[str, str]],
    compile_command: str,
    backend: Backend,
) -> CodeRunResult | None:
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
                return CodeRunResult(
                    stdin="",
                    stdout="",
                    stderr=f"Compilation timed out ({ale_bench.constants.COMPILE_TIMEOUT}s).",
                    exit_status=ExitStatus.COMPILE_ERROR.value,
                    execution_time=0.0,
                    memory_usage=0,
                )
        except Exception:
            return CodeRunResult(
                stdin="",
                stdout="",
                stderr="Failed to compile the code due to an unexpected error.",
                exit_status=ExitStatus.COMPILE_ERROR.value,
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
        return CodeRunResult(
            stdin="",
            stdout="",
            stderr=stderr,
            exit_status=ExitStatus.COMPILE_ERROR.value,
            execution_time=0.0,
            memory_usage=0,
        )
    return None


def run_run_container(
    code_language: CodeLanguage,
    judge_version: JudgeVersion,
    time_limit: float,
    run_volumes: dict[str, dict[str, str]],
    run_command: str,
    stdin: str,
    backend: Backend,
) -> CodeRunResult | tuple[float, str]:
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
            return CodeRunResult(
                stdin=stdin,
                stdout="",
                stderr=stderr,
                exit_status=ExitStatus.TIME_LIMIT_EXCEEDED.value,
                execution_time=min(execution_time_host, time_limit + 0.1),
                memory_usage=0,
            )
        else:
            return CodeRunResult(
                stdin=stdin,
                stdout="",
                stderr=stderr,
                exit_status=exit_code,
                execution_time=execution_time_host,
                memory_usage=0,
            )
    return execution_time_host, stderr


def parse_profiles(
    time_limit: float,
    memory_limit: int,
    profiles_content: str,
    execution_time_host: float,
    stdin: str,
    stdout: str,
    stderr: str,
) -> CodeRunResult | tuple[float, int]:
    """Parse the profiles content and check for time limit, memory limit, and exit status."""
    assert execution_time_host >= 0.0, "execution_time_host must be non-negative"
    is_tle = False
    if profiles_content == "":
        if execution_time_host > time_limit:
            return CodeRunResult(
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                exit_status=ExitStatus.TIME_LIMIT_EXCEEDED.value,
                execution_time=min(execution_time_host, time_limit + 0.1),
                memory_usage=0,
            )
        else:
            return CodeRunResult(
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                exit_status=ExitStatus.RUNTIME_ERROR.value,
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
        return CodeRunResult(
            stdin=stdin,
            stdout=stdout,
            stderr=f"Failed to parse profiles.\nStandard error:\n{stderr}",
            exit_status=ExitStatus.RUNTIME_ERROR.value,
            execution_time=min(execution_time_host, time_limit + 0.1),
            memory_usage=0,
        )
    try:
        profiles = Profiles(**profiles_dict)
    except ValueError:
        return CodeRunResult(
            stdin=stdin,
            stdout=stdout,
            stderr=f"Invalid profiles format.\nStandard error:\n{stderr}",
            exit_status=ExitStatus.RUNTIME_ERROR.value,
            execution_time=min(execution_time_host, time_limit + 0.1),
            memory_usage=0,
        )
    exit_status = profiles.exit_status
    execution_time = max(profiles.elapsed_time_seconds, profiles.user_cpu_seconds + profiles.system_cpu_seconds)
    memory_usage = profiles.max_resident_set_size_kbytes * 1024
    if exit_status != 0:
        return CodeRunResult(
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            exit_status=exit_status,
            execution_time=min(execution_time, time_limit + 0.1),
            memory_usage=memory_usage,
        )
    elif execution_time > time_limit or is_tle:
        return CodeRunResult(
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            exit_status=ExitStatus.TIME_LIMIT_EXCEEDED.value,
            execution_time=min(execution_time, time_limit + 0.1),
            memory_usage=memory_usage,
        )
    elif memory_usage > memory_limit:
        return CodeRunResult(
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            exit_status=ExitStatus.MEMORY_LIMIT_EXCEEDED.value,
            execution_time=execution_time,
            memory_usage=memory_usage,
        )
    return execution_time, memory_usage


def _run_code_modal(
    *,
    code: str,
    code_language: CodeLanguage,
    judge_version: JudgeVersion,
    stdin: str,
    time_limit: float,
    memory_limit: int,
    backend: ModalBackend,
) -> CodeRunResult:
    """Run code using Modal backend primitives (no local temp files)."""
    from ale_bench.code_language import get_compile_command, get_run_command
    import math

    # Write code file
    submission_rel = get_submission_file_path(code_language, judge_version)
    submission_path = f"{ale_bench.constants.WORK_DIR}/{submission_rel}"
    backend.write_file(submission_path, code)

    # Object file path
    object_rel = get_object_file_path(code_language, judge_version)
    object_path = f"/tmp/{object_rel}"
    backend.write_file(object_path, "")

    # Compile
    compile_cmd = get_compile_command(code_language, judge_version)
    compile_cmd += f"; cp {ale_bench.constants.WORK_DIR}/{object_rel} /tmp/{object_rel}"
    compile_cmd += f"; chmod 744 /tmp/{object_rel}"

    exit_code, comp_stdout, comp_stderr = backend.exec_command(
        compile_cmd, workdir=ale_bench.constants.WORK_DIR, timeout=ale_bench.constants.COMPILE_TIMEOUT
    )

    # Check compilation
    object_size = backend.file_size(object_path)
    if any([
        exit_code != 0,
        code_language != CodeLanguage.PYTHON and object_size == 0,
        code_language == CodeLanguage.PYTHON and "SyntaxError" in comp_stderr,
    ]):
        return CodeRunResult(
            stdin="",
            stdout="",
            stderr=comp_stderr.strip(),
            exit_status=ExitStatus.COMPILE_ERROR.value,
            execution_time=0.0,
            memory_usage=0,
        )

    # Write input + output/profiles placeholders in single round-trip
    backend.write_files({
        ale_bench.constants.INPUT_FILE: stdin,
        ale_bench.constants.OUTPUT_FILE: "",
        ale_bench.constants.PROFILES_FILE: "",
    })

    # Build run command
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

    # Run
    start_at = time.perf_counter()
    exit_code, run_stdout, run_stderr = backend.exec_command(
        run_command, workdir=ale_bench.constants.WORK_DIR
    )
    end_at = time.perf_counter()
    execution_time_host = end_at - start_at

    if exit_code != 0:
        if execution_time_host > time_limit:
            return CodeRunResult(
                stdin=stdin,
                stdout="",
                stderr=run_stderr.strip(),
                exit_status=ExitStatus.TIME_LIMIT_EXCEEDED.value,
                execution_time=min(execution_time_host, time_limit + 0.1),
                memory_usage=0,
            )
        else:
            return CodeRunResult(
                stdin=stdin,
                stdout="",
                stderr=run_stderr.strip(),
                exit_status=exit_code,
                execution_time=execution_time_host,
                memory_usage=0,
            )

    # Batch read output + profiles in single round-trip
    stdout_content, profiles_content = backend.read_files([
        ale_bench.constants.OUTPUT_FILE, ale_bench.constants.PROFILES_FILE
    ])

    profiles_result = parse_profiles(
        time_limit,
        memory_limit,
        profiles_content,
        execution_time_host,
        stdin,
        stdout_content,
        run_stderr.strip(),
    )
    if isinstance(profiles_result, CodeRunResult):
        return profiles_result

    execution_time, memory_usage = profiles_result
    return CodeRunResult(
        stdin=stdin,
        stdout=stdout_content,
        stderr=run_stderr.strip(),
        exit_status=ExitStatus.SUCCESS.value,
        execution_time=execution_time,
        memory_usage=memory_usage,
    )


def _run_code_docker(
    *,
    code: str,
    code_language: CodeLanguage,
    judge_version: JudgeVersion,
    stdin: str,
    time_limit: float,
    memory_limit: int,
    backend: Backend,
) -> CodeRunResult:
    """Run code using Docker backend (existing local temp file approach)."""
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        # Compilation
        host_paths_compile = setup_paths_compile(temp_dir, code, code_language, judge_version)
        compile_volumes = get_compile_volumes(host_paths_compile, temp_dir)
        compile_command = build_compile_command(
            code_language, judge_version, host_paths_compile.object_file.relative_to(temp_dir)
        )
        compile_result = run_compile_container(
            code_language,
            judge_version,
            host_paths_compile,
            compile_volumes,
            compile_command,
            backend,
        )
        if compile_result is not None:
            return compile_result
        # Running
        host_paths_run = setup_paths_batch_run(host_paths_compile, temp_dir, stdin)
        run_volumes = get_batch_run_volumes(host_paths_run, temp_dir)
        run_command = build_batch_run_command(code_language, judge_version, time_limit)
        run_result = run_run_container(code_language, judge_version, time_limit, run_volumes, run_command, stdin, backend)
        if isinstance(run_result, CodeRunResult):
            return run_result
        assert isinstance(run_result, tuple), "Run result must be a tuple"
        execution_time_host, stderr = run_result
        stdout = host_paths_run.output_file.read_text()
        profiles_content = host_paths_run.profiles_file.read_text()
        profiles_result = parse_profiles(
            time_limit,
            memory_limit,
            profiles_content,
            execution_time_host,
            stdin,
            stdout,
            stderr,
        )
        if isinstance(profiles_result, CodeRunResult):
            return profiles_result
        assert isinstance(profiles_result, tuple), "Profiles result must be a tuple"
        execution_time, memory_usage = profiles_result

        return CodeRunResult(
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            exit_status=ExitStatus.SUCCESS.value,
            execution_time=execution_time,
            memory_usage=memory_usage,
        )


def run_code(
    *,
    code: str,
    code_language: CodeLanguage,
    judge_version: JudgeVersion,
    stdin: str,
    time_limit: float,
    memory_limit: int,
    backend: Backend,
) -> CodeRunResult:
    """Run the given code with the specified language and judge version.

    Args:
        code (str): The code to run.
        code_language (CodeLanguage): The code language.
        judge_version (JudgeVersion): The judge version.
        stdin (str): The input string to be provided to the program.
        time_limit (float): The time limit in seconds.
        memory_limit (int): The memory limit in bytes.
        backend (Backend): Execution backend to use.

    Returns:
        CodeRunResult: The result of the code execution.
    """
    if isinstance(backend, (ModalBackend, LocalBackend)):
        return _run_code_modal(
            code=code,
            code_language=code_language,
            judge_version=judge_version,
            stdin=stdin,
            time_limit=time_limit,
            memory_limit=memory_limit,
            backend=backend,
        )
    else:
        return _run_code_docker(
            code=code,
            code_language=code_language,
            judge_version=judge_version,
            stdin=stdin,
            time_limit=time_limit,
            memory_limit=memory_limit,
            backend=backend,
        )
