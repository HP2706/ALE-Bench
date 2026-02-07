import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

import ale_bench.constants
from ale_bench.backends import Backend
from ale_bench.backends.local_backend import LocalBackend
from ale_bench.backends.modal_backend import ModalBackend
from ale_bench.error import AleBenchError


class HostPathsGen(BaseModel):
    """Paths on the host for the generator tool."""

    model_config = ConfigDict(frozen=True)

    seeds_file: Path = Field(description="The seeds file")
    input_dir: Path = Field(description="The directory for the generated input cases")


def setup_paths_gen(temp_dir: Path, seeds: list[int]) -> HostPathsGen:
    """Setup the paths for the generator tool."""
    seeds_file = temp_dir / ale_bench.constants.SEEDS_FILE.split("/")[-1]
    seeds_file.write_text("\n".join([str(seed) for seed in seeds]) + "\n")
    input_dir = temp_dir / ale_bench.constants.IN_DIR.split("/")[-1]
    input_dir.mkdir()
    return HostPathsGen(seeds_file=seeds_file, input_dir=input_dir)


def get_gen_volumes(host_paths: HostPathsGen, tool_dir: Path) -> dict[str, dict[str, str]]:
    """Get the volumes for the generator tool with the setup."""
    return {
        str(host_paths.seeds_file): {"bind": ale_bench.constants.SEEDS_FILE, "mode": "ro"},
        str(tool_dir / "tools" / "target" / "release" / "gen"): {"bind": ale_bench.constants.GEN_BIN, "mode": "ro"},
        str(host_paths.input_dir): {"bind": f"{ale_bench.constants.IN_DIR}", "mode": "rw"},
        str(host_paths.input_dir.parent): {"bind": ale_bench.constants.WORK_DIR, "mode": "rw"},
    }


def build_gen_command(gen_kwargs: dict[str, Any]) -> str:
    """Build the command for the generator tool."""
    gen_command = ale_bench.constants.GEN_BIN
    for key, value in gen_kwargs.items():
        if key == "dir":
            warnings.warn("`dir` is a reserved keyword and will be ignored.")
            continue
        gen_command += f" --{key}={value}"
    gen_command += f" {ale_bench.constants.SEEDS_FILE}"
    return gen_command


def run_gen_container(
    gen_volumes: dict[str, dict[str, str]],
    gen_command: str,
    timeout: int,
    backend: Backend,
) -> None:
    """Run the generator tool using the specified backend (Docker path)."""
    container = backend.run_container(
        image=ale_bench.constants.RUST_TOOL_DOCKER_IMAGE,
        command=f"/bin/sh -c '{gen_command}'",
        volumes=gen_volumes,
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
            container.wait(timeout=timeout)
            exit_code = container.attrs["State"]["ExitCode"]
        except Exception:
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8").strip()
            if len(stderr) > 0:
                raise AleBenchError(f"Failed to generate the case. The standard error is:\n{stderr}")
            else:
                raise AleBenchError(f"Failed to generate the case. Timeout after {timeout} seconds.")
    finally:
        container.remove(force=True)
    if exit_code != 0:
        raise AleBenchError("Failed to generate the case.")


def _generate_inputs_modal(seeds: list[int], gen_kwargs: dict[str, Any], tool_dir: Path, backend: ModalBackend) -> list[str]:
    """Generate input cases using Modal backend primitives (no local temp files).

    Retries once if the sandbox dies mid-operation.
    """
    """Implementation of Modal generate_inputs."""
    # Setup tool links so gen binary is at /judge/target/release/gen
    backend.setup_tool_links(str(tool_dir))
        
    # Write seeds file directly in sandbox
    seeds_content = "\n".join(str(s) for s in seeds) + "\n"
    backend.write_file(ale_bench.constants.SEEDS_FILE, seeds_content)

    # Clean and create working dirs (previous runs may have left files)
    backend.exec_command(f"rm -rf {ale_bench.constants.IN_DIR}", timeout=10)
    backend.mkdir(ale_bench.constants.IN_DIR)

    # Build and run command
    gen_command = build_gen_command(gen_kwargs)
    timeout = ale_bench.constants.GENERATION_TIMEOUT

    exit_code, stdout, stderr = backend.exec_command(
        gen_command, workdir=ale_bench.constants.WORK_DIR, timeout=timeout
    )
    if exit_code != 0:
        if stderr:
            raise AleBenchError(f"Failed to generate the case. The standard error is:\n{stderr}")
        else:
            raise AleBenchError("Failed to generate the case.")

    # Read results directly from sandbox (batch read: 1 round-trip for all files)
    input_files = backend.list_files(ale_bench.constants.IN_DIR, "*.txt")
    for idx, input_file in enumerate(input_files):
        filename = input_file.split("/")[-1]
        assert filename == f"{idx:04d}.txt", (
            "The generated case files must be named `0000.txt`, `0001.txt`, ..."
        )
    return backend.read_files(input_files)


def _generate_inputs_docker(seeds: list[int], gen_kwargs: dict[str, Any], tool_dir: Path, backend: Backend) -> list[str]:
    """Generate input cases using Docker backend (existing local temp file approach)."""
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        gen_host_paths = setup_paths_gen(temp_dir, seeds)
        gen_volumes = get_gen_volumes(gen_host_paths, tool_dir)
        gen_command = build_gen_command(gen_kwargs)

        timeout = ale_bench.constants.GENERATION_TIMEOUT
        run_gen_container(gen_volumes, gen_command, timeout, backend)

        input_files = sorted(list(gen_host_paths.input_dir.glob("*.txt")))
        generated_cases = []
        for idx, input_file in enumerate(input_files):
            assert input_file.name == f"{idx:04d}.txt", (
                "The generated case files must be named `0000.txt`, `0001.txt`, ..."
            )
            generated_cases.append(input_file.read_text())

    return generated_cases


def generate_inputs(seeds: list[int], gen_kwargs: dict[str, Any], tool_dir: Path, backend: Backend) -> list[str]:
    """Generate input cases using the generator tool.

    Args:
        seeds (list[int]): The list of seeds for the generation.
        gen_kwargs (dict[str, Any]): The keyword arguments for the generator tool.
        tool_dir (Path): The directory of the tools.
        backend (Backend): Execution backend to use.

    Returns:
        list[str]: The list of generated input cases.
    """
    if isinstance(backend, (ModalBackend, LocalBackend)):
        return _generate_inputs_modal(seeds, gen_kwargs, tool_dir, backend)
    else:
        return _generate_inputs_docker(seeds, gen_kwargs, tool_dir, backend)
