"""Docker backend implementation for ALE-Bench."""

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import docker

from .base import Backend

logger = logging.getLogger(__name__)


class DockerBackend(Backend):
    """Docker-based execution backend."""

    def __init__(self):
        """Initialize Docker backend with a Docker client."""
        self.client = docker.from_env()
        logger.info("[DOCKER] Docker backend initialized")

    def build_tools(self, problem_id: str, tool_dir: Path) -> None:
        """Build Rust tools using Docker container."""
        logger.info(f"[DOCKER] Building Rust tools for {problem_id} at {tool_dir}")
        volumes = {
            str(tool_dir): {"bind": "/work", "mode": "rw"}
        }
        container = self.client.containers.run(
            image="rust:1.75",
            command="cargo build --release",
            volumes=volumes,
            working_dir="/work",
            remove=True,
            detach=False
        )
        logger.info(f"[DOCKER] Rust tools built successfully for {problem_id}")

    def run_container(
        self,
        image: str,
        command: str,
        volumes: Dict[str, Dict[str, str]],
        working_dir: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Any:
        """Run a command in a Docker container."""
        logger.info(f"[DOCKER] Running container with image: {image}")
        logger.info(f"[DOCKER] Command: {command[:200]}...")

        detach = kwargs.get("detach", True)
        remove = kwargs.get("remove", False)
        ports = kwargs.get("ports", None)

        container = self.client.containers.run(
            image=image,
            command=command,
            volumes=volumes,
            working_dir=working_dir,
            environment=environment,
            detach=detach,
            remove=remove,
            ports=ports,
        )

        logger.info(f"[DOCKER] Container started: {container.id if hasattr(container, 'id') else 'N/A'}")
        return container

    def write_file(self, remote_path: str, content: str | bytes) -> None:
        """Write file to local filesystem."""
        p = Path(remote_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content)

    def read_file(self, remote_path: str) -> str:
        """Read file from local filesystem."""
        return Path(remote_path).read_text()

    def list_files(self, remote_path: str, pattern: str = "*") -> list[str]:
        """List files matching pattern on local filesystem."""
        return sorted([str(p) for p in Path(remote_path).glob(pattern)])

    def file_size(self, remote_path: str) -> int:
        """Get file size on local filesystem."""
        return Path(remote_path).stat().st_size

    def mkdir(self, remote_path: str) -> None:
        """Create directory on local filesystem."""
        Path(remote_path).mkdir(parents=True, exist_ok=True)

    def exec_command(self, command: str, workdir: str | None = None, timeout: int = 3600) -> tuple[int, str, str]:
        """Execute command via Docker container using run_container internally."""
        # For Docker, we use subprocess since files are local
        result = subprocess.run(
            ["/bin/sh", "-c", command],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return (result.returncode, result.stdout, result.stderr)

    def setup_tool_links(self, tool_dir: str) -> None:
        """No-op for Docker — tools are mounted via volumes."""
        pass

    def close(self) -> None:
        """Close Docker client connection."""
        if self.client:
            logger.info("[DOCKER] Closing Docker client")
            self.client.close()
