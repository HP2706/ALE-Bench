"""Abstract base class for ALE-Bench execution backends."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import ale_bench.constants


class Backend(ABC):
    """Abstract backend interface for executing ALE-Bench operations."""

    @abstractmethod
    def build_tools(self, problem_id: str, tool_dir: Path) -> None:
        """Build Rust tools (gen/tester/vis) for a problem.

        Args:
            problem_id: Problem identifier (e.g., "ahc001")
            tool_dir: Directory containing Cargo.toml and tool source code
        """
        pass

    @abstractmethod
    def run_container(
        self,
        image: str,
        command: str,
        volumes: Dict[str, Dict[str, str]],
        working_dir: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Any:
        """Run a command in an isolated execution environment.

        Args:
            image: Container/sandbox image name
            command: Command to execute
            volumes: Volume mounts mapping {host_path: {"bind": container_path, "mode": "rw"}}
            working_dir: Working directory for command execution
            environment: Environment variables
            **kwargs: Additional backend-specific parameters

        Returns:
            Container/sandbox result object with logs() and wait() methods
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up backend resources (sandboxes, connections, etc.)."""
        pass

    @abstractmethod
    def write_file(self, remote_path: str, content: str | bytes) -> None:
        """Write content to a file in the execution environment.

        Args:
            remote_path: Path in the execution environment
            content: File content (str or bytes)
        """
        pass

    @abstractmethod
    def read_file(self, remote_path: str) -> str:
        """Read a file from the execution environment.

        Args:
            remote_path: Path in the execution environment

        Returns:
            File content as string
        """
        pass

    @abstractmethod
    def list_files(self, remote_path: str, pattern: str = "*") -> list[str]:
        """List files matching a pattern in the execution environment.

        Args:
            remote_path: Directory path
            pattern: Glob pattern (default "*")

        Returns:
            Sorted list of matching file paths
        """
        pass

    @abstractmethod
    def file_size(self, remote_path: str) -> int:
        """Get size of a file in the execution environment.

        Args:
            remote_path: Path to file

        Returns:
            File size in bytes
        """
        pass

    @abstractmethod
    def mkdir(self, remote_path: str) -> None:
        """Create directory (and parents) in the execution environment.

        Args:
            remote_path: Directory path to create
        """
        pass

    @abstractmethod
    def exec_command(self, command: str, workdir: str | None = None, timeout: int = 3600) -> tuple[int, str, str]:
        """Execute a command in the execution environment.

        Args:
            command: Shell command to execute
            workdir: Working directory (optional)
            timeout: Timeout in seconds

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        pass

    def read_files(self, remote_paths: list[str]) -> list[str]:
        """Read multiple files. Default implementation calls read_file sequentially."""
        return [self.read_file(p) for p in remote_paths]

    def write_files(self, files: dict[str, str]) -> None:
        """Write multiple files. Default implementation calls write_file sequentially."""
        for path, content in files.items():
            self.write_file(path, content)

    @abstractmethod
    def setup_tool_links(self, tool_dir: str) -> None:
        """Setup tool binary links/paths so judge binaries are accessible at standard paths.

        Args:
            tool_dir: Directory containing built tools (e.g., with tools/target/release/)
        """
        pass

    @property
    def judge_dir(self) -> str:
        """Directory where judge binaries (gen, tester, vis) are located."""
        return ale_bench.constants.JUDGE_DIR

    @property
    def work_dir(self) -> str:
        """Directory for code files and working directory during execution."""
        return ale_bench.constants.WORK_DIR

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
