"""Local backend implementation for ALE-Bench.

Runs everything via local subprocess + local filesystem — no Docker, no Modal sandbox.
Designed for use inside a Modal function where the host and sandbox share a filesystem.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from .base import Backend

logger = logging.getLogger(__name__)

# Use /tmp for judge/work dirs when /judge and /workdir are read-only (e.g. sandboxed environments)
_DEFAULT_LOCAL_JUDGE_DIR = os.environ.get("ALE_BENCH_JUDGE_DIR", "/tmp/ale-bench-judge")
_DEFAULT_LOCAL_WORK_DIR = os.environ.get("ALE_BENCH_WORK_DIR", "/tmp/ale-bench-workdir")


class LocalBackend(Backend):
    """Local subprocess-based execution backend.

    Uses direct filesystem operations and subprocess calls.
    Fastest backend when running inside an environment that already has
    the required toolchains (e.g., inside a Modal function).
    """

    def __init__(
        self,
        judge_dir: Optional[str] = None,
        work_dir: Optional[str] = None,
    ):
        self._judge_dir = judge_dir or _DEFAULT_LOCAL_JUDGE_DIR
        self._work_dir = work_dir or _DEFAULT_LOCAL_WORK_DIR
        logger.info(f"[LOCAL] Local backend initialized (judge_dir={self._judge_dir}, work_dir={self._work_dir})")

    @property
    def judge_dir(self) -> str:
        """Writable directory for judge binaries (avoids read-only /judge in sandboxes)."""
        return self._judge_dir

    @property
    def work_dir(self) -> str:
        """Writable directory for code and working files (avoids read-only /workdir in sandboxes)."""
        return self._work_dir

    def build_tools(self, problem_id: str, tool_dir: Path) -> None:
        """Build Rust tools locally via subprocess."""
        from ale_bench.data import build_rust_tools_local
        logger.info(f"[LOCAL] Building Rust tools for {problem_id} at {tool_dir}")
        build_rust_tools_local(tool_dir)
        logger.info(f"[LOCAL] Rust tools built successfully for {problem_id}")

    def run_container(
        self,
        image: str,
        command: str,
        volumes: Dict[str, Dict[str, str]],
        working_dir: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Any:
        raise NotImplementedError("LocalBackend uses primitives, not containers")

    def write_file(self, remote_path: str, content: str | bytes) -> None:
        p = Path(remote_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content)

    def read_file(self, remote_path: str) -> str:
        return Path(remote_path).read_text()

    def list_files(self, remote_path: str, pattern: str = "*") -> list[str]:
        return sorted([str(p) for p in Path(remote_path).glob(pattern)])

    def file_size(self, remote_path: str) -> int:
        return Path(remote_path).stat().st_size

    def mkdir(self, remote_path: str) -> None:
        Path(remote_path).mkdir(parents=True, exist_ok=True)

    def exec_command(self, command: str, workdir: str | None = None, timeout: int = 3600) -> tuple[int, str, str]:
        result = subprocess.run(
            ["/bin/sh", "-c", command],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return (result.returncode, result.stdout, result.stderr)

    def setup_tool_links(self, tool_dir: str) -> None:
        """Create symlinks so judge binaries are at judge_dir/target/release/."""
        judge_release = Path(self._judge_dir) / "target" / "release"
        judge_release.mkdir(parents=True, exist_ok=True)

        tool_path = Path(tool_dir)
        release_dir = tool_path / "tools" / "target" / "release"
        if not release_dir.exists():
            release_dir = tool_path / "target" / "release"

        for tool in ["gen", "tester", "vis"]:
            src = release_dir / tool
            dst = judge_release / tool
            if src.exists():
                dst.unlink(missing_ok=True)
                os.symlink(src, dst)
                logger.info(f"[LOCAL] Symlinked {dst} -> {src}")

    def close(self) -> None:
        logger.info("[LOCAL] Local backend closed (no-op)")
