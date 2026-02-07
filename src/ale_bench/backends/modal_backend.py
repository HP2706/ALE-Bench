"""Native Modal Sandbox backend implementation for ALE-Bench."""

import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import modal

from .base import Backend

if TYPE_CHECKING:
    from ale_bench.data import Problem, Seeds, Standings, RankPerformanceMap

logger = logging.getLogger(__name__)


def get_modal_volume():
    """Get or create persistent Modal volume for ALE-Bench cache."""
    volume = modal.Volume.from_name("ale-bench-cache", create_if_missing=True)
    return volume


def get_alebench_modal_image():
    """Get Modal image for ALE-Bench with native toolchains."""
    image = modal.Image.from_registry(
        "ubuntu:22.04", add_python="3.12"
    ).apt_install(
        'curl', 'wget', 'git', 'build-essential', 'ca-certificates',
        "gcc-12", "g++-12", "libeigen3-dev", "libgmp-dev", "time",
        "python3.11", "python3.11-dev", 'unzip', 'zip',
    ).run_commands(
        # Boost 1.82
        "wget -q https://sourceforge.net/projects/boost/files/boost/1.82.0/boost_1_82_0.tar.gz/download -O /tmp/boost.tar.gz && "
        "tar -xzf /tmp/boost.tar.gz -C /tmp && "
        "cd /tmp/boost_1_82_0 && "
        "./bootstrap.sh --prefix=/opt/boost/gcc --with-toolset=gcc && "
        "./b2 install -j$(nproc) && "
        "rm -rf /tmp/boost_1_82_0 /tmp/boost.tar.gz",
    ).run_commands(
        # AC Library
        "wget -q https://github.com/atcoder/ac-library/releases/download/v1.5.1/ac-library.zip -O /tmp/ac-library.zip && "
        "mkdir -p /opt/ac-library && "
        "unzip -o /tmp/ac-library.zip -d /opt/ac-library && "
        "rm /tmp/ac-library.zip",
    ).uv_pip_install(
        "numpy", "scipy", "networkx", "sympy", "sortedcontainers",
        "more-itertools", "shapely", "bitarray", "PuLP", "mpmath",
        "pandas", "z3-solver", "scikit-learn", "ortools", "polars",
        "lightgbm", "gmpy2", "numba", "ac-library-python", "torch",
        "docker", "Pillow", "huggingface_hub", "pydantic", "ahocorapy",
        "cairosvg", "cloudpickle", "modal",
    ).run_commands(
        # Rust
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && "
        "export PATH=$HOME/.cargo/bin:$PATH && "
        "rustc --version && cargo --version",
    ).env({
        "PATH": "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONPATH": "/root/ALE-Bench/src",
        "ALE_BENCH_NO_DOCKER": "1",
        "ALE_BENCH_DATA": "/root/ALE-Bench/data",
    }).add_local_dir(
        local_path="ALE-Bench",
        remote_path="/root/ALE-Bench",
        copy=False,
    )

    return image


class ModalBackend(Backend):
    """Native Modal Sandbox backend with persistent volumes."""

    def __init__(self):
        """Initialize Modal backend with app and volume."""
        try:
            self.app = modal.App.lookup("ale-bench", create_if_missing=True)
            self.volume = get_modal_volume()
            self.image = get_alebench_modal_image()
            self.sandbox = None
            self._last_verified: float = 0.0  # timestamp of last successful sandbox ping
            logger.info("[MODAL] Modal backend initialized")
        except Exception as e:
            logger.error(f"[MODAL] Failed to initialize backend: {e}")
            raise

    def load_problem(
        self,
        problem_id: str,
        lite_version: bool
    ) -> tuple["Problem", "Seeds", "Standings", "RankPerformanceMap", Path]:
        """Run load_problem inside the Modal sandbox and transfer results back."""
        import json
        import cloudpickle

        sandbox = self._ensure_sandbox()

        MARKER = "===ALE_BENCH_RESULT==="
        script = f'''
import sys, os, base64, cloudpickle
os.environ["HF_HOME"] = "/root/.cache/ale-bench/hf"
os.environ["ALE_BENCH_CACHE"] = "/root/.cache/ale-bench/datasets"

from ale_bench.data import load_problem

result = load_problem("{problem_id}", {lite_version})
data = base64.b64encode(cloudpickle.dumps(result)).decode("ascii")
print("{MARKER}")
print(data)
'''

        logger.info(f"[MODAL] Running load_problem for {problem_id} in sandbox")
        proc = sandbox.exec("python", "-c", script, timeout=300)
        proc.wait()

        stdout = (proc.stdout.read() if proc.stdout else "").strip()
        stderr = proc.stderr.read() if proc.stderr else ""

        if stderr:
            logger.debug(f"[MODAL] load_problem stderr: {stderr[:500]}")

        if MARKER not in stdout:
            raise RuntimeError(f"[MODAL] load_problem failed. stderr: {stderr}\nstdout: {stdout}")

        b64_data = stdout.split(MARKER, 1)[1].strip()
        problem, seeds, standings, rank_performance_map, data_root = cloudpickle.loads(base64.b64decode(b64_data))
        return problem, seeds, standings, rank_performance_map, data_root

    def build_rust_tools(self, problem_id: str, tool_dir: Path) -> None:
        """Build Rust tools in Modal Sandbox with persistent caching."""
        logger.info(f"[MODAL] Building Rust tools for {problem_id}")
        sandbox = self._ensure_sandbox()
        script = f"""
import os
from pathlib import Path
os.environ["ALE_BENCH_CACHE"] = "/root/.cache/ale-bench/datasets"
from ale_bench.data import build_rust_tools_local
build_rust_tools_local(Path("{tool_dir}"))
"""
        proc = sandbox.exec("python", "-c", script, timeout=600)
        proc.wait()
        stderr = proc.stderr.read() if proc.stderr else ""
        if proc.returncode != 0:
            raise RuntimeError(f"[MODAL] build_rust_tools failed: {stderr}")
        logger.info(f"[MODAL] build_rust_tools completed for {problem_id}")

    def _create_sandbox(self) -> modal.Sandbox:
        """Create a new sandbox instance."""
        logger.info("[MODAL] Creating new sandbox with persistent volume")
        try:
            with modal.enable_output():
                sandbox = modal.Sandbox.create(
                    image=self.image,
                    app=self.app,
                    timeout=3600,
                    volumes={"/root/.cache/ale-bench": self.volume}
                )
            logger.info("[MODAL] Sandbox created successfully")
            return sandbox
        except Exception as e:
            logger.error(f"[MODAL] Failed to create sandbox: {e}")
            raise

    def _ensure_sandbox(self) -> modal.Sandbox:
        """Ensure sandbox exists and is alive, recreate if needed."""
        import time as _time
        if self.sandbox is None:
            self.sandbox = self._create_sandbox()
            self._last_verified = _time.monotonic()
        else:
            now = _time.monotonic()
            # Only ping if we haven't verified recently (avoid excessive round-trips)
            if now - self._last_verified > 30.0:
                try:
                    proc = self.sandbox.exec("/bin/sh", "-c", "echo ok", timeout=10)
                    proc.wait()
                    self._last_verified = now
                except Exception as e:
                    logger.warning(f"[MODAL] Sandbox appears dead, recreating: {e}")
                    self.sandbox = self._create_sandbox()
                    self._last_verified = _time.monotonic()
        return self.sandbox

    def _exec_with_retry(self, fn, *args, **kwargs):
        """Execute a function that uses the sandbox, retrying once if sandbox is dead."""
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "finished" in str(e) or "NotFound" in type(e).__name__:
                import time as _time
                logger.warning(f"[MODAL] Sandbox dead during operation, recreating: {e}")
                self.sandbox = self._create_sandbox()
                self._last_verified = _time.monotonic()
                return fn(*args, **kwargs)
            raise

    def build_tools(self, problem_id: str, tool_dir: Path) -> None:
        """Build Rust tools in Modal Sandbox with persistent caching."""
        logger.info(f"[MODAL] Building Rust tools for {problem_id}")
        sandbox = self._ensure_sandbox()

        cache_path = f"/root/.cache/ale-bench/tools/{problem_id}/target/release"
        check_cmd = f"test -d {cache_path} && echo 'exists' || echo 'missing'"

        proc = sandbox.exec("/bin/sh", "-c", check_cmd, timeout=10)
        proc.wait()
        result = proc.stdout.read() if proc.stdout else ""

        if "exists" in result:
            logger.info(f"[MODAL] Rust tools already built for {problem_id} (cached)")
            return

        logger.info(f"[MODAL] Building Rust tools (first time) for {problem_id}")
        tool_cache_dir = f"/root/.cache/ale-bench/tools/{problem_id}"
        sandbox.exec("/bin/sh", "-c", f"mkdir -p {tool_cache_dir}", timeout=10).wait()

        build_cmd = f"cd {tool_cache_dir} && cargo build --release"
        proc = sandbox.exec("/bin/sh", "-c", build_cmd, timeout=600)
        proc.wait()

        if proc.returncode != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            logger.error(f"[MODAL] Build failed: {stderr}")
            raise RuntimeError(f"Failed to build Rust tools for {problem_id}: {stderr}")

        logger.info(f"[MODAL] Rust tools built successfully for {problem_id}")

    def _mark_verified(self):
        """Mark sandbox as recently verified (call after successful exec)."""
        import time as _time
        self._last_verified = _time.monotonic()

    def write_file(self, remote_path: str, content: str | bytes) -> None:
        """Write file to sandbox via base64 encoding."""
        sandbox = self._ensure_sandbox()
        parent = str(Path(remote_path).parent)

        if isinstance(content, str):
            data = content.encode("utf-8")
        else:
            data = content

        # Handle empty files directly
        if len(data) == 0:
            proc = sandbox.exec("/bin/sh", "-c", f"mkdir -p {parent} && : > {remote_path}", timeout=10)
            proc.wait()
            if proc.returncode != 0:
                stderr = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(f"Failed to write empty file {remote_path}: {stderr}")
            self._mark_verified()
            return

        sandbox.exec("/bin/sh", "-c", f"mkdir -p {parent}", timeout=10).wait()

        b64 = base64.b64encode(data).decode("ascii")

        # Write in chunks to avoid command-line length limits
        chunk_size = 50000
        num_chunks = (len(b64) + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(b64))
            chunk = b64[start:end]
            redirect = ">" if i == 0 else ">>"
            proc = sandbox.exec("/bin/sh", "-c", f"echo -n '{chunk}' {redirect} /tmp/_upload.b64", timeout=30)
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"Failed to upload chunk {i}")

        proc = sandbox.exec("/bin/sh", "-c", f"base64 -d < /tmp/_upload.b64 > {remote_path} && rm /tmp/_upload.b64", timeout=30)
        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"Failed to write file {remote_path}: {stderr}")

        self._mark_verified()
        logger.debug(f"[MODAL] Wrote {len(data)} bytes to {remote_path}")

    def read_file(self, remote_path: str) -> str:
        """Read file from sandbox."""
        sandbox = self._ensure_sandbox()
        proc = sandbox.exec("/bin/sh", "-c", f"cat {remote_path}", timeout=30)
        proc.wait()
        stdout = proc.stdout.read() if proc.stdout else ""
        if proc.returncode != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"Failed to read file {remote_path}: {stderr}")
        self._mark_verified()
        return stdout

    def read_files(self, remote_paths: list[str]) -> list[str]:
        """Read multiple files from sandbox in a single round-trip."""
        if not remote_paths:
            return []
        if len(remote_paths) == 1:
            return [self.read_file(remote_paths[0])]
        sandbox = self._ensure_sandbox()
        import json as _json
        paths_json = _json.dumps(remote_paths)
        script = f"import json; paths = {paths_json}; print(json.dumps([open(p).read() for p in paths]))"
        proc = sandbox.exec("python3", "-c", script, timeout=60)
        proc.wait()
        stdout = proc.stdout.read() if proc.stdout else ""
        if proc.returncode != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"Failed to read files: {stderr}")
        self._mark_verified()
        return _json.loads(stdout)

    def write_files(self, files: dict[str, str]) -> None:
        """Write multiple files to sandbox in a single round-trip."""
        if not files:
            return
        if len(files) == 1:
            path, content = next(iter(files.items()))
            self.write_file(path, content)
            return
        sandbox = self._ensure_sandbox()
        import json as _json
        files_json = _json.dumps(files)
        # Use base64 to avoid escaping issues
        import base64 as _b64
        encoded = _b64.b64encode(files_json.encode()).decode()
        script = (
            "import json, base64, os; "
            f"files = json.loads(base64.b64decode('{encoded}')); "
            "[os.makedirs(os.path.dirname(p), exist_ok=True) or None for p in files]; "
            "[open(p, 'w').write(c) for p, c in files.items()]"
        )
        proc = sandbox.exec("python3", "-c", script, timeout=60)
        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"Failed to write files: {stderr}")
        self._mark_verified()

    def list_files(self, remote_path: str, pattern: str = "*") -> list[str]:
        """List files in sandbox directory matching pattern."""
        sandbox = self._ensure_sandbox()
        # Use find for glob-like matching
        if pattern == "*":
            cmd = f"find {remote_path} -maxdepth 1 -type f | sort"
        else:
            cmd = f"find {remote_path} -maxdepth 1 -type f -name '{pattern}' | sort"
        proc = sandbox.exec("/bin/sh", "-c", cmd, timeout=30)
        proc.wait()
        stdout = proc.stdout.read() if proc.stdout else ""
        if not stdout.strip():
            return []
        return [line for line in stdout.strip().split("\n") if line]

    def file_size(self, remote_path: str) -> int:
        """Get file size in sandbox."""
        sandbox = self._ensure_sandbox()
        proc = sandbox.exec("/bin/sh", "-c", f"stat -c%s {remote_path}", timeout=10)
        proc.wait()
        stdout = proc.stdout.read() if proc.stdout else ""
        if proc.returncode != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"Failed to stat {remote_path}: {stderr}")
        return int(stdout.strip())

    def mkdir(self, remote_path: str) -> None:
        """Create directory in sandbox."""
        sandbox = self._ensure_sandbox()
        sandbox.exec("/bin/sh", "-c", f"mkdir -p {remote_path}", timeout=10).wait()

    def exec_command(self, command: str, workdir: str | None = None, timeout: int = 3600) -> tuple[int, str, str]:
        """Execute command in sandbox."""
        sandbox = self._ensure_sandbox()

        if workdir:
            sandbox.exec("/bin/sh", "-c", f"mkdir -p {workdir}", timeout=10).wait()

        kwargs = {"timeout": timeout}
        if workdir:
            kwargs["workdir"] = workdir

        proc = sandbox.exec("/bin/sh", "-c", command, **kwargs)
        proc.wait()

        stdout = proc.stdout.read() if proc.stdout else ""
        stderr = proc.stderr.read() if proc.stderr else ""
        exit_code = proc.returncode

        self._mark_verified()
        logger.debug(f"[MODAL] exec_command exit={exit_code} cmd={command[:100]}...")
        return (exit_code, stdout, stderr)

    def setup_tool_links(self, tool_dir: str) -> None:
        """Create symlinks so judge binaries are at /judge/target/release/."""
        sandbox = self._ensure_sandbox()
        cmd = f"mkdir -p /judge/target/release && ln -sf {tool_dir}/tools/target/release/gen /judge/target/release/gen && ln -sf {tool_dir}/tools/target/release/tester /judge/target/release/tester && ln -sf {tool_dir}/tools/target/release/vis /judge/target/release/vis"
        proc = sandbox.exec("/bin/sh", "-c", cmd, timeout=10)
        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            logger.warning(f"[MODAL] setup_tool_links failed: {stderr}")

    def run_container(
        self,
        image: str,
        command: str,
        volumes: Dict[str, Dict[str, str]],
        working_dir: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Any:
        """Run command in Modal Sandbox (legacy interface, used by Docker-style callers)."""
        logger.info(f"[MODAL] run_container (legacy): {command[:200]}...")

        sandbox = self._ensure_sandbox()

        if working_dir:
            sandbox.exec("/bin/sh", "-c", f"mkdir -p {working_dir}").wait()

        if volumes:
            for host_path, mount_spec in volumes.items():
                bind_path = mount_spec["bind"]
                host = Path(host_path)
                if host.is_dir():
                    self._upload_directory(sandbox, host_path, bind_path)
                elif host.is_file():
                    parent = str(Path(bind_path).parent)
                    sandbox.exec("/bin/sh", "-c", f"mkdir -p {parent}").wait()
                    self._upload_file(sandbox, host_path, bind_path)
                else:
                    logger.warning(f"[MODAL] Host path does not exist: {host_path}")

        process = sandbox.exec(
            "/bin/sh", "-c", command,
            workdir=working_dir,
            timeout=kwargs.get("timeout", 3600),
        )

        return _ModalContainerResult(sandbox, process, self, volumes)

    def _upload_file(self, sandbox: modal.Sandbox, local_path: str, remote_path: str):
        """Upload a single file to the sandbox using tar + base64."""
        import tarfile
        import tempfile

        local_path_obj = Path(local_path)
        logger.info(f"[MODAL] Uploading file: {local_path} -> {remote_path}")

        remote_dir = str(Path(remote_path).parent)
        sandbox.exec("/bin/sh", "-c", f"mkdir -p {remote_dir}", timeout=10).wait()

        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_tar:
            tar_path = tmp_tar.name
            with tarfile.open(tar_path, 'w:gz') as tar:
                tar.add(local_path, arcname=Path(remote_path).name)

            with open(tar_path, 'rb') as f:
                tar_data = f.read()

            tar_b64 = base64.b64encode(tar_data).decode('ascii')

            chunk_size = 50000
            num_chunks = (len(tar_b64) + chunk_size - 1) // chunk_size

            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(tar_b64))
                chunk = tar_b64[start:end]
                redirect = ">" if i == 0 else ">>"
                chunk_cmd = f"echo -n '{chunk}' {redirect} /tmp/upload.b64"
                proc = sandbox.exec("/bin/sh", "-c", chunk_cmd, timeout=30)
                proc.wait()
                if proc.returncode != 0:
                    raise RuntimeError(f"Failed to upload chunk {i}")

            extract_cmd = f"base64 -d < /tmp/upload.b64 | tar -xzf - -C {remote_dir} && rm /tmp/upload.b64"
            proc = sandbox.exec("/bin/sh", "-c", extract_cmd, timeout=60)
            proc.wait()

            if proc.returncode != 0:
                stderr = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(f"Failed to extract file: {stderr}")

            Path(tar_path).unlink()

        is_executable = local_path_obj.stat().st_mode & 0o111
        name = local_path_obj.name
        looks_like_binary = (
            name.endswith('.out') or
            name in ('a.out', 'Main') or
            '.' not in name or
            name.endswith('.sh')
        )
        if is_executable or looks_like_binary:
            sandbox.exec("/bin/sh", "-c", f"chmod +x {remote_path}", timeout=10).wait()

    def _upload_directory(self, sandbox: modal.Sandbox, local_path: str, remote_path: str):
        """Upload a directory to the sandbox using tar + base64."""
        import tarfile
        import tempfile

        logger.info(f"[MODAL] Uploading directory: {local_path} -> {remote_path}")

        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_tar:
            tar_path = tmp_tar.name
            with tarfile.open(tar_path, 'w:gz') as tar:
                tar.add(local_path, arcname='.')

            with open(tar_path, 'rb') as f:
                tar_data = f.read()

            tar_b64 = base64.b64encode(tar_data).decode('ascii')

            chunk_size = 50000
            num_chunks = (len(tar_b64) + chunk_size - 1) // chunk_size

            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(tar_b64))
                chunk = tar_b64[start:end]
                redirect = ">" if i == 0 else ">>"
                chunk_cmd = f"echo -n '{chunk}' {redirect} /tmp/upload.b64"
                proc = sandbox.exec("/bin/sh", "-c", chunk_cmd, timeout=30)
                proc.wait()
                if proc.returncode != 0:
                    raise RuntimeError(f"Failed to upload chunk {i}")

            extract_cmd = f"mkdir -p {remote_path} && base64 -d < /tmp/upload.b64 | tar -xzf - -C {remote_path} && rm /tmp/upload.b64"
            proc = sandbox.exec("/bin/sh", "-c", extract_cmd, timeout=60)
            proc.wait()

            if proc.returncode != 0:
                stderr = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(f"Failed to extract: {stderr}")

            Path(tar_path).unlink()
            logger.info(f"[MODAL] Upload completed: {remote_path}")

    def _download_directory(self, sandbox: modal.Sandbox, remote_path: str, local_path: str):
        """Download a directory from the sandbox using tar + base64."""
        import tarfile
        import tempfile

        logger.info(f"[MODAL] Downloading directory: {remote_path} -> {local_path}")

        tar_cmd = f"cd {remote_path} && tar -czf /tmp/download.tar.gz . && base64 < /tmp/download.tar.gz && rm /tmp/download.tar.gz"
        proc = sandbox.exec("/bin/sh", "-c", tar_cmd, timeout=60)
        proc.wait()

        if proc.returncode != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"Failed to create tar: {stderr}")

        tar_b64 = proc.stdout.read() if proc.stdout else ""
        if not tar_b64:
            raise RuntimeError("No data received from sandbox")

        tar_data = base64.b64decode(tar_b64)

        Path(local_path).mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_tar:
            tmp_tar.write(tar_data)
            tar_path = tmp_tar.name

        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(local_path)
        finally:
            Path(tar_path).unlink()

    def _download_file(self, sandbox: modal.Sandbox, remote_path: str, local_path: str):
        """Download a single file from the sandbox using base64."""
        logger.info(f"[MODAL] Downloading file: {remote_path} -> {local_path}")

        proc = sandbox.exec("/bin/sh", "-c", f"base64 < {remote_path}", timeout=60)
        proc.wait()

        if proc.returncode != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"Failed to read remote file: {stderr}")

        file_b64 = proc.stdout.read() if proc.stdout else ""
        if not file_b64:
            raise RuntimeError("No data received from sandbox")

        file_data = base64.b64decode(file_b64)

        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(file_data)

    def close(self) -> None:
        """Terminate sandbox and clean up resources."""
        if self.sandbox:
            logger.info("[MODAL] Terminating sandbox")
            try:
                self.sandbox.terminate()
            except Exception as e:
                logger.warning(f"[MODAL] Error terminating sandbox: {e}")
            finally:
                self.sandbox = None


class _ModalContainerResult:
    """Wrapper for Modal Sandbox execution result to match Docker API."""

    def __init__(self, sandbox: modal.Sandbox, process: Any, backend: ModalBackend, volumes: Dict[str, Dict[str, str]]):
        self.sandbox = sandbox
        self.process = process
        self.backend = backend
        self.volumes = volumes
        self.attrs = {"State": {"ExitCode": None}}
        self._stdout = ""
        self._stderr = ""
        self._completed = False

    def wait(self, timeout=None):
        if not self._completed:
            self.process.wait()
            self._stdout = self.process.stdout.read() if self.process.stdout else ""
            self._stderr = self.process.stderr.read() if self.process.stderr else ""
            self.attrs["State"]["ExitCode"] = self.process.returncode
            self._completed = True

            logger.info(f"[MODAL] Process completed with exit code: {self.process.returncode}")

            if self.process.returncode == 0 and self.volumes:
                for host_path, mount_spec in self.volumes.items():
                    bind_path = mount_spec["bind"]
                    mode = mount_spec.get("mode", "rw")
                    if mode == "ro":
                        continue
                    try:
                        check_proc = self.sandbox.exec("/bin/sh", "-c", f"test -d {bind_path} && echo dir || (test -f {bind_path} && echo file || echo missing)", timeout=10)
                        check_proc.wait()
                        check_output = check_proc.stdout.read() if check_proc.stdout else ""
                        if "dir" in check_output:
                            self.backend._download_directory(self.sandbox, bind_path, host_path)
                        elif "file" in check_output:
                            self.backend._download_file(self.sandbox, bind_path, host_path)
                    except Exception as e:
                        logger.error(f"[MODAL] Failed to download {bind_path}: {e}")

        return {"StatusCode": self.attrs["State"]["ExitCode"]}

    def logs(self, stdout=False, stderr=True):
        if not self._completed:
            self.wait()
        result = ""
        if stdout and not stderr:
            result = self._stdout
        elif stderr and not stdout:
            result = self._stderr
        elif stdout and stderr:
            result = self._stdout + self._stderr
        else:
            result = self._stderr
        if isinstance(result, str):
            return result.encode("utf-8")
        return result

    def remove(self, force=False):
        pass
