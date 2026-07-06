import asyncio
import os
import shlex
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from prime_sandboxes import CommandTimeoutError, CreateSandboxRequest
from prime_sandboxes.exceptions import SandboxFileNotFoundError, SandboxImagePullError, SandboxNotRunningError
from prime_sandboxes.models import BackgroundJob, BackgroundJobStatus, CommandResponse, FileUploadResponse


class LocalDockerSandboxClient:
    """Small async subset of Prime's sandbox client backed by local Docker."""

    def __init__(self) -> None:
        self._containers: set[str] = set()

    def teardown(self) -> None:
        pass

    async def _run(self, *args: str, timeout: int | None = None, input_data: bytes | None = None) -> tuple[int, bytes, bytes]:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE if input_data is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(input_data), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise
        return proc.returncode or 0, stdout, stderr

    async def create(self, request: CreateSandboxRequest):
        sandbox_id = f"local-{uuid.uuid4().hex[:12]}"
        memory = max(float(request.memory_gb or 4), 1.0)
        cpus = max(float(request.cpu_cores or 4), 1.0)
        cmd = request.start_command or "tail -f /dev/null"
        env_args: list[str] = []
        for key, value in (request.environment_vars or {}).items():
            env_args.extend(["-e", f"{key}={value}"])
        args = [
            "docker",
            "run",
            "-d",
            "--name",
            sandbox_id,
            "--cpus",
            str(cpus),
            "--memory",
            f"{memory}g",
            *env_args,
            "--entrypoint",
            "/bin/bash",
            request.docker_image,
            "-lc",
            cmd,
        ]
        code, stdout, stderr = await self._run(*args, timeout=600)
        if code != 0:
            text = (stderr or stdout).decode(errors="replace")[:1000]
            raise SandboxImagePullError(sandbox_id, message=f"Local Docker failed for {request.docker_image}: {text}")
        self._containers.add(sandbox_id)
        return SimpleNamespace(id=sandbox_id)

    async def wait_for_creation(self, sandbox_id: str, *args: Any, **kwargs: Any) -> None:
        code, stdout, _ = await self._run("docker", "inspect", "-f", "{{.State.Running}}", sandbox_id, timeout=30)
        if code != 0 or stdout.decode().strip() != "true":
            raise SandboxNotRunningError(sandbox_id)

    async def delete(self, sandbox_id: str) -> None:
        await self._run("docker", "rm", "-f", sandbox_id, timeout=60)
        self._containers.discard(sandbox_id)

    async def execute_command(
        self,
        sandbox_id: str,
        command: str,
        timeout: int | None = None,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> CommandResponse:
        exec_args = ["docker", "exec"]
        if working_dir:
            exec_args.extend(["-w", working_dir])
        for key, value in (env or {}).items():
            exec_args.extend(["-e", f"{key}={value}"])
        exec_args.extend([sandbox_id, "bash", "-lc", command])
        try:
            code, stdout, stderr = await self._run(*exec_args, timeout=timeout)
        except asyncio.TimeoutError:
            raise CommandTimeoutError(sandbox_id=sandbox_id, command=command, timeout=int(timeout or 0))
        return CommandResponse(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            exit_code=code,
        )

    async def upload_file(
        self,
        sandbox_id: str,
        file_path: str,
        local_file_path: str,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> FileUploadResponse:
        parent = os.path.dirname(file_path) or "/"
        await self.execute_command(sandbox_id, f"mkdir -p {shlex.quote(parent)}", timeout=timeout or 30)
        code, stdout, stderr = await self._run(
            "docker",
            "cp",
            local_file_path,
            f"{sandbox_id}:{file_path}",
            timeout=timeout or 300,
        )
        if code != 0:
            raise RuntimeError((stderr or stdout).decode(errors="replace"))
        return FileUploadResponse(
            success=True,
            path=file_path,
            size=Path(local_file_path).stat().st_size,
            timestamp=datetime.now(timezone.utc),
        )

    async def upload_bytes(
        self,
        sandbox_id: str,
        file_path: str,
        file_bytes: bytes,
        filename: str | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> FileUploadResponse:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            return await self.upload_file(sandbox_id, file_path, tmp_path, timeout=timeout)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    async def download_file(
        self,
        sandbox_id: str,
        file_path: str,
        local_file_path: str,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> None:
        Path(local_file_path).parent.mkdir(parents=True, exist_ok=True)
        code, stdout, stderr = await self._run(
            "docker",
            "cp",
            f"{sandbox_id}:{file_path}",
            local_file_path,
            timeout=timeout or 300,
        )
        if code != 0:
            raise SandboxFileNotFoundError((stderr or stdout).decode(errors="replace"))

    async def start_background_job(
        self,
        sandbox_id: str,
        command: str,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> BackgroundJob:
        job_id = uuid.uuid4().hex[:8]
        stdout_log_file = f"/tmp/job_{job_id}.stdout.log"
        stderr_log_file = f"/tmp/job_{job_id}.stderr.log"
        exit_file = f"/tmp/job_{job_id}.exit"
        env_prefix = ""
        if env:
            env_prefix = "; ".join(f"export {k}={shlex.quote(v)}" for k, v in env.items()) + "; "
        dir_prefix = f"cd {shlex.quote(working_dir)} && " if working_dir else ""
        body = f"{env_prefix}{dir_prefix}{command}"
        wrapped = (
            f"({body}) > {shlex.quote(stdout_log_file)} 2> {shlex.quote(stderr_log_file)}; "
            f"echo $? > {shlex.quote(exit_file)}"
        )
        await self.execute_command(sandbox_id, f"nohup bash -lc {shlex.quote(wrapped)} < /dev/null >/dev/null 2>&1 &", timeout=30)
        return BackgroundJob(
            job_id=job_id,
            sandbox_id=sandbox_id,
            stdout_log_file=stdout_log_file,
            stderr_log_file=stderr_log_file,
            exit_file=exit_file,
        )

    async def _read_or_empty(self, sandbox_id: str, path: str, timeout: int | None = None) -> str:
        result = await self.execute_command(sandbox_id, f"cat {shlex.quote(path)} 2>/dev/null || true", timeout=timeout or 30)
        return result.stdout

    async def get_background_job(
        self,
        sandbox_id: str,
        job: BackgroundJob,
        timeout: int | None = None,
    ) -> BackgroundJobStatus:
        exit_content = await self._read_or_empty(sandbox_id, job.exit_file, timeout=timeout)
        if not exit_content.strip():
            return BackgroundJobStatus(job_id=job.job_id, completed=False)
        try:
            exit_code = int(exit_content.strip())
        except ValueError:
            return BackgroundJobStatus(job_id=job.job_id, completed=False)
        stdout = await self._read_or_empty(sandbox_id, job.stdout_log_file, timeout=timeout)
        stderr = await self._read_or_empty(sandbox_id, job.stderr_log_file, timeout=timeout)
        return BackgroundJobStatus(job_id=job.job_id, completed=True, exit_code=exit_code, stdout=stdout, stderr=stderr)
