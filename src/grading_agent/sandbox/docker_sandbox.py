"""Run untrusted student programs in a restricted Docker container."""

from __future__ import annotations

import base64
import io
import math
import shlex
import tarfile
import uuid
from collections.abc import Sequence
from typing import Any

import docker
from docker.errors import DockerException

from grading_agent.config import settings


class DockerSandbox:
    """Manage one isolated container for a grading session.

    The root filesystem is read-only. Only the size-limited ``/workspace``
    and ``/tmp`` tmpfs mounts are writable. The container has no network,
    dropped Linux capabilities, and configured CPU, memory, and PID limits.
    """

    def __init__(
        self,
        image: str | None = None,
        memory_limit: str | None = None,
        cpu_limit: float | None = None,
        max_output_bytes: int | None = None,
    ) -> None:
        self.image = image or settings.DOCKER_SANDBOX_IMAGE
        self.memory_limit = memory_limit or settings.DOCKER_MEMORY_LIMIT
        self.cpu_limit = (
            settings.DOCKER_CPU_LIMIT if cpu_limit is None else cpu_limit
        )
        self.max_output_bytes = (
            settings.DOCKER_MAX_OUTPUT_BYTES
            if max_output_bytes is None
            else max_output_bytes
        )

        if (
            isinstance(self.cpu_limit, bool)
            or not isinstance(self.cpu_limit, (int, float))
            or not math.isfinite(self.cpu_limit)
            or self.cpu_limit <= 0
        ):
            raise ValueError("cpu_limit must be a finite positive number.")
        if (
            isinstance(self.max_output_bytes, bool)
            or not isinstance(self.max_output_bytes, int)
            or self.max_output_bytes < 1
        ):
            raise ValueError("max_output_bytes must be a positive integer.")

        self.client = docker.from_env()
        self.container: Any | None = None
        self._closed = False

    def start(self) -> None:
        """Create and start the restricted container."""
        if self._closed:
            raise RuntimeError("This sandbox has already been closed.")
        if self.container is not None:
            raise RuntimeError("Sandbox is already running.")

        try:
            self.client.ping()
            self.container = self.client.containers.run(
                self.image,
                name=f"agent-sandbox-{uuid.uuid4().hex[:12]}",
                detach=True,
                command=["sleep", "infinity"],
                working_dir="/workspace",
                user=(
                    f"{settings.DOCKER_CONTAINER_UID}:"
                    f"{settings.DOCKER_CONTAINER_GID}"
                ),
                network_mode=settings.DOCKER_NETWORK_MODE,
                mem_limit=self.memory_limit,
                # Equal to the memory cap disables additional swap use.
                memswap_limit=self.memory_limit,
                nano_cpus=int(self.cpu_limit * 1_000_000_000),
                pids_limit=settings.DOCKER_PIDS_LIMIT,
                read_only=settings.DOCKER_READ_ONLY_ROOTFS,
                tmpfs={
                    "/workspace": (
                        "rw,exec,nosuid,nodev,"
                        f"size={settings.DOCKER_WORKSPACE_TMPFS_SIZE},"
                        f"uid={settings.DOCKER_CONTAINER_UID},"
                        f"gid={settings.DOCKER_CONTAINER_GID}"
                    ),
                    "/tmp": (
                        "rw,nosuid,nodev,"
                        f"size={settings.DOCKER_TMPFS_SIZE},mode=1777"
                    ),
                },
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                init=True,
            )
        except Exception:
            self.close()
            raise

    def _require_container(self):
        if self.container is None:
            raise RuntimeError("Start the sandbox first.")
        return self.container

    @staticmethod
    def _validate_filename(filename: str) -> None:
        """Accept a simple filename only; paths are deliberately forbidden."""
        if (
            not isinstance(filename, str)
            or not filename
            or filename in {".", ".."}
            or "/" in filename
            or "\\" in filename
            or "\x00" in filename
        ):
            raise ValueError("Only simple filenames are allowed.")

    def _exec_as_sandbox(self, command: Sequence[str]):
        """Run a short trusted helper command as the sandbox user."""
        container = self._require_container()
        return container.exec_run(
            list(command),
            workdir="/workspace",
            user=(
                f"{settings.DOCKER_CONTAINER_UID}:"
                f"{settings.DOCKER_CONTAINER_GID}"
            ),
            demux=False,
        )

    def write_file(self, filename: str, content: str) -> None:
        """Write UTF-8 text into the workspace without Docker put_archive.

        Docker's put_archive API can fail when the container root filesystem is
        read-only. This method instead writes base64-encoded chunks through an
        exec process into the writable /workspace tmpfs.
        """
        self._require_container()
        self._validate_filename(filename)
        if not isinstance(content, str):
            raise TypeError("content must be a string.")

        data = content.encode("utf-8")
        if len(data) > settings.DOCKER_MAX_FILE_BYTES:
            raise ValueError(
                "File exceeds the configured limit of "
                f"{settings.DOCKER_MAX_FILE_BYTES} bytes."
            )

        target = f"/workspace/{filename}"
        quoted_target = shlex.quote(target)

        # Remove an existing file or symlink before creating the replacement.
        # The target name has already been restricted to a simple filename.
        result = self._exec_as_sandbox(["rm", "-f", "--", filename])
        if result.exit_code != 0:
            output = (result.output or b"").decode("utf-8", errors="replace")
            raise RuntimeError(f"Could not prepare {filename}: {output}")

        encoded = base64.b64encode(data).decode("ascii")
        chunk_size = settings.DOCKER_WRITE_CHUNK_CHARS
        chunks = [encoded[i : i + chunk_size] for i in range(0, len(encoded), chunk_size)]

        if not chunks:
            chunks = [""]

        for index, chunk in enumerate(chunks):
            operator = ">" if index == 0 else ">>"
            script = (
                f"umask 077; printf '%s' {shlex.quote(chunk)} "
                f"| base64 -d {operator} {quoted_target}"
            )
            result = self._exec_as_sandbox(["/bin/sh", "-c", script])
            if result.exit_code != 0:
                output = (result.output or b"").decode("utf-8", errors="replace")
                raise RuntimeError(f"Could not write {filename}: {output}")

    def run(
        self,
        command: str | Sequence[str],
        timeout_s: float | None = None,
        stdin: str | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Run a command with a deadline and a captured-output limit.

        Prefer a sequence, for example ``["gcc", "main.c", "-o", "main"]``.
        A string is accepted for convenience and split with ``shlex``; it is
        not treated as a shell script.
        """
        container = self._require_container()

        # ``timeout`` is retained as a backwards-compatible alias for older
        # callers; new code should use ``timeout_s``.
        if timeout is not None:
            if timeout_s is not None:
                raise ValueError("Specify only one of timeout and timeout_s.")
            timeout_s = timeout

        if isinstance(command, str):
            command = shlex.split(command)
        elif isinstance(command, Sequence) and not isinstance(command, bytes):
            command = list(command)
        else:
            raise TypeError("command must be a string or sequence of strings.")

        if not command or any(
            not isinstance(arg, str) or "\x00" in arg for arg in command
        ):
            raise ValueError("Invalid command arguments.")
        if not command[0]:
            raise ValueError("The executable name cannot be empty.")

        timeout = (
            settings.DOCKER_DEFAULT_TIMEOUT_S
            if timeout_s is None
            else timeout_s
        )
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0 < timeout <= settings.DOCKER_MAX_TIMEOUT_S
        ):
            raise ValueError(
                "timeout_s must be greater than zero and no more than "
                f"{settings.DOCKER_MAX_TIMEOUT_S} seconds."
            )

        if stdin is not None and not isinstance(stdin, str):
            raise TypeError("stdin must be a string or None.")

        input_filename = None
        try:
            input_redirect = ""
            if stdin is not None:
                input_filename = f".grader-input-{uuid.uuid4().hex}.txt"
                self.write_file(input_filename, stdin)
                input_redirect = f" < {shlex.quote('/workspace/' + input_filename)}"

            safe_command = shlex.join(
                [
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=1s",
                    f"{timeout:g}s",
                    *command,
                ]
            )
            # PIPESTATUS is a Bash feature. head stops output capture at the
            # configured limit; the tested command's exit status is returned.
            script = (
                "set +e; "
                f"{safe_command}{input_redirect} 2>&1 "
                f"| head -c {self.max_output_bytes}; "
                'status=${PIPESTATUS[0]}; exit "$status"'
            )
            result = container.exec_run(
                ["/bin/bash", "-c", script],
                workdir="/workspace",
                user=(
                    f"{settings.DOCKER_CONTAINER_UID}:"
                    f"{settings.DOCKER_CONTAINER_GID}"
                ),
                demux=False,
            )
            output_bytes = result.output or b""
            return {
                "exit_code": result.exit_code,
                "output": output_bytes.decode("utf-8", errors="replace"),
                "timed_out": result.exit_code in {124, 137},
                "output_may_be_truncated": (
                    len(output_bytes) >= self.max_output_bytes
                    or result.exit_code == 141
                ),
            }
        finally:
            if input_filename is not None:
                try:
                    self._exec_as_sandbox(["rm", "-f", "--", input_filename])
                except DockerException:
                    # Container cleanup will remove this temporary input too.
                    pass

    def read_file(self, filename: str) -> str:
        """Read a regular file from the container workspace."""
        container = self._require_container()
        self._validate_filename(filename)
        target = f"/workspace/{filename}"

        # Do not follow a symlink supplied or created by executed code.
        check = self._exec_as_sandbox(
            ["/bin/sh", "-c", f"test -f {shlex.quote(target)} && test ! -L {shlex.quote(target)}"]
        )
        if check.exit_code != 0:
            raise FileNotFoundError(
                f"{filename!r} does not exist or is not a regular file."
            )

        chunks, metadata = container.get_archive(target)
        reported_size = metadata.get("size")
        if (
            isinstance(reported_size, int)
            and reported_size > settings.DOCKER_MAX_FILE_BYTES
        ):
            raise ValueError(
                "File exceeds the configured read limit of "
                f"{settings.DOCKER_MAX_FILE_BYTES} bytes."
            )

        archive = io.BytesIO(b"".join(chunks))
        with tarfile.open(fileobj=archive, mode="r:*") as tar:
            member = next((item for item in tar.getmembers() if item.isfile()), None)
            if member is None:
                raise FileNotFoundError(filename)
            if member.size > settings.DOCKER_MAX_FILE_BYTES:
                raise ValueError(
                    "File exceeds the configured read limit of "
                    f"{settings.DOCKER_MAX_FILE_BYTES} bytes."
                )
            extracted = tar.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Could not read {filename}.")
            return extracted.read().decode("utf-8", errors="replace")

    def close(self) -> None:
        """Remove the container and close the Docker client."""
        if self._closed:
            return

        container = self.container
        self.container = None
        try:
            if container is not None:
                container.remove(force=True)
        finally:
            self.client.close()
            self._closed = True

    def __enter__(self) -> DockerSandbox:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False
