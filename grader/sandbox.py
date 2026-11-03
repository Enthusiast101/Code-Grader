from __future__ import annotations

import json
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any


class SandboxError(RuntimeError):
    pass


class DockerSandbox:
    """Restricted Docker runner; student processes never execute on the host."""

    def __init__(self, image: str, memory_mb: int = 256, cpu_limit: float = 1.0, max_output: int = 65536):
        self.image = image
        self.memory_mb = memory_mb
        self.cpu_limit = cpu_limit
        self.max_output = max_output
        self.container_id: str | None = None
        self.image_id: str | None = None

    @staticmethod
    def _command(command: list[str], timeout: float = 20, **kwargs: Any) -> subprocess.CompletedProcess:
        try:
            result = subprocess.run(command, capture_output=True, timeout=timeout, **kwargs)
        except FileNotFoundError as exc:
            raise SandboxError("Docker CLI is not installed or is not available on PATH.") from exc
        except subprocess.TimeoutExpired as exc:
            raise SandboxError(f"Docker command timed out after {timeout:g} seconds: {' '.join(command[:4])}") from exc
        return result

    def __enter__(self) -> "DockerSandbox":
        version = self._command(["docker", "version", "--format", "{{.Server.Version}}"], timeout=8)
        if version.returncode != 0 or not version.stdout.strip():
            detail = (version.stderr or version.stdout).decode("utf-8", errors="replace")[:1000]
            raise SandboxError("Docker daemon is unavailable. Start Docker and retry. " + detail)
        inspect = self._command(["docker", "image", "inspect", self.image, "--format", "{{.Id}}"], timeout=8)
        if inspect.returncode != 0:
            raise SandboxError(f"Sandbox image '{self.image}' is missing. Build it with: docker build -t {self.image} .")
        self.image_id = inspect.stdout.decode().strip()
        name = "code-grader-" + uuid.uuid4().hex[:12]
        command = [
            "docker", "run", "-d", "--name", name,
            "--network", "none", "--read-only",
            "--pids-limit", "64", "--memory", f"{self.memory_mb}m",
            "--memory-swap", f"{self.memory_mb}m", "--cpus", str(self.cpu_limit),
            "--ulimit", "core=0:0", "--ulimit", "nofile=64:64",
            "--ulimit", "fsize=16777216:16777216",
            # These capabilities belong only to the root supervisor and are
            # removed from every student/compiler process by setpriv.
            "--cap-drop", "ALL", "--cap-add", "CHOWN", "--cap-add", "DAC_OVERRIDE", "--cap-add", "KILL",
            "--cap-add", "SETPCAP", "--cap-add", "SETUID", "--cap-add", "SETGID",
            "--security-opt", "no-new-privileges", "--user", "0:0",
            "--tmpfs", "/work:rw,exec,nosuid,nodev,size=128m,uid=1000,gid=1000,mode=0700",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=32m,uid=1000,gid=1000,mode=0700",
            # Sticky bit lets UID 1000 create the binary but prevents replacing
            # or removing it after the root supervisor locks it.
            "--tmpfs", "/locked:rw,exec,nosuid,nodev,size=128m,uid=0,gid=1000,mode=1770",
            self.image, "sh", "-lc", "sleep infinity",
        ]
        created = self._command(command, timeout=15)
        if created.returncode != 0:
            detail = (created.stderr or created.stdout).decode("utf-8", errors="replace")[:1500]
            raise SandboxError("Could not create restricted sandbox: " + detail)
        self.container_id = created.stdout.decode().strip()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.container_id:
            self._command(["docker", "rm", "-f", self.container_id], timeout=10)
            self.container_id = None

    def put_text(self, path_in_container: str, content: str) -> None:
        if not self.container_id:
            raise SandboxError("Sandbox is not running.")
        if not re.fullmatch(r"/work/[A-Za-z0-9_.-]+", path_in_container):
            raise SandboxError("Unsafe container path.")
        temp = Path("/tmp") / f"cg_{uuid.uuid4().hex}.txt"
        temp.write_text(content, encoding="utf-8")
        try:
            copied = self._command(["docker", "cp", str(temp), f"{self.container_id}:{path_in_container}"], timeout=10)
            if copied.returncode != 0:
                detail = (copied.stderr or copied.stdout).decode("utf-8", errors="replace")[:1000]
                raise SandboxError("Could not copy a file into the sandbox: " + detail)
        finally:
            temp.unlink(missing_ok=True)

    def exec(self, args: list[str], *, timeout: float = 10, stdin: str | None = None, user: str = "1000:1000") -> dict[str, Any]:
        if not self.container_id:
            raise SandboxError("Sandbox is not running.")
        if user == "1000:1000":
            # docker exec inherits the container capability set. Use setpriv
            # as root to drop the complete bounding/permitted/inheritable set
            # before changing to the untrusted UID.
            command = ["docker", "exec", "--user", "0:0", self.container_id,
                       "setpriv", "--bounding-set=-all", "--inh-caps=-all", "--ambient-caps=-all",
                       "--reuid=1000", "--regid=1000", "--groups=1000", "--", *args]
        elif user == "0:0":
            command = ["docker", "exec", "--user", "0:0", self.container_id, *args]
        else:
            raise SandboxError("Only the fixed sandbox identities 0:0 and 1000:1000 are allowed.")
        try:
            result = self._command(command, timeout=timeout, input=stdin.encode("utf-8") if stdin is not None else None)
            return {
                "stdout": result.stdout.decode("utf-8", errors="replace")[:self.max_output],
                "stderr": result.stderr.decode("utf-8", errors="replace")[:self.max_output],
                "exit_code": result.returncode,
                "timed_out": False,
            }
        except SandboxError as exc:
            if "Docker command timed out after" not in str(exc):
                raise
            # A timed-out docker exec may leave compiler or submission children
            # behind. The root supervisor has CAP_KILL solely for this cleanup.
            if user != "0:0":
                self._command(["docker", "exec", "--user", "0:0", self.container_id, "pkill", "-9", "-u", "1000"], timeout=3)
            return {"stdout": "", "stderr": str(exc), "exit_code": None, "timed_out": True}

    def lock_source(self, work_name: str, locked_name: str) -> None:
        """Copy a source file to immutable storage before any untrusted binary runs."""
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", work_name) or not re.fullmatch(r"[A-Za-z0-9_.-]+", locked_name):
            raise SandboxError("Unsafe source name.")
        copied = self.exec(["cp", f"/work/{work_name}", f"/locked/{locked_name}"], timeout=5, user="1000:1000")
        if copied["exit_code"] != 0:
            raise SandboxError("Could not stage the source file: " + copied["stderr"] + copied["stdout"])
        locked = self.exec(["sh", "-lc", f"chown 0:0 /locked/{locked_name} && chmod 0444 /locked/{locked_name}"], timeout=5, user="0:0")
        if locked["exit_code"] != 0:
            raise SandboxError("Could not protect the source file: " + locked["stderr"] + locked["stdout"])

    def lock_binary(self, work_name: str, locked_name: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", work_name) or not re.fullmatch(r"[A-Za-z0-9_.-]+", locked_name):
            raise SandboxError("Unsafe executable name.")
        # The compiler runs as UID 1000 and owns /work. Copy as that user into
        # a sticky, group-writable staging directory, then transfer ownership
        # to root and make the executable immutable to the student UID.
        copied = self.exec(["cp", f"/work/{work_name}", f"/locked/{locked_name}"], timeout=5, user="1000:1000")
        if copied["exit_code"] != 0:
            raise SandboxError("Could not stage the compiled executable: " + copied["stderr"] + copied["stdout"])
        locked = self.exec(["sh", "-lc", f"chown 0:0 /locked/{locked_name} && chmod 0555 /locked/{locked_name}"], timeout=5, user="0:0")
        if locked["exit_code"] != 0:
            raise SandboxError("Could not protect the compiled executable: " + locked["stderr"] + locked["stdout"])

    def _kill_submission_processes(self) -> None:
        # The idle supervisor runs as root with a narrow set of cleanup/staging capabilities.
        result = self.exec(["pkill", "-9", "-u", "1000"], timeout=3, user="0:0")
        if result.get("timed_out") or result.get("exit_code") not in (0, 1):
            raise SandboxError("Could not clean up student processes: " + result.get("stderr", ""))

    def clear_workspaces(self) -> None:
        """Remove all oracle artifacts before staging the student submission."""
        self._kill_submission_processes()
        for directory in ("/work", "/locked"):
            result = self.exec(
                ["find", directory, "-mindepth", "1", "-maxdepth", "1", "-exec", "rm", "-rf", "--", "{}", "+"],
                timeout=8, user="0:0",
            )
            if result.get("timed_out") or result.get("exit_code") != 0:
                raise SandboxError(f"Could not clear the reference workspace at {directory}: " + result.get("stderr", ""))

    def compile(self, source_name: str, output_name: str, *, flags: list[str] | None = None, timeout: float = 15) -> dict[str, Any]:
        if source_name not in {"submission.c", "reference.c"}:
            raise SandboxError("Unknown protected source name.")
        if output_name not in {"student", "student_san", "reference"}:
            raise SandboxError("Unknown compiler output name.")
        flags = flags or ["-std=c11", "-O2", "-pipe", "-fno-ident", "-fdiagnostics-color=never", "-Wall", "-Wextra", "-Wpedantic"]
        # Source files are immutable root-owned copies in /locked; only the
        # compiler output is written to the UID-1000 workspace.
        args = ["gcc", *flags, f"/locked/{source_name}", "-o", f"/work/{output_name}"]
        result = self.exec(args, timeout=timeout)
        result["warnings"] = [line.strip() for line in (result["stderr"] + result["stdout"]).splitlines() if "warning:" in line]
        return result

    def run_program(self, program_name: str, stdin: str, *, timeout_s: float = 2.0) -> dict[str, Any]:
        if not self.container_id:
            raise SandboxError("Sandbox is not running.")
        if program_name not in {"student", "reference", "student_san"}:
            raise SandboxError("Unknown protected executable name.")
        # stdin is piped directly; no attacker-writable input/output file is
        # shared with the submission. The executable itself lives in a root-
        # owned, read-only-to-student tmpfs directory.
        script = (
            "set +e; "
            f"timeout -k 1s {max(timeout_s, 0.01):.3f}s /locked/{program_name} "
            f"2> >(head -c 8192 >&2) | head -c {self.max_output + 1}; "
            "rc=${PIPESTATUS[0]}; "
            "printf '\n__CG_RC__%s\n' \"$rc\""
        )
        try:
            result = self._command(
                ["docker", "exec", "-i", "--user", "0:0", self.container_id,
                 "setpriv", "--bounding-set=-all", "--inh-caps=-all", "--ambient-caps=-all",
                 "--reuid=1000", "--regid=1000", "--groups=1000", "--", "bash", "-lc", script],
                timeout=timeout_s + 4,
                input=stdin.encode("utf-8"),
            )
        except SandboxError as exc:
            self._kill_submission_processes()
            return {"stdout": "", "stderr": str(exc), "exit_code": None, "timed_out": True, "output_truncated": False}
        output = result.stdout
        marker = b"\n__CG_RC__"
        position = output.rfind(marker)
        if position < 0:
            self._kill_submission_processes()
            return {"stdout": "", "stderr": result.stderr.decode("utf-8", errors="replace")[:8192] or "Sandbox output protocol failed.",
                    "exit_code": result.returncode, "timed_out": False, "output_truncated": False}
        captured = output[:position]
        rc_line = output[position + len(marker):].splitlines()[0].decode("ascii", errors="replace")
        try:
            exit_code = int(rc_line)
        except ValueError:
            exit_code = result.returncode
        truncated = len(captured) > self.max_output
        stdout = captured[:self.max_output].decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")[:8192]
        self._kill_submission_processes()
        return {"stdout": stdout, "stderr": stderr, "exit_code": exit_code,
                "timed_out": exit_code in (124, 137), "output_truncated": truncated}

    def compiler_version(self) -> str:
        result = self.exec(["gcc", "--version"], timeout=5)
        return (result["stdout"] or result["stderr"]).splitlines()[0] if result["exit_code"] == 0 else "unavailable"
