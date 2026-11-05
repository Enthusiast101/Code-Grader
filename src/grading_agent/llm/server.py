"""Start and supervise the local llama.cpp server using central settings."""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time

import requests

from grading_agent.config import settings

logger = logging.getLogger(__name__)


def _verify_json_schema_compatible_server(executable: str) -> str:
    """Reject llama.cpp builds that cannot handle Qwen empty-think JSON schemas.

    llama.cpp PR #31380 fixed sampler initialization for Qwen/LFM templates
    when ``reasoning_format=none`` and JSON-schema output are used together.
    Build b12005 (commit 3c809f44) is the first release verified upstream.
    """
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "Could not verify llama-server version. Install a llama.cpp build "
            "that includes PR #31380 (b12005 / commit 3c809f44 or newer)."
        ) from exc

    version_text = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if result.returncode != 0 or not version_text:
        raise RuntimeError(
            "llama-server --version did not return version information. "
            "Install a build that includes llama.cpp PR #31380 (b12005 or newer)."
        )

    # Standard llama.cpp output resembles: version: 0.0.0 (build 12005, commit ...).
    # Release binaries print: version: 12005 (3c809f44).
    build_match = re.search(r"\bbuild\s*[=:]?\s*(\d+)\b", version_text, re.IGNORECASE)
    if not build_match:
        build_match = re.search(r"\bb(\d{4,})\b", version_text, re.IGNORECASE)
    if not build_match:
        build_match = re.search(r"\bversion:\s*(\d{4,})\s*\(", version_text, re.IGNORECASE)
    # Semantic-version packages (e.g. Homebrew 0.9.0, cut from nightly b12110)
    # postdate b12005; 0.9.0 is the first such release verified to include PR #31380.
    semver_match = re.search(r"\bversion:?\s*v?(\d+)\.(\d+)\.(\d+)", version_text, re.IGNORECASE)
    if not build_match and semver_match:
        semver = tuple(int(part) for part in semver_match.groups())
        if semver >= (0, 9, 0):
            logger.info("llama-server JSON-schema compatibility verified: version %s", ".".join(map(str, semver)))
            return version_text
    commit_match = re.search(r"\bcommit\s+([0-9a-f]{7,40})\b", version_text, re.IGNORECASE)
    known_fixed_commits = ("3c809f44", "1052f7e")
    has_known_fix = bool(
        commit_match and any(commit_match.group(1).lower().startswith(prefix) for prefix in known_fixed_commits)
    )

    if build_match:
        build_number = int(build_match.group(1))
        minimum = int(settings.LLAMA_MIN_JSON_SCHEMA_BUILD)
        if build_number < minimum and not has_known_fix:
            raise RuntimeError(
                f"Incompatible llama.cpp build {build_number}; Qwen3 JSON-schema grading "
                f"requires build b{minimum} or newer (PR #31380 fixes the empty <think> "
                "prefix / grammar sampler error). Update llama.cpp and restart Gradio. "
                f"Detected version: {version_text[:300]}"
            )
        logger.info("llama-server JSON-schema compatibility verified: build b%s", build_number)
        return version_text

    if has_known_fix:
        logger.info("llama-server JSON-schema compatibility verified by fixed commit %s", commit_match.group(1))
        return version_text

    if settings.LLAMA_ALLOW_UNVERIFIED_SERVER:
        logger.warning(
            "Could not parse llama.cpp build number; continuing because "
            "LLAMA_ALLOW_UNVERIFIED_SERVER=true. Version: %s", version_text[:300]
        )
        return version_text

    raise RuntimeError(
        "Could not parse the llama.cpp build number, so JSON-schema compatibility "
        "cannot be verified. Require b12005 or newer (PR #31380), or set "
        "LLAMA_ALLOW_UNVERIFIED_SERVER=true only if this custom build contains that fix. "
        f"Detected version: {version_text[:300]}"
    )


def initiate_server() -> subprocess.Popen:
    """Start llama-server and wait until its health endpoint responds."""
    executable = shutil.which(settings.LLAMA_SERVER)

    if executable is None:
        candidate = settings.LLAMA_SERVER
        if candidate and os.path.isfile(candidate):
            executable = os.path.abspath(candidate)

    if executable is None:
        raise FileNotFoundError(
            "Cannot find llama-server. Install llama.cpp and add "
            "llama-server to PATH, or set LLAMA_SERVER in "
            "src/grading_agent/config/settings.py (or its environment override)."
        )

    # Fail before model loading or any grading/compression request. Older
    # llama.cpp builds emit an empty <think> prefill that breaks JSON grammar
    # sampler initialization; retries cannot repair a server-side incompatibility.
    _verify_json_schema_compatible_server(executable)

    reasoning_enabled = bool(settings.runtime_value("LLAMA_REASONING_ENABLED"))
    reasoning_format = str(settings.runtime_value("LLAMA_REASONING_FORMAT")) if reasoning_enabled else "none"
    reasoning_budget = int(settings.runtime_value("LLAMA_REASONING_BUDGET")) if reasoning_enabled else 0
    command = [
        executable,
        "-hf", str(settings.runtime_value("MODEL")),
        "--temp", str(settings.runtime_value("TEMPERATURE")),
        "--seed", str(settings.SEED),
        "--ctx-size", str(settings.LLAMA_CONTEXT_SIZE),
        "--host", str(settings.HOST),
        "--port", str(settings.PORT),
        "--reasoning", "on" if reasoning_enabled else "off",
        "--reasoning-format", reasoning_format,
        "--reasoning-budget", str(reasoning_budget),
    ]

    logger.info("Starting llama-server with model %s", settings.runtime_value("MODEL"))
    logger.info(
        "llama-server configuration: host=%s port=%s context=%s reasoning=%s "
        "reasoning_format=%s reasoning_budget=%s",
        settings.HOST, settings.PORT, settings.LLAMA_CONTEXT_SIZE,
        reasoning_enabled, reasoning_format,
        reasoning_budget,
    )

    process = subprocess.Popen(command)
    deadline = time.monotonic() + settings.LLAMA_STARTUP_TIMEOUT_S

    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with code {process.returncode}. "
                    "Check the server output for details."
                )

            try:
                response = requests.get(
                    settings.LLAMA_HEALTH_URL,
                    timeout=settings.LLAMA_STARTUP_HEALTH_TIMEOUT_S,
                )
                if response.status_code == 200:
                    logger.info("llama-server is ready.")
                    return process
            except requests.RequestException:
                pass

            time.sleep(settings.LLAMA_HEALTH_POLL_INTERVAL_S)

        raise TimeoutError(
            f"llama-server did not become ready within "
            f"{settings.LLAMA_STARTUP_TIMEOUT_S:g} seconds: "
            f"{settings.LLAMA_HEALTH_URL}"
        )

    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=settings.LLAMA_SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


def main() -> None:
    process = initiate_server()
    try:
        process.wait()
    except KeyboardInterrupt:
        logger.warning("Stopping llama-server...")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=settings.LLAMA_SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
