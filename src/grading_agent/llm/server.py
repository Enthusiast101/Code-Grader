"""Start and supervise the local llama.cpp server using central settings."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time

import requests

from grading_agent.config import settings

logger = logging.getLogger(__name__)


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

    command = [
        executable,
        "-hf", str(settings.MODEL),
        "--temp", str(settings.TEMPERATURE),
        "--seed", str(settings.SEED),
        "--ctx-size", str(settings.LLAMA_CONTEXT_SIZE),
        "--host", str(settings.HOST),
        "--port", str(settings.PORT),
        "--reasoning", "on" if settings.LLAMA_REASONING_ENABLED else "off",
        "--reasoning-format", str(settings.LLAMA_REASONING_FORMAT),
        "--reasoning-budget", str(settings.LLAMA_REASONING_BUDGET),
    ]

    logger.info("Starting llama-server with model %s", settings.MODEL)
    logger.info(
        "llama-server configuration: host=%s port=%s context=%s reasoning=%s "
        "reasoning_format=%s reasoning_budget=%s",
        settings.HOST, settings.PORT, settings.LLAMA_CONTEXT_SIZE,
        settings.LLAMA_REASONING_ENABLED, settings.LLAMA_REASONING_FORMAT,
        settings.LLAMA_REASONING_BUDGET,
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
