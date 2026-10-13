
import logging
import os
import shutil
import subprocess
import time

import requests

from grading_agent.config.settings import HOST, PORT, MODEL

logger = logging.getLogger(__name__)

LLAMA_SERVER = os.getenv("LLAMA_SERVER", "llama-server")


def initiate_server():
    executable = shutil.which(LLAMA_SERVER)

    if executable is None and os.path.isfile(LLAMA_SERVER):
        executable = os.path.abspath(LLAMA_SERVER)

    if executable is None:
        raise FileNotFoundError(
            "Cannot find llama-server. Install llama.cpp and add "
            "llama-server.exe to PATH, or set LLAMA_SERVER to its full path."
        )

    command = [
        executable,
        "-hf",
        MODEL,
        "--host",
        HOST,
        "--port",
        str(PORT),
    ]

    logger.info("Starting llama-server...")
    logger.info("Command: %s", " ".join(command))

    process = subprocess.Popen(command)

    health_host = "127.0.0.1" if HOST in ("0.0.0.0", "::") else HOST
    health_url = f"http://{health_host}:{PORT}/health"
    deadline = time.monotonic() + 300

    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with code {process.returncode}. "
                    "Check the server output for details."
                )

            try:
                response = requests.get(health_url, timeout=1)
                if response.status_code == 200:
                    logger.info("llama-server is ready.")
                    return process
            except requests.exceptions.RequestException:
                pass

            time.sleep(0.5)

        raise TimeoutError(
            f"llama-server did not become ready within 300 seconds: "
            f"{health_url}"
        )

    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


if __name__ == "__main__":
    process = initiate_server()

    try:
        process.wait()
    except KeyboardInterrupt:
        logger.warning("Stopping llama-server...")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
