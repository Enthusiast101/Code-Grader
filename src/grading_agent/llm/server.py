import logging
import subprocess
import time
import requests
from grading_agent.config.settings import HOST, PORT, MODEL

logger = logging.getLogger(__name__)


def initiate_server():
    command = [
        "llama",
        "serve",
        "-hf",
        MODEL,
        "--host",
        HOST,
        "--port",
        str(PORT),
    ]

    logger.info("Starting llama-server...")
    logger.info("Command:", " ".join(command))

    process = subprocess.Popen(
        command,
        stdout=None,
        stderr=None,
    )

    logger.info(f"Waiting for llama-server at http://{HOST}:{PORT}...")

    while True:
        if process.poll() is not None:
            raise RuntimeError(
                f"llama-server exited with code {process.returncode}"
            )

        try:
            response = requests.get(
                f"http://{HOST}:{PORT}/health",
                timeout=1,
            )

            if response.status_code == 200:
                logger.info("llama-server is ready.")
                return process
            
            else:
                logger.warning(f"Health check returned non-200 status: {response.status_code}")

        except requests.exceptions.RequestException:
            pass

        time.sleep(0.5)

    return 


if __name__ == "__main__":
    process = initiate_server()

    try:
        process.wait()

    except KeyboardInterrupt:
        print("\nStopping llama-server...", flush=True)
        process.terminate()
        process.wait()