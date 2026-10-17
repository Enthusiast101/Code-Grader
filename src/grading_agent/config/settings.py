"""Central configuration for the local LLM and Docker sandbox."""

# ---------------------------------------------------------------------------
# Local Llama server
# ---------------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 8080
LLAMA_BASE_URL = f"http://{HOST}:{PORT}"

MODEL = "Qwen/Qwen3-4B-GGUF:Q4_K_M"
TEMPERATURE = 0.2
MAX_TOKENS = 2048

# ---------------------------------------------------------------------------
# Docker sandbox
# ---------------------------------------------------------------------------
DOCKER_SANDBOX_IMAGE = "agent-sandbox:latest"

# Resource limits applied to each sandbox container.
DOCKER_MEMORY_LIMIT = "512m"
DOCKER_CPU_LIMIT = 1.0
DOCKER_PIDS_LIMIT = 64
DOCKER_NETWORK_MODE = "none"
DOCKER_READ_ONLY_ROOTFS = True

# Writable, size-limited filesystems inside the otherwise read-only container.
DOCKER_WORKSPACE_TMPFS_SIZE = "256m"
DOCKER_TMPFS_SIZE = "64m"
DOCKER_CONTAINER_UID = 10001
DOCKER_CONTAINER_GID = 10001

# Execution and file handling limits.
DOCKER_DEFAULT_TIMEOUT_S = 5
DOCKER_MAX_TIMEOUT_S = 120
DOCKER_MAX_OUTPUT_BYTES = 64 * 1024
DOCKER_MAX_FILE_BYTES = 2 * 1024 * 1024
DOCKER_WRITE_CHUNK_CHARS = 32 * 1024
