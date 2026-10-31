"""Single source of truth for Code Checking Assistant configuration.

Change runtime/model/sandbox/UI/benchmark settings here rather than editing
values inside the implementation modules. Environment overrides are read here
only, so configuration remains discoverable in one place.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}.") from exc


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be numeric, got {value!r}.") from exc


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be true/false, got {value!r}.")



# ---------------------------------------------------------------------------
# Project paths and logging
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RULES_PATH = PROJECT_ROOT / "rules.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
GRADIO_STAGING_DIR = PROJECT_ROOT / ".gradio_runs"
GRADIO_REPORTS_DIR = REPORTS_DIR / "gradio"
FORM_AI_GRADIO_RUNS_DIR = PROJECT_ROOT / "benchmark" / "gradio_runs"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT_APP = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_FORMAT_BENCHMARK = "%(asctime)s | %(levelname)s | %(message)s"

# ---------------------------------------------------------------------------
# Local LLM / llama-server
# ---------------------------------------------------------------------------
LLAMA_SERVER = os.getenv("LLAMA_SERVER", "llama-server")
HOST = os.getenv("LLAMA_HOST", "127.0.0.1")
PORT = _env_int("LLAMA_PORT", 8080)
LLAMA_BASE_URL = f"http://{HOST}:{PORT}"
LLAMA_API_PATH = "/v1/chat/completions"
LLAMA_HEALTH_PATH = "/health"
LLAMA_HEALTH_HOST = "127.0.0.1" if HOST in {"0.0.0.0", "::"} else HOST
LLAMA_HEALTH_URL = f"http://{LLAMA_HEALTH_HOST}:{PORT}{LLAMA_HEALTH_PATH}"

MODEL = os.getenv("LLAMA_MODEL", "Qwen/Qwen3-4B-GGUF:Q4_K_M")
TEMPERATURE = _env_float("LLAMA_TEMPERATURE", 0.0)
SEED = _env_int("LLAMA_SEED", 42)
MAX_TOKENS = _env_int("LLAMA_MAX_TOKENS", 3072)
LLAMA_CONTEXT_SIZE = _env_int("LLAMA_CONTEXT_SIZE", 8192)
LLAMA_REASONING_ENABLED = _env_bool("LLAMA_REASONING_ENABLED", True)
LLAMA_REASONING_FORMAT = os.getenv("LLAMA_REASONING_FORMAT", "deepseek")
LLAMA_REASONING_BUDGET = _env_int("LLAMA_REASONING_BUDGET", 1536)
LLAMA_CHAT_TEMPLATE_KWARGS = {"enable_thinking": LLAMA_REASONING_ENABLED}

# Server lifecycle and HTTP request timeouts (connect timeout, read timeout).
LLAMA_STARTUP_TIMEOUT_S = _env_float("LLAMA_STARTUP_TIMEOUT_S", 300.0)
LLAMA_STARTUP_HEALTH_TIMEOUT_S = _env_float("LLAMA_STARTUP_HEALTH_TIMEOUT_S", 1.0)
LLAMA_HEALTH_TIMEOUT_S = _env_float("LLAMA_HEALTH_TIMEOUT_S", 2.0)
LLAMA_HEALTH_POLL_INTERVAL_S = _env_float("LLAMA_HEALTH_POLL_INTERVAL_S", 0.5)
LLAMA_SHUTDOWN_TIMEOUT_S = _env_float("LLAMA_SHUTDOWN_TIMEOUT_S", 5.0)
LLAMA_REQUEST_CONNECT_TIMEOUT_S = _env_float("LLAMA_REQUEST_CONNECT_TIMEOUT_S", 5.0)
LLAMA_REQUEST_READ_TIMEOUT_S = _env_float("LLAMA_REQUEST_READ_TIMEOUT_S", 180.0)
LLAMA_REQUEST_TIMEOUT = (LLAMA_REQUEST_CONNECT_TIMEOUT_S, LLAMA_REQUEST_READ_TIMEOUT_S)

# ---------------------------------------------------------------------------
# LLM review / RSI behaviour
# ---------------------------------------------------------------------------
LLM_REVIEW_MAX_TOKENS = _env_int("LLM_REVIEW_MAX_TOKENS", MAX_TOKENS)
LLM_SOURCE_AUDIT_MAX_TOKENS = _env_int("LLM_SOURCE_AUDIT_MAX_TOKENS", MAX_TOKENS)
LLM_RSI_MAX_TOKENS = _env_int("LLM_RSI_MAX_TOKENS", MAX_TOKENS)
LLM_RETRY_ATTEMPTS = _env_int("LLM_RETRY_ATTEMPTS", 2)
LLM_REVIEW_MAX_ITERATIONS = _env_int("LLM_REVIEW_MAX_ITERATIONS", 3)
LLM_DEFAULT_MAX_FINDINGS = _env_int("LLM_DEFAULT_MAX_FINDINGS", 30)

LLM_MAX_EVIDENCE_CHARS = _env_int("LLM_MAX_EVIDENCE_CHARS", 4200)
LLM_MAX_SOURCE_CHARS = _env_int("LLM_MAX_SOURCE_CHARS", 6000)
LLM_MAX_REVIEW_STRING_CHARS = _env_int("LLM_MAX_REVIEW_STRING_CHARS", 240)
LLM_RSI_PROMPT_CHAR_LIMIT = _env_int("LLM_RSI_PROMPT_CHAR_LIMIT", 4200)
LLM_SOURCE_CHUNK_OVERLAP_LINES = _env_int("LLM_SOURCE_CHUNK_OVERLAP_LINES", 12)
LLM_SOURCE_CHUNK_OVERLAP_CHARS = _env_int("LLM_SOURCE_CHUNK_OVERLAP_CHARS", 200)

# ---------------------------------------------------------------------------
# Docker sandbox
# ---------------------------------------------------------------------------
DOCKER_SANDBOX_IMAGE = os.getenv("DOCKER_SANDBOX_IMAGE", "agent-sandbox:latest")
DOCKER_MEMORY_LIMIT = os.getenv("DOCKER_MEMORY_LIMIT", "512m")
DOCKER_CPU_LIMIT = _env_float("DOCKER_CPU_LIMIT", 1.0)
DOCKER_PIDS_LIMIT = _env_int("DOCKER_PIDS_LIMIT", 64)
DOCKER_NETWORK_MODE = os.getenv("DOCKER_NETWORK_MODE", "none")
DOCKER_READ_ONLY_ROOTFS = _env_bool("DOCKER_READ_ONLY_ROOTFS", True)
DOCKER_WORKSPACE_TMPFS_SIZE = os.getenv("DOCKER_WORKSPACE_TMPFS_SIZE", "256m")
DOCKER_TMPFS_SIZE = os.getenv("DOCKER_TMPFS_SIZE", "64m")
DOCKER_CONTAINER_UID = _env_int("DOCKER_CONTAINER_UID", 10001)
DOCKER_CONTAINER_GID = _env_int("DOCKER_CONTAINER_GID", 10001)
DOCKER_DEFAULT_TIMEOUT_S = _env_float("DOCKER_DEFAULT_TIMEOUT_S", 5.0)
DOCKER_MAX_TIMEOUT_S = _env_float("DOCKER_MAX_TIMEOUT_S", 120.0)
DOCKER_MAX_OUTPUT_BYTES = _env_int("DOCKER_MAX_OUTPUT_BYTES", 64 * 1024)
DOCKER_MAX_FILE_BYTES = _env_int("DOCKER_MAX_FILE_BYTES", 2 * 1024 * 1024)
DOCKER_WRITE_CHUNK_CHARS = _env_int("DOCKER_WRITE_CHUNK_CHARS", 32 * 1024)

# ---------------------------------------------------------------------------
# FormAI-v2 benchmark defaults
# ---------------------------------------------------------------------------
FORM_AI_DEFAULT_TOOL_TIMEOUT_S = _env_float("FORM_AI_DEFAULT_TOOL_TIMEOUT_S", 20.0)
FORM_AI_DEFAULT_MAX_FINDINGS = _env_int("FORM_AI_DEFAULT_MAX_FINDINGS", LLM_DEFAULT_MAX_FINDINGS)
FORM_AI_DEFAULT_RSI_ITERATIONS = _env_int("FORM_AI_DEFAULT_RSI_ITERATIONS", LLM_REVIEW_MAX_ITERATIONS)
FORM_AI_DEFAULT_MEMORY_LIMIT = os.getenv("FORM_AI_DEFAULT_MEMORY_LIMIT", DOCKER_MEMORY_LIMIT)
FORM_AI_LINE_TOLERANCE = _env_int("FORM_AI_LINE_TOLERANCE", 2)
FORM_AI_DEFAULT_MAX_SAMPLES = None  # None means evaluate the complete supplied split.
FORM_AI_PREDICTION_MODES = ("tools_only", "llm_only", "hybrid", "hybrid_rsi")
FORM_AI_RSI_METRIC_DEFAULTS = (
    ("structure", "cppcheck_source", 0.5),
    ("memory_safety", "llm_source_audit", 0.5),
)

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
UI_HOST = os.getenv("GRADIO_HOST", "127.0.0.1")
UI_PORT = _env_int("GRADIO_PORT", 7860)
UI_SHOW_ERROR = _env_bool("GRADIO_SHOW_ERROR", True)
UI_MAX_UPLOAD_BYTES = _env_int("UI_MAX_UPLOAD_BYTES", DOCKER_MAX_FILE_BYTES)
UI_DEFAULT_TOOL_TIMEOUT_S = FORM_AI_DEFAULT_TOOL_TIMEOUT_S
UI_DEFAULT_MAX_FINDINGS = FORM_AI_DEFAULT_MAX_FINDINGS
UI_DEFAULT_RSI_ITERATIONS = FORM_AI_DEFAULT_RSI_ITERATIONS
UI_DOCKER_MEMORY_CHOICES = ("512m", "1g", "2g")
UI_DEFAULT_MEMORY_LIMIT = FORM_AI_DEFAULT_MEMORY_LIMIT

# ---------------------------------------------------------------------------
# Analyzer defaults (rules.json parameters override these values)
# ---------------------------------------------------------------------------
DEFAULT_SOURCE_FILENAME = "main.c"
ANALYZER_COMPILE_TIMEOUT_S = _env_float("ANALYZER_COMPILE_TIMEOUT_S", 10.0)
ANALYZER_IO_RUN_TIMEOUT_S = _env_float("ANALYZER_IO_RUN_TIMEOUT_S", 2.0)
ANALYZER_EDGE_RUN_TIMEOUT_S = _env_float("ANALYZER_EDGE_RUN_TIMEOUT_S", 2.0)
ANALYZER_RUNTIME_RUN_TIMEOUT_S = _env_float("ANALYZER_RUNTIME_RUN_TIMEOUT_S", 2.0)
ANALYZER_MEMORY_RUN_TIMEOUT_S = _env_float("ANALYZER_MEMORY_RUN_TIMEOUT_S", 5.0)
ANALYZER_RUN_SANITIZERS = _env_bool("ANALYZER_RUN_SANITIZERS", True)

ANALYZER_COMPILER_TIMEOUT_S = _env_float("ANALYZER_COMPILER_TIMEOUT_S", 10.0)
ANALYZER_COMPILER_FLAGS = ("-std=c11", "-Wall", "-Wextra", "-Wpedantic")
ANALYZER_PREFLIGHT_FLAGS = ("-std=c11", "-Wall", "-Wextra", "-O2")
ANALYZER_PREFLIGHT_TIMEOUT_S = _env_float("ANALYZER_PREFLIGHT_TIMEOUT_S", 10.0)
ANALYZER_TOOL_VERSION_TIMEOUT_S = _env_float("ANALYZER_TOOL_VERSION_TIMEOUT_S", 3.0)
ANALYZER_COMPILER_WARNING_PENALTY = _env_float("ANALYZER_COMPILER_WARNING_PENALTY", 5.0)
ANALYZER_STATIC_TIMEOUT_S = _env_float("ANALYZER_STATIC_TIMEOUT_S", 15.0)
ANALYZER_STATIC_PENALTIES = {
    "error": 25.0,
    "warning": 10.0,
    "style": 3.0,
    "performance": 4.0,
    "portability": 3.0,
    "information": 0.5,
}

ANALYZER_RUNTIME_TARGET_MS = _env_float("ANALYZER_RUNTIME_TARGET_MS", 50.0)
ANALYZER_RUNTIME_REPETITIONS = _env_int("ANALYZER_RUNTIME_REPETITIONS", 3)
ANALYZER_RUNTIME_MAX_REPETITIONS = _env_int("ANALYZER_RUNTIME_MAX_REPETITIONS", 10)

ANALYZER_MIN_FUNCTIONS = _env_int("ANALYZER_MIN_FUNCTIONS", 2)
ANALYZER_EXPECT_HELPERS = _env_bool("ANALYZER_EXPECT_HELPERS", True)
ANALYZER_MAX_CYCLOMATIC = _env_int("ANALYZER_MAX_CYCLOMATIC", 8)
ANALYZER_MAX_FUNCTION_LENGTH = _env_int("ANALYZER_MAX_FUNCTION_LENGTH", 40)
ANALYZER_COMPLEXITY_PENALTY = _env_float("ANALYZER_COMPLEXITY_PENALTY", 8.0)
ANALYZER_LENGTH_PENALTY_PER_LINE = _env_float("ANALYZER_LENGTH_PENALTY_PER_LINE", 0.5)
ANALYZER_MAX_LINE_LENGTH = _env_int("ANALYZER_MAX_LINE_LENGTH", 100)
ANALYZER_LONG_LINE_PENALTY = _env_float("ANALYZER_LONG_LINE_PENALTY", 1.0)
ANALYZER_SINGLE_CHAR_IDENTIFIER_PENALTY = _env_float("ANALYZER_SINGLE_CHAR_IDENTIFIER_PENALTY", 1.0)
ANALYZER_MISSING_COMMENT_PENALTY = _env_float("ANALYZER_MISSING_COMMENT_PENALTY", 5.0)
ANALYZER_SIMILARITY_THRESHOLD = _env_float("ANALYZER_SIMILARITY_THRESHOLD", 0.85)
ANALYZER_SIMILARITY_MAX_FILES = _env_int("ANALYZER_SIMILARITY_MAX_FILES", 5000)
ANALYZER_SIMILARITY_SHINGLE_SIZE = _env_int("ANALYZER_SIMILARITY_SHINGLE_SIZE", 5)
ANALYZER_SIMILARITY_MIN_SHINGLE_SIZE = 2
ANALYZER_SIMILARITY_MAX_FILE_BYTES = _env_int("ANALYZER_SIMILARITY_MAX_FILE_BYTES", 2_000_000)

# ---------------------------------------------------------------------------
# FormAI dataset preparation defaults
# ---------------------------------------------------------------------------
FORM_AI_SAMPLE_PER_CLASS = _env_int("FORM_AI_SAMPLE_PER_CLASS", 250)
FORM_AI_SAMPLE_SEED = _env_int("FORM_AI_SAMPLE_SEED", 42)
FORM_AI_SAMPLE_SOURCE_OUTPUT = "benchmark/formai_inputs.jsonl"
FORM_AI_SAMPLE_LABELS_OUTPUT = "benchmark/formai_ground_truth.json"
FORM_AI_SPLIT_VALIDATION_FRACTION = _env_float("FORM_AI_SPLIT_VALIDATION_FRACTION", 0.2)
FORM_AI_SPLIT_TEST_FRACTION = _env_float("FORM_AI_SPLIT_TEST_FRACTION", 0.2)
FORM_AI_SPLIT_SEED = _env_int("FORM_AI_SPLIT_SEED", 42)
FORM_AI_SPLIT_OUTPUT_DIR = "benchmark/splits"
FORM_AI_EVALUATION_OUTPUT = "reports/formai-evaluation.json"

# UI sample-count default: zero means process the complete uploaded test split.
UI_DEFAULT_MAX_SAMPLES = _env_int("UI_DEFAULT_MAX_SAMPLES", 0)
UI_DEFAULT_BENCHMARK_MODE = "hybrid_rsi"
