from __future__ import annotations

from unittest.mock import MagicMock, patch

from grading_agent.config import settings
from grading_agent.llm import server


def test_default_model_is_qwen35_2b_q4km() -> None:
    assert settings.MODEL == "gguf/Qwen3.5-2B-GGUF:Q4_K_M"


def test_previous_bundled_default_is_migrated_to_qwen35() -> None:
    assert settings._resolve_model_from_env("Qwen/Qwen3-4B-GGUF:Q4_K_M") == settings.DEFAULT_MODEL


def test_custom_model_override_is_preserved() -> None:
    custom = "some-org/custom-coder-GGUF:Q5_K_M"
    assert settings._resolve_model_from_env(custom) == custom


def test_managed_server_passes_selected_model_to_llama_cpp() -> None:
    process = MagicMock()
    process.poll.return_value = None
    healthy = MagicMock(status_code=200)
    with patch.object(server.shutil, "which", return_value="/usr/bin/llama-server"), \
         patch.object(server, "_verify_json_schema_compatible_server"), \
         patch.object(server.subprocess, "Popen", return_value=process) as popen, \
         patch.object(server.requests, "get", return_value=healthy):
        server.initiate_server()

    command = popen.call_args.args[0]
    model_index = command.index("-hf") + 1
    assert command[model_index] == settings.MODEL
    assert command[model_index] == "gguf/Qwen3.5-2B-GGUF:Q4_K_M"
