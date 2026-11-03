from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any


class LLMError(RuntimeError):
    pass


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    # Recover the first complete JSON object without being confused by braces
    # inside quoted strings. This handles harmless prose/code fences around JSON.
    start = text.find("{")
    while start >= 0:
        depth, in_string, escaped = 0, False, False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start:index + 1])
                        if isinstance(value, dict):
                            return value
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise LLMError("The language model response did not contain a valid JSON object.")


def extract_c_code(text: str) -> str:
    text = (text or "").strip()
    match = re.search(r"```(?:c|C)?\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        text = match.group(1).strip()
    # If the model prefaced the code, keep the complete translation unit.
    include = text.find("#include")
    if include >= 0:
        text = text[include:]
    if not re.search(r"\bmain\s*\(", text) or not re.search(r"\breturn\b|\bprintf\s*\(|\bputs\s*\(", text):
        raise LLMError("The generated reference response does not look like a complete C program.")
    return text.strip() + "\n"


class LLMClient:
    """Small OpenAI-compatible client for a local llama-server."""

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        base_url = str(os.getenv("LLAMA_BASE_URL") or config.get("base_url", "http://127.0.0.1:8080/v1")).rstrip("/")
        self.base_url = base_url if base_url.endswith("/v1") else base_url + "/v1"
        inferred_health = self.base_url[:-3].rstrip("/") + "/health"
        health_url = os.getenv("LLAMA_HEALTH_URL") or config.get("health_url") or inferred_health
        self.health_url = str(health_url).rstrip("/")
        self.model = str(os.getenv("LLAMA_MODEL") or config.get("model", "gguf/Qwen3.5-2B-GGUF:Q4_K_M"))
        self.server_executable = str(os.getenv("LLAMA_SERVER") or config.get("server_executable", "llama-server"))
        self.host = str(os.getenv("LLAMA_HOST") or config.get("host", "127.0.0.1"))
        self.port = int(os.getenv("LLAMA_PORT") or config.get("port", 8080))
        self.context = int(os.getenv("LLAMA_CONTEXT_SIZE") or config.get("context_size", 8192))
        self.seed = int(os.getenv("LLAMA_SEED") or config.get("seed", 42))
        self.temperature = 0.0
        self.timeout = float(os.getenv("LLAMA_TIMEOUT_S") or config.get("timeout_s", 180))
        self.max_tokens = int(os.getenv("LLAMA_MAX_TOKENS") or config.get("max_tokens", 4096))
        self.start_server = bool(config.get("start_server", True))
        self.process: subprocess.Popen | None = None
        self.started_here = False
        self._server_log = None

    def _healthy(self) -> bool:
        try:
            req = urllib.request.Request(self.health_url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status == 200
        except Exception:
            return False

    def ensure_ready(self) -> None:
        if self._healthy():
            return
        if not self.start_server:
            raise LLMError(f"No local LLM server is responding at {self.health_url}.")
        executable = shutil.which(self.server_executable)
        if executable is None and os.path.isfile(self.server_executable):
            executable = os.path.abspath(self.server_executable)
        if executable is None:
            raise LLMError(
                f"llama-server was not found ({self.server_executable}). Install llama.cpp, "
                "add llama-server to PATH, or reuse a running OpenAI-compatible local server."
            )
        command = [
            executable, "-hf", self.model,
            "--host", self.host, "--port", str(self.port),
            "--ctx-size", str(self.context), "--temp", "0", "--seed", str(self.seed),
            "--reasoning", "off",
        ]
        self._server_log = tempfile.TemporaryFile(mode="w+b")
        try:
            self.process = subprocess.Popen(command, stdout=self._server_log, stderr=subprocess.STDOUT)
        except OSError as exc:
            self.close()
            raise LLMError(f"Could not start llama-server: {exc}") from exc
        self.started_here = True
        deadline = time.monotonic() + float(os.getenv("LLAMA_STARTUP_TIMEOUT_S", "300"))
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                code = self.process.returncode
                detail = self._server_log_tail()
                self.close()
                suffix = f" Server log: {detail}" if detail else ""
                raise LLMError(f"llama-server exited during startup with code {code}.{suffix}")
            if self._healthy():
                return
            time.sleep(0.5)
        detail = self._server_log_tail()
        self.close()
        suffix = f" Server log: {detail}" if detail else ""
        raise LLMError(f"llama-server did not become ready at {self.health_url}.{suffix}")

    def _server_log_tail(self, limit: int = 2000) -> str:
        if self._server_log is None:
            return ""
        try:
            self._server_log.flush()
            self._server_log.seek(0, os.SEEK_END)
            size = self._server_log.tell()
            self._server_log.seek(max(0, size - limit))
            return self._server_log.read(limit).decode("utf-8", errors="replace").strip()
        except (OSError, ValueError):
            return ""

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None
        if self._server_log is not None:
            self._server_log.close()
            self._server_log = None

    def _request(self, messages: list[dict[str, str]], max_tokens: int | None = None, *, json_mode: bool = False) -> str:
        self.ensure_ready()
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "seed": self.seed,
            "max_tokens": int(max_tokens or self.max_tokens),
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1500]
            raise LLMError(f"LLM HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"LLM request failed: {type(exc).__name__}: {exc}") from exc
        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = message.get("content", "")
            if isinstance(content, list):
                content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
            content = str(content or "").strip()
            finish = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("The LLM response did not contain choices[0].message.content.") from exc
        if not content:
            raise LLMError("The LLM returned an empty content field.")
        if finish == "length":
            raise LLMError("The LLM response was truncated at its token limit; increase llm.max_tokens.")
        return content

    def json(self, system: str, user: str, *, max_tokens: int | None = None) -> dict[str, Any]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        content = self._request(messages, max_tokens=max_tokens, json_mode=True)
        try:
            return _extract_json(content)
        except LLMError:
            # A single deterministic-format repair pass protects the pipeline
            # from wrappers/code fences while preserving the original task.
            repair = [
                {"role": "system", "content": "Return the supplied object as valid JSON only. Preserve all information. Do not add commentary or markdown."},
                {"role": "user", "content": "Convert this response to one valid JSON object.\n\n" + content[:20000]},
            ]
            repaired = self._request(repair, max_tokens=max_tokens, json_mode=True)
            return _extract_json(repaired)

    def text(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        return self._request(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
        )
