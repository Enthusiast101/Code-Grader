from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


DEFAULT_METRICS = [
    {"name": "correctness", "analyzer": "io_tests", "weight": 0.50, "enabled": True, "params": {}},
    {"name": "syntax", "analyzer": "compiler_diagnostics", "weight": 0.15, "enabled": True, "params": {"warning_penalty": 5}},
    {"name": "structure", "analyzer": "static_analysis", "weight": 0.10, "enabled": True, "params": {"max_cyclomatic": 12, "max_function_length": 80, "violation_penalty": 15}},
    {"name": "memory_safety", "analyzer": "memory_safety", "weight": 0.20, "enabled": True, "params": {"run_sanitizers": True}},
    {"name": "readability", "analyzer": "readability", "weight": 0.05, "enabled": True, "params": {"max_line_length": 100, "long_line_penalty": 2}},
]

SUPPORTED_ANALYZERS = {
    "io_tests", "correctness", "compiler_diagnostics", "compiler_warnings",
    "static_analysis", "structure", "edge_case_runner", "logical",
    "memory_safety", "sanitizer_run", "readability", "modularity",
}

PLACEHOLDERS = (
    "replace_with", "replace this", "todo", "tbd", "paste the assignment",
    "insert assignment", "<assignment", "your assignment here",
)


class RulesError(ValueError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_weights(metrics: list[dict[str, Any]]) -> None:
    enabled = [m for m in metrics if m["enabled"] and m["weight"] > 0]
    if not enabled:
        raise RulesError("At least one metric must be enabled with a positive weight.")
    total = sum(m["weight"] for m in enabled)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise RulesError(f"Enabled metric weights must sum to 1.0; received {total:.8f}.")


def _validate_metric_params(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Reject malformed scoring controls instead of failing midway through a run."""
    params = dict(params)

    def positive_int(key: str, default: int, maximum: int = 100000) -> None:
        value = params.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise RulesError(f"Metric '{name}' parameter '{key}' must be an integer from 1 to {maximum}.")
        params[key] = value

    def bounded_number(key: str, default: float, maximum: float = 100.0) -> None:
        value = params.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= maximum:
            raise RulesError(f"Metric '{name}' parameter '{key}' must be a finite number from 0 to {maximum:g}.")
        params[key] = float(value)

    if name == "syntax":
        bounded_number("warning_penalty", 5.0)
    elif name == "structure":
        positive_int("min_functions", 1)
        positive_int("max_cyclomatic", 12)
        positive_int("max_function_length", 80)
        bounded_number("violation_penalty", 15.0)
    elif name == "logical":
        pass
    elif name == "memory_safety":
        value = params.get("run_sanitizers", True)
        if not isinstance(value, bool):
            raise RulesError(f"Metric '{name}' parameter 'run_sanitizers' must be a boolean.")
        params["run_sanitizers"] = value
    elif name == "readability":
        positive_int("max_line_length", 100, 10000)
        bounded_number("long_line_penalty", 2.0)
    elif name == "modularity":
        positive_int("min_functions", 1)
        positive_int("min_helpers", max(1, params.get("min_functions", 2) - 1))
        value = params.get("expect_helpers", False)
        if not isinstance(value, bool):
            raise RulesError(f"Metric '{name}' parameter 'expect_helpers' must be a boolean.")
        params["expect_helpers"] = value
    return params


def _normalize_metrics(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        metrics = [dict(item, params=dict(item["params"])) for item in DEFAULT_METRICS]
    elif isinstance(raw, list):
        metrics = []
        aliases = {
            "correctness": "correctness", "io_tests": "correctness",
            "syntax": "syntax", "compiler_diagnostics": "syntax", "compiler_warnings": "syntax",
            "structure": "structure", "static_analysis": "structure",
            "logical": "logical", "edge_case_runner": "logical",
            "memory_safety": "memory_safety", "sanitizer_run": "memory_safety",
            "readability": "readability", "modularity": "modularity",
        }
        for item in raw:
            if not isinstance(item, dict):
                raise RulesError("Each rules.metrics item must be a JSON object.")
            analyzer = str(item.get("analyzer", item.get("name", ""))).strip()
            name = str(item.get("name", analyzer)).strip()
            weight = item.get("weight", 0.0)
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
                raise RulesError(f"Metric '{name}' weight must be a finite non-negative number.")
            enabled = item.get("enabled", True)
            if not isinstance(enabled, bool):
                raise RulesError(f"Metric '{name}' enabled must be a boolean.")
            canonical = aliases.get(analyzer, aliases.get(name))
            if canonical is None:
                # Older project rules contain disabled timing/Big-O/similarity
                # entries. Ignore those safely; fail when they carry grade weight.
                if not enabled or float(weight) == 0.0:
                    continue
                raise RulesError(f"Enabled analyzer '{analyzer}' has no deterministic implementation. Disable it or remove its weight.")
            params = item.get("params", {})
            if not isinstance(params, dict):
                raise RulesError(f"Metric '{name}' params must be an object.")
            params = _validate_metric_params(canonical, params)
            metrics.append({"name": canonical, "analyzer": canonical, "weight": float(weight) if enabled else 0.0, "enabled": enabled and float(weight) > 0, "params": params})
    elif isinstance(raw, dict):
        metrics = []
        for name, item in raw.items():
            if not isinstance(item, dict):
                raise RulesError(f"Metric '{name}' must be a JSON object.")
            metrics.append({"name": name, "analyzer": name, **item})
        return _normalize_metrics(metrics)
    else:
        raise RulesError("rules.metrics must be an array or object.")

    deduplicated: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        name = metric["name"]
        if name in deduplicated:
            raise RulesError(f"Duplicate metric '{name}'.")
        deduplicated[name] = metric
    result = [deduplicated[name] for name in sorted(deduplicated)]
    _validate_weights(result)
    return result


def discover_source(project_root: Path, configured: Any = None) -> Path:
    if configured:
        if not isinstance(configured, str) or not configured.strip():
            raise RulesError("submission.source_file must be a non-empty path.")
        candidate = Path(configured)
        resolved = candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()
        if not resolved.is_relative_to(project_root.resolve()) or not resolved.is_file():
            raise RulesError(f"C source file does not exist inside the project directory: {configured}")
        if resolved.suffix.lower() != ".c":
            raise RulesError("The submission must be a .c file.")
        return resolved

    candidates = sorted(p.resolve() for p in project_root.glob("*.c") if p.is_file() and not p.is_symlink())
    candidates = [p for p in candidates if p.is_relative_to(project_root.resolve()) and not p.name.startswith(("reference_", "oracle_", "generated_"))]
    if len(candidates) != 1:
        names = ", ".join(p.name for p in candidates) or "none"
        raise RulesError(
            "Place exactly one student .c file beside rules.json, or set "
            f"submission.source_file. Found: {names}."
        )
    return candidates[0]


def load_rules(path: str | Path, *, source_override: str | Path | None = None) -> dict[str, Any]:
    rules_path = Path(path).resolve()
    if not rules_path.is_file():
        raise FileNotFoundError(f"Rules file not found: {rules_path}")
    try:
        raw = json.loads(rules_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RulesError(f"Invalid JSON in {rules_path.name}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RulesError("rules.json must contain a JSON object.")

    problem = raw.get("problem", raw.get("assignment", ""))
    if not isinstance(problem, str) or len(problem.strip()) < 30:
        raise RulesError("rules.json -> problem must contain the actual assignment statement (not just a title).")
    normalized_problem = " ".join(problem.casefold().split())
    if any(marker in normalized_problem for marker in PLACEHOLDERS):
        raise RulesError("rules.json still contains a placeholder in the problem statement.")

    limits = raw.get("limits", {})
    if not isinstance(limits, dict):
        raise RulesError("rules.json -> limits must be an object.")
    time_limit = raw.get("time_limit_s", limits.get("time_limit_s", 2.0))
    memory_limit = raw.get("memory_limit_mb", limits.get("memory_limit_mb", 256))
    if isinstance(time_limit, bool) or not isinstance(time_limit, (int, float)) or not math.isfinite(time_limit) or time_limit <= 0 or time_limit > 30:
        raise RulesError("time_limit_s must be a number in the range (0, 30].")
    if isinstance(memory_limit, bool) or not isinstance(memory_limit, int) or not 32 <= memory_limit <= 4096:
        raise RulesError("memory_limit_mb must be an integer from 32 to 4096.")

    submission = raw.get("submission", {})
    if isinstance(submission, str):
        submission = {"source_file": submission}
    if not isinstance(submission, dict):
        raise RulesError("submission must be an object or a source-file string.")
    configured_source = str(source_override) if source_override is not None else submission.get("source_file", raw.get("source_file"))
    source = discover_source(rules_path.parent, configured_source)

    contract = raw.get("problem_contract", {})
    if contract is None:
        contract = {}
    if not isinstance(contract, dict):
        raise RulesError("problem_contract must be a JSON object when provided.")

    examples = raw.get("examples", [])
    if not isinstance(examples, list) or len(examples) > 500:
        raise RulesError("examples must be an array of at most 500 {input, expected} objects.")
    for i, case in enumerate(examples, 1):
        if not isinstance(case, dict) or not isinstance(case.get("input"), str) or not isinstance(case.get("expected"), str):
            raise RulesError(f"Example {i} must contain string input and expected fields.")
        if not case["input"].strip() or len(case["input"]) > 20000 or len(case["expected"]) > 20000:
            raise RulesError(f"Example {i} input must be non-empty and both input/output must be at most 20000 characters.")

    originality = raw.get("originality", {})
    if not isinstance(originality, dict):
        raise RulesError("originality must be a JSON object.")
    originality_enabled = originality.get("enabled", False)
    if not isinstance(originality_enabled, bool):
        raise RulesError("originality.enabled must be a boolean.")
    corpus_dir = originality.get("corpus_dir", "submissions")
    if not isinstance(corpus_dir, str) or not corpus_dir.strip() or Path(corpus_dir).is_absolute():
        raise RulesError("originality.corpus_dir must be a non-empty relative path.")
    corpus_path = (rules_path.parent / corpus_dir).resolve()
    if not corpus_path.is_relative_to(rules_path.parent.resolve()):
        raise RulesError("originality.corpus_dir must remain inside the project directory.")
    threshold = originality.get("threshold", 0.85)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0.5 <= threshold <= 1.0:
        raise RulesError("originality.threshold must be between 0.50 and 1.00.")
    min_tokens = originality.get("min_tokens", 80)
    max_files = originality.get("max_files", 500)
    max_file_bytes = originality.get("max_file_bytes", 1_000_000)
    for label, value, minimum, maximum in (("min_tokens", min_tokens, 20, 10000), ("max_files", max_files, 1, 10000), ("max_file_bytes", max_file_bytes, 1024, 20_000_000)):
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise RulesError(f"originality.{label} must be an integer from {minimum} to {maximum}.")

    test_generation = raw.get("test_generation", {})
    if not isinstance(test_generation, dict):
        raise RulesError("test_generation must be an object.")
    passes = test_generation.get("rsi_passes", 3)
    max_cases = test_generation.get("max_cases", 40)
    seed = test_generation.get("seed", 42)
    for label, value, low, high in (("rsi_passes", passes, 1, 3), ("max_cases", max_cases, 4, 80), ("seed", seed, 0, 2**31 - 1)):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise RulesError(f"test_generation.{label} must be an integer from {low} to {high}.")
    generation_enabled = test_generation.get("enabled", True)
    if not isinstance(generation_enabled, bool):
        raise RulesError("test_generation.enabled must be a boolean.")
    temperature = test_generation.get("temperature", 0.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or float(temperature) != 0.0:
        raise RulesError("For reproducible generation, test_generation.temperature must be 0.0.")

    llm_raw = raw.get("llm", {})
    if not isinstance(llm_raw, dict):
        raise RulesError("llm must be an object.")
    llm = dict(llm_raw)
    defaults = {"base_url": "http://127.0.0.1:8080/v1",
                "model": "gguf/Qwen3.5-2B-GGUF:Q4_K_M", "server_executable": "llama-server",
                "host": "127.0.0.1", "port": 8080, "context_size": 8192,
                "seed": seed, "temperature": 0.0, "max_tokens": 4096,
                "timeout_s": 180, "start_server": True}
    for key, default in defaults.items():
        llm.setdefault(key, default)
    for key in ("base_url", "model", "server_executable", "host"):
        if not isinstance(llm[key], str) or not llm[key].strip():
            raise RulesError(f"llm.{key} must be a non-empty string.")
    if "health_url" in llm and (not isinstance(llm["health_url"], str) or not llm["health_url"].strip()):
        raise RulesError("llm.health_url must be a non-empty string when provided.")
    for key, minimum, maximum in (("port", 1, 65535), ("context_size", 512, 131072), ("max_tokens", 128, 32768), ("seed", 0, 2**31 - 1)):
        value = llm[key]
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise RulesError(f"llm.{key} must be an integer from {minimum} to {maximum}.")
    timeout_value = llm["timeout_s"]
    if isinstance(timeout_value, bool) or not isinstance(timeout_value, (int, float)) or not math.isfinite(timeout_value) or not 1 <= timeout_value <= 600:
        raise RulesError("llm.timeout_s must be a finite number from 1 to 600.")
    llm["timeout_s"] = float(timeout_value)
    if not isinstance(llm["start_server"], bool):
        raise RulesError("llm.start_server must be a boolean.")
    if isinstance(llm["temperature"], bool) or not isinstance(llm["temperature"], (int, float)) or float(llm["temperature"]) != 0.0:
        raise RulesError("llm.temperature must be 0.0; numeric scoring is deterministic and LLM output is advisory.")
    cache_dir = test_generation.get("cache_dir", ".grader_cache")
    if not isinstance(cache_dir, str) or not cache_dir.strip() or Path(cache_dir).is_absolute():
        raise RulesError("test_generation.cache_dir must be a non-empty relative path.")
    reference_path = test_generation.get("reference_solution_path")
    if reference_path is not None and (not isinstance(reference_path, str) or not reference_path.strip() or Path(reference_path).is_absolute()):
        raise RulesError("test_generation.reference_solution_path must be a relative C source path when provided.")

    if "metrics" in raw:
        metrics_raw = raw["metrics"]
    else:
        metrics_raw = None
    metrics = _normalize_metrics(metrics_raw)

    cpu_limit = raw.get("cpu_limit", 1.0)
    if isinstance(cpu_limit, bool) or not isinstance(cpu_limit, (int, float)) or not math.isfinite(cpu_limit) or cpu_limit <= 0 or cpu_limit > 8:
        raise RulesError("cpu_limit must be a finite number in the range (0, 8].")

    comparison = raw.get("output_comparison", "exact")
    if not isinstance(comparison, str) or comparison not in {"exact", "tokens", "trimmed"}:
        raise RulesError("output_comparison must be 'exact', 'trimmed', or 'tokens'.")

    docker_image = raw.get("docker_image", "code-grader-sandbox:latest")
    if not isinstance(docker_image, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,254}", docker_image):
        raise RulesError("docker_image must be a valid non-empty image reference without whitespace.")

    return {
        "path": rules_path,
        "root": rules_path.parent,
        "sha256": _sha256(rules_path.read_bytes()),
        "problem": problem.strip(),
        "problem_contract": contract,
        "source_path": source,
        "source_name": source.name,
        "time_limit_s": float(time_limit),
        "memory_limit_mb": memory_limit,
        "cpu_limit": float(cpu_limit),
        "metrics": metrics,
        "examples": examples,
        "originality": {"enabled": originality_enabled, "corpus_dir": corpus_path, "threshold": float(threshold), "min_tokens": min_tokens, "max_files": max_files, "max_file_bytes": max_file_bytes},
        "test_generation": {"enabled": generation_enabled, "rsi_passes": passes, "max_cases": max_cases, "seed": seed, "temperature": 0.0, "cache_dir": cache_dir, "reference_solution_path": reference_path},
        "llm": llm,
        "output_comparison": comparison,
        "docker_image": docker_image,
        "raw": raw,
    }
