"""Run a stratified FormAI benchmark of the grader's vulnerability detection.

One run: draw a stratified random sample from the indexed dataset, predict
every program from its source alone, then (only after all predictions are
written) read the ESBMC labels and evaluate with ``formai_metrics``.

Each run folder under ``reports/formai/`` holds:
``config.json``, ``sample.json`` (strata, weights, file names), ``inputs.jsonl``
(source only), ``predictions.jsonl``, ``labels.json``, ``evaluation.json`` and
``report.md``.

CLI::

    python -m grading_agent.benchmark.formai_benchmark --dataset v2 -n 50 --mode hybrid_rsi
"""
from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from grading_agent.benchmark import formai_dataset, formai_metrics, formai_sampling
from grading_agent.benchmark.formai_predict import PredictionSession
from grading_agent.config import settings

logger = logging.getLogger("formai_benchmark")

MODES: dict[str, str] = {
    "tools_only": "Static tools only (cppcheck + syntax-tree property checks)",
    "llm_only": "LLM source audit only",
    "hybrid": "Static tools + LLM audit (unverified)",
    "hybrid_verify": "Static tools + LLM audit + iterative claim verification",
    "hybrid_rsi": "Static tools + LLM audit + full RSI evidence review (slow, conservative)",
}


@dataclass
class BenchmarkConfig:
    dataset: str = "v2"
    sample_size: int = 20
    stratify_by: str = "label_family"
    allocation: str = "proportional"
    seed: int = settings.FORM_AI_SAMPLE_SEED
    mode: str = settings.UI_DEFAULT_BENCHMARK_MODE
    tool_timeout: float = settings.FORM_AI_DEFAULT_TOOL_TIMEOUT_S
    max_llm_findings: int = settings.FORM_AI_DEFAULT_MAX_FINDINGS
    rsi_iterations: int = settings.FORM_AI_DEFAULT_RSI_ITERATIONS
    memory_limit: str = settings.FORM_AI_DEFAULT_MEMORY_LIMIT
    cpu_limit: float = settings.DOCKER_CPU_LIMIT
    line_tolerance: int = settings.FORM_AI_LINE_TOLERANCE
    bootstrap_samples: int = 1000

    def validate(self) -> None:
        if self.dataset not in formai_dataset.CATALOG:
            raise ValueError(f"Unknown dataset {self.dataset!r}.")
        if self.sample_size < 1:
            raise ValueError("Choose at least one program.")
        if self.stratify_by not in formai_sampling.STRATIFY_OPTIONS:
            raise ValueError(f"Unknown stratification {self.stratify_by!r}.")
        if self.allocation not in formai_sampling.ALLOCATIONS:
            raise ValueError(f"Unknown allocation {self.allocation!r}.")
        if self.mode not in MODES:
            raise ValueError(f"Unknown mode {self.mode!r}.")
        if self.tool_timeout <= 0 or self.max_llm_findings < 1:
            raise ValueError("Tool timeout and finding limit must be positive.")
        if not 1 <= self.rsi_iterations <= settings.LLM_REVIEW_MAX_ITERATIONS:
            raise ValueError(f"RSI iterations must be 1-{settings.LLM_REVIEW_MAX_ITERATIONS}.")
        if not 0.25 <= self.cpu_limit <= 16:
            raise ValueError("Docker CPU limit must be between 0.25 and 16 cores.")
        if self.line_tolerance < 0:
            raise ValueError("Line tolerance cannot be negative.")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


STATIC_SOURCES = {"cppcheck", "syntax_tree"}


def inconclusive_reason(prediction: dict[str, Any]) -> str | None:
    """Why a program got no vulnerable/safe decision (None when decided)."""
    if isinstance(prediction.get("predicted_vulnerable"), bool):
        return None
    errors = prediction.get("stage_errors") or {}
    if not prediction:
        return "no prediction written"
    if errors.get("session") or errors.get("input"):
        return "prediction failed"
    llm_error = str(prediction.get("llm_error") or errors.get("llm") or "")
    if llm_error and not prediction.get("llm_findings"):
        return "LLM output truncated" if "finish_reason=length" in llm_error else "LLM audit failed"
    if prediction.get("llm_findings_not_confirmed_by_rsi"):
        return "LLM candidates not confirmed by RSI"
    if prediction.get("llm_unresolved_positive_summary") or prediction.get("llm_unresolved_invalid_candidates"):
        return "LLM claim not grounded to a source line"
    if prediction.get("tool_status") not in {None, "completed", "not_run"}:
        return "static tools incomplete"
    if prediction.get("rsi_review_status") not in {None, "completed", "completed_with_warnings", "not_run"}:
        return "RSI review incomplete"
    return "other"


def positive_evidence(prediction: dict[str, Any]) -> str | None:
    """Which kind of evidence a vulnerable prediction rests on."""
    if prediction.get("predicted_vulnerable") is not True:
        return None
    sources = {f.get("source") for f in prediction.get("predicted_findings") or []}
    static = bool(sources & STATIC_SOURCES)
    model = bool(sources - STATIC_SOURCES - {None})
    if static and model:
        return "static tools + LLM/RSI"
    if model:
        return "LLM/RSI only"
    return "static tools only" if static else "unattributed"


def evaluation_records(sample: dict[str, Any], predictions: dict[str, dict[str, Any]],
                       labels: dict[str, formai_dataset.ProgramLabel]) -> list[dict[str, Any]]:
    """Join sample design, predictions and labels into one record per program."""
    records = []
    for program in sample["programs"]:
        name = program["file_name"]
        label = labels[name]
        prediction = predictions.get(name, {})
        raw = prediction.get("predicted_vulnerable")
        records.append({
            "file_name": name,
            "stratum": program["stratum"],
            "weight": program["weight"],
            "truth": label.label == formai_dataset.VULNERABLE,
            "truth_lines": label.lines,
            "truth_families": label.families,
            "truth_error_types": label.error_types,
            "truth_vulnerabilities": [
                {"line": v["line"], "family": v["family"], "error_type": v["error_type"]} for v in label.vulnerabilities],
            "verification_complete": label.verification_complete,
            "predicted": raw if isinstance(raw, bool) else None,
            "predicted_line": prediction.get("predicted_line"),
            "predicted_family": prediction.get("predicted_family"),
            "findings": prediction.get("predicted_findings") or [],
            "prediction_status": prediction.get("prediction_status", "missing"),
            "inconclusive_reason": inconclusive_reason(prediction),
            "positive_evidence": positive_evidence(prediction),
            "seconds": (prediction.get("timings") or {}).get("total_s"),
        })
    return records


def iter_benchmark(config: BenchmarkConfig, *, session_factory: Any = PredictionSession) -> Iterator[dict[str, Any]]:
    """Run one benchmark, yielding progress events; the last event is ``done``."""
    config.validate()
    if not formai_dataset.is_ready(config.dataset):
        raise FileNotFoundError(f"{formai_dataset.CATALOG[config.dataset].name} is not prepared yet.")
    connection = formai_dataset.connect(config.dataset)
    try:
        sample = formai_sampling.draw_sample(
            connection, dataset=config.dataset, n=config.sample_size, stratify_by=config.stratify_by,
            allocation=config.allocation, seed=config.seed,
        ).to_json()
        names = [program["file_name"] for program in sample["programs"]]
        sources = formai_dataset.load_sources(connection, names)

        run_dir = Path(settings.FORM_AI_RUNS_DIR) / f"{time.strftime('%Y%m%d-%H%M%S')}-{config.dataset}-{config.mode}-{uuid.uuid4().hex[:6]}"
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_json(run_dir / "config.json", asdict(config))
        _write_json(run_dir / "sample.json", sample)
        with (run_dir / "inputs.jsonl").open("w", encoding="utf-8") as stream:
            for name in names:
                stream.write(json.dumps({"file_name": name, "source_code": sources[name]}, ensure_ascii=False) + "\n")
        yield {"event": "sampled", "run_dir": str(run_dir), "sample": sample}

        predictions: dict[str, dict[str, Any]] = {}
        started = time.monotonic()
        with session_factory(
            mode=config.mode, tool_timeout=config.tool_timeout, max_llm_findings=config.max_llm_findings,
            rsi_iterations=config.rsi_iterations, memory_limit=config.memory_limit, cpu_limit=config.cpu_limit,
        ) as session, (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as output:
            for index, name in enumerate(names, 1):
                stages: queue.Queue[str] = queue.Queue()
                holder: dict[str, Any] = {}

                def work(item: dict[str, Any] = {"file_name": name, "source_code": sources[name]}) -> None:
                    try:
                        holder["prediction"] = session.predict(item, on_stage=stages.put)
                    except Exception as exc:  # one program never stops the run
                        logger.exception("Prediction failed for %s", item["file_name"])
                        holder["prediction"] = {
                            "file_name": item["file_name"], "predicted_vulnerable": None,
                            "prediction_status": "inconclusive", "predicted_findings": [],
                            "stage_errors": {"session": f"{type(exc).__name__}: {exc}"},
                        }

                worker = threading.Thread(target=work, daemon=True)
                worker.start()
                yield {"event": "program", "index": index, "total": len(names), "file_name": name, "stage": "starting",
                       "elapsed_s": time.monotonic() - started}
                while True:
                    worker.join(timeout=0.5)
                    # Drain after the final join too: a fast program can finish
                    # before its stage events are read.
                    while not stages.empty():
                        yield {"event": "program", "index": index, "total": len(names), "file_name": name,
                               "stage": stages.get_nowait(), "elapsed_s": time.monotonic() - started}
                    if not worker.is_alive():
                        break
                prediction = holder["prediction"]
                predictions[name] = prediction
                output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                output.flush()
                yield {"event": "predicted", "index": index, "total": len(names), "file_name": name,
                       "predicted": prediction.get("predicted_vulnerable"), "status": prediction.get("prediction_status"),
                       "elapsed_s": time.monotonic() - started}

        # Labels are read only now, after every prediction has been written.
        labels = formai_dataset.load_labels(connection, names)
        _write_json(run_dir / "labels.json", {name: asdict(label) for name, label in labels.items()})
        records = evaluation_records(sample, predictions, labels)
        evaluation = formai_metrics.evaluate(
            records, strata=sample["strata"], line_tolerance=config.line_tolerance,
            bootstrap_samples=config.bootstrap_samples, seed=config.seed,
        )
        seconds = [r["seconds"] for r in records if isinstance(r.get("seconds"), (int, float))]
        payload = {
            "dataset": formai_dataset.CATALOG[config.dataset].name,
            "config": asdict(config),
            "index": {key: value for key, value in formai_dataset.index_summary(config.dataset).items()
                      if key in {"archive_sha256", "labels", "eligible_programs", "index_schema", "label_policy"}},
            "sample": {key: sample[key] for key in ("stratify_by", "allocation", "seed", "requested", "size", "population")},
            "evaluation": evaluation,
            "runtime": {
                "total_s": round(time.monotonic() - started, 1),
                "mean_s_per_program": round(sum(seconds) / len(seconds), 2) if seconds else None,
            },
            "model": settings.runtime_value("MODEL") if config.mode != "tools_only" else None,
            "labels_read_after_predictions": True,
        }
        _write_json(run_dir / "evaluation.json", payload)
        report = summary_markdown(payload)
        (run_dir / "report.md").write_text(report, encoding="utf-8")
        yield {"event": "done", "run_dir": str(run_dir), "evaluation": payload, "records": records, "report": report}
    finally:
        connection.close()


def run_benchmark(config: BenchmarkConfig, **kwargs: Any) -> dict[str, Any]:
    """Blocking helper: run to completion and return the ``done`` event."""
    last: dict[str, Any] = {}
    for event in iter_benchmark(config, **kwargs):
        last = event
        if event["event"] == "predicted":
            logger.info("[%d/%d] %s -> %s (%s)", event["index"], event["total"], event["file_name"],
                        event["predicted"], event["status"])
    return last


def _fmt(value: Any, percent: bool = True) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.1f}%" if percent else f"{value:.3f}"


def _with_ci(metrics: dict[str, Any], name: str, percent: bool = True) -> str:
    interval = (metrics.get("ci95") or {}).get(name)
    text = _fmt(metrics.get(name), percent)
    if interval and metrics.get(name) is not None:
        text += f" ({_fmt(interval[0], percent)}–{_fmt(interval[1], percent)})"
    return text


def summary_markdown(payload: dict[str, Any]) -> str:
    """Human-readable report of one run (also shown in the app)."""
    evaluation = payload["evaluation"]
    config = payload["config"]
    population = evaluation["metrics"]["population"]
    sample_view = evaluation["metrics"]["sample"]
    decided = evaluation["metrics"]["decided"]
    loc = evaluation["localisation"]
    sample = payload["sample"]
    lines = [
        f"## {payload['dataset']} benchmark: {MODES.get(config['mode'], config['mode'])}",
        f"{evaluation['programs']} programs ({evaluation['vulnerable_programs']} vulnerable, {evaluation['safe_programs']} safe), "
        f"stratified by {formai_sampling.STRATIFY_OPTIONS[sample['stratify_by']].lower()}, "
        f"{sample['allocation']} allocation, seed {sample['seed']}, from {sample['population']:,} eligible programs.",
        "",
        f"**Decided:** {evaluation['decided']}/{evaluation['programs']} "
        f"({_fmt(evaluation['coverage'])}); inconclusive: {evaluation['inconclusive_vulnerable']} vulnerable, "
        f"{evaluation['inconclusive_safe']} safe (scored as errors below).",
    ]
    reasons = evaluation.get("inconclusive_reasons") or {}
    if reasons:
        lines.append("**Why undecided:** " + "; ".join(
            f"{reason} {c['vulnerable'] + c['safe']} ({c['vulnerable']} vulnerable, {c['safe']} safe)"
            for reason, c in reasons.items()) + ".")
    evidence = evaluation.get("positive_evidence") or {}
    if evidence:
        lines.append("**Vulnerable predictions rest on:** " + "; ".join(
            f"{source} {c['tp'] + c['fp']} (TP {c['tp']}, FP {c['fp']})" for source, c in evidence.items()) + ".")
    lines += [
        "",
        "| Metric | Dataset estimate (95% CI) | This sample | Decided only |",
        "|---|---|---|---|",
    ]
    for name, title in (("recall", "Recall (vulnerable found)"), ("specificity", "Specificity (safe kept safe)"),
                        ("precision", "Precision"), ("f1", "F1"), ("balanced_accuracy", "Balanced accuracy"),
                        ("accuracy", "Accuracy")):
        lines.append(f"| {title} | {_with_ci(population, name)} | {_fmt(sample_view.get(name))} | {_fmt(decided.get(name))} |")
    lines.append(f"| MCC | {_with_ci(population, 'mcc', False)} | {_fmt(sample_view.get('mcc'), False)} | {_fmt(decided.get('mcc'), False)} |")
    confusion = sample_view["confusion"]
    lines += [
        "",
        f"**Confusion (sample, strict):** TP {confusion['tp']} · FN {confusion['fn']} · FP {confusion['fp']} · TN {confusion['tn']}",
        "",
        f"**Location (±{loc['line_tolerance']} lines, detected vulnerable programs, n={loc['detected_vulnerable_programs']}):** "
        f"primary finding on an ESBMC line {_fmt(loc['primary_line_hit']['population'])}, "
        f"any finding on an ESBMC line {_fmt(loc['any_finding_line_hit']['population'])}, "
        f"same family {_fmt(loc['family_match']['population'])}, "
        f"same line and family {_fmt(loc['line_and_family_match']['population'])}.",
        f"ESBMC vulnerabilities located: {_fmt(loc['esbmc_vulnerability_recall_line'])} by line, "
        f"{_fmt(loc['esbmc_vulnerability_recall_line_and_family'])} by line and family "
        f"(of {loc['esbmc_vulnerabilities']}). Findings on ESBMC lines: {_fmt(loc['finding_line_precision_lower_bound'])} "
        "(a lower bound: ESBMC checks a bounded program).",
    ]
    notes = []
    if evaluation["inconclusive"] and (evaluation["coverage"] or 0) < 0.8:
        notes.append(
            f"Only {_fmt(evaluation['coverage'])} of programs were decided. The *Decided only* column drops the "
            f"{evaluation['inconclusive']} undecided programs, which are likely the hardest cases, so it overstates "
            "accuracy; use the dataset estimate.")
    if evaluation["unsampled_strata"]:
        notes.append(
            f"{len(evaluation['unsampled_strata'])} strata had no sampled program, so dataset estimates cover "
            f"{_fmt(evaluation['population_covered_by_sampled_strata'])} of the eligible programs. Increase the sample size to cover all strata.")
    if evaluation["programs"] < 30:
        notes.append("Small sample: confidence intervals are wide; treat the figures as indicative.")
    notes.append("Dataset estimates weight each program by its stratum's population/sample ratio; "
                 "inconclusive results count as a missed vulnerability or a false alarm.")
    lines += ["", *[f"> {note}" for note in notes]]
    runtime = payload.get("runtime", {})
    if runtime.get("mean_s_per_program") is not None:
        lines.append(f"\nRuntime {runtime['total_s']} s ({runtime['mean_s_per_program']} s per program)"
                     + (f", model {payload['model']}." if payload.get("model") else "."))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(formai_dataset.CATALOG), default="v2")
    parser.add_argument("-n", "--sample-size", type=int, default=20)
    parser.add_argument("--stratify-by", choices=sorted(formai_sampling.STRATIFY_OPTIONS), default="label_family")
    parser.add_argument("--allocation", choices=sorted(formai_sampling.ALLOCATIONS), default="proportional")
    parser.add_argument("--seed", type=int, default=settings.FORM_AI_SAMPLE_SEED)
    parser.add_argument("--mode", choices=sorted(MODES), default=settings.UI_DEFAULT_BENCHMARK_MODE)
    parser.add_argument("--tool-timeout", type=float, default=settings.FORM_AI_DEFAULT_TOOL_TIMEOUT_S)
    parser.add_argument("--max-llm-findings", type=int, default=settings.FORM_AI_DEFAULT_MAX_FINDINGS)
    parser.add_argument("--rsi-iterations", type=int, default=settings.FORM_AI_DEFAULT_RSI_ITERATIONS)
    parser.add_argument("--memory-limit", default=settings.FORM_AI_DEFAULT_MEMORY_LIMIT)
    parser.add_argument("--cpu-limit", type=float, default=settings.DOCKER_CPU_LIMIT)
    parser.add_argument("--line-tolerance", type=int, default=settings.FORM_AI_LINE_TOLERANCE)
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples for 95%% CIs (0 = off).")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # Windows consoles cannot print every symbol
    logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT_BENCHMARK)
    config = BenchmarkConfig(
        dataset=args.dataset, sample_size=args.sample_size, stratify_by=args.stratify_by, allocation=args.allocation,
        seed=args.seed, mode=args.mode, tool_timeout=args.tool_timeout, max_llm_findings=args.max_llm_findings,
        rsi_iterations=args.rsi_iterations, memory_limit=args.memory_limit, cpu_limit=args.cpu_limit,
        line_tolerance=args.line_tolerance, bootstrap_samples=args.bootstrap,
    )
    try:
        result = run_benchmark(config)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(result["report"])
    print(f"\nRun folder: {result['run_dir']}")


if __name__ == "__main__":
    main()
