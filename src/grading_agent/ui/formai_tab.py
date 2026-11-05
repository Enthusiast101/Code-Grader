"""Gradio tab: stratified FormAI benchmark of the grader's vulnerability detection."""
from __future__ import annotations

import html
import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import gradio as gr

from grading_agent.benchmark import formai_dataset, formai_sampling
from grading_agent.benchmark.formai_benchmark import MODES, BenchmarkConfig, iter_benchmark
from grading_agent.config import settings

logger = logging.getLogger(__name__)

DATASET_CHOICES = [(f"{spec.name} — {spec.notes}", key) for key, spec in formai_dataset.CATALOG.items()]
STRATIFY_CHOICES = [(label, key) for key, label in formai_sampling.STRATIFY_OPTIONS.items()]
ALLOCATION_CHOICES = [(label, key) for key, label in formai_sampling.ALLOCATIONS.items()]
MODE_CHOICES = [(label, key) for key, label in MODES.items()]
# Typical seconds per program with the default local model (measured on this
# project's hardware); shown only as a rough runtime estimate.
SECONDS_PER_PROGRAM = {"tools_only": 0.2, "llm_only": 13.0, "hybrid": 13.0, "hybrid_verify": 25.0, "hybrid_rsi": 95.0}
STRATA_HEADERS = ["Stratum", "Programs in dataset", "Share %", "Sampled", "Weight (N/n)"]
RESULT_STRATA_HEADERS = ["Stratum", "Sampled", "TP", "FN", "FP", "TN", "Inconclusive", "Metric", "Value"]
FAMILY_HEADERS = ["ESBMC family", "Programs", "Detected", "Detection rate", "Located (line + family)", "Location rate"]
PROGRAM_HEADERS = ["Program", "Stratum", "ESBMC label", "Prediction", "Outcome", "First finding", "ESBMC lines", "Seconds"]


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def dataset_status(version: str) -> str:
    spec = formai_dataset.CATALOG[version]
    if not formai_dataset.is_ready(version):
        return (f"**{spec.name} is not prepared.** Click *Prepare dataset* to download the published archives "
                "(SHA-256 verified), extract them and build the per-program index. FormAI-v2 needs about 120 MB "
                "of download and 4 GB of disk; it takes 2–3 minutes.")
    summary = formai_dataset.index_summary(version)
    labels = summary["labels"]
    return (
        f"**{spec.name} ready:** {summary['programs']:,} programs · "
        f"{labels.get('VULNERABLE', 0):,} vulnerable · {labels.get('SAFE', 0):,} safe · "
        f"{labels.get('UNKNOWN', 0):,} excluded (ESBMC time-out or parse error). "
        f"Generators: {', '.join(summary['generators'])}."
    )


def prepare_dataset(version: str) -> Iterator[tuple[str, str]]:
    """Download, verify and index a dataset, streaming progress messages."""
    messages: list[str] = []
    started = time.monotonic()

    import queue
    import threading

    updates: queue.Queue[str] = queue.Queue()
    outcome: dict[str, Any] = {}

    def work() -> None:
        try:
            outcome["summary"] = formai_dataset.prepare(version, progress=updates.put)
        except Exception as exc:  # shown in the UI
            logger.exception("FormAI preparation failed")
            outcome["error"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    while worker.is_alive():
        worker.join(timeout=1.0)
        while not updates.empty():
            messages.append(updates.get_nowait())
        last = html.escape(messages[-1]) if messages else "Starting…"
        yield _progress_html("Preparing dataset", last, None, None, started), gr.skip()
    if "error" in outcome:
        yield _progress_html("Preparation failed", outcome["error"], None, None, started, state="failed"), dataset_status(version)
    else:
        yield _progress_html("Dataset ready", messages[-1] if messages else "Index ready.", 1, 1, started, state="done"), dataset_status(version)


def preview_allocation(version: str, n: float, stratify_by: str, allocation: str, mode: str):
    """Allocation table for the current slider value (no programs are run)."""
    if not formai_dataset.is_ready(version):
        return gr.update(value=[], headers=STRATA_HEADERS), "Prepare the dataset to see the stratified allocation."
    try:
        with formai_dataset.connect(version) as connection:
            strata = formai_sampling.allocate(
                formai_sampling.population_strata(connection, stratify_by), int(n), allocation)
    except ValueError as exc:
        return gr.update(value=[], headers=STRATA_HEADERS), f"{exc}"
    rows = formai_sampling.allocation_table(strata)
    sampled = sum(s.allocated for s in strata)
    empty = sum(1 for s in strata if not s.allocated)
    seconds = sampled * SECONDS_PER_PROGRAM.get(mode, 15.0)
    note = (f"**{sampled} programs** across {len(strata) - empty} of {len(strata)} strata · "
            f"vulnerable {sum(s.allocated for s in strata if s.label == 'VULNERABLE')}, "
            f"safe {sum(s.allocated for s in strata if s.label == 'SAFE')} · "
            f"estimated runtime ≈ {_duration(seconds)} with {MODES[mode].lower()}.")
    if empty:
        note += f" {empty} strata get no program at this size; dataset estimates then exclude them."
    return gr.update(value=rows, headers=STRATA_HEADERS), note


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, round(seconds))} s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"


def _progress_html(title: str, detail: str, done: int | None, total: int | None, started: float,
                   state: str = "running", stage: str | None = None) -> str:
    if state == "done":
        percent = 100
    elif done is not None and total:
        percent = max(2, min(99, int(100 * done / total)))
    else:
        percent = 3
    elapsed = time.monotonic() - started
    meta = f"{percent}% · elapsed {_duration(elapsed)}"
    if state == "running" and done and total and done < total:
        meta += f" · about {_duration(elapsed / done * (total - done))} left"
    card_class = {"done": " is-done", "failed": " is-failed"}.get(state, "")
    chips = ""
    if stage:
        chips = f'<ul class="progress-stages"><li class="current">{html.escape(stage)}</li></ul>'
    return (
        f'<section class="progress-card{card_class}" role="status" aria-live="polite">'
        f'<div class="progress-head"><span class="progress-title">{html.escape(title)}</span>'
        f'<span class="progress-meta">{meta}</span></div>'
        f'<p class="progress-detail">{detail}</p>'
        f'<div class="progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percent}" '
        f'aria-label="{html.escape(title)}"><div class="progress-fill" style="width:{percent}%"></div></div>{chips}</section>'
    )


def _outcome(record: dict[str, Any]) -> str:
    predicted = record.get("predicted")
    if predicted is None:
        reason = record.get("inconclusive_reason")
        return f"inconclusive: {reason}" if reason else "inconclusive"
    if record["truth"]:
        return "TP" if predicted else "FN"
    return "FP" if predicted else "TN"


def _result_tables(evaluation: dict[str, Any], records: list[dict[str, Any]]):
    strata_rows = [
        [row["stratum"], row["sampled"], row["tp"], row["fn"], row["fp"], row["tn"], row["inconclusive"],
         row["metric"], _pct(row["value"])]
        for row in evaluation["per_stratum"] if row["sampled"]
    ]
    family_rows = [
        [row["family"], row["programs"], row["detected"], _pct(row["detection_rate"]),
         row["located_same_family"], _pct(row["location_rate"])]
        for row in evaluation["per_family"]
    ]
    program_rows = []
    for record in records:
        first = record["findings"][0] if record.get("findings") else None
        program_rows.append([
            record["file_name"], record["stratum"], "VULNERABLE" if record["truth"] else "SAFE",
            {True: "vulnerable", False: "safe", None: "inconclusive"}[record.get("predicted")],
            _outcome(record),
            f"line {first['line']} · {first['family']} ({first['source']})" if first else "",
            ", ".join(str(line) for line in record.get("truth_lines", [])[:8]),
            record.get("seconds"),
        ])
    return strata_rows, family_rows, program_rows


def run_benchmark_ui(version: str, n: float, stratify_by: str, allocation: str, seed: float, mode: str,
                     tool_timeout: float, max_findings: float, rsi_iterations: float, memory_limit: str,
                     cpu_limit: float, line_tolerance: float):
    """Stream progress, then the evaluation (summary, tables, files)."""
    started = time.monotonic()
    blank = (gr.skip(),) * 6
    busy = gr.update(interactive=False)
    try:
        config = BenchmarkConfig(
            dataset=version, sample_size=int(n), stratify_by=stratify_by, allocation=allocation, seed=int(seed),
            mode=mode, tool_timeout=float(tool_timeout), max_llm_findings=int(max_findings),
            rsi_iterations=int(rsi_iterations), memory_limit=memory_limit, cpu_limit=float(cpu_limit),
            line_tolerance=int(line_tolerance),
        )
        yield (_progress_html("Benchmark starting", "Drawing the stratified sample…", 0, int(n), started), *blank, busy)
        total = int(n)
        done = 0
        for event in iter_benchmark(config):
            kind = event["event"]
            if kind == "sampled":
                total = event["sample"]["size"]
                yield (_progress_html("Benchmark running", f"Sampled {total} programs; starting predictions.", 0, total,
                                      started), *blank, busy)
            elif kind == "program":
                detail = f"Program {event['index']} of {event['total']}: <code>{html.escape(event['file_name'])}</code>"
                yield (_progress_html("Benchmark running", detail, done, total, started, stage=event["stage"]), *blank, busy)
            elif kind == "predicted":
                done = event["index"]
            elif kind == "done":
                strata_rows, family_rows, program_rows = _result_tables(event["evaluation"]["evaluation"], event["records"])
                run_dir = Path(event["run_dir"])
                files = [str(run_dir / name) for name in ("evaluation.json", "predictions.jsonl", "sample.json", "report.md")]
                yield (
                    _progress_html("Benchmark complete", f"{total} programs evaluated. Files: <code>{html.escape(str(run_dir))}</code>",
                                   total, total, started, state="done"),
                    event["report"],
                    gr.update(value=strata_rows, headers=RESULT_STRATA_HEADERS),
                    gr.update(value=family_rows, headers=FAMILY_HEADERS),
                    gr.update(value=program_rows, headers=PROGRAM_HEADERS),
                    files,
                    gr.update(value=None),
                    gr.update(interactive=True),
                )
                return
    except Exception as exc:
        logger.exception("FormAI benchmark failed")
        yield (_progress_html("Benchmark stopped", html.escape(f"{type(exc).__name__}: {exc}"), None, None, started,
                              state="failed"),
               f"## Benchmark could not complete\n`{type(exc).__name__}: {exc}`",
               gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.update(interactive=True))


def build_formai_tab() -> None:
    """Add the benchmark tab to the current Blocks context."""
    with gr.Tab("FormAI benchmark"):
        gr.Markdown(
            "Measure how well the grader detects vulnerabilities on "
            "[FormAI](https://github.com/FormAI-Dataset/FormAI-dataset): AI-generated C programs labelled by the "
            "ESBMC model checker. A stratified random sample is drawn, every program is analysed from its source "
            "alone, and the ESBMC labels are read only after all predictions are written.",
            elem_classes=["compact-note"],
        )
        with gr.Row(equal_height=True):
            dataset = gr.Dropdown(DATASET_CHOICES, value="v2", label="Dataset", scale=3)
            prepare_button = gr.Button("Prepare dataset", scale=1)
        status = gr.Markdown(dataset_status("v2") if "v2" in formai_dataset.CATALOG else "")
        prepare_progress = gr.HTML(elem_classes=["progress-host"])
        with gr.Row():
            with gr.Column(scale=3):
                sample_size = gr.Slider(1, settings.FORM_AI_UI_MAX_SAMPLES, value=settings.FORM_AI_UI_DEFAULT_SAMPLES, step=1,
                                        label="Number of programs to test",
                                        info="Drawn by stratified random sampling; the same seed always gives the same programs.")
                with gr.Row():
                    stratify_by = gr.Dropdown(STRATIFY_CHOICES, value="label_family", label="Stratify by")
                    allocation = gr.Radio(ALLOCATION_CHOICES, value="proportional", label="Allocation")
                with gr.Row():
                    mode = gr.Radio(MODE_CHOICES, value=settings.UI_DEFAULT_BENCHMARK_MODE, label="What to test")
                    seed = gr.Number(value=settings.FORM_AI_SAMPLE_SEED, precision=0, label="Seed")
            with gr.Column(scale=2):
                allocation_note = gr.Markdown()
                allocation_table = gr.Dataframe(headers=STRATA_HEADERS, interactive=False, wrap=True,
                                                label="Stratified allocation")
        with gr.Accordion("Analysis settings", open=False):
            with gr.Row():
                tool_timeout = gr.Number(value=settings.FORM_AI_DEFAULT_TOOL_TIMEOUT_S, precision=0, label="cppcheck timeout (s)")
                max_findings = gr.Number(value=settings.FORM_AI_DEFAULT_MAX_FINDINGS, precision=0, label="Maximum LLM findings")
                rsi_iterations = gr.Slider(1, settings.LLM_REVIEW_MAX_ITERATIONS, value=settings.FORM_AI_DEFAULT_RSI_ITERATIONS,
                                           step=1, label="RSI iterations")
                line_tolerance = gr.Slider(0, 10, value=settings.FORM_AI_LINE_TOLERANCE, step=1, label="Line tolerance (±)")
            with gr.Row():
                memory_limit = gr.Dropdown(list(settings.UI_DOCKER_MEMORY_CHOICES), value=settings.UI_DEFAULT_MEMORY_LIMIT,
                                           label="Docker memory limit")
                cpu_limit = gr.Slider(0.25, 16, value=settings.DOCKER_CPU_LIMIT, step=0.25, label="Docker CPU limit (cores)")
        run_button = gr.Button("Run benchmark", variant="primary", elem_id="formai-run-button")
        progress = gr.HTML(elem_classes=["progress-host"])
        summary = gr.Markdown()
        with gr.Tabs():
            with gr.Tab("By stratum"):
                strata_table = gr.Dataframe(headers=RESULT_STRATA_HEADERS, interactive=False, wrap=True)
            with gr.Tab("By vulnerability family"):
                family_table = gr.Dataframe(headers=FAMILY_HEADERS, interactive=False, wrap=True)
            with gr.Tab("Programs"):
                program_table = gr.Dataframe(headers=PROGRAM_HEADERS, interactive=False, wrap=True)
            with gr.Tab("Files"):
                files = gr.File(label="Run files", file_count="multiple")
        hidden = gr.State()

        preview_inputs = [dataset, sample_size, stratify_by, allocation, mode]
        for control in preview_inputs:
            control.change(preview_allocation, inputs=preview_inputs, outputs=[allocation_table, allocation_note],
                           show_progress="hidden")
        dataset.change(dataset_status, inputs=dataset, outputs=status, show_progress="hidden")
        prepare_button.click(prepare_dataset, inputs=dataset, outputs=[prepare_progress, status], show_progress="hidden").then(
            preview_allocation, inputs=preview_inputs, outputs=[allocation_table, allocation_note], show_progress="hidden")
        run_button.click(
            run_benchmark_ui,
            inputs=[dataset, sample_size, stratify_by, allocation, seed, mode, tool_timeout, max_findings,
                    rsi_iterations, memory_limit, cpu_limit, line_tolerance],
            outputs=[progress, summary, strata_table, family_table, program_table, files, hidden, run_button],
            show_progress="hidden",
        )
        return dataset, sample_size, stratify_by, allocation, mode, allocation_table, allocation_note
