"""Local Gradio interface for C assignment grading and FormAI-v2 evaluation."""
from __future__ import annotations

import json
import logging
from argparse import Namespace
from pathlib import Path
import re
import shutil
import uuid
from typing import Any

import gradio as gr

from grading_agent.benchmark.formai_evaluate import evaluate_mode
from grading_agent.benchmark.formai_predict import run as run_formai_prediction
from grading_agent.main import PROJECT_ROOT, grade_with_rules_file
from grading_agent.config import settings

logger = logging.getLogger("grading_agent.ui")


def _upload_path(upload: Any, label: str) -> Path:
    if upload is None:
        raise ValueError(f"Upload the {label} file first.")
    raw = upload if isinstance(upload, (str, Path)) else getattr(upload, "name", None)
    if not raw:
        raise ValueError(f"Could not resolve the uploaded {label} file.")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Uploaded {label} file does not exist: {path}")
    return path


def _metric_summary(report: dict[str, Any]) -> str:
    scoring = report.get("scoring", {})
    lines = ["## Final grading result"]
    lines.append(f"**Overall grade: {scoring.get('score', 0.0):.2f}/100**")
    lines.append(f"**Status:** `{scoring.get('status', 'completed')}`")
    lines.append("\n### Metric scores")
    for metric in report.get("configuration", {}).get("enabled_metrics", []):
        name = metric.get("name")
        result = report.get(name, {})
        lines.append(
            f"- **{name}** — **{result.get('score', 0.0):.2f}/100** "
            f"(LLM RSI: {result.get('llm_rsi_score') if result.get('llm_rsi_score') is not None else 'N/A'}, "
            f"analyzer fallback: {result.get('analyzer_score') if result.get('analyzer_score') is not None else 'N/A'})"
        )
        rationale = result.get("llm_rsi", {}).get("rationale") if isinstance(result.get("llm_rsi"), dict) else None
        if rationale:
            lines.append(f"  - RSI: {rationale}")
        basis = result.get("scoring_basis")
        if basis:
            lines.append(f"  - Score basis: {basis}")

    rsi = report.get("llm_rsi_evaluation", {})
    lines.append("\n### LLM evidence-based RSI")
    lines.append(f"Status: `{rsi.get('status', 'unavailable')}`")
    if rsi.get("error"):
        lines.append(f"LLM note: {rsi['error']}")
    if rsi.get("missing_metrics"):
        lines.append("Metrics without an LLM RSI score used the deterministic analyzer score where available: " + ", ".join(rsi["missing_metrics"]))

    warnings = scoring.get("warnings", [])
    if warnings:
        lines.append("\n### Scoring notes")
        lines.extend(f"- {warning}" for warning in warnings)
    if report.get("sandbox", {}).get("status") == "unavailable":
        lines.append("\n**Docker was unavailable:** dynamic analyzer results are shown as unavailable; RSI may still estimate metrics from the source code.")
    return "\n".join(lines)


def grade_uploaded_submission(source_upload: Any, rules_upload: Any):
    """Stage uploads safely, then execute the same project grading pipeline."""
    stage_dir: Path | None = None
    try:
        source_path = _upload_path(source_upload, "C source")
        rules_path = _upload_path(rules_upload, "rules JSON")
        if source_path.suffix.lower() != ".c":
            raise ValueError("The submission must be a .c file.")
        if source_path.stat().st_size > settings.UI_MAX_UPLOAD_BYTES:
            raise ValueError(f"The source file exceeds the {settings.UI_MAX_UPLOAD_BYTES} byte project limit.")

        rules = json.loads(rules_path.read_text(encoding="utf-8"))
        if not isinstance(rules, dict) or not isinstance(rules.get("submission"), dict):
            raise ValueError("The rules JSON must contain a submission object.")

        run_id = uuid.uuid4().hex[:12]
        stage_dir = settings.GRADIO_STAGING_DIR / run_id
        stage_dir.mkdir(parents=True, exist_ok=False)
        staged_source = stage_dir / "submission.c"
        shutil.copyfile(source_path, staged_source)
        rules["submission"]["source_file"] = str(staged_source.relative_to(PROJECT_ROOT))
        staged_rules = stage_dir / "rules.json"
        staged_rules.write_text(json.dumps(rules, indent=2), encoding="utf-8")

        report_dir = settings.GRADIO_REPORTS_DIR / run_id
        report, json_path, text_path = grade_with_rules_file(
            staged_rules,
            project_root=PROJECT_ROOT,
            report_dir=report_dir,
        )
        return _metric_summary(report), str(json_path), str(text_path)
    except Exception as exc:
        logger.exception("Gradio grading request failed")
        return f"## Could not complete grading\n`{type(exc).__name__}: {exc}`", None, None
    finally:
        if stage_dir is not None:
            shutil.rmtree(stage_dir, ignore_errors=True)


def _prediction_stage_counts(prediction_path: Path) -> dict[str, int]:
    """Count completed/failed analysis stages for transparent UI feedback."""
    counts = {
        "samples": 0, "llm_completed": 0, "llm_failed": 0,
        "cppcheck_completed": 0, "cppcheck_unavailable": 0,
        "rsi_completed": 0, "rsi_unavailable": 0,
        "partial_predictions": 0, "inconclusive_predictions": 0,
    }
    with prediction_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            counts["samples"] += 1
            if item.get("llm_audit_status") in {"completed", "completed_with_warnings"}:
                counts["llm_completed"] += 1
            else:
                counts["llm_failed"] += 1
            if item.get("tool_status") == "completed":
                counts["cppcheck_completed"] += 1
            elif item.get("tool_status") in {"unavailable", "inconclusive"}:
                counts["cppcheck_unavailable"] += 1
            if item.get("rsi_review_status") in {"completed", "completed_with_warnings"}:
                counts["rsi_completed"] += 1
            elif item.get("rsi_review_status") in {"unavailable", "skipped"}:
                counts["rsi_unavailable"] += 1
            if item.get("prediction_status") == "completed_with_warnings":
                counts["partial_predictions"] += 1
            elif item.get("prediction_status") == "inconclusive":
                counts["inconclusive_predictions"] += 1
    return counts


def run_formai_combined(
    inputs_upload: Any,
    labels_upload: Any,
    mode: str,
    max_samples: float | int | None,
    tool_timeout: float,
    max_llm_findings: float | int,
    rsi_iterations: float | int,
    memory_limit: str,
):
    """Run LLM + cppcheck fusion, with optional evidence-review refinement."""
    try:
        inputs_path = _upload_path(inputs_upload, "source-only test JSONL")
        labels_path = _upload_path(labels_upload, "held-out labels JSON")
        if mode not in {"hybrid", "hybrid_rsi"}:
            raise ValueError("Choose a combined mode so both LLM and cppcheck are attempted.")
        if tool_timeout <= 0 or int(max_llm_findings) < 1 or not 1 <= int(rsi_iterations) <= 3:
            raise ValueError("Invalid timeout, finding limit, or RSI iteration count.")

        run_id = uuid.uuid4().hex[:12]
        out_dir = settings.FORM_AI_GRADIO_RUNS_DIR / run_id
        out_dir.mkdir(parents=True, exist_ok=False)
        prediction_path = out_dir / f"{mode}.jsonl"
        evaluation_path = out_dir / f"{mode}-evaluation.json"
        args = Namespace(
            inputs=inputs_path,
            mode=mode,
            output=prediction_path,
            max_samples=int(max_samples) if max_samples and int(max_samples) > 0 else None,
            tool_timeout=float(tool_timeout),
            max_llm_findings=int(max_llm_findings),
            rsi_iterations=int(rsi_iterations),
            memory_limit=memory_limit,
        )
        run_formai_prediction(args)
        evaluation = evaluate_mode(labels_path, prediction_path, line_tolerance=settings.FORM_AI_LINE_TOLERANCE)
        stage_counts = _prediction_stage_counts(prediction_path)
        evaluation_payload = {
            "dataset": "FormAI-v2",
            "mode": mode,
            "prediction_file": str(prediction_path),
            "evaluation": evaluation,
            "stage_counts": stage_counts,
            "fusion_policy": "OR of all completed primary stages (cppcheck and LLM); RSI may add evidence-backed positives but cannot veto primary outputs.",
        }
        evaluation_path.write_text(json.dumps(evaluation_payload, indent=2), encoding="utf-8")
        confusion = evaluation["confusion_matrix"]
        summary = (
            "## FormAI-v2 combined evaluation\n"
            f"**Mode:** `{mode}`  \n"
            f"**Evaluated:** {evaluation['evaluated_records']}  \n"
            f"**Coverage:** {evaluation['prediction_coverage']}  \n"
            f"**Precision / Recall / F1:** {evaluation['precision']} / {evaluation['recall']} / {evaluation['f1_score']}  \n"
            f"**Accuracy / Balanced accuracy:** {evaluation['accuracy']} / {evaluation['balanced_accuracy']}  \n"
            f"**TP / FP / FN / TN:** {confusion['tp']} / {confusion['fp']} / {confusion['fn']} / {confusion['tn']}  \n"
            f"**Inconclusive:** {evaluation['inconclusive_predictions']}  \n"
            f"**LLM audits completed:** {stage_counts['llm_completed']}/{stage_counts['samples']}  \n"
            f"**cppcheck completed:** {stage_counts['cppcheck_completed']}/{stage_counts['samples']}  \n"
            f"**Evidence reviews completed:** {stage_counts['rsi_completed']}/{stage_counts['samples']}  \n"
            f"**Partial / inconclusive predictions:** {stage_counts['partial_predictions']} / {stage_counts['inconclusive_predictions']}  \n\n"
            "The evaluator reads labels only after prediction generation; ground truth is not passed to either analysis stage."
        )
        return summary, str(prediction_path), str(evaluation_path)
    except Exception as exc:
        logger.exception("FormAI Gradio request failed")
        return f"## Benchmark could not complete\n`{type(exc).__name__}: {exc}`", None, None


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Code Checking Assistant") as demo:
        gr.Markdown(
            "# Code Checking Assistant\n"
            "Rule-driven C grading with deterministic analyzer evidence and an independent local LLM review. "
            "Infrastructure failures are reported per stage; one unavailable analyzer does not stop the remaining pipeline."
        )
        with gr.Tab("Grade a C submission"):
            gr.Markdown("Upload the C source and the instructor's `rules.json`. Test files and similarity corpora remain resolved relative to the project root.")
            with gr.Row():
                source_file = gr.File(label="C source file", file_types=[".c"], type="filepath")
                rules_file = gr.File(label="Instructor rules JSON", file_types=[".json"], type="filepath")
            grade_button = gr.Button("Run grading pipeline", variant="primary")
            grade_summary = gr.Markdown()
            with gr.Row():
                report_json = gr.File(label="Full JSON report")
                report_text = gr.File(label="Readable text report")
            grade_button.click(
                grade_uploaded_submission,
                inputs=[source_file, rules_file],
                outputs=[grade_summary, report_json, report_text],
            )

        with gr.Tab("FormAI-v2 combined benchmark"):
            gr.Markdown(
                "Run the combined source-level pipeline on a held-out split. Both cppcheck and the local LLM are attempted independently; "
                "`hybrid_rsi` adds evidence-led review but never discards a valid primary result."
            )
            with gr.Row():
                source_inputs = gr.File(label="test_inputs.jsonl (source only)", file_types=[".jsonl"], type="filepath")
                ground_truth = gr.File(label="test_labels.json (evaluator only)", file_types=[".json"], type="filepath")
            with gr.Row():
                benchmark_mode = gr.Radio(
                    choices=[("Hybrid: LLM + cppcheck", "hybrid"), ("Hybrid + evidence review (RSI)", "hybrid_rsi")],
                    value=settings.UI_DEFAULT_BENCHMARK_MODE, label="Combined prediction mode",
                )
                sample_limit = gr.Number(value=settings.UI_DEFAULT_MAX_SAMPLES, precision=0, label="Max samples (0 = all)")
            with gr.Row():
                tool_timeout = gr.Number(value=settings.UI_DEFAULT_TOOL_TIMEOUT_S, precision=0, label="cppcheck timeout (seconds)")
                max_findings = gr.Number(value=settings.UI_DEFAULT_MAX_FINDINGS, precision=0, label="Maximum LLM findings")
                rsi_iterations = gr.Slider(1, settings.LLM_REVIEW_MAX_ITERATIONS, value=settings.UI_DEFAULT_RSI_ITERATIONS, step=1, label="RSI iterations")
                memory_limit = gr.Dropdown(list(settings.UI_DOCKER_MEMORY_CHOICES), value=settings.UI_DEFAULT_MEMORY_LIMIT, label="Docker memory limit")
            benchmark_button = gr.Button("Run combined evaluation", variant="primary")
            benchmark_summary = gr.Markdown()
            with gr.Row():
                predictions_file = gr.File(label="Prediction JSONL")
                evaluation_file = gr.File(label="Evaluation JSON")
            benchmark_button.click(
                run_formai_combined,
                inputs=[source_inputs, ground_truth, benchmark_mode, sample_limit, tool_timeout, max_findings, rsi_iterations, memory_limit],
                outputs=[benchmark_summary, predictions_file, evaluation_file],
            )
        gr.Markdown(
            "**Scoring policy:** every enabled metric receives an analyzer score, an LLM RSI score, or both. "
            "Python combines both when present and always computes the configured weighted total."
        )
    return demo


def main() -> None:
    logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT_APP)
    app = build_app()
    app.queue().launch(server_name=settings.UI_HOST, server_port=settings.UI_PORT, show_error=settings.UI_SHOW_ERROR)


if __name__ == "__main__":
    main()
