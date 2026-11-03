#!/usr/bin/env python3
"""Apply the fail-closed N/A presentation fix to the existing Code-Grader tree.

Run from the repository root. The script refuses to overwrite files when the
expected source snippets do not match, so it is safe against an unrelated version.
"""
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"Refusing to patch {path}: expected one matching block, found {count}.")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")


app = Path("src/grading_agent/ui/app.py")
replace_once(app,
'''    lines = ["## Final grading result"]
    lines.append(f"**Overall grade: {number(scoring.get('score')):.2f}/100**")
    lines.append(f"**Status:** `{scoring.get('status', 'completed_with_warnings')}`")
''',
'''    lines = ["## Final grading result"]
    scoring_status = str(scoring.get("status", "unknown"))
    overall_value = scoring.get("score")
    if scoring_status == "completed" and isinstance(overall_value, (int, float)) and not isinstance(overall_value, bool) and math.isfinite(float(overall_value)):
        overall_display = f"{float(overall_value):.2f}/100"
    else:
        overall_display = "N/A (grade not finalized)"
    lines.append(f"**Overall grade: {overall_display}**")
    lines.append(f"**Status:** `{scoring_status}`")
''')
replace_once(app,
'''        lines.append(f"- **{name}: {number(result.get('score')):.2f}/100**")
        rationale = compact_rationale(result.get("score_rationale", ""), max_words=18, max_chars=120)
''',
'''        metric_status = str(result.get("status", "unknown"))
        metric_value = result.get("score")
        if metric_status == "completed" and isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool) and math.isfinite(float(metric_value)):
            metric_display = f"{float(metric_value):.2f}/100"
        else:
            metric_reason = str(result.get("reason") or result.get("error") or metric_status)
            metric_display = f"N/A ({metric_status}: {metric_reason[:160]})"
        lines.append(f"- **{name}: {metric_display}**")
        rationale = compact_rationale(result.get("score_rationale", ""), max_words=18, max_chars=120)
''')
replace_once(app,
'''        try:
            score = float(scoring.get("score", 0.0))
            if not math.isfinite(score):
                score = 0.0
        except (TypeError, ValueError):
            score = 0.0
        lines = [
            "## Final grading result",
            f"**Overall grade: {score:.2f}/100**",
            f"**Status:** `{scoring.get('status', 'completed_with_warnings')}`",
''',
'''        raw_score = scoring.get("score")
        score_is_valid = (
            scoring.get("status") == "completed"
            and isinstance(raw_score, (int, float))
            and not isinstance(raw_score, bool)
            and math.isfinite(float(raw_score))
        )
        grade_display = f"{float(raw_score):.2f}/100" if score_is_valid else "N/A (grade not finalized)"
        lines = [
            "## Final grading result",
            f"**Overall grade: {grade_display}**",
            f"**Status:** `{scoring.get('status', 'completed_with_warnings')}`",
''')
replace_once(app,
'''            f"Formatting diagnostic: `{type(exc).__name__}: {exc}`",
        ]
''',
'''            f"Formatting diagnostic: `{type(exc).__name__}: {exc}`",
        ]
        for missing in scoring.get("missing_metrics", []) if isinstance(scoring.get("missing_metrics"), list) else []:
            if isinstance(missing, dict):
                lines.append(f"- **{missing.get('metric', 'unknown')}: N/A** — {str(missing.get('reason', 'not measured'))[:300]}")
''')

# Show automatic test-generation provenance and label the LLM assessment advisory.
replace_once(app,
    "    rsi_overview = report.get(\"llm_rsi_evaluation\", {})\n",
    """    test_generation = report.get(\"test_generation\", {})
    if isinstance(test_generation, dict) and test_generation:
        generation_status = str(test_generation.get(\"status\", \"unknown\"))
        if generation_status == \"completed\":
            lines.append(
                f\"**Automatic test suite:** {test_generation.get('total_test_count', 0)} cases \"
                f\"({test_generation.get('instructor_test_count', 0)} instructor + generated), \"
                f\"{test_generation.get('edge_test_count', 0)} edge cases; source=`{test_generation.get('source', 'unknown')}`\"
            )
            if test_generation.get(\"suite_sha256\"):
                lines.append(f\"**Test suite SHA-256:** `{str(test_generation['suite_sha256'])[:16]}…`\")
            if test_generation.get(\"oracle_caveat\"):
                lines.append(\"**Oracle note:** \" + str(test_generation[\"oracle_caveat\"])[:400])
        elif generation_status == \"failed\":
            lines.append(\"**Automatic test generation failed:** \" + str(test_generation.get(\"error\", \"unknown cause\"))[:500])
    rsi_overview = report.get(\"llm_rsi_evaluation\", {})
""",
)
replace_once(app,
    "        lines.append(\"\\n### Executive assessment\")\n",
    "        lines.append(\"\\n### LLM assessment (advisory; deterministic analyzers govern the grade)\")\n",
)
replace_once(app,
    "        rationale = compact_rationale(result.get(\"score_rationale\", \"\"), max_words=18, max_chars=120)\n        if rationale:\n            lines.append(f\"  - Assessment: {rationale}\")\n",
    "        rationale = compact_rationale(result.get(\"score_rationale\", \"\"), max_words=18, max_chars=120)\n        if rationale:\n            lines.append(f\"  - Assessment: {rationale}\")\n        llm_review = result.get(\"llm_review\", {})\n        if isinstance(llm_review, dict) and llm_review.get(\"score_band\"):\n            lines.append(f\"  - LLM advisory band: `{llm_review['score_band']}` (not used in final grade)\")\n",
)

report = Path("src/grading_agent/reporting/text_report.py")
replace_once(report,
'''        score = result.get("score")
        status = result.get("status", "completed" if score is not None else "inconclusive")
        score_text = f"{float(score):.2f}/100" if isinstance(score, (int, float)) else "N/A"
''',
'''        score = result.get("score")
        status = result.get("status", "completed" if score is not None else "inconclusive")
        score_valid = status == "completed" and isinstance(score, (int, float)) and not isinstance(score, bool)
        score_text = f"{float(score):.2f}/100" if score_valid else "N/A"
''')
replace_once(report,
'''    if scoring.get("status") == "completed":
        lines.append(f"Final grade: {scoring.get('score'):.2f}/100")
        for item in scoring.get("contributions", []):
            lines.append(
                f"  {item['metric']}: {item['score']:.2f} × {item['weight']:.3f} = {item['contribution']:.2f}"
            )
    else:
        lines.append(f"Final grade: {float(scoring.get('score', 0.0)):.2f}/100")
''',
'''    overall_value = scoring.get("score")
    if scoring.get("status") == "completed" and isinstance(overall_value, (int, float)) and not isinstance(overall_value, bool):
        lines.append(f"Final grade: {float(overall_value):.2f}/100")
        for item in scoring.get("contributions", []):
            lines.append(
                f"  {item['metric']}: {item['score']:.2f} × {item['weight']:.3f} = {item['contribution']:.2f}"
            )
    else:
        lines.append(f"Final grade: N/A (status: {scoring.get('status', 'unknown')})")
        for missing in scoring.get("missing_metrics", []):
            if isinstance(missing, dict):
                lines.append(
                    f"  Missing metric: {missing.get('metric', 'unknown')} "
                    f"[{missing.get('status', 'unknown')}]: {str(missing.get('reason', 'not measured'))[:500]}"
                )
''')
# Display the generated suite fingerprint and oracle provenance in the text report.
replace_once(report,
    "    config = report.get(\"configuration\", {})\n",
    """    test_generation = report.get(\"test_generation\", {})
    if isinstance(test_generation, dict) and test_generation:
        lines.extend([\"\", \"AUTOMATIC ASSIGNMENT TEST GENERATION\", \"-\" * 72])
        lines.append(f\"Status: {test_generation.get('status', 'unknown')}\")
        if test_generation.get(\"status\") == \"completed\":
            lines.append(f\"Total cases: {test_generation.get('total_test_count', 0)}\")
            lines.append(f\"Instructor cases: {test_generation.get('instructor_test_count', 0)}\")
            lines.append(f\"Edge cases: {test_generation.get('edge_test_count', 0)}\")
            lines.append(f\"Suite fingerprint: {test_generation.get('assignment_fingerprint', 'unknown')}\")
            lines.append(f\"Reference SHA-256: {test_generation.get('reference_solution_sha256', 'unknown')}\")
            if test_generation.get(\"oracle_caveat\"):
                lines.append(f\"Oracle limitation: {test_generation['oracle_caveat']}\")
        elif test_generation.get(\"error\"):
            lines.append(f\"Error: {test_generation['error']}\")

    config = report.get(\"configuration\", {})
""",
)
replace_once(report,
    "        lines.extend([\"\", \"EXECUTIVE ASSESSMENT\", \"-\" * 72, str(rsi_summary[\"overall_assessment\"]).replace(\"\\n\", \" \")])\n",
    "        lines.extend([\"\", \"LLM ASSESSMENT (ADVISORY; DETERMINISTIC SCORES GOVERN GRADE)\", \"-\" * 72, str(rsi_summary[\"overall_assessment\"]).replace(\"\\n\", \" \")])\n",
)
replace_once(report,
    "        if rationale:\n            lines.append(f\"  Assessment: {rationale}\")\n        if name == \"correctness\":\n",
    "        if rationale:\n            lines.append(f\"  Assessment: {rationale}\")\n        llm_review = result.get(\"llm_review\", {})\n        if isinstance(llm_review, dict) and llm_review.get(\"score_band\"):\n            lines.append(f\"  LLM advisory band: {llm_review['score_band']} (not used in final grade)\")\n        if name == \"correctness\":\n",
)
print("Applied fail-closed N/A reporting changes to Gradio and text report.")
