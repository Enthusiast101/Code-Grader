"""Benchmark metrics for stratified FormAI runs.

Scoring rules (all visible in the evaluation JSON):

* Ground truth is program-level: VULNERABLE (ESBMC counterexample) or SAFE
  (verified within the bound). Programs ESBMC could not decide are never sampled.
* A prediction is vulnerable, safe, or inconclusive (a stage failed or the
  evidence was contradictory). Headline metrics are **strict**: an inconclusive
  result counts as an error, i.e. a missed vulnerability on a vulnerable
  program and a false alarm on a safe one. Metrics over decided programs only
  are reported next to them.
* **Population estimates** weight each program by its stratum's design weight
  ``N_h / n_h`` so the figures describe the whole eligible dataset, not the
  sample's class mix. Precision in particular depends on prevalence, so the
  unweighted sample value is only comparable under proportional allocation.
* 95% confidence intervals come from a stratified bootstrap (resampling within
  each stratum, fixed seed).
* Location: a finding hits a vulnerability when its line is within
  ``line_tolerance`` lines of an ESBMC vulnerable line; "same family" also
  requires the vulnerability family to match. ESBMC checks a bounded program,
  so findings away from its lines are not necessarily wrong; location precision
  is therefore a lower bound.
* Undefined ratios (zero denominator) are null, never zero.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable

from grading_agent.benchmark.formai_dataset import FAMILIES

HEADLINE = ("precision", "recall", "specificity", "f1", "accuracy", "balanced_accuracy", "mcc")


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator <= 0 else numerator / denominator


def binary_metrics(tp: float, fp: float, fn: float, tn: float) -> dict[str, float | None]:
    """Standard binary metrics; counts may be fractional (weighted)."""
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    accuracy = _ratio(tp + tn, tp + fp + fn + tn)
    balanced = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / math.sqrt(denominator) if denominator > 0 else None
    return {
        "precision": precision, "recall": recall, "specificity": specificity, "f1": f1,
        "accuracy": accuracy, "balanced_accuracy": balanced, "mcc": mcc,
        "false_positive_rate": _ratio(fp, fp + tn), "false_negative_rate": _ratio(fn, fn + tp),
    }


def outcome(record: dict[str, Any], *, strict: bool) -> str | None:
    """tp/fp/fn/tn for one program, or None when excluded (non-strict inconclusive)."""
    truth, predicted = bool(record["truth"]), record.get("predicted")
    if predicted is None:
        if not strict:
            return None
        return "fn" if truth else "fp"
    if truth:
        return "tp" if predicted else "fn"
    return "fp" if predicted else "tn"


def _confusion(records: Iterable[dict[str, Any]], *, strict: bool, weighted: bool) -> dict[str, float]:
    counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    for record in records:
        kind = outcome(record, strict=strict)
        if kind:
            counts[kind] += float(record.get("weight") or 1.0) if weighted else 1.0
    return counts


def _round(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: _round(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item, digits) for item in value]
    return value


def _within(line: Any, targets: Iterable[int], tolerance: int) -> bool:
    return isinstance(line, int) and any(abs(line - target) <= tolerance for target in targets)


def _rate(flags: list[tuple[bool, float]], weighted: bool) -> float | None:
    if not flags:
        return None
    total = sum(weight if weighted else 1.0 for _, weight in flags)
    return _ratio(sum((weight if weighted else 1.0) for hit, weight in flags if hit), total)


def localisation(records: list[dict[str, Any]], tolerance: int) -> dict[str, Any]:
    """Line and family agreement with ESBMC on vulnerable programs."""
    detected = [r for r in records if r["truth"] and r.get("predicted") is True]
    primary, any_line, family, line_family = [], [], [], []
    for r in detected:
        weight = float(r.get("weight") or 1.0)
        truth_lines = r.get("truth_lines") or []
        truth_vulns = r.get("truth_vulnerabilities") or []
        findings = r.get("findings") or []
        primary.append((_within(r.get("predicted_line"), truth_lines, tolerance), weight))
        any_line.append((any(_within(f.get("line"), truth_lines, tolerance) for f in findings), weight))
        family.append((bool({f.get("family") for f in findings} & set(r.get("truth_families") or [])), weight))
        line_family.append((any(
            f.get("family") == v.get("family") and _within(f.get("line"), [v["line"]], tolerance)
            for f in findings for v in truth_vulns if v.get("line")), weight))

    # Finding level, over every sampled vulnerable program (detected or not).
    findings_total = findings_on_lines = 0
    truth_total = truth_found = truth_found_family = 0
    for r in records:
        if not r["truth"]:
            continue
        findings = r.get("findings") or []
        truth_lines = r.get("truth_lines") or []
        findings_total += len(findings)
        findings_on_lines += sum(_within(f.get("line"), truth_lines, tolerance) for f in findings)
        seen: set[tuple[int, str]] = set()
        for v in r.get("truth_vulnerabilities") or []:
            if not v.get("line") or (v["line"], v.get("family")) in seen:
                continue
            seen.add((v["line"], v.get("family")))
            truth_total += 1
            truth_found += any(_within(f.get("line"), [v["line"]], tolerance) for f in findings)
            truth_found_family += any(
                f.get("family") == v.get("family") and _within(f.get("line"), [v["line"]], tolerance) for f in findings)
    safe = [r for r in records if not r["truth"]]
    return {
        "line_tolerance": tolerance,
        "detected_vulnerable_programs": len(detected),
        "primary_line_hit": {"sample": _rate(primary, False), "population": _rate(primary, True)},
        "any_finding_line_hit": {"sample": _rate(any_line, False), "population": _rate(any_line, True)},
        "family_match": {"sample": _rate(family, False), "population": _rate(family, True)},
        "line_and_family_match": {"sample": _rate(line_family, False), "population": _rate(line_family, True)},
        "findings_on_vulnerable_programs": findings_total,
        "finding_line_precision_lower_bound": _ratio(findings_on_lines, findings_total),
        "esbmc_vulnerabilities": truth_total,
        "esbmc_vulnerability_recall_line": _ratio(truth_found, truth_total),
        "esbmc_vulnerability_recall_line_and_family": _ratio(truth_found_family, truth_total),
        "mean_findings_per_safe_program": _ratio(sum(len(r.get("findings") or []) for r in safe), len(safe)),
    }


def per_family(records: list[dict[str, Any]], tolerance: int) -> list[dict[str, Any]]:
    """Detection and location by ESBMC family (a program counts under each family it has)."""
    rows = []
    for family_name, description in FAMILIES.items():
        members = [r for r in records if r["truth"] and family_name in (r.get("truth_families") or [])]
        if not members:
            continue
        detected = sum(r.get("predicted") is True for r in members)
        located = sum(any(
            f.get("family") == family_name and _within(
                f.get("line"), [v["line"] for v in r.get("truth_vulnerabilities") or []
                                if v.get("family") == family_name and v.get("line")], tolerance)
            for f in r.get("findings") or []) for r in members)
        rows.append({
            "family": family_name, "description": description, "programs": len(members),
            "detected": detected, "detection_rate": _ratio(detected, len(members)),
            "located_same_family": located, "location_rate": _ratio(located, len(members)),
        })
    return rows


def per_stratum(records: list[dict[str, Any]], strata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_stratum[record["stratum"]].append(record)
    rows = []
    for stratum in strata:
        members = by_stratum.get(stratum["key"], [])
        counts = _confusion(members, strict=False, weighted=False)
        inconclusive = sum(r.get("predicted") is None for r in members)
        strict = _confusion(members, strict=True, weighted=False)
        vulnerable = stratum["label"] == "VULNERABLE"
        rows.append({
            "stratum": stratum["key"], "label": stratum["label"], "population": stratum["population"],
            "sampled": len(members), "weight": stratum.get("weight"),
            **{key: int(value) for key, value in counts.items()}, "inconclusive": inconclusive,
            # The only defined rate inside one-label stratum: recall or specificity.
            "metric": "recall" if vulnerable else "specificity",
            "value": _ratio(strict["tp"], strict["tp"] + strict["fn"]) if vulnerable
            else _ratio(strict["tn"], strict["tn"] + strict["fp"]),
        })
    return rows


def _bootstrap(records: list[dict[str, Any]], samples: int, seed: int) -> dict[str, dict[str, list[float]]]:
    """Percentile CIs from resampling programs with replacement inside each stratum."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["stratum"]].append(record)
    rng = random.Random(f"formai-bootstrap:{seed}")
    draws: dict[str, dict[str, list[float]]] = {
        view: {name: [] for name in HEADLINE} for view in ("population", "sample", "decided")}
    for _ in range(samples):
        resample = [member for group in groups.values() for member in rng.choices(group, k=len(group))]
        for view, strict, weighted in (("population", True, True), ("sample", True, False), ("decided", False, False)):
            metrics = binary_metrics(**_confusion(resample, strict=strict, weighted=weighted))
            for name in HEADLINE:
                if metrics[name] is not None:
                    draws[view][name].append(metrics[name])
    intervals: dict[str, dict[str, list[float]]] = {}
    for view, by_metric in draws.items():
        intervals[view] = {}
        for name, values in by_metric.items():
            if len(values) >= max(20, samples // 2):
                values.sort()
                low = values[int(0.025 * (len(values) - 1))]
                high = values[int(math.ceil(0.975 * (len(values) - 1)))]
                intervals[view][name] = [low, high]
    return intervals


def evaluate(
    records: list[dict[str, Any]],
    *,
    strata: list[dict[str, Any]],
    line_tolerance: int = 2,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Full evaluation of one stratified run (see module docstring)."""
    if not records:
        raise ValueError("No predictions to evaluate.")
    sampled_keys = {r["stratum"] for r in records}
    eligible = sum(s["population"] for s in strata)
    covered = sum(s["population"] for s in strata if s["key"] in sampled_keys)
    inconclusive = [r for r in records if r.get("predicted") is None]
    decided = len(records) - len(inconclusive)
    views = {
        "population": _confusion(records, strict=True, weighted=True),
        "sample": _confusion(records, strict=True, weighted=False),
        "decided": _confusion(records, strict=False, weighted=False),
    }
    intervals = _bootstrap(records, bootstrap_samples, seed) if bootstrap_samples > 0 else {}
    metrics = {}
    for view, counts in views.items():
        values = binary_metrics(**counts)
        metrics[view] = {
            "confusion": counts if view == "population" else {key: int(value) for key, value in counts.items()},
            **values,
            "ci95": intervals.get(view, {}),
        }
    unsampled = [s["key"] for s in strata if s["key"] not in sampled_keys and s["population"]]
    reasons: dict[str, dict[str, int]] = defaultdict(lambda: {"vulnerable": 0, "safe": 0})
    for r in inconclusive:
        reasons[r.get("inconclusive_reason") or "unrecorded"]["vulnerable" if r["truth"] else "safe"] += 1
    evidence: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0})
    for r in records:
        if r.get("predicted") is True:
            evidence[r.get("positive_evidence") or "unrecorded"]["tp" if r["truth"] else "fp"] += 1
    return _round({
        "programs": len(records),
        "vulnerable_programs": sum(bool(r["truth"]) for r in records),
        "safe_programs": sum(not r["truth"] for r in records),
        "decided": decided,
        "inconclusive": len(inconclusive),
        "inconclusive_vulnerable": sum(bool(r["truth"]) for r in inconclusive),
        "inconclusive_safe": sum(not r["truth"] for r in inconclusive),
        "inconclusive_reasons": dict(sorted(reasons.items(), key=lambda kv: -sum(kv[1].values()))),
        "positive_evidence": dict(sorted(evidence.items(), key=lambda kv: -sum(kv[1].values()))),
        "coverage": _ratio(decided, len(records)),
        "eligible_population": eligible,
        "population_covered_by_sampled_strata": _ratio(covered, eligible),
        "unsampled_strata": unsampled,
        "metrics": metrics,
        "localisation": localisation(records, line_tolerance),
        "per_family": per_family(records, line_tolerance),
        "per_stratum": per_stratum(records, strata),
        "bootstrap_samples": bootstrap_samples,
        "definitions": {
            "population": "Strict (inconclusive = error), weighted by N_h/n_h to estimate the whole eligible dataset.",
            "sample": "Strict (inconclusive = error), unweighted over the sampled programs.",
            "decided": "Unweighted over programs with a vulnerable/safe decision; inconclusive programs excluded.",
            "ci95": "Stratified bootstrap percentile interval.",
        },
    })
