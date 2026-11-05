"""Stratified random sampling of FormAI programs with design weights.

Programs are grouped into strata (label, plus vulnerability family or
generating model), a sample size is allocated to each stratum, and programs
are drawn at random inside each stratum. Every sampled program carries the
design weight ``N_h / n_h`` (population size over sample size of its stratum)
so metrics can be projected back to the whole eligible dataset even when the
allocation over-samples rare strata.

Labels are used here only to form strata; the prediction pipeline receives
source code alone.
"""
from __future__ import annotations

import math
import random
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any

from grading_agent.benchmark.formai_dataset import FAMILIES, SAFE, VULNERABLE

STRATIFY_OPTIONS: dict[str, str] = {
    "label_family": "Label × vulnerability family",
    "label": "Label only (vulnerable / safe)",
    "label_generator": "Label × generating model",
}
ALLOCATIONS: dict[str, str] = {
    "proportional": "Proportional: the sample mirrors the dataset's class mix",
    "balanced": "Balanced: half vulnerable, half safe, proportional inside each half",
}


@dataclass
class Stratum:
    key: str
    label: str
    population: int
    allocated: int = 0

    @property
    def weight(self) -> float | None:
        return self.population / self.allocated if self.allocated else None


@dataclass
class Sample:
    dataset: str
    stratify_by: str
    allocation: str
    seed: int
    requested: int
    strata: list[Stratum]
    programs: list[dict[str, Any]] = field(default_factory=list)  # file_name, stratum, weight

    @property
    def size(self) -> int:
        return len(self.programs)

    @property
    def population(self) -> int:
        return sum(stratum.population for stratum in self.strata)

    def to_json(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "stratify_by": self.stratify_by,
            "allocation": self.allocation,
            "seed": self.seed,
            "requested": self.requested,
            "size": self.size,
            "population": self.population,
            "strata": [{**asdict(stratum), "weight": stratum.weight} for stratum in self.strata],
            "programs": self.programs,
        }


def _stratum_sql(stratify_by: str) -> str:
    if stratify_by == "label":
        return "label"
    if stratify_by == "label_family":
        return "CASE WHEN label='VULNERABLE' THEN 'VULNERABLE:' || COALESCE(primary_family, 'other') ELSE label END"
    if stratify_by == "label_generator":
        return "label || ':' || generator"
    raise ValueError(f"Unknown stratification {stratify_by!r}; choose one of {sorted(STRATIFY_OPTIONS)}.")


def _sort_key(key: str) -> tuple:
    label, _, rest = key.partition(":")
    family_order = list(FAMILIES)
    return (0 if label == VULNERABLE else 1, family_order.index(rest) if rest in family_order else 99, rest)


def population_strata(connection: sqlite3.Connection, stratify_by: str) -> list[Stratum]:
    """Eligible programs (VULNERABLE or SAFE) per stratum."""
    expression = _stratum_sql(stratify_by)
    rows = connection.execute(
        f"SELECT {expression} AS stratum, label, COUNT(*) AS n FROM programs "
        "WHERE label IN ('VULNERABLE', 'SAFE') GROUP BY stratum, label"
    ).fetchall()
    strata = [Stratum(key=row[0], label=row[1], population=row[2]) for row in rows]
    return sorted(strata, key=lambda stratum: _sort_key(stratum.key))


def _largest_remainder(sizes: list[int], total: int, *, minimum_one: bool) -> list[int]:
    """Split ``total`` across groups proportionally to ``sizes`` (Hamilton method)."""
    population = sum(sizes)
    total = min(total, population)
    if total <= 0 or population == 0:
        return [0] * len(sizes)
    quotas = [total * size / population for size in sizes]
    counts = [min(size, math.floor(quota)) for size, quota in zip(sizes, quotas)]
    if minimum_one and total >= sum(1 for size in sizes if size):
        counts = [max(count, 1) if size else 0 for count, size in zip(counts, sizes)]
    # Hand out (or take back) units by largest remainder until the sum matches.
    order = sorted(range(len(sizes)), key=lambda i: (-(quotas[i] - math.floor(quotas[i])), -sizes[i], i))
    while sum(counts) < total:
        progressed = False
        for i in order:
            if sum(counts) >= total:
                break
            if counts[i] < sizes[i]:
                counts[i] += 1
                progressed = True
        if not progressed:
            break
    while sum(counts) > total:
        # Take from the most over-represented stratum that keeps at least one.
        candidates = [i for i in range(len(sizes)) if counts[i] > 1]
        if not candidates:
            break
        i = max(candidates, key=lambda j: (counts[j] - quotas[j], counts[j]))
        counts[i] -= 1
    return counts


def allocate(strata: list[Stratum], n: int, allocation: str) -> list[Stratum]:
    """Set ``allocated`` on each stratum; every stratum gets >= 1 when n allows."""
    if allocation not in ALLOCATIONS:
        raise ValueError(f"Unknown allocation {allocation!r}; choose one of {sorted(ALLOCATIONS)}.")
    if n < 1:
        raise ValueError("Sample size must be at least 1.")
    if allocation == "proportional":
        counts = _largest_remainder([s.population for s in strata], n, minimum_one=True)
    else:
        groups = {label: [i for i, s in enumerate(strata) if s.label == label] for label in (VULNERABLE, SAFE)}
        available = {label: sum(strata[i].population for i in idx) for label, idx in groups.items()}
        # Half each; any shortfall in one label is given to the other.
        want = {VULNERABLE: math.ceil(n / 2), SAFE: n // 2}
        for label, other in ((VULNERABLE, SAFE), (SAFE, VULNERABLE)):
            if want[label] > available[label]:
                want[other] += want[label] - available[label]
                want[label] = available[label]
        counts = [0] * len(strata)
        for label, idx in groups.items():
            split = _largest_remainder([strata[i].population for i in idx], want[label], minimum_one=True)
            for i, count in zip(idx, split):
                counts[i] = count
    for stratum, count in zip(strata, counts):
        stratum.allocated = count
    return strata


def draw_sample(
    connection: sqlite3.Connection,
    *,
    dataset: str,
    n: int,
    stratify_by: str = "label_family",
    allocation: str = "proportional",
    seed: int = 42,
) -> Sample:
    """Draw a reproducible stratified sample (same inputs -> same programs)."""
    strata = allocate(population_strata(connection, stratify_by), n, allocation)
    expression = _stratum_sql(stratify_by)
    programs: list[dict[str, Any]] = []
    for stratum in strata:
        if not stratum.allocated:
            continue
        names = [row[0] for row in connection.execute(
            f"SELECT file_name FROM programs WHERE label IN ('VULNERABLE', 'SAFE') AND {expression} = ? "
            "ORDER BY file_name", (stratum.key,))]
        # A per-stratum shuffle keyed by the seed: a larger sample from the same
        # seed extends a smaller one inside each stratum.
        random.Random(f"formai:{seed}:{stratum.key}").shuffle(names)
        for name in names[:stratum.allocated]:
            programs.append({"file_name": name, "stratum": stratum.key, "weight": stratum.weight})
    total = sum(s.population for s in strata)
    return Sample(dataset=dataset, stratify_by=stratify_by, allocation=allocation, seed=seed,
                  requested=min(n, total), strata=strata, programs=programs)


def allocation_table(strata: list[Stratum]) -> list[list[Any]]:
    """Rows for display: stratum, population, share, sampled, weight."""
    total = sum(s.population for s in strata) or 1
    return [
        [s.key, s.population, round(100 * s.population / total, 2), s.allocated,
         round(s.weight, 1) if s.weight else None]
        for s in strata
    ]
