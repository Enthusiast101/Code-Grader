#!/usr/bin/env python3
"""Reject machine-dependent wall-clock runtime as a weighted metric."""
from pathlib import Path
path = Path("src/grading_agent/config/rule_loader.py")
s = path.read_text(encoding="utf-8")
old = '''        if analyzer == "valgrind_memcheck":
            analyzer = "memory_safety"

        enabled = metric.get("enabled", True)
'''
new = '''        if analyzer == "valgrind_memcheck":
            analyzer = "memory_safety"

        enabled = metric.get("enabled", True)
        if enabled and analyzer == "runtime_profile":
            raise RuleValidationError(
                f"Metric '{name}' uses wall-clock runtime_profile, which is machine-dependent "
                "and cannot contribute to a reproducible grade. Disable/remove this metric; "
                "use the algorithmic complexity metric and keep runtime measurements as telemetry only."
            )
'''
if s.count(old) != 1:
    raise SystemExit(f"Refusing runtime-scoring guard patch: expected one matching block, found {s.count(old)}.")
path.write_text(s.replace(old, new, 1), encoding="utf-8")
print("Applied guard against machine-dependent runtime scoring.")
