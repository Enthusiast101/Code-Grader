#!/usr/bin/env python3
"""Compile reference_solution.c and check it against every stored test file."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    cases = []
    for name in ("testcases.json", "edge_cases.json", "memory_cases.json"):
        cases += [(name, case) for case in json.loads((ROOT / name).read_text(encoding="utf-8"))]
    with tempfile.TemporaryDirectory() as temp_dir:
        binary = Path(temp_dir) / ("reference.exe" if os.name == "nt" else "reference")
        build = subprocess.run(["gcc", "-std=c11", "-O2", "-Wall", "-Wextra", str(ROOT / "reference_solution.c"), "-o", str(binary)],
                               capture_output=True, text=True)
        if build.returncode != 0:
            print(build.stderr)
            return 1
        failures = 0
        for source, case in cases:
            run = subprocess.run([str(binary)], input=case["input"], capture_output=True, text=True, timeout=30)
            if run.stdout.strip() != case["expected"].strip():
                failures += 1
                print(f"FAIL {source}:{case['id']}")
        print(f"{len(cases) - failures}/{len(cases)} cases match the reference")
        return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
