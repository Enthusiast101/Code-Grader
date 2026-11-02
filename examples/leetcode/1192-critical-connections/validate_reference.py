#!/usr/bin/env python3
"""Compile the correct C reference and verify the fixture's expected outputs."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_cases(filename: str) -> list[dict[str, str]]:
    with (ROOT / filename).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{filename} must contain a JSON array")
    return data


def main() -> int:
    source = ROOT / "reference_solution.c"
    regular = load_cases("testcases.json")
    edge = load_cases("edge_cases.json")
    runtime = load_cases("runtime_cases.json")
    cases = [("testcases", c) for c in regular] + [("edge_cases", c) for c in edge] + [("runtime_cases", c) for c in runtime]

    with tempfile.TemporaryDirectory(prefix="critical_connections_") as temp_dir:
        binary = Path(temp_dir) / ("reference.exe" if __import__("os").name == "nt" else "reference")
        compile_result = subprocess.run(
            ["gcc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Wpedantic", str(source), "-o", str(binary)],
            capture_output=True, text=True, check=False,
        )
        if compile_result.returncode != 0:
            print(compile_result.stdout)
            print(compile_result.stderr)
            raise SystemExit("Reference solution failed to compile")

        failures = 0
        for suite, case in cases:
            result = subprocess.run(
                [str(binary)], input=case["input"], capture_output=True,
                text=True, timeout=10, check=False,
            )
            expected = case["expected"]
            if result.returncode != 0 or result.stdout != expected:
                failures += 1
                print(f"FAIL [{suite}] {case.get('id', '<unnamed>')}")
                print(f"  exit: {result.returncode}")
                print(f"  expected: {expected[:200]!r}")
                print(f"  actual:   {result.stdout[:200]!r}")
                if result.stderr:
                    print(f"  stderr: {result.stderr[:500]}")
            else:
                print(f"PASS [{suite}] {case.get('id', '<unnamed>')}")

        print(f"\n{len(cases) - failures}/{len(cases)} cases passed")
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
