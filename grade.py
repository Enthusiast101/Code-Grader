from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from grader.grading import grade


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic, evidence-backed C code grader")
    parser.add_argument("--rules", default="rules.json", help="Rules JSON path (default: ./rules.json)")
    parser.add_argument("--source", default=None, help="Optional C source path; otherwise auto-detect one .c file next to rules.json")
    parser.add_argument("--refresh-suite", action="store_true", help="Regenerate tests and reference oracle (normally the frozen cache is reused)")
    parser.add_argument("--json-only", action="store_true", help="Print JSON report only")
    args = parser.parse_args()
    try:
        report, json_path, text_path = grade(args.rules, source_override=args.source, refresh_suite=args.refresh_suite)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.json_only:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(text_path.read_text(encoding="utf-8"))
        print(f"JSON report: {json_path}")
        print(f"Text report: {text_path}")
    score = report.get("scoring", {})
    return 0 if score.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
