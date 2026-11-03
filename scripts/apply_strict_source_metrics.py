#!/usr/bin/env python3
"""Require the pinned Lizard implementation; no host-dependent lexical fallback."""
from pathlib import Path
path = Path("src/grading_agent/analyzers/source_metrics.py")
s = path.read_text(encoding="utf-8")
old = '''def _lizard_metrics(filename: str, source: str) -> dict[str, Any] | None:
    """Use Lizard when installed; return None for the local lexical fallback."""
    try:
        import lizard  # type: ignore
    except ImportError:
        return None
    try:
'''
new = '''def _lizard_metrics(filename: str, source: str) -> dict[str, Any]:
    """Use exactly the pinned Lizard implementation or fail the metric."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        import lizard  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Strict reproducibility requires lizard==1.17.10; install requirements.txt.") from exc
    try:
        installed_version = version("lizard")
    except PackageNotFoundError as exc:
        raise RuntimeError("Strict reproducibility requires the pinned lizard package metadata.") from exc
    if installed_version != "1.17.10":
        raise RuntimeError(f"Strict reproducibility requires lizard==1.17.10; found {installed_version}.")
    try:
'''
if s.count(old) != 1:
    raise SystemExit(f"Expected one _lizard_metrics header, found {s.count(old)}")
s=s.replace(old,new,1)
old2='''    except Exception:
        # A parse error should not silently crash all grading. The explicit
        # note makes the fallback visible in the report.
        return None
'''
new2='''    except Exception as exc:
        # Never switch to a different lexical implementation: that would make
        # scores depend on whether Lizard parsed this source successfully.
        raise RuntimeError(f"Pinned Lizard failed to analyze {filename}: {type(exc).__name__}: {exc}") from exc
'''
if s.count(old2) != 1:
    raise SystemExit(f"Expected one Lizard fallback block, found {s.count(old2)}")
s=s.replace(old2,new2,1)
# A strict analyzer result cannot describe a fallback that no longer exists.
s=s.replace('''    if lizard_result is not None:
        functions = lizard_result["functions"]
''','''    functions = lizard_result["functions"]
''')
s=s.replace('''        "measurement_note": (
            lizard_result["measurement_note"] if lizard_result is not None else
            "Lizard unavailable or unable to parse this file; function and cyclomatic-complexity values are lexical estimates, not a full compiler AST analysis."
        ),
''','''        "measurement_note": lizard_result["measurement_note"],
''')
path.write_text(s, encoding="utf-8")
print("Applied strict pinned-Lizard requirement; analyzer drift fallbacks are disabled.")
