# FormAI-v2 program-level aggregation patch

This focused patch fixes the FormAI-v2 benchmark's handling of multiple verification records for one C source file.

## Replace these files

- `src/grading_agent/benchmark/formai_sample.py`
- `src/grading_agent/benchmark/formai_evaluate.py`

Copy `tests/test_formai_dataset_aggregation.py` into the project's `tests/` directory.

## What changes

- Reservoir-samples **unique `file_name` programs**, not raw verification rows.
- Aggregates all known vulnerable lines and error types for each vulnerable source program.
- Labels a program `VULNERABLE` when any record confirms a vulnerability.
- Labels a program `SAFE` only when it has non-vulnerable records and no timeout/unknown verification record.
- Excludes unknown/timeout-only programs from binary safe-vulnerable sampling.
- Rejects inconsistent source code for the same filename rather than silently merging it.
- Evaluates one prediction per program; localization succeeds when the predicted line is within tolerance of any known vulnerable line, and type accuracy checks against any recorded error type.
- Rejects duplicate prediction filenames and reports predictions that do not occur in the selected ground truth.

## Test

From the repository root:

```bash
PYTHONPATH=src python -m unittest tests.test_formai_dataset_aggregation -v
```

The prediction modes and their CLI remain unchanged.
