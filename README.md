# Focused LLM Review + Modularity Fix

This is a **small patch**, not a full project replacement. Copy these files into the root of your existing Code-Checking-Assistant repository, preserving the directory structure:

- `src/grading_agent/llm/reviewer.py`
- `src/grading_agent/reporting/text_report.py`
- `rules.json`

The three files under `tests/` are new/updated regression tests; copy them into the existing `tests/` directory.

## What changed

1. The general LLM reviewer requests JSON-mode output from llama.cpp, validates the required fields, retries an empty/invalid response once per pass, and never labels an empty response as `completed`.
2. If final synthesis fails, the latest valid earlier review is retained and the overall status becomes `partial`; if all attempts fail, the status is `unavailable` with a non-empty explanation. Failed passes are retained as diagnostics.
3. The text report displays `completed`, `partial`, and `unavailable` states, including failure details, rather than suppressing non-completed review information.
4. `MAX_SOURCE_CHARS = 6000` is defined in `reviewer.py` for the dedicated memory-safety source audit.
5. The `sum_two_integers` rules no longer require a helper function: `expect_helpers=false`, `min_functions=1`. This avoids penalizing a one-function implementation for unnecessary decomposition. Keep helper requirements enabled for assignments that explicitly demand them.

## Run focused tests

```bash
PYTHONPATH=src python -m unittest \
  tests.test_reviewer_context \
  tests.test_modularity_simple_task \
  tests.test_text_report \
  tests.test_memory_safety_scoring \
  tests.test_rule_loader -v
```

Then rerun grading:

```bash
python -m grading_agent.main
```

Changing `rules.json` changes its SHA-256, so the next report will show a new rules hash. Recomputing your supplied report with the revised modularity rule gives **98.80/100**, assuming all other metric results remain unchanged.

The unit tests validate JSON-mode requests, retry behavior, valid-review fallback, status reporting, and modularity scoring. The live request to your local llama.cpp server must still be confirmed in your environment.
