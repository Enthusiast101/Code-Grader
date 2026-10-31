# Code Checking Assistant

A rule-driven C assignment grader. Python loads instructor-provided `rules.json`, dispatches independent analyzers, runs submitted code in a restricted Docker sandbox, collects structured evidence and raw tool logs, applies deterministic weighted scoring, and uses a local `llama.cpp` model for bounded evidence review and feedback.

## Architecture implemented

- **Inputs:** `rules.json` and the selected C source file are resolved together by the Python coordinator.
- **Sandbox:** Docker runs compilation, tests, cppcheck, Valgrind, and sanitizers with no network, read-only root filesystem, dropped capabilities, CPU/memory/PID limits, and writable workspace/tmpfs only.
- **Static/source analyzers:** GCC diagnostics, cppcheck, Lizard-backed complexity when installed (lexical fallback otherwise), modularity, readability heuristics, and token-shingle similarity against an instructor-authorized submissions corpus.
- **Dynamic analyzers:** I/O correctness, instructor-defined edge/corner tests, repeated runtime profiling, Valgrind Memcheck, and supplementary ASan/UBSan runs.
- **Composite memory safety:** one instructor metric invokes Valgrind and an independent LLM source audit. The default component weights are 70% Valgrind and 30% LLM audit; defaults and severity penalties are in `config/settings.py`. ASan/UBSan diagnostics are retained as supplementary evidence and are not silently assigned a third weight.
- **LLM review:** one source-level audit plus a bounded 1–3 pass review of all analyzer evidence. The LLM does not choose or change numerical scores.
- **Scoring:** Python computes the composite memory score and overall weighted score. If any enabled metric is unavailable or inconclusive, no official overall grade is issued and weights are not renormalized.
- **FormAI:** independent benchmark sampling and evaluation utilities compare predictions against held-out FormAI labels. The dataset itself is external and is not bundled or passed to the prediction-time LLM.

## Prerequisites

- Python 3.11+
- Docker Engine running
- `llama-server` from `llama.cpp` available on `PATH`, or set `LLAMA_SERVER` to its full path
- A compatible GGUF model supported by your `llama.cpp` build

## Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

`lizard` is a Python dependency and is used for cyclomatic-complexity/function metrics. If it is unavailable, the source metrics module explicitly reports that it used a lightweight lexical fallback.

## Build the sandbox image

Rebuild after changes to the installed tools:

```bash
docker build --no-cache -f Dockerfile.sandbox -t agent-sandbox:latest .
```

The image installs GCC, cppcheck, Valgrind, Bash, and core utilities. GCC's AddressSanitizer/UndefinedBehaviorSanitizer support is used for supplementary runs.

## Configure a grading run

Edit `rules.json` in the repository root. The example enables:

| Metric | Analyzer | Purpose |
|---|---|---|
| correctness | `io_tests` | Compile and run expected-output tests |
| syntax | `compiler_diagnostics` | GCC syntax errors and warnings |
| structure | `static_analysis` | cppcheck diagnostics |
| modularity | `modularity` | Function decomposition heuristics |
| complexity | `complexity_estimator` | Cyclomatic complexity and function length |
| logical | `edge_case_runner` | Instructor-provided boundary/corner cases |
| runtime_efficiency | `runtime_profile` | Repeated per-input wall-time measurements |
| memory_safety | `memory_safety` | Valgrind plus LLM source audit; ASan/UBSan supplementary |
| readability | `readability` | Deterministic style/readability heuristics |
| originality | `similarity_flag` | Token-shingle similarity against prior submissions |

Enabled top-level metric weights must sum to `1.0`. The example `originality` metric is deliberately disabled until the `submissions/` corpus contains authorized comparison files. To enable it, set `enabled` to `true` and adjust the other enabled weights so their sum remains exactly `1.0`. Similarity is only a flag for human review; it is not proof of plagiarism or intent.

The memory-safety metric is configured once. Its internal default weights are Valgrind `0.70` and LLM source audit `0.30`; if only one subweight is specified, the other is set to its complement. Missing/inconclusive mandatory components withhold the composite score. Severity penalties are defaults in `config/settings.py` and can be overridden in that metric's `params`.

`llm_review.max_iterations` accepts 1–3 and defaults to 3. These are bounded evidence-review passes, not a guarantee of mathematical convergence. The official numeric grade remains the deterministic Python result.

## Run

From the repository root:

```bash
python -m grading_agent.main
```

The coordinator validates `rules.json`, runs a compile/smoke preflight (reported separately from weighted metrics), skips execution-dependent analyzers after a preflight compile failure/crash/timeout while retaining source-only checks, dispatches the remaining enabled analyzers, starts or reuses the local Llama server, performs source-level memory audit and bounded general evidence review, computes the weighted grade, and writes:

```text
reports/<problem>.json
reports/<problem>.txt
```

The JSON contains full analyzer details and raw logs (subject to the sandbox's configured file-size limits); the LLM request uses bounded log excerpts. The text report is a compact human-readable summary.

## Similarity corpus

Place authorized prior student submissions/reference C files under `submissions/` (subdirectories are supported). The current example `originality` metric remains disabled until the corpus is ready. Increase `max_files` if the course corpus is larger than the default scan cap. Similarity scores should be reviewed by an instructor; common boilerplate and assignment scaffolding can produce legitimate overlap.

## FormAI-v2 independent evaluation

FormAI-v2 contains a large collection of AI-generated C programs and formal-verification-based vulnerability labels. It is used as an **independent benchmark**, not as an instructor's ordinary assignment test set and not as an input to the prediction-time LLM.

The repository is not included because the dataset is multi-gigabyte. Follow the dataset authors' instructions at <https://github.com/FormAI-Dataset/FormAI-dataset> to download and extract `FormAI-v2.json` outside this repository. Then install the streaming JSON dependency:

```bash
python -m pip install -e '.[benchmark]'
```

Create a deterministic stratified sample with labels separated from source inputs:

```bash
python -m grading_agent.benchmark.formai_sample \
  --dataset /path/to/FormAI-v2.json \
  --per-class 250 \
  --seed 42 \
  --source-output benchmark/formai_inputs.jsonl \
  --labels-output benchmark/formai_ground_truth.json
```

Split that sample into reproducible train, validation, and held-out test partitions. Source-only inputs and labels remain in separate files for every partition:

```bash
python -m grading_agent.benchmark.formai_split \
  --inputs benchmark/formai_inputs.jsonl \
  --labels benchmark/formai_ground_truth.json \
  --validation-fraction 0.2 \
  --test-fraction 0.2 \
  --seed 42 \
  --output-dir benchmark/splits
```

Use the training split for development, the validation split for threshold/prompt tuning, and the test split only for final reporting. The prediction runner implements three baselines: **tools-only** (cppcheck source diagnostics), **LLM-only** (local LLM source audit), and **hybrid** (positive if either source-level method flags a vulnerability; inconclusive if a required method fails). These are source-level benchmark modes: FormAI does not provide a standardized per-program input oracle, so this benchmark runner does not execute arbitrary dataset programs.

Run each mode against the *same held-out source-only test split*:

```bash
python -m grading_agent.benchmark.formai_predict \
  --inputs benchmark/splits/test_inputs.jsonl \
  --mode tools_only \
  --output benchmark/predictions/tools_only.jsonl

python -m grading_agent.benchmark.formai_predict \
  --inputs benchmark/splits/test_inputs.jsonl \
  --mode llm_only \
  --output benchmark/predictions/llm_only.jsonl

python -m grading_agent.benchmark.formai_predict \
  --inputs benchmark/splits/test_inputs.jsonl \
  --mode hybrid \
  --output benchmark/predictions/hybrid.jsonl
```

Keep `formai_ground_truth.json` separate from all prediction runs. A prediction JSONL record has this shape:

```json
{"file_name":"sample.c","predicted_vulnerable":true,"predicted_line":42,"predicted_error_type":"array bounds violated"}
```

Compare the same held-out records for the tools-only, LLM-only, and hybrid predictions:

```bash
python -m grading_agent.benchmark.formai_evaluate \
  --ground-truth benchmark/splits/test_labels.json \
  --predictions tools_only=benchmark/predictions/tools_only.jsonl \
  --predictions llm_only=benchmark/predictions/llm_only.jsonl \
  --predictions hybrid=benchmark/predictions/hybrid.jsonl \
  --line-tolerance 2 \
  --output reports/formai-evaluation.json
```

The evaluator reports Precision, Recall, F1, Accuracy, Specificity, False Positive Rate, False Negative Rate, balanced accuracy, line-localization accuracy, and error-type accuracy. Inconclusive predictions are excluded from confusion-matrix calculations and contribute to the skipped count; missing prediction records are treated as negative predictions and reported explicitly. Use the validation labels for threshold/prompt tuning only. The held-out test labels are not read by the prediction runner and are used only by the evaluator after all prediction files are complete. The hybrid OR rule is a simple baseline, not a claim that it is optimal; compare it empirically and consider a validated fusion rule in later work.

## Evaluation interpretation and limitations

- Passing tests do not prove program correctness for all inputs.
- Valgrind and sanitizers only observe executed paths; source-level LLM review can identify untested risks but can also produce false positives/negatives.
- Cyclomatic complexity/function-size values depend on Lizard when installed; the fallback is lexical and approximate.
- Runtime values include Docker execution overhead and should only be compared on the same environment.
- Similarity detection is a near-duplicate heuristic and requires instructor review.
- FormAI's formal-verification labels and dynamic memory tools measure overlapping but non-identical properties; report results by benchmark category and tool capability rather than assuming every clean runtime sample is universally safe.

## Tests

From the repository root:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
