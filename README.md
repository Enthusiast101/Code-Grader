# Code Checking Assistant

A rule-driven C assignment grader. Supply `rules.json` and a C source file; Python runs the configured static and dynamic analyzers, collects their evidence, asks a local LLM to perform evidence-based RSI scoring for every enabled metric, and calculates the final weighted grade.

## Architecture implemented

- **Inputs:** `rules.json` and the selected C source file are resolved together by the Python coordinator.
- **Sandbox:** Docker runs compilation, tests, cppcheck, Valgrind, and sanitizers with no network, read-only root filesystem, dropped capabilities, CPU/memory/PID limits, and writable workspace/tmpfs only.
- **Static/source analyzers:** GCC diagnostics, cppcheck, Lizard-backed complexity when installed (lexical fallback otherwise), modularity, readability heuristics, and token-shingle similarity against an instructor-authorized submissions corpus.
- **Dynamic analyzers:** I/O correctness, instructor-defined edge/corner tests, repeated runtime profiling, Valgrind Memcheck, and supplementary ASan/UBSan runs.
- **Memory safety:** Valgrind provides the deterministic analyzer score; ASan/UBSan are supplementary evidence; the shared LLM RSI evaluator provides the LLM score for the same metric.
- **LLM RSI evaluation:** the LLM scores every enabled rubric metric using the C source, rubric parameters, and analyzer evidence. Missing test/tool evidence does not cancel the evaluation; the LLM makes a clearly labelled source-based estimate instead.
- **Final metric score:** the LLM/RSI assigns each metric score from the rubric, C source, and analyzer evidence. If the LLM score is unavailable, Python uses that metric's deterministic analyzer score. If both are unavailable, zero is used and the report warns about it.
- **Overall grade:** Python always calculates `sum(metric_weight * final_metric_score)` across all enabled metrics. Instructor weights are not renormalized.
- **FormAI:** independent benchmark sampling and evaluation utilities compare predictions against held-out FormAI labels. The dataset itself is external and is not bundled or passed to the prediction-time LLM.

## Repository layout and file responsibilities

- `rules.json`: instructor-owned assignment contract, metric weights, thresholds, and test-set paths.
- `src/grading_agent/main.py`: orchestration only; loads rules, runs analyzers, performs the LLM stages, computes the grade, and writes reports.
- `src/grading_agent/config/rule_loader.py`: validates rule schemas, weights, paths, and test-case metadata before execution.
- `src/grading_agent/sandbox/docker_sandbox.py` and `Dockerfile.sandbox`: isolated execution boundary and the pinned tool environment.
- `src/grading_agent/analyzers/`: one module per analysis function (compiler, static analysis, correctness, edge cases, runtime, memory, complexity, modularity, readability, and similarity).
- `src/grading_agent/scoring/`: selects each LLM RSI score (or analyzer fallback) and computes the weighted total in Python.
- `src/grading_agent/llm/evidence.py`: creates the evidence ledger and rejects unknown IDs, rule-conflicting recommendations, universal safety claims, and selected claims contradicted by exact test records.
- `src/grading_agent/llm/reviewer.py`: runs one bounded evidence-based RSI request per enabled metric and the separate FormAI source audit.
- `src/grading_agent/reporting/`: JSON and human-readable grading reports.
- `src/grading_agent/benchmark/`: FormAI source sampling, leakage-safe splitting, prediction, and evaluation utilities.
- `tests/`: unit/regression tests; `tests/sum/` holds the example assignment's instructor test cases.
- `examples/`: the sample student program used by the bundled rules.
- `resources/flowchart.jpg`: pipeline architecture figure.
- `submissions/`: optional, instructor-authorized similarity corpus; do not place unapproved student work here.

Generated reports, Docker workspaces, Python caches, virtual environments, and downloaded FormAI data are excluded from source control and the clean source bundle.

## Prerequisites

- Python 3.11+
- Docker Engine running
- `llama-server` from `llama.cpp` available on `PATH`, or set `LLAMA_SERVER` to its full path
- A compatible GGUF model supported by your `llama.cpp` build

The application starts `llama-server` with an 8,192-token context by default. Set `LLAMA_CONTEXT_SIZE` before launch to override it, for example `LLAMA_CONTEXT_SIZE=4096` on a memory-constrained machine. RSI prompts are sent one metric at a time and are capped to keep them comfortably below the configured context size.

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
| memory_safety | `memory_safety` | Valgrind score plus the shared LLM RSI score; ASan/UBSan supplementary |
| readability | `readability` | Deterministic style/readability heuristics |
| originality | `similarity_flag` | Token-shingle similarity against prior submissions |

Enabled top-level metric weights must sum to `1.0`. The example `originality` metric is deliberately disabled until the `submissions/` corpus contains authorized comparison files. To enable it, set `enabled` to `true` and adjust the other enabled weights so their sum remains exactly `1.0`. Similarity is only a flag for human review; it is not proof of plagiarism or intent.

The memory-safety analyzer uses the measured Valgrind pass rate as its deterministic score. ASan/UBSan results are supplementary evidence. The same LLM RSI pass used for all other metrics evaluates the memory-safety metric from the source and collected tool evidence.

There is no extra scoring ratio to configure: analyzer outputs are evidence for the LLM/RSI metric evaluation. The LLM's per-metric score is used; a deterministic analyzer score is the fallback if the LLM cannot score that metric.

## Run

From the repository root:

```bash
python -m grading_agent.main
```

The coordinator validates `rules.json`, runs the configured static and dynamic analyzers, starts or reuses the local Llama server, evaluates every enabled rubric metric in a separate bounded RSI request, applies the instructor weights in Python, and writes:

```text
reports/<problem>.json
reports/<problem>.txt
```

The JSON contains analyzer details, the evidence ledger, the LLM RSI score/rationale per metric, the selected final score for each metric, and the final weighted score. The text report includes the metric score breakdown and failing test inputs/expected/actual output.

### Evidence-based RSI and scoring

The LLM receives one metric at a time: that metric's configured parameters, its analyzer result, a small set of related evidence records, and selected source lines. This prevents the combined rubric/source/evidence prompt from exceeding a small local model context window. The LLM returns a score for every enabled metric. When dynamic evidence is absent, it must still evaluate the supplied C source and identify the score as a source-only or limited-evidence estimate; it must not discard the entire evaluation. Python accepts each valid metric score independently, so one malformed/missing metric entry does not discard the others.

For each metric, Python applies this simple rule:

```text
if llm_rsi_score is available:
    final_metric_score = llm_rsi_score
elif analyzer_score is available:
    final_metric_score = analyzer_score  # fallback
else:
    final_metric_score = 0

final_grade = sum(metric_weight * final_metric_score)
```

All enabled metric weights must sum to `1.0`; weights are not renormalized. If both analysis sources fail for a metric, that metric contributes zero and the report identifies the missing evidence. The weighted arithmetic is deterministic Python code; model-generated RSI scores use the configured local model with temperature `0` and seed `42`. The LLM/RSI score is the metric score, and the analyzer score is only a fallback.

## Similarity corpus

Place authorized prior student submissions/reference C files under `submissions/` (subdirectories are supported). The current example `originality` metric remains disabled until the corpus is ready. Increase `max_files` if the course corpus is larger than the default scan cap. Similarity scores should be reviewed by an instructor; common boilerplate and assignment scaffolding can produce legitimate overlap.

## Gradio frontend

Install the optional web UI dependency and start the local interface:

```bash
python -m pip install -e ".[ui]"
python -m grading_agent.ui.app
```

Open `http://127.0.0.1:7860`. The **Grade a C submission** tab accepts a C file and the instructor's rules JSON, then runs the same grading backend and returns JSON/text reports. The **FormAI-v2 combined benchmark** tab accepts the source-only held-out `test_inputs.jsonl` and the separate `test_labels.json`; choose `hybrid` or `hybrid_rsi` to run both cppcheck and the LLM, then evaluate after prediction generation. Test inputs and labels remain separate; labels are used only by the evaluator.

## Failure isolation and combined predictions

Docker, cppcheck, LLM, and RSI are independent pipeline stages. A missing Docker SDK/daemon or a failed cppcheck invocation is recorded in that stage's result and does not stop LLM source analysis. In `hybrid` and `hybrid_rsi`, every sample is evaluated by the LLM independently of deterministic-tool availability. Completed primary methods are fused with an OR rule; an unavailable method cannot veto a positive result from the other method. `hybrid_rsi` runs evidence review whenever either primary stage produces a usable result, including cppcheck findings emitted before a timeout; it uses source-line, static-finding, and model-audit evidence. RSI may add validated evidence-backed findings, but it never erases a primary finding. If a sample-level stage fails, the output JSONL still receives a record with the error and the other stages continue.

LLM source findings are validated independently: an unsupported or malformed finding is omitted without discarding the audit summary or other findings that have exact source-line excerpts. Files longer than the per-request source limit are audited in overlapping chunks, and chunk-relative findings are mapped back to original file line numbers; a failed chunk is reported as a warning without discarding valid results from other chunks. For the ordinary assignment workflow, LLM RSI scores contribute to each metric as configured above; the final weighted arithmetic is always performed by Python.

## FormAI-v2 independent evaluation

FormAI-v2 contains a large collection of AI-generated C programs and formal-verification-based vulnerability labels. It is used as an **independent benchmark**, not as an instructor's ordinary assignment test set and not as an input to the prediction-time LLM.

The repository is not included because the dataset is multi-gigabyte. Follow the dataset authors' instructions at <https://github.com/FormAI-Dataset/FormAI-dataset> to download and extract `FormAI-v2.json` outside this repository. Then install the streaming JSON dependency:

```bash
python -m pip install -e '.[benchmark]'
```

Create a deterministic stratified sample with labels separated from source inputs:

```bash
python -m grading_agent.benchmark.formai_sample \
  --dataset data/FormAI-dataset/FormAI-v2.json \
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

Use the training split for development, the validation split for threshold/prompt tuning, and the test split only for final reporting. The prediction runner implements four modes: **tools-only** (cppcheck source diagnostics), **LLM-only** (local LLM source audit), **hybrid** (positive if either source-level method flags a vulnerability), and **hybrid+RSI** (the same two signals followed by bounded evidence review; evidence review can add a validated positive result but cannot veto either primary signal). A tools-only or LLM-only mode is inconclusive when its sole stage fails. Combined modes preserve any completed primary-stage prediction and mark partial infrastructure failures as `completed_with_warnings`; only a record with no completed primary stage is inconclusive. These are source-level benchmark modes: FormAI does not provide a standardized per-program input oracle, so this benchmark runner does not execute arbitrary dataset programs.

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

python -m grading_agent.benchmark.formai_predict \
  --inputs benchmark/splits/test_inputs.jsonl \
  --mode hybrid_rsi \
  --rsi-iterations 3 \
  --output benchmark/predictions/hybrid_rsi.jsonl
```

Keep `formai_ground_truth.json` separate from all prediction runs. A prediction JSONL record has this shape:

```json
{"file_name":"sample.c","predicted_vulnerable":true,"predicted_line":42,"predicted_error_type":"array bounds violated"}
```

Compare the same held-out records for all four prediction modes:

```bash
python -m grading_agent.benchmark.formai_evaluate \
  --ground-truth benchmark/splits/test_labels.json \
  --predictions tools_only=benchmark/predictions/tools_only.jsonl \
  --predictions llm_only=benchmark/predictions/llm_only.jsonl \
  --predictions hybrid=benchmark/predictions/hybrid.jsonl \
  --predictions hybrid_rsi=benchmark/predictions/hybrid_rsi.jsonl \
  --line-tolerance 2 \
  --output reports/formai-evaluation.json
```

The evaluator reports Precision, Recall, F1, Accuracy, Specificity, False Positive Rate, False Negative Rate, balanced accuracy, line-localization accuracy, and error-type accuracy. Inconclusive predictions are excluded from confusion-matrix calculations and contribute to the skipped count; missing prediction records are treated as negative predictions and reported explicitly. Use the validation labels for threshold/prompt tuning only. The held-out test labels are not read by the prediction runner and are used only by the evaluator after all prediction files are complete. The hybrid OR rule is a simple baseline, not a claim that it is optimal. A combined prediction is emitted whenever at least one primary stage completes; unavailable stages are identified in the prediction record. The hybrid+RSI mode is an experimental evidence-review layer that can add validated evidence-backed findings but cannot downgrade or discard primary outputs. Compare all four modes empirically on the same held-out split, report confidence intervals where practical, and select thresholds/prompts using validation data only.

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
