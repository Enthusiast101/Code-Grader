# Code Checking Assistant

A Python-based C-program grading prototype using Docker isolation, GCC, Valgrind, configurable JSON rules, and a locally hosted `llama.cpp` model for evidence-based feedback.

## Prerequisites

- Python 3.11+
- Docker Engine running
- `llama-server` available on `PATH`, or set `LLAMA_SERVER` to its executable path
- A compatible GGUF model or Hugging Face model supported by the installed `llama.cpp` build

## Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Build the sandbox image

From the repository root:

```bash
docker build -f Dockerfile.sandbox -t agent-sandbox:latest .
```

The image includes GCC, Bash/coreutils, and Valgrind. Student programs run in a container with no network, a read-only root filesystem, restricted capabilities, and configured resource limits.

## Configure grading

Edit `rules.json` in the repository root. It selects the C source file, time and memory limits, enabled metrics, analyzer identifiers, weights, and analyzer parameters.

Currently supported analyzer identifiers:

- `io_tests`: compile and execute the program against JSON input/expected-output tests.
- `memory_safety`: one instructor metric that runs both Valgrind Memcheck on configured test inputs and an LLM source-code memory-safety audit. Raw Valgrind logs and parsed summaries are preserved. Its default component weights are 70% Valgrind and 30% LLM audit; supplying one component weight makes the other its complement.
- `static_analysis`: GCC syntax/compiler diagnostics with a configurable warning penalty.

Enabled metric weights must sum to `1.0`. Test-set paths and source paths must remain inside the repository root. The composite `memory_safety` metric defaults to Valgrind weight `0.70` and LLM source-audit weight `0.30`; these can be overridden in that metric's `params`. If only one subweight is supplied, the other is set to its complement so both checks continue to run and the subweights sum to `1.0`. Severity penalties also have defaults in `config/settings.py` (critical: 40, high: 25, medium: 10, low: 3 points deducted per unique finding, capped at 100). If either sub-audit is missing or inconclusive, the composite memory-safety score is withheld rather than treating missing evidence as a clean result.

## Run

From the repository root:

```bash
python -m grading_agent.main
```

The coordinator loads and validates `rules.json`, dispatches enabled analyzers, saves analyzer evidence, launches or reuses the local Llama server for feedback review, and writes a JSON report under `reports/`.

Set `LLAMA_SERVER` if the executable is not on `PATH`. Model, host, port, sampling, and Docker defaults are in `src/grading_agent/config/settings.py`.

## Current implementation status

Implemented and integrated:

- Docker sandbox with network disabled, read-only root filesystem, CPU/memory/PID restrictions, bounded command execution, and workspace file helpers.
- External `rules.json` loading and validation, including supported analyzer names, path checks, test-case schema checks, and enabled-weight validation.
- I/O correctness tests.
- Composite memory-safety workflow: Valgrind Memcheck plus a dedicated LLM source audit, with configurable/default component weights and deterministic Python aggregation.
- Valgrind raw logs and parsed summaries preserved as evidence.
- GCC static compiler/syntax diagnostics.
- Analyzer registry in `analyzers/registry.py`, currently exposing I/O tests, composite memory safety, and static compiler analysis.
- Local Llama server lifecycle and evidence-review request.

Still to implement:

- Overall deterministic weighted scoring across all instructor-defined metrics and score calibration (the memory-safety subscore is implemented separately).
- Structure, modularity, complexity, readability, and plagiarism/similarity analyzers.
- FormAI benchmark ingestion, stratified evaluation, and comparison of tools-only, LLM-only, and hybrid configurations.
- Final human-readable report generation and expanded failure/edge-case tests.

The LLM identifies source-level risks and provides explanations; Python applies the configured severity-penalty rubric and combines the LLM-derived component with Valgrind using the instructor-defined subweights. The model does not directly choose the numerical score. Missing or inconclusive analyzer evidence withholds the composite memory-safety score rather than being treated as clean. The overall weighted grade across all top-level metrics is still pending.
