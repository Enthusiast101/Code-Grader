# C Code Grader

Rule-driven grading for C assignments using deterministic analyzers, a local LLM, and weighted metric scoring. The project-level `rules.json` is assignment-neutral; replace its task placeholder and configure the source file and any assignment-specific test sets before grading.

## Grading pipeline

1. Deterministic analyzers collect compiler, static-analysis, test, runtime, readability, complexity, and memory-safety evidence where configured.
2. The configured LLM reads the submitted source and assignment, then describes actual source behavior separately from intended behavior.
3. RSI pass 1 identifies whole-program issues supported by the source or evidence ledger.
4. RSI pass 2 receives pass 1's complete response and rechecks it for omissions, false positives, unsupported claims, and incorrect locations.
5. RSI pass 3 rechecks pass 2 in the same way. RSI passes are findings-only; they do not assign scores.
6. One separate LLM scoring request evaluates metrics that are not defined by executed tests, using the task, rubric, numbered source, validated findings, and analyzer evidence.
7. Executed correctness and logical test results determine those metrics by pass rate. Python validates other LLM scores and applies the configured weights; completed analyzers are fallbacks when an LLM score is unavailable. Score sources are shown in the report.

Each structured grading stage allows up to three bounded attempts: the initial request plus up to two corrective re-prompts when JSON or stage-specific evidence validation fails. The normal successful path remains five grading stages: one program-understanding stage, three sequential RSI passes, and one separate metric-scoring stage; only failed stages incur additional repair calls. Same-model compression requests are added only when an oversized semantic section or RSI history actually needs compression, and each compression payload gets one corrective retry. If the local model/server fails or returns an unusable response, the report remains renderable and uses a clearly identified evidence-based fallback; a fallback is never presented as a successful LLM evaluation.

## Default model: Qwen3.5-2B

The project defaults to **`gguf/Qwen3.5-2B-GGUF:Q4_K_M`** for both grading and same-model context compression. This is the llama.cpp Hugging Face model identifier documented by the GGUF repository; the Q4_K_M model file is approximately 1.31 GB. The Gradio model textbox inherits this default from `settings.MODEL`, and the server is launched with the same identifier.

To use it, copy `.env.example` to `.env` (or update the existing `.env`) and set:

```dotenv
LLAMA_MODEL=gguf/Qwen3.5-2B-GGUF:Q4_K_M
```

The application downloads the selected GGUF through llama.cpp on first startup; no separate compression model is configured. If an existing `.env` still contains the previous bundled default `Qwen/Qwen3-4B-GGUF:Q4_K_M`, the settings loader automatically migrates that exact legacy value to Qwen3.5-2B. Any other custom `LLAMA_MODEL` value is preserved. This grader uses text-only C source, so it does not need the repository's optional multimodal projector.

## JSON output, Qwen reasoning, and llama.cpp version requirement

Structured grading and same-model compression use explicit JSON Schema through llama.cpp `/v1/chat/completions`. Schema-constrained requests retain the configured `enable_thinking` setting while enforcing JSON Schema on the final response. The client validates parsed JSON and stage-specific evidence; malformed responses receive bounded corrective retries. Compatibility still depends on the installed llama.cpp build and its Qwen chat template, so verify the server build if structured output fails.

**Required llama.cpp build:** Qwen3/Qwen3.5 chat templates can prefill an empty `<think></think>` block. Older llama.cpp builds attempt to feed that prefix into the JSON grammar and fail with `Failed to initialize samplers: std::exception` before generation. Upstream fixed this in [llama.cpp PR #31380](https://github.com/ggml-org/llama.cpp/pull/31380); the first verified release reported upstream is build `b12005` (commit `3c809f44`). The application checks `llama-server --version` before starting and refuses to run on an older or unverifiable build. Do not bypass this check unless your custom build includes that fix (`LLAMA_ALLOW_UNVERIFIED_SERVER=true`).

Update/build instructions are in `scripts/update_llama_cpp.sh`. On Linux, run it with `LLAMA_CPP_DIR` set to your existing llama.cpp checkout; set `GGML_CUDA=ON` when you need a CUDA build. Stage generation limits are controlled by `LLM_PROGRAM_UNDERSTANDING_MAX_TOKENS`, `LLM_GLOBAL_RSI_MAX_TOKENS`, `LLM_GLOBAL_SCORE_MAX_TOKENS`, and `LLM_CONTEXT_COMPRESSION_MAX_TOKENS`. Restart Gradio after updating llama.cpp.

## Same-model context compression

The configured `LLAMA_MODEL` is used for both grading and compression through the same local `llama-server`; no separate compression model is selected. Compression runs only when a previous RSI response or semantic section exceeds its configured threshold. The compressor may shorten narrative prose, but it cannot change submitted source code, line numbers, metric identities, finding identities, severities, analyzer values, or evidence IDs. The original anchors are reattached and validated before compressed text is accepted.

If model compression does not reduce the payload, changes protected information, returns an invalid object, or the local endpoint is unavailable, deterministic anchor-preserving compaction is used. The report differentiates an actual same-model compression call, deterministic fallback, and a run where compression was not needed. Compression makes a same-model request for each oversized payload and retries once with a repair instruction if the returned JSON is invalid; if compression still fails validation, deterministic anchor-preserving compaction is used.

## Metric scoring and reports

Executed correctness and logical tests set their metrics to the observed pass rate; the LLM is not asked to rescore those outcomes. For other metrics, Python validates each LLM score is finite and within 0-100, then applies the configured weights. A completed deterministic analyzer score is used when a valid LLM score is unavailable. Inconclusive evidence remains unscored, and the overall grade is withheld when a required metric cannot be measured. Reports identify the source used for each score.

Every metric explanation in the Gradio and text reports preserves up to three concise, metric-specific sentences (up to 60 words/420 characters) so the rationale can explain evidence, impact, and rubric alignment. Generic RSI filler and separate “Score basis” boilerplate are omitted. Successful compiler diagnostics such as `compiled=True; warnings=0; errors=0` are not emitted as syntax defects.

## Generic assignment rules and test data

The root `rules.json` is not tied to addition or any other specific task. Before grading:

- Replace `problem` with the complete instructor assignment statement.
- Set `submission.source_file` to the submitted `.c` file.
- Fill `problem_contract` only with requirements that genuinely apply to the task.
- Optionally add instructor test files and set the relevant `metrics[].params.testset_path` fields.
- Assignment test generation can infer the I/O contract from the natural-language problem, produce normal/boundary/adversarial inputs and a C reference oracle, then recheck the complete specification, oracle, and cases in sequential RSI passes. The reference is compiled and executed in the restricted Docker sandbox; generated suites are fingerprinted and frozen. If the assignment is ambiguous or the oracle fails validation, generation is marked unavailable rather than silently trusting expected outputs.
- Ensure enabled metric weights sum to `1.0`.

The test-based analyzers (`io_tests`, `edge_case_runner`, `runtime_profile`, and runtime memory checks) use optional test paths. With a path set to `null` or omitted, the corresponding execution analyzer reports `inconclusive`/N/A rather than an artificial zero. Source-based analyzers and the LLM evaluation continue. GCC `-fanalyzer` can still inspect memory behavior without dynamic test inputs; Valgrind and ASan/UBSan require executable test cases.

Example test-case file:

```json
[
  {
    "id": "basic-1",
    "input": "<stdin for this assignment>\\n",
    "expected": "<expected stdout>\\n",
    "description": "Basic valid input"
  }
]
```

## Setup and launch

Use Python 3.11 or newer:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .
```

Build the sandbox image with the project’s existing Dockerfile (the image tag
matches `DOCKER_SANDBOX_IMAGE`):

```bash
docker build --no-cache -f Dockerfile.sandbox -t agent-sandbox:latest .
```

Launch the existing Gradio application:

```bash
python -m grading_agent.ui.app
```

Install/build the configured Docker sandbox image if Docker-backed analyzers are enabled. The Gradio app can start and supervise `llama-server` when the executable is available on `PATH`; otherwise start a compatible llama.cpp server separately using the same model and configured host/port.

Run CLI grading:

```bash
PYTHONPATH=src python -m grading_agent.main
```

Run Gradio:

```bash
PYTHONPATH=src python -m grading_agent.ui.app
```

The default UI is `http://127.0.0.1:7860`. Set `GRADIO_HOST` and `GRADIO_PORT` to change the bind address.

## Gradio controls

The **Grade a C submission** tab exposes per-run controls for execution timeout, Docker memory/CPU, ASan/UBSan, RSI pass count, token budgets, maximum findings, source/evidence limits, model identifier, temperature, request timeout, thinking mode, and same-model compression threshold. The model selection and the grading/compression model are shared. Schema-constrained grading and compression inherit the configured reasoning/thinking setting; JSON validity and evidence are checked after generation with bounded repair attempts.

The editable JSON panel controls the task statement, problem contract, source path, metric enablement/weights, test-case paths, and analyzer-specific parameters. Uploading a new rules file takes precedence over stale editor state; edits are preserved when they still belong to the currently uploaded file. Each report records the assignment and rules fingerprint used for that run.

The FormAI-v2 benchmark tab has its own mode, timeout, finding, RSI, Docker, and sample-count controls. Its evaluation path remains separate from ordinary assignment grading.

## Focused checks

The targeted regression tests for structured JSON generation, same-model compression, score authority, compiler-finding filtering, report length, and Gradio rendering are:

```bash
PYTHONPATH=src pytest -q tests/test_pristine_llm_path.py
PYTHONPATH=src python -m compileall -q src
```

The targeted tests mock the local LLM endpoint. A live run against the selected local model/server is still required to verify actual tokenization, grammar support, latency, and model quality on the target machine.


### llama.cpp structured output

The grader and the same-model context compressor use llama.cpp's native
`/v1/chat/completions` schema format: `response_format: {"type": "json_schema",
"schema": ...}`. This intentionally uses the direct `schema` member documented by
llama.cpp rather than the OpenAI SDK's nested `json_schema` wrapper. Structured
requests disable Qwen thinking for that request so grammar-constrained decoding
starts at the JSON content; unstructured model calls retain the configured
reasoning mode. Compression uses the configured grading model and validates
that all finding locations and evidence IDs are preserved before accepting a
compressed result.
