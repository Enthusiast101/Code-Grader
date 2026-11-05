# C Code Grader

Rule-driven grading for C assignments using deterministic analyzers, a local LLM, and weighted metric scoring. The project-level `rules.json` is assignment-neutral; replace its task placeholder and configure the source file and any assignment-specific test sets before grading. In the Gradio workflow, test generation and grading are separate actions: generate and review the test suite first, then explicitly start grading against that saved suite.

## Grading pipeline

1. Deterministic analyzers collect compiler, static-analysis, test, runtime, readability, complexity, and memory-safety evidence where configured.
2. The configured LLM reads the submitted source and assignment, then describes actual source behavior separately from intended behavior.
3. RSI pass 1 identifies whole-program defects. The evidence ledger it receives lists observed defects first (failed tests, analyzer, sanitizer and Valgrind findings, non-clean metric summaries), then assignment context, then clean summaries, so a fixed evidence budget never drops a defect in favour of an "all passed" summary.
4. Each later RSI pass receives the previous pass's complete response plus the list of defect evidence records that no finding cites yet, and must explain each one or justify why it is not a defect of the submission. RSI passes are findings-only.
5. RSI stops early when a pass reproduces the previous findings exactly (a fixed point at temperature 0); set `LLM_RSI_STOP_ON_CONVERGENCE=false` to always run every configured pass.
6. Surviving findings are deduplicated (one finding per metric and exact line range, with evidence merged) and adjudicated. The adjudicator writes its reasoning first and then five explicit checks; a finding is kept only when every check is true.
7. One scoring request rates, on an integer 0-10 scale, only the metrics that no completed analyzer or executed test measures. Python converts levels to points. Executed tests and completed analyzers (`io_tests`, `edge_case_runner`, `compiler_diagnostics`, `memory_safety`, `runtime_profile`) are authoritative and are not sent for rating; a runtime metric without a completed profile stays unmeasured. The final grade is out of 100.

Each structured grading stage allows up to three bounded attempts: the initial request plus up to two corrective re-prompts when JSON or stage-specific validation fails. A rating response that omits a metric is re-prompted with the missing metric names. If the local model/server fails or returns an unusable response, the report remains renderable and uses a clearly identified evidence-based fallback; a fallback is never presented as a successful LLM evaluation.

### Reproducibility

Grading requests use temperature 0, a fixed seed, and `cache_prompt: false`. llama-server's prompt cache changes batch boundaries between requests, which changes logits slightly and can make identical requests decode differently even at temperature 0. Keep `LLAMA_CACHE_PROMPT=false` (the default) when repeatable grades matter; enabling it trades repeatability for speed.

## Default model: Qwen3.5-2B

The project defaults to **`gguf/Qwen3.5-2B-GGUF:Q4_K_M`** for both grading and same-model context compression. This is the llama.cpp Hugging Face model identifier documented by the GGUF repository; the Q4_K_M model file is approximately 1.31 GB. The Gradio model textbox inherits this default from `settings.MODEL`, and the server is launched with the same identifier.

To use it, copy `.env.example` to `.env` (or update the existing `.env`) and set:

```dotenv
LLAMA_MODEL=gguf/Qwen3.5-2B-GGUF:Q4_K_M
```

The application downloads the selected GGUF through llama.cpp on first startup; no separate compression model is configured. If an existing `.env` still contains the previous bundled default `Qwen/Qwen3-4B-GGUF:Q4_K_M`, the settings loader automatically migrates that exact legacy value to Qwen3.5-2B. Any other custom `LLAMA_MODEL` value is preserved. This grader uses text-only C source, so it does not need the repository's optional multimodal projector.

## JSON output, Qwen reasoning, and llama.cpp version requirement

Structured grading and same-model compression use explicit JSON Schema through llama.cpp `/v1/chat/completions`, sent as `response_format: {"type": "json_schema", "json_schema": {"name", "schema"}}`. llama-server only reads the nested `json_schema.schema` member for this type; a top-level `schema` member is silently ignored and leaves decoding unconstrained. Schema-constrained requests disable template thinking so the schema applies from the first generated token. The client validates parsed JSON and stage-specific evidence; malformed responses receive bounded corrective retries. Compatibility still depends on the installed llama.cpp build and its Qwen chat template, so verify the server build if structured output fails.

**Required llama.cpp build:** Qwen3/Qwen3.5 chat templates can prefill an empty `<think></think>` block. Older llama.cpp builds attempt to feed that prefix into the JSON grammar and fail with `Failed to initialize samplers: std::exception` before generation. Upstream fixed this in [llama.cpp PR #31380](https://github.com/ggml-org/llama.cpp/pull/31380); the first verified release reported upstream is build `b12005` (commit `3c809f44`). The application checks `llama-server --version` before starting and refuses to run on an older or unverifiable build. Do not bypass this check unless your custom build includes that fix (`LLAMA_ALLOW_UNVERIFIED_SERVER=true`).

Update/build instructions are in `scripts/update_llama_cpp.sh`. On Linux, run it with `LLAMA_CPP_DIR` set to your existing llama.cpp checkout; set `GGML_CUDA=ON` when you need a CUDA build. Stage generation limits are controlled by `LLM_PROGRAM_UNDERSTANDING_MAX_TOKENS`, `LLM_GLOBAL_RSI_MAX_TOKENS`, `LLM_GLOBAL_SCORE_MAX_TOKENS`, and `LLM_CONTEXT_COMPRESSION_MAX_TOKENS`. Restart Gradio after updating llama.cpp.

## Same-model context compression

The configured `LLAMA_MODEL` is used for both grading and compression through the same local `llama-server`; no separate compression model is selected. Every grading stage measures its prompt with llama-server's `/tokenize` endpoint (falling back to a conservative character estimate) and reserves a quarter of the context for the answer. Compression runs only when the prompt would not otherwise fit; when it fits, the prompt, including the previous RSI response, is sent verbatim. The compressor may shorten narrative prose, but it cannot change submitted source code, line numbers, metric identities, finding identities, severities, analyzer values, or evidence IDs. The original anchors are reattached and validated before compressed text is accepted.

If model compression does not reduce the payload, changes protected information, returns an invalid object, or the local endpoint is unavailable, deterministic anchor-preserving compaction is used. The report differentiates an actual same-model compression call, deterministic fallback, and a run where compression was not needed. Compression makes a same-model request for each oversized payload and retries once with a repair instruction if the returned JSON is invalid; if compression still fails validation, deterministic anchor-preserving compaction is used.

## Metric scoring and reports

The LLM returns an integer 0–10 level for each metric that no completed analyzer or executed test measures, using the anchored levels in `rubrics/default_metric_rubric.json`; Python converts levels to points. Executed correctness and logical test pass rates and completed analyzers remain authoritative and are not sent for rating. Metric scores are shown against the configured `max_points` caps; the root rules allocate 20 points to correctness and 10 points to each of syntax, structure, modularity, logical, memory safety, complexity, readability, and similarity. A completed deterministic analyzer score is used when a valid LLM rating is unavailable. Inconclusive evidence remains unscored, and the overall grade is withheld when a required metric cannot be measured. Reports show both the earned metric points and LLM rating where available; the overall grade is out of 100.

Every metric explanation in the Gradio and text reports preserves up to three concise, metric-specific sentences (up to 60 words/420 characters) so the rationale can explain evidence, impact, and rubric alignment. Generic RSI filler and separate “Score basis” boilerplate are omitted. Successful compiler diagnostics such as `compiled=True; warnings=0; errors=0` are not emitted as syntax defects.

### Similarity metric

`similarity` (analyzer `similarity_flag`, 10 points) replaces the former runtime-efficiency metric. It compares the submission with every other `.c` file in `params.corpus_dir` using normalized 8-token shingles: comments, literals and identifier names are ignored, so renaming variables or reformatting does not hide copying. The highest similarity `s` to any other file is scored as:

- no `corpus_dir`, a missing directory, or no other C files: full marks;
- `s <= low_threshold` (default 0.35): full marks;
- `s >= flag_threshold` (default 0.85): zero, and the pair is flagged for instructor review;
- in between: linear partial credit.

The report also includes the full pairwise similarity matrix of every file in the directory, so copied pairs that do not involve the graded student are visible too. A high score is a review signal, not proof of plagiarism. The submission's own file (same path, or same name and contents) is never compared with itself.

## Generic assignment rules and test data

The root `rules.json` is not tied to addition or any other specific task. Before grading:

- Replace `problem` with the complete instructor assignment statement.
- Set `submission.source_file` to the submitted `.c` file.
- Fill `problem_contract` only with requirements that genuinely apply to the task.
- Optionally add instructor test files and set the relevant `metrics[].params.testset_path` fields.
- In Gradio, choose the maximum number of **additional** LLM-generated cases with the slider and click **Generate Test Cases**. The LLM first analyzes the exact submitted C source only to identify useful test dimensions; it does not use student behavior to decide correctness. A generation call proposes exact stdin/expected-stdout pairs from the assignment and instructor contract. A second, independent LLM call solves each proposed input from the assignment specification and checks its expected output. Invalid, ambiguous, duplicate, malformed, or mismatched candidates are rejected; one bounded refill is attempted only when the first batch has no independently verified cases; every replacement goes through the same independent verification. Empty stdin is allowed for assignments that do not read input; explicit contract constraints and the independent verifier reject it when input is required. No C reference solution is generated, compiled, or executed. Instructor test cases are preserved and supplied as format/behavior hints. Review the preview, then click **Grade Submission**. Grading consumes the saved generated suite and does not invoke test generation again; if the source, rules, count, or focus changes after generation, generate again before grading.

- When using the CLI instead of Gradio, execution-based metrics with missing testcase paths may still trigger automatic suite generation during the grading run. Test generation performs one optional source-understanding call, one input/output-pair generation call, and one independent expected-output verification call. If the first batch produces no independently verified cases, one bounded replacement-generation and verification cycle is allowed. It does not generate, compile, or execute a C reference program; correctness is checked by the independent verifier against the assignment and instructor examples. Only accepted input/output pairs are fingerprinted and frozen. The code grader's own RSI/review loop remains separate and unchanged.

- Ensure enabled metric weights sum to `1.0`.

The test-based analyzers (`io_tests`, `edge_case_runner`, `runtime_profile`, and runtime memory checks) use instructor test paths when supplied and generated testsets otherwise. If automatic generation cannot produce a validated suite, affected execution metrics are inconclusive/N/A rather than an artificial zero. Source-based analyzers and the LLM evaluation can still run. GCC `-fanalyzer` can inspect memory behavior without dynamic test inputs; Valgrind and ASan/UBSan require executable test cases.

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

## Installation, updates, and launch

Use Python 3.11 or newer. Run these commands from the repository root. Create and activate a virtual environment once per checkout (activate it again in each new terminal):

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Build the sandbox image (the tag matches `DOCKER_SANDBOX_IMAGE`):

```bash
docker build --no-cache -f Dockerfile.sandbox -t agent-sandbox:latest .
```

### Updating an existing checkout

After pulling new source changes, refresh the editable install and rebuild the sandbox image so Python dependencies and sandbox tools stay in sync:

```bash
git pull
pip install -e .
docker build --no-cache -f Dockerfile.sandbox -t agent-sandbox:latest .
```

The editable install points at the current checkout, so source changes take effect on the next application start. Re-run `pip install -e .` when project dependencies or packaging metadata change. Rebuild the Docker image when the sandbox Dockerfile or installed sandbox tools change.

### Launch Gradio

```bash
LLAMA_SERVER="$HOME/.local/bin/llama-server" PYTHONPATH=src python -m grading_agent.ui.app
```

Keep Docker running when using Docker-backed analyzers. The Gradio app can start and supervise `llama-server` when the executable is available on `PATH`; otherwise start a compatible llama.cpp server separately using the same model and configured host/port.

Run CLI grading:

```bash
python -m grading_agent.main
```

The default UI is `http://127.0.0.1:7860`. Set `GRADIO_HOST` and `GRADIO_PORT` to change the bind address.

## Gradio controls

The **Grade a C submission** tab has two explicit steps. **Step 1 — Generate Test Cases:** choose the requested number of additional cases and optional coverage focus, then click **Generate Test Cases**. The LLM analyzes the submitted C code only to choose useful test dimensions, then proposes stdin/expected-output pairs. A second LLM call independently solves each input from the assignment specification and checks the proposed output. One bounded refill is attempted only if the first batch yields no independently verified cases. Empty stdin is accepted only when the assignment requires no input; no C reference program is generated, compiled, or executed. The UI displays the combined instructor and generated cases for review. **Step 2 — Grade Submission:** click this button only after generation completes. The grading run uses the saved suite and does not generate a second suite. Changes to the source, rules, requested count, or focus invalidate the saved generation state and require regeneration.

Execution timeout, Docker memory/CPU, ASan/UBSan, RSI pass count, token budgets, maximum findings, source/evidence limits, model identifier, temperature, request timeout, thinking mode, and same-model compression threshold remain available as per-run settings. The model selection and the grading/compression model are shared. Schema-constrained grading and compression inherit the configured reasoning/thinking setting; JSON validity and evidence are checked after generation with bounded repair attempts.

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
