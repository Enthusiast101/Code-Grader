# Program understanding, sequential RSI, and LLM-led scoring

The C-assignment grading path follows this sequence:

1. Deterministic analyzers collect objective results in a structured evidence ledger.
2. The configured LLM understands the submitted source, separating actual source behavior from assignment intent.
3. RSI pass 1 identifies whole-program defects grounded in the source and evidence ledger.
4. RSI pass 2 receives pass 1's complete response and rechecks it for missed defects, false positives, unsupported claims, and incorrect source locations.
5. RSI pass 3 repeats the recheck using pass 2's complete response. RSI passes are findings-only.
6. One separate LLM scoring request evaluates every enabled metric from the assignment, rubric, actual source, validated findings, and analyzer evidence. It returns a score and a concise rationale for each metric.
7. Python validates score ranges and applies rubric weights. The LLM's valid metric scores are authoritative; a completed deterministic analyzer is only a fallback when the LLM score is unavailable.

Each grading stage makes one request. There are no repair/retry generations. Structured stages send an explicit llama.cpp GBNF grammar, activated after `</think>` when reasoning is enabled. This constrains the final response to syntactically valid JSON while allowing bounded model thinking. If the local server is unavailable or rejects a request, the pipeline retains a clearly labelled evidence-based fallback rather than issuing another generation.

## Reasoning and compression

- Model thinking is enabled by default for program understanding, RSI, scoring, and context compression, with a bounded reasoning budget.
- Context compression uses the same `LLAMA_MODEL` and local `llama-server` as grading; no second model is configured.
- Same-model compression is invoked only when a review history or semantic section exceeds its configured threshold. It may shorten prose but cannot modify source code, metric identities, finding anchors, line numbers, or evidence IDs.
- Compressed output is validated before use. If it is not shorter or fails validation, deterministic anchor-preserving compaction is used without a second model request.

## Reporting

The Gradio and text reports show each metric score with a model-authored or fallback explanation limited to 24 words and 160 characters. They omit generic RSI boilerplate and score-basis labels. Successful compiler diagnostics are not reported as syntax defects.

A valid LLM score of zero is retained as the model's decision; Python does not silently replace it. When the LLM score is unavailable, a completed analyzer score may be used as a fallback. If neither source measured the metric, the report labels a neutral provisional score rather than inventing a zero.

The FormAI benchmark path remains separate from ordinary assignment grading.


## Structured JSON reliability

- Program understanding, each whole-program RSI pass, metric scoring, and same-model compression use the configured grading model with llama.cpp's `/v1/chat/completions` `response_format` JSON Schema contract.
- These schema-constrained requests explicitly disable the Qwen reasoning prefix so the schema is active from the first assistant token; this avoids the Qwen reasoning-prefix/grammar prefill issue. Legacy unstructured requests may still use the configured reasoning mode.
- Metric explanations are limited to 18 words and 120 characters (normally no more than 2-3 display lines).


## llama.cpp structured-output transport

- Structured grading and same-model compression use the documented `/v1/chat/completions` `response_format: {type: json_schema, json_schema: {name, strict, schema}}` field. They do not send the raw `grammar` field, which is for the completion endpoint and may be ignored by chat-server builds.
- Template thinking is disabled per structured request so the JSON Schema grammar can start at the first assistant output token; the same configured model is used for all grading and compression requests.
