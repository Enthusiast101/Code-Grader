# Program understanding, sequential RSI, and LLM-led scoring

The C-assignment grading path follows this sequence:

1. Deterministic analyzers collect objective results in a structured evidence ledger.
2. The configured LLM understands the submitted source, separating actual source behavior from assignment intent.
3. RSI pass 1 identifies whole-program defects. Evidence is ordered defects-first so the evidence budget never drops a failed test or analyzer/sanitizer finding.
4. Each later pass receives the previous complete response plus the defect evidence records no finding cites yet, and must explain or dismiss each. Passes are findings-only.
5. RSI stops when a pass reproduces the previous findings exactly (`LLM_RSI_STOP_ON_CONVERGENCE`, default true).
6. Findings are deduplicated per metric and exact line range, then adjudicated: reasoning first, then five checks; a finding survives only if all checks hold.
7. One scoring request rates, 0-10, only metrics without an authoritative measurement. Executed tests and completed analyzers are authoritative; a runtime metric without a profile stays unmeasured. Python converts levels to points and applies weights.

## Structured output

- Requests send `response_format: {type: json_schema, json_schema: {name, schema}}`. llama-server ignores a top-level `schema` member for type `json_schema`; that shape was previously used and left every stage unconstrained.
- Template thinking is disabled for schema-constrained requests so the schema applies from the first token.
- Each stage allows up to three attempts; repair prompts name the specific problem (for example, missing metrics).

## Context fitting and compression

- Every grading stage is fitted to the context window using exact token counts from `/tokenize` when available.
- A quarter of the context (512-4096 tokens) is reserved for the answer, independent of the requested ceiling.
- Same-model compression runs only when the prompt does not fit; otherwise the prompt, including the previous RSI response, is sent verbatim.
- Compressed output is validated; deterministic anchor-preserving compaction is the fallback.

## Reproducibility

- Temperature 0, fixed seed, and `cache_prompt: false` (`LLAMA_CACHE_PROMPT`). With prompt caching enabled, identical requests were observed to decode differently on a warm server.
