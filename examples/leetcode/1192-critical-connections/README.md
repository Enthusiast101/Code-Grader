# Graph Hard Architecture Fixture — Critical Connections (LeetCode 1192)

This fixture adapts the Critical Connections in a Network / graph-bridges problem into a stdin/stdout C assignment for Code-Grader.

## Files

- `assignment.md`: task, constraints, input/output contract, expected algorithmic bound.
- `submission.c`: intentionally buggy C submission. It uses `>=` instead of `>` in Tarjan's bridge predicate.
- `reference_solution.c`: correct iterative Tarjan implementation; uses an explicit DFS stack to avoid recursion-stack dependence.
- `rules.json`: all nine example rubric metrics enabled, with the weights from the assignment PDF.
- `testcases.json`: ordinary correctness inputs.
- `edge_cases.json`: cycles, minimum two-vertex input, star, cycle, tail, and articulation-vertex cases.
- `memory_cases.json`: small deterministic input for Valgrind / ASan / UBSan.
- `runtime_cases.json`: maximum-size sparse cycle to exercise O(V+E) traversal without huge expected output.
- `instructor_expected_findings.json`: hidden oracle for the deliberately planted defect. Do not pass it to the LLM.
- `originality_corpus/`: unrelated C baseline so originality can run without returning N/A.
- `originality_corpus_copied/`: optional near-duplicate reference file for a controlled similarity-positive test.
- `validate_reference.py`: compiles the reference solution and checks all stored correctness and edge tests.

## Run in Code-Grader

Preserve this directory layout under your repository root:

```text
examples/leetcode/1192-critical-connections/
```

Then in Gradio, upload this folder's `rules.json` and `submission.c`. The rules use paths relative to the repository root. The `submission.source_file` field already points to the fixture path; update it if you copy the C source elsewhere. All nine rubric metrics are enabled with the assignment-example weights summing to 1.00. The originality analyzer uses the unrelated baseline corpus by default so it completes with a real comparison. To deliberately test a high-similarity flag, change `metrics[name=originality].params.corpus_dir` to `examples/leetcode/1192-critical-connections/originality_corpus_copied` (this is a synthetic positive control, not evidence of student plagiarism).

Use `assignment.md` as the assignment prompt. Do not upload `instructor_expected_findings.json` or `reference_solution.c` to the blind grader.

## Validate the reference implementation

From a shell with GCC installed:

```bash
python examples/leetcode/1192-critical-connections/validate_reference.py
```

The script checks the reference solution against all regular and edge-case tests. It also runs the 100,000-vertex runtime input and checks that the expected output is `0`.

## Expected defect

The student submission intentionally uses:

```c
low[child] >= discovery[parent]
```

The correct criterion is:

```c
low[child] > discovery[parent]
```

A child subtree with `low[child] == discovery[parent]` has a back edge to the parent, meaning the tree edge is part of a cycle and is not a bridge. The intentionally flawed program can therefore return false-positive critical connections on cycles.
