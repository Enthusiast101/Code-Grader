# Sliding Window Maximum (LeetCode 239, Hard)

A stdin/stdout C assignment adapted from the LeetCode problem, written for Code-Grader.

## Files

- `assignment.md`: the task, constraints, input/output format and expected complexity.
- `submission.c`: the student submission to grade. Output is correct, but every window is rescanned (O(n*k)), so the large edge case times out.
- `reference_solution.c`: a correct solution, used only by `validate_reference.py`. Do not give it to the grader.
- `rules.json`: all nine metrics enabled (correctness 20 points, the other eight 10 points each).
- `testcases.json`, `edge_cases.json`, `memory_cases.json`: correctness, boundary and memory-check inputs. Expected outputs come from an independent Python reference.
- `instructor_expected_findings.json`: hidden oracle for the planted defect. Do not pass it to the LLM.
- `validate_reference.py`: compiles the reference and checks every stored test.

## Similarity

This example has no comparison folder: `corpus_dir` is `null` in `rules.json`, so the similarity metric awards full marks. Set it to a folder of other submissions to compare.

## Run

In Gradio, upload `examples/leetcode/0239-sliding-window-maximum/rules.json` and `submission.c`. Paths in the rules are relative to the repository root.
