# Number of Islands (LeetCode 200, Medium)

A stdin/stdout C assignment adapted from the LeetCode problem, written for Code-Grader.

## Files

- `assignment.md`: the task, constraints, input/output format and expected complexity.
- `submission.c`: the student submission to grade. Iterative flood fill is correct and stack-safe, but the grid is never freed (memory leak).
- `reference_solution.c`: the instructor's correct solution, named by `reference_solution` in `rules.json`. Test generation runs it in the sandbox to compute the expected output of every generated test case; it is never shown to the model.
- `rules.json`: all nine metrics enabled (correctness 20 points, the other eight 10 points each).
- `testcases.json`, `edge_cases.json`, `memory_cases.json`: correctness, boundary and memory-check inputs. Expected outputs come from an independent Python reference.
- `instructor_expected_findings.json`: hidden oracle for the planted defect. Do not pass it to the LLM.
- `validate_reference.py`: compiles the reference and checks every stored test.
- `submissions/`: other students' solutions used as the similarity comparison set.

## Similarity

The graded submission is original: its highest similarity is about 0.25, so it gets full credit. Within `submissions/`, `student_01.c` (BFS) and `student_03.c` are the same program with renamed identifiers, so the pairwise matrix flags that pair at 1.0 for review; `student_02.c` (union-find) is independent.

## Run

In Gradio, upload `examples/leetcode/0200-number-of-islands/rules.json` and `submission.c`. Paths in the rules are relative to the repository root.
