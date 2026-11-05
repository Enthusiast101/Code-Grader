# Merge Intervals (LeetCode 56, Medium)

A stdin/stdout C assignment adapted from the LeetCode problem, written for Code-Grader.

## Files

- `assignment.md`: the task, constraints, input/output format and expected complexity.
- `submission.c`: the student submission to grade. Sort-and-sweep is right, but the overlap test uses `<` instead of `<=`, so touching intervals are not merged.
- `reference_solution.c`: the instructor's correct solution, named by `reference_solution` in `rules.json`. Test generation runs it in the sandbox to compute the expected output of every generated test case; it is never shown to the model.
- `rules.json`: all nine metrics enabled (correctness 20 points, the other eight 10 points each).
- `testcases.json`, `edge_cases.json`, `memory_cases.json`: correctness, boundary and memory-check inputs. Expected outputs come from an independent Python reference.
- `instructor_expected_findings.json`: hidden oracle for the planted defect. Do not pass it to the LLM.
- `validate_reference.py`: compiles the reference and checks every stored test.
- `submissions/`: other students' solutions used as the similarity comparison set.

## Similarity

`submissions/student_kim.c` copies the submission's struct, comparator and input handling but writes its own merge loop (similarity about 0.67, partial credit). `student_lee.c` (coverage marking) and `student_moe.c` (insertion sort) are independent.

## Run

In Gradio, upload `examples/leetcode/0056-merge-intervals/rules.json` and `submission.c`. Paths in the rules are relative to the repository root.
