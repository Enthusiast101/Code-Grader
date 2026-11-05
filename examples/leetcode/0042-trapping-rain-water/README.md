# Trapping Rain Water (LeetCode 42, Hard)

A stdin/stdout C assignment adapted from the LeetCode problem, written for Code-Grader.

## Files

- `assignment.md`: the task, constraints, input/output format and expected complexity.
- `submission.c`: the student submission to grade. Correct two-pointer algorithm, but the total is a 32-bit `int`, which overflows on large inputs.
- `reference_solution.c`: the instructor's correct solution, named by `reference_solution` in `rules.json`. Test generation runs it in the sandbox to compute the expected output of every generated test case; it is never shown to the model.
- `rules.json`: all nine metrics enabled (correctness 20 points, the other eight 10 points each).
- `testcases.json`, `edge_cases.json`, `memory_cases.json`: correctness, boundary and memory-check inputs. Expected outputs come from an independent Python reference.
- `instructor_expected_findings.json`: hidden oracle for the planted defect. Do not pass it to the LLM.
- `validate_reference.py`: compiles the reference and checks every stored test.
- `submissions/`: other students' solutions used as the similarity comparison set.

## Similarity

`submissions/` simulates a class: `student_alex.c` is the graded submission with renamed variables (similarity 1.0, so 0/10), while `student_bea.c` (monotonic stack), `student_chen.c` (prefix maxima) and `student_dana.c` (brute force) are independent (about 0.15-0.19).

## Run

In Gradio, upload `examples/leetcode/0042-trapping-rain-water/rules.json` and `submission.c`. Paths in the rules are relative to the repository root.
