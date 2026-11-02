# Grid Paths With Obstacles

## Student task

An `R x C` grid contains `0` for an open cell and `1` for a blocked cell. Starting at the top-left cell, count the number of paths to the bottom-right cell when only moves **right** and **down** are allowed. A path may not enter a blocked cell. If the start or destination is blocked, the answer is zero.

The answer fits in a signed 64-bit integer for the supplied constraints.

## Input

- First line: `R C`.
- Next `R` lines: `C` integers, each either 0 (open) or 1 (blocked).

## Output

Print the number of valid paths, followed by a newline.

## Constraints

`1 <= R, C <= 30`.

## Submission note

`submission.c` intentionally ignores blocked cells in its DP transition. `reference_solution.c` is provided only to validate the test suite.

## Grading

Upload this folder's `rules.json` and `submission.c`. Test paths are relative to the repository root. Keep `instructor_expected_findings.json` hidden from model input.
