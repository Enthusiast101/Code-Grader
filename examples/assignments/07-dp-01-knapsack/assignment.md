# 0/1 Knapsack

## Student task

There are `n` items. Item `i` has positive weight `w[i]` and nonnegative value `v[i]`. Choose a subset whose total weight does not exceed capacity `W`, maximizing total value. Each item may be selected **at most once**.

Use 0/1 dynamic programming. The item transition must not allow the same item to be reused during one iteration.

## Input

- First line: `n W`.
- Second line: `n` item weights.
- Third line: `n` item values.

## Output

Print the maximum achievable value, followed by a newline.

## Constraints

`1 <= n <= 100`, `0 <= W <= 100000`, `1 <= w[i] <= 100000`, `0 <= v[i] <= 1000000000`.

## Submission note

`submission.c` intentionally iterates capacities in the wrong direction, turning the 0/1 transition into an unbounded-knapsack transition. `reference_solution.c` is only for validating the test suite.

## Grading

Upload this folder's `rules.json` and `submission.c`. Test paths are relative to the repository root. Keep `instructor_expected_findings.json` hidden from model input.
