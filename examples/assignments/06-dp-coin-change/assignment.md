# Minimum Coin Change (DP)

## Student task

Given `n` distinct positive coin denominations and a target amount `A`, return the minimum number of coins needed to form exactly `A`. Each denomination may be used an unlimited number of times. Print `-1` if the amount cannot be formed.

A greedy largest-first strategy is not guaranteed to be optimal. Implement a correct dynamic-programming solution.

## Input

- First line: `n A`.
- Second line: `n` distinct positive integer denominations.

## Output

Print the minimum number of coins, or `-1`, followed by a newline.

## Constraints

`1 <= n <= 100`, `0 <= A <= 100000`, `1 <= denomination <= 10000`.

## Submission note

`submission.c` is intentionally flawed: it uses a greedy strategy. `reference_solution.c` is provided only to validate the tests; do not submit it when checking whether the grader detects the seeded defect.

## Grading

Upload this folder's `rules.json` and `submission.c` in the Gradio grading tab. All test paths are relative to the repository root. Keep `instructor_expected_findings.json` out of the model input.
