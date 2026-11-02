# Longest Strictly Increasing Subsequence

## Student task

Given an integer array, output the length of its **strictly increasing** subsequence of maximum length. A subsequence preserves relative order but its elements do not need to be adjacent. Equal values cannot extend a strictly increasing subsequence.

The maximum input size is large enough that an O(n^2) dynamic program is not acceptable. Target O(n log n) time and O(n) auxiliary memory.

## Input

- First line: `n`.
- Second line: `n` signed 32-bit integers.

## Output

Print the LIS length, followed by a newline.

## Constraints

`1 <= n <= 20000`, `-2147483648 <= a[i] <= 2147483647`.

## Submission note

`submission.c` has two seeded defects: it uses `<=` (non-decreasing rather than strictly increasing) and an O(n^2) transition. The runtime case is intended to expose the scaling problem; its threshold is environment-dependent. `reference_solution.c` is the O(n log n) reference.

## Grading

Upload this folder's `rules.json` and `submission.c`. Test paths are relative to the repository root. Keep `instructor_expected_findings.json` hidden from model input.
