# Merge Intervals (LeetCode 56, Medium)

You are given closed intervals `[start, end]`. Merge every group of overlapping intervals and print the result sorted by start. Two intervals overlap when they share at least one point, so `[1, 4]` and `[4, 5]` merge into `[1, 5]`.

## Input (stdin)

- Line 1: an integer `n` (1 <= n <= 10000).
- Next `n` lines: two integers `start end` with 0 <= start <= end <= 10000. Intervals may appear in any order.

## Output (stdout)

- Line 1: `m`, the number of merged intervals.
- Next `m` lines: `start end` of each merged interval, in increasing order of start.

## Notes

Expected time O(n log n) (sort, then one pass), extra space O(n).

## Example

Input:

```text
4
8 10
1 3
2 6
15 18
```

Output:

```text
3
1 6
8 10
15 18
```
