# Sliding Window Maximum (LeetCode 239, Hard)

A window of width `k` slides over an integer array from left to right, one position at a time. For every position of the window, report the largest value inside it.

## Input (stdin)

- Line 1: two integers `n k` (1 <= k <= n <= 100000).
- Line 2: `n` integers, each with -10000 <= value <= 10000.

## Output (stdout)

Print the `n - k + 1` window maxima on one line, separated by single spaces, followed by a newline.

## Notes

- Expected time O(n) using a monotonic deque; extra space O(k).
- An O(n * k) rescan of every window is too slow for n = 100000 and k = 50000.

## Example

Input:

```text
8 3
1 3 -1 -3 5 3 6 7
```

Output:

```text
3 3 5 5 6 7
```
