# Trapping Rain Water (LeetCode 42, Hard)

Bars of width 1 stand side by side; their heights are given from left to right. After rain, water collects in the dips between taller bars. Compute the total volume of water held.

## Input (stdin)

- Line 1: an integer `n` (1 <= n <= 20000), the number of bars.
- Line 2: `n` integers `h[0] .. h[n-1]` separated by spaces, each with 0 <= h[i] <= 1000000.

## Output (stdout)

Print a single integer, the total trapped water, followed by a newline.

## Notes

- The total can exceed the range of a 32-bit integer (up to about 2 * 10^10); use a 64-bit type.
- Expected time O(n), extra space O(1) (two pointers) or O(n) (prefix maxima).

## Example

Input:

```text
12
0 1 0 2 1 0 1 3 2 1 2 1
```

Output:

```text
6
```
