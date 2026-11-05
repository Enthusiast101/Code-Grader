# Number of Islands (LeetCode 200, Medium)

A map is a grid of land (`1`) and water (`0`) cells. An island is a maximal group of land cells joined horizontally or vertically (not diagonally). Count the islands.

## Input (stdin)

- Line 1: two integers `R C` (1 <= R, C <= 300), the number of rows and columns.
- Next `R` lines: a string of exactly `C` characters, each `0` or `1`.

## Output (stdout)

Print the number of islands followed by a newline.

## Notes

- Expected time O(R * C), extra space O(R * C).
- A grid can be up to 300 x 300, so a single island may contain 90,000 cells; make sure the traversal does not exhaust the call stack.
- Release every allocation before exiting.

## Example

Input:

```text
4 5
11000
11000
00100
00011
```

Output:

```text
3
```
