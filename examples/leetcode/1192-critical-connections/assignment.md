# LeetCode Hard — Critical Connections in a Network (Graph Bridges)

## Problem

You are given an undirected network with `n` vertices numbered `0` through `n - 1` and `m` connections. The input graph is connected, as in the original LeetCode problem. This fixture additionally sorts the output edges lexicographically for deterministic grading. A connection is **critical** if removing that edge increases the number of connected components in the graph.

Find every critical connection. Print each critical edge once, with its smaller endpoint first, and sort the resulting edges lexicographically by `(u, v)`.

This is the graph-bridge problem commonly known as **Critical Connections in a Network (LeetCode 1192, Hard)**. A linear-time DFS low-link algorithm (Tarjan bridge detection) is expected.

## Input format

```text
n m
u1 v1
u2 v2
...
um vm
```

- `2 <= n <= 100000`
- `n - 1 <= m <= 100000`
- `0 <= ui, vi < n`
- The input has no self-loops or duplicate undirected edges.
- The graph is connected, with no self-loops or repeated undirected edges.

## Output format

Print `k`, the number of critical connections, on the first line. Then print the `k` edges, one per line, as `u v`, with `u < v`. Sort these lines in ascending lexicographic order. If there are no critical connections, print `0` only.

## Example

Input:
```text
4 4
0 1
1 2
2 0
1 3
```

Output:
```text
1
1 3
```

## Expected algorithm and complexity

Use discovery times and low-link values. For a DFS tree edge `(u, v)`, it is a bridge exactly when `low[v] > discovery[u]`. Expected time complexity: `O(V + E)`; expected auxiliary space: `O(V + E)`.

A solution that recomputes connectivity after deleting every edge is too slow for the maximum constraints.

## Submission fixture

`submission.c` is intentionally flawed: it uses `low[v] >= discovery[u]` instead of the strict bridge condition. This produces false positives for some cycle edges. Do not treat it as a correct reference implementation.
