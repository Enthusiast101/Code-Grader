#include <stdio.h>
#include <stdlib.h>

#define MAX_N 100000
#define MAX_M 100000

typedef struct {
    int u;
    int v;
} Bridge;

static int min_int(int a, int b) {
    return (a < b) ? a : b;
}

static void add_directed_edge(int from, int to, int edge_id,
                              int *head, int *to_array,
                              int *next_array, int *id_array,
                              int *cursor) {
    to_array[*cursor] = to;
    id_array[*cursor] = edge_id;
    next_array[*cursor] = head[from];
    head[from] = *cursor;
    ++(*cursor);
}

static int compare_bridges(const void *lhs, const void *rhs) {
    const Bridge *a = (const Bridge *)lhs;
    const Bridge *b = (const Bridge *)rhs;
    if (a->u != b->u) return (a->u > b->u) - (a->u < b->u);
    return (a->v > b->v) - (a->v < b->v);
}

static void find_bridges(int n, int m, const int *head,
                         const int *to_array, const int *next_array,
                         const int *id_array, int *discovery, int *low,
                         int *parent, int *parent_edge, int *iterator,
                         int *stack, Bridge *bridges, int *bridge_count) {
    int timer = 0;
    *bridge_count = 0;

    /* Iterative DFS avoids relying on the process recursion-stack limit. */
    for (int root = 0; root < n; ++root) {
        if (discovery[root] != 0) continue;

        parent[root] = -1;
        parent_edge[root] = -1;
        discovery[root] = low[root] = ++timer;
        iterator[root] = head[root];

        int top = 0;
        stack[top++] = root;

        while (top > 0) {
            int u = stack[top - 1];
            int edge = iterator[u];

            if (edge != -1) {
                iterator[u] = next_array[edge];
                if (id_array[edge] == parent_edge[u]) continue;

                int v = to_array[edge];
                if (discovery[v] == 0) {
                    parent[v] = u;
                    parent_edge[v] = id_array[edge];
                    discovery[v] = low[v] = ++timer;
                    iterator[v] = head[v];
                    stack[top++] = v;
                } else {
                    low[u] = min_int(low[u], discovery[v]);
                }
            } else {
                --top;
                if (parent[u] != -1) {
                    int p = parent[u];
                    if (low[u] > discovery[p]) {
                        int a = p;
                        int b = u;
                        if (a > b) {
                            int temporary = a;
                            a = b;
                            b = temporary;
                        }
                        bridges[*bridge_count].u = a;
                        bridges[*bridge_count].v = b;
                        ++(*bridge_count);
                    }
                    low[p] = min_int(low[p], low[u]);
                }
            }
        }
    }

    (void)m;
    qsort(bridges, (size_t)*bridge_count, sizeof(*bridges), compare_bridges);
}

int main(void) {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2 || n < 1 || n > MAX_N || m < 0 || m > MAX_M) {
        return 1;
    }

    int *head = (int *)malloc((size_t)n * sizeof(*head));
    size_t directed_capacity = (size_t)(m > 0 ? 2 * m : 1);
    int *to_array = (int *)malloc(directed_capacity * sizeof(*to_array));
    int *next_array = (int *)malloc(directed_capacity * sizeof(*next_array));
    int *id_array = (int *)malloc(directed_capacity * sizeof(*id_array));
    int *discovery = (int *)calloc((size_t)n, sizeof(*discovery));
    int *low = (int *)malloc((size_t)n * sizeof(*low));
    int *parent = (int *)malloc((size_t)n * sizeof(*parent));
    int *parent_edge = (int *)malloc((size_t)n * sizeof(*parent_edge));
    int *iterator = (int *)malloc((size_t)n * sizeof(*iterator));
    int *stack = (int *)malloc((size_t)n * sizeof(*stack));
    Bridge *bridges = (Bridge *)malloc((size_t)(m > 0 ? m : 1) * sizeof(*bridges));

    if (head == NULL || to_array == NULL || next_array == NULL || id_array == NULL ||
        discovery == NULL || low == NULL || parent == NULL || parent_edge == NULL ||
        iterator == NULL || stack == NULL || bridges == NULL) {
        free(head); free(to_array); free(next_array); free(id_array);
        free(discovery); free(low); free(parent); free(parent_edge);
        free(iterator); free(stack); free(bridges);
        return 1;
    }

    for (int i = 0; i < n; ++i) head[i] = -1;

    int cursor = 0;
    for (int edge_id = 0; edge_id < m; ++edge_id) {
        int u, v;
        if (scanf("%d %d", &u, &v) != 2 || u < 0 || u >= n || v < 0 || v >= n || u == v) {
            free(head); free(to_array); free(next_array); free(id_array);
            free(discovery); free(low); free(parent); free(parent_edge);
            free(iterator); free(stack); free(bridges);
            return 1;
        }
        add_directed_edge(u, v, edge_id, head, to_array, next_array, id_array, &cursor);
        add_directed_edge(v, u, edge_id, head, to_array, next_array, id_array, &cursor);
    }

    int bridge_count = 0;
    find_bridges(n, m, head, to_array, next_array, id_array,
                 discovery, low, parent, parent_edge, iterator,
                 stack, bridges, &bridge_count);

    printf("%d\n", bridge_count);
    for (int i = 0; i < bridge_count; ++i) {
        printf("%d %d\n", bridges[i].u, bridges[i].v);
    }

    free(head); free(to_array); free(next_array); free(id_array);
    free(discovery); free(low); free(parent); free(parent_edge);
    free(iterator); free(stack); free(bridges);
    return 0;
}
