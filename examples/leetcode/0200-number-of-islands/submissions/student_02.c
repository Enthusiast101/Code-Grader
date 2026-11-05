#include <stdio.h>
#include <stdlib.h>

/* Union-find over land cells: every land cell starts as its own set and is
 * joined with land neighbours to the right and below. The answer is the
 * number of successful unions subtracted from the land count. */
static int *parent;

static int find(int x)
{
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

static int unite(int a, int b)
{
    int ra = find(a), rb = find(b);
    if (ra == rb) return 0;
    parent[ra] = rb;
    return 1;
}

int main(void)
{
    int rows, cols;
    if (scanf("%d %d", &rows, &cols) != 2) return 1;
    char *cells = malloc((size_t)rows * cols + 1);
    char line[512];
    parent = malloc(sizeof(int) * (size_t)rows * cols);
    if (!cells || !parent) return 1;

    for (int r = 0; r < rows; r++) {
        if (scanf("%511s", line) != 1) return 1;
        for (int c = 0; c < cols; c++) cells[r * cols + c] = line[c];
    }

    int land = 0, merges = 0;
    for (int i = 0; i < rows * cols; i++) {
        parent[i] = i;
        if (cells[i] == '1') land++;
    }
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int id = r * cols + c;
            if (cells[id] != '1') continue;
            if (c + 1 < cols && cells[id + 1] == '1') merges += unite(id, id + 1);
            if (r + 1 < rows && cells[id + cols] == '1') merges += unite(id, id + cols);
        }
    }
    printf("%d\n", land - merges);
    free(cells);
    free(parent);
    return 0;
}
