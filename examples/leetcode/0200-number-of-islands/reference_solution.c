#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void free_grid(char **grid, int rows)
{
    for (int r = 0; r < rows; r++) {
        free(grid[r]);
    }
    free(grid);
}

static void sink_island(char **grid, int rows, int cols, int start, int *stack)
{
    int top = 0;
    stack[top++] = start;
    grid[start / cols][start % cols] = '0';
    while (top > 0) {
        int cell = stack[--top];
        int r = cell / cols;
        int c = cell % cols;
        const int dr[4] = {1, -1, 0, 0};
        const int dc[4] = {0, 0, 1, -1};
        for (int k = 0; k < 4; k++) {
            int nr = r + dr[k];
            int nc = c + dc[k];
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == '1') {
                grid[nr][nc] = '0';
                stack[top++] = nr * cols + nc;
            }
        }
    }
}

int main(void)
{
    int rows, cols;
    if (scanf("%d %d", &rows, &cols) != 2 || rows < 1 || cols < 1) {
        return 1;
    }
    char **grid = calloc((size_t)rows, sizeof *grid);
    int *stack = malloc((size_t)rows * (size_t)cols * sizeof *stack);
    if (grid == NULL || stack == NULL) {
        free(grid);
        free(stack);
        return 1;
    }
    for (int r = 0; r < rows; r++) {
        grid[r] = malloc((size_t)cols + 1);
        if (grid[r] == NULL || scanf("%300s", grid[r]) != 1 || (int)strlen(grid[r]) != cols) {
            free_grid(grid, rows);
            free(stack);
            return 1;
        }
    }
    int islands = 0;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (grid[r][c] == '1') {
                islands++;
                sink_island(grid, rows, cols, r * cols + c, stack);
            }
        }
    }
    printf("%d\n", islands);
    free_grid(grid, rows);
    free(stack);
    return 0;
}
