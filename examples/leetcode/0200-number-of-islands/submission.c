#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Flood fill with an explicit stack of cell indices, so a 300 x 300 island
 * cannot overflow the call stack. */
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

static int count_islands(char **grid, int rows, int cols)
{
    int *stack = malloc((size_t)rows * (size_t)cols * sizeof *stack);
    if (stack == NULL) {
        return -1;
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
    free(stack);
    return islands;
}

int main(void)
{
    int rows, cols;
    if (scanf("%d %d", &rows, &cols) != 2 || rows < 1 || cols < 1) {
        return 1;
    }
    char **grid = malloc((size_t)rows * sizeof *grid);
    if (grid == NULL) {
        return 1;
    }
    for (int r = 0; r < rows; r++) {
        grid[r] = malloc((size_t)cols + 1);
        if (grid[r] == NULL || scanf("%300s", grid[r]) != 1 || (int)strlen(grid[r]) != cols) {
            return 1;
        }
    }
    int islands = count_islands(grid, rows, cols);
    if (islands < 0) {
        return 1;
    }
    printf("%d\n", islands);
    return 0;
}
