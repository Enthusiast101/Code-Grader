#include <stdio.h>

int main(void) {
    int rows, cols;
    int blocked[30][30];
    long long dp[30][30] = {{0}};
    if (scanf("%d %d", &rows, &cols) != 2 || rows < 1 || rows > 30 || cols < 1 || cols > 30) {
        return 1;
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (scanf("%d", &blocked[i][j]) != 1 || (blocked[i][j] != 0 && blocked[i][j] != 1)) return 1;
        }
    }

    dp[0][0] = 1;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (i == 0 && j == 0) continue;
            /* BUG: this recurrence counts paths through blocked cells. */
            dp[i][j] = (i > 0 ? dp[i - 1][j] : 0) + (j > 0 ? dp[i][j - 1] : 0);
        }
    }

    printf("%lld\n", dp[rows - 1][cols - 1]);
    return 0;
}
