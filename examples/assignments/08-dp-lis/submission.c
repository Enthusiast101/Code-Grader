#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int n;
    if (scanf("%d", &n) != 1 || n < 1 || n > 20000) return 1;

    int *values = malloc((size_t)n * sizeof(*values));
    int *dp = malloc((size_t)n * sizeof(*dp));
    if (values == NULL || dp == NULL) {
        free(values); free(dp);
        return 1;
    }
    for (int i = 0; i < n; ++i) {
        if (scanf("%d", &values[i]) != 1) {
            free(values); free(dp);
            return 1;
        }
    }

    int best = 1;
    for (int i = 0; i < n; ++i) {
        dp[i] = 1;
        for (int j = 0; j < i; ++j) {
            if (values[j] <= values[i] && dp[j] + 1 > dp[i]) {
                dp[i] = dp[j] + 1;
            }
        }
        if (dp[i] > best) best = dp[i];
    }

    printf("%d\n", best);
    free(values); free(dp);
    return 0;
}
