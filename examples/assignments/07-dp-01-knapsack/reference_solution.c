#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int n, capacity;
    if (scanf("%d %d", &n, &capacity) != 2 || n < 1 || n > 100 || capacity < 0 || capacity > 100000) {
        return 1;
    }

    int *weights = malloc((size_t)n * sizeof(*weights));
    long long *values = malloc((size_t)n * sizeof(*values));
    long long *dp = calloc((size_t)capacity + 1U, sizeof(*dp));
    if (weights == NULL || values == NULL || dp == NULL) {
        free(weights); free(values); free(dp);
        return 1;
    }
    for (int i = 0; i < n; ++i) {
        if (scanf("%d", &weights[i]) != 1 || weights[i] <= 0) {
            free(weights); free(values); free(dp);
            return 1;
        }
    }
    for (int i = 0; i < n; ++i) {
        if (scanf("%lld", &values[i]) != 1 || values[i] < 0) {
            free(weights); free(values); free(dp);
            return 1;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int current = capacity; current >= weights[i]; --current) {
            long long candidate = dp[current - weights[i]] + values[i];
            if (candidate > dp[current]) dp[current] = candidate;
        }
    }

    printf("%lld\n", dp[capacity]);
    free(weights); free(values); free(dp);
    return 0;
}
