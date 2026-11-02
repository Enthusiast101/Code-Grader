#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int n, amount;
    if (scanf("%d %d", &n, &amount) != 2 || n < 1 || n > 100 || amount < 0 || amount > 100000) {
        return 1;
    }

    int *coins = malloc((size_t)n * sizeof(*coins));
    int *dp = malloc(((size_t)amount + 1U) * sizeof(*dp));
    if (coins == NULL || dp == NULL) {
        free(coins);
        free(dp);
        return 1;
    }
    for (int i = 0; i < n; ++i) {
        if (scanf("%d", &coins[i]) != 1 || coins[i] <= 0) {
            free(coins);
            free(dp);
            return 1;
        }
    }

    for (int value = 0; value <= amount; ++value) dp[value] = INT_MAX / 4;
    dp[0] = 0;
    for (int value = 1; value <= amount; ++value) {
        for (int i = 0; i < n; ++i) {
            if (coins[i] <= value && dp[value - coins[i]] + 1 < dp[value]) {
                dp[value] = dp[value - coins[i]] + 1;
            }
        }
    }

    printf("%d\n", dp[amount] >= INT_MAX / 4 ? -1 : dp[amount]);
    free(coins);
    free(dp);
    return 0;
}
