#include <stdio.h>
#include <stdlib.h>

static int compare_desc(const void *left, const void *right) {
    int a = *(const int *)left;
    int b = *(const int *)right;
    return (a < b) - (a > b);
}

int main(void) {
    int n, amount;
    if (scanf("%d %d", &n, &amount) != 2 || n < 1 || n > 100 || amount < 0 || amount > 100000) {
        return 1;
    }

    int *coins = malloc((size_t)n * sizeof(*coins));
    if (coins == NULL) return 1;
    for (int i = 0; i < n; ++i) {
        if (scanf("%d", &coins[i]) != 1 || coins[i] <= 0) {
            free(coins);
            return 1;
        }
    }

    qsort(coins, (size_t)n, sizeof(*coins), compare_desc);
    int remaining = amount;
    int used = 0;
    for (int i = 0; i < n; ++i) {
        used += remaining / coins[i];
        remaining %= coins[i];
    }

    printf("%d\n", remaining == 0 ? used : -1);
    free(coins);
    return 0;
}
