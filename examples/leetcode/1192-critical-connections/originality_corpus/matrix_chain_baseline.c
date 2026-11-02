#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

static long long minimum_cost(const int *dimensions, int count) {
    long long *table = (long long *)calloc((size_t)count * (size_t)count, sizeof(*table));
    if (table == NULL) return -1;
    for (int width = 2; width < count; ++width) {
        for (int left = 0; left + width < count; ++left) {
            int right = left + width;
            long long best = LLONG_MAX;
            for (int split = left + 1; split < right; ++split) {
                long long candidate = table[left * count + split] +
                    table[split * count + right] +
                    (long long)dimensions[left] * dimensions[split] * dimensions[right];
                if (candidate < best) best = candidate;
            }
            table[left * count + right] = best;
        }
    }
    long long answer = table[count - 1];
    free(table);
    return answer;
}

int main(void) {
    int count;
    if (scanf("%d", &count) != 1 || count < 2 || count > 500) return 1;
    int *dimensions = (int *)malloc((size_t)count * sizeof(*dimensions));
    if (dimensions == NULL) return 1;
    for (int index = 0; index < count; ++index) {
        if (scanf("%d", &dimensions[index]) != 1) {
            free(dimensions);
            return 1;
        }
    }
    printf("%lld\\n", minimum_cost(dimensions, count));
    free(dimensions);
    return 0;
}
