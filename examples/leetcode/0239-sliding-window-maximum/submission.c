#include <stdio.h>
#include <stdlib.h>

/* Rescan every window to find its maximum. */
static int window_max(const int *values, int start, int width)
{
    int best = values[start];
    for (int i = start + 1; i < start + width; i++) {
        if (values[i] > best) {
            best = values[i];
        }
    }
    return best;
}

int main(void)
{
    int n, k;
    if (scanf("%d %d", &n, &k) != 2 || k < 1 || n < k) {
        return 1;
    }
    int *values = malloc((size_t)n * sizeof *values);
    if (values == NULL) {
        return 1;
    }
    for (int i = 0; i < n; i++) {
        if (scanf("%d", &values[i]) != 1) {
            free(values);
            return 1;
        }
    }
    for (int start = 0; start + k <= n; start++) {
        printf(start == 0 ? "%d" : " %d", window_max(values, start, k));
    }
    printf("\n");
    free(values);
    return 0;
}
