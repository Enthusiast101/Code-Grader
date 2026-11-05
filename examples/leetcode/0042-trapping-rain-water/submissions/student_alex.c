#include <stdio.h>
#include <stdlib.h>

// compute the rain water using two indices from both ends
static int rain(const int *h, int n)
{
    int lo = 0;
    int hi = n - 1;
    int best_lo = 0;
    int best_hi = 0;
    int water = 0;

    while (lo < hi) {
        if (h[lo] < h[hi]) {
            if (h[lo] >= best_lo) {
                best_lo = h[lo];
            } else {
                water += best_lo - h[lo];
            }
            lo++;
        } else {
            if (h[hi] >= best_hi) {
                best_hi = h[hi];
            } else {
                water += best_hi - h[hi];
            }
            hi--;
        }
    }
    return water;
}

int main(void)
{
    int n;
    if (scanf("%d", &n) != 1 || n < 1) {
        return 1;
    }
    int *h = malloc((size_t)n * sizeof *h);
    if (h == NULL) {
        return 1;
    }
    for (int k = 0; k < n; k++) {
        if (scanf("%d", &h[k]) != 1) {
            free(h);
            return 1;
        }
    }
    printf("%d\n", rain(h, n));
    free(h);
    return 0;
}
