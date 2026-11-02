#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int n;
    if (scanf("%d", &n) != 1 || n < 1 || n > 20000) return 1;

    int *tails = malloc((size_t)n * sizeof(*tails));
    if (tails == NULL) return 1;

    int length = 0;
    for (int i = 0; i < n; ++i) {
        int value;
        if (scanf("%d", &value) != 1) {
            free(tails);
            return 1;
        }
        int low = 0, high = length;
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (tails[mid] < value) low = mid + 1;
            else high = mid;
        }
        tails[low] = value;
        if (low == length) ++length;
    }

    printf("%d\n", length);
    free(tails);
    return 0;
}
