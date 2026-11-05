#include <stdio.h>
#include <stdlib.h>

/* Monotonic deque of indices whose values decrease from front to back; the
 * front is always the maximum of the current window. */
int main(void)
{
    int n, k;
    if (scanf("%d %d", &n, &k) != 2 || k < 1 || n < k) {
        return 1;
    }
    int *values = malloc((size_t)n * sizeof *values);
    int *deque = malloc((size_t)n * sizeof *deque);
    if (values == NULL || deque == NULL) {
        free(values);
        free(deque);
        return 1;
    }
    for (int i = 0; i < n; i++) {
        if (scanf("%d", &values[i]) != 1) {
            free(values);
            free(deque);
            return 1;
        }
    }
    int head = 0;
    int tail = 0;
    for (int i = 0; i < n; i++) {
        if (head < tail && deque[head] <= i - k) {
            head++;
        }
        while (head < tail && values[deque[tail - 1]] <= values[i]) {
            tail--;
        }
        deque[tail++] = i;
        if (i >= k - 1) {
            printf(i == k - 1 ? "%d" : " %d", values[deque[head]]);
        }
    }
    printf("\n");
    free(values);
    free(deque);
    return 0;
}
