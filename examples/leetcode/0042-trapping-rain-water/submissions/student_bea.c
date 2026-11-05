#include <stdio.h>
#include <stdlib.h>

/* Monotonic stack of indices with non-increasing heights. When a taller bar
 * arrives, every lower bar popped forms the floor of a pool bounded by the
 * new bar and the bar beneath it on the stack. */
int main(void)
{
    int n;
    if (scanf("%d", &n) != 1 || n <= 0) {
        return 1;
    }

    long long *bars = calloc((size_t)n, sizeof(long long));
    int *stack = calloc((size_t)n, sizeof(int));
    if (!bars || !stack) {
        free(bars);
        free(stack);
        return 1;
    }
    for (int i = 0; i < n; ++i) {
        if (scanf("%lld", bars + i) != 1) {
            free(bars);
            free(stack);
            return 1;
        }
    }

    long long pooled = 0;
    int top = -1;
    for (int i = 0; i < n; ++i) {
        while (top >= 0 && bars[i] > bars[stack[top]]) {
            int floor_index = stack[top--];
            if (top < 0) {
                break;
            }
            int wall = stack[top];
            long long depth = (bars[i] < bars[wall] ? bars[i] : bars[wall]) - bars[floor_index];
            pooled += depth * (long long)(i - wall - 1);
        }
        stack[++top] = i;
    }

    printf("%lld\n", pooled);
    free(bars);
    free(stack);
    return 0;
}
