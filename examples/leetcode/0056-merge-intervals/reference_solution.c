#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int start;
    int end;
} Interval;

static int compare_by_start(const void *a, const void *b)
{
    const Interval *x = a;
    const Interval *y = b;
    if (x->start != y->start) {
        return (x->start > y->start) - (x->start < y->start);
    }
    return (x->end > y->end) - (x->end < y->end);
}

/* Closed intervals: touching endpoints (start == current end) also merge. */
static int merge_intervals(Interval *items, int count, Interval *out)
{
    qsort(items, (size_t)count, sizeof *items, compare_by_start);
    int merged = 0;
    out[merged++] = items[0];
    for (int i = 1; i < count; i++) {
        Interval *last = &out[merged - 1];
        if (items[i].start <= last->end) {
            if (items[i].end > last->end) {
                last->end = items[i].end;
            }
        } else {
            out[merged++] = items[i];
        }
    }
    return merged;
}

int main(void)
{
    int count;
    if (scanf("%d", &count) != 1 || count < 1) {
        return 1;
    }
    Interval *items = malloc((size_t)count * sizeof *items);
    Interval *out = malloc((size_t)count * sizeof *out);
    if (items == NULL || out == NULL) {
        free(items);
        free(out);
        return 1;
    }
    for (int i = 0; i < count; i++) {
        if (scanf("%d %d", &items[i].start, &items[i].end) != 2) {
            free(items);
            free(out);
            return 1;
        }
    }
    int merged = merge_intervals(items, count, out);
    printf("%d\n", merged);
    for (int i = 0; i < merged; i++) {
        printf("%d %d\n", out[i].start, out[i].end);
    }
    free(items);
    free(out);
    return 0;
}
