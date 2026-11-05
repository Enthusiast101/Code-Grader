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

int main(void)
{
    int count;
    if (scanf("%d", &count) != 1 || count < 1) {
        return 1;
    }
    Interval *items = malloc((size_t)count * sizeof *items);
    if (items == NULL) {
        return 1;
    }
    for (int i = 0; i < count; i++) {
        if (scanf("%d %d", &items[i].start, &items[i].end) != 2) {
            free(items);
            return 1;
        }
    }
    qsort(items, (size_t)count, sizeof *items, compare_by_start);

    /* Merge in place: keep a write index and grow the interval behind it. */
    int write = 0;
    for (int read = 1; read < count; read++) {
        if (items[read].start <= items[write].end) {
            if (items[read].end > items[write].end)
                items[write].end = items[read].end;
        } else {
            write++;
            items[write] = items[read];
        }
    }
    printf("%d\n", write + 1);
    for (int i = 0; i <= write; i++)
        printf("%d %d\n", items[i].start, items[i].end);
    free(items);
    return 0;
}
