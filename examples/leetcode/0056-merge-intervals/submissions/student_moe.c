#include <stdio.h>
#include <stdlib.h>

struct span { int lo, hi; };

/* Insertion sort is fine for the class tests; then a single sweep. */
static void sort_spans(struct span *s, int n)
{
    for (int i = 1; i < n; i++) {
        struct span key = s[i];
        int j = i - 1;
        while (j >= 0 && (s[j].lo > key.lo || (s[j].lo == key.lo && s[j].hi > key.hi))) {
            s[j + 1] = s[j];
            j--;
        }
        s[j + 1] = key;
    }
}

int main(void)
{
    int n;
    if (scanf("%d", &n) != 1 || n <= 0) return 1;
    struct span *s = calloc(n, sizeof *s);
    struct span *res = calloc(n, sizeof *res);
    if (!s || !res) { free(s); free(res); return 1; }
    for (int i = 0; i < n; i++) {
        if (scanf("%d %d", &s[i].lo, &s[i].hi) != 2) { free(s); free(res); return 1; }
    }
    sort_spans(s, n);

    int m = 0;
    for (int i = 0; i < n; i++) {
        if (m > 0 && s[i].lo <= res[m - 1].hi) {
            if (s[i].hi > res[m - 1].hi) res[m - 1].hi = s[i].hi;
        } else {
            res[m++] = s[i];
        }
    }
    printf("%d\n", m);
    for (int i = 0; i < m; i++) printf("%d %d\n", res[i].lo, res[i].hi);
    free(s);
    free(res);
    return 0;
}
