#include <stdio.h>
#include <string.h>

/* Coordinates are small (0..10000), so mark coverage instead of sorting.
 * covered[x] marks the point x; joined[x] marks that x and x+1 lie in one
 * input interval, so separate runs of points are not glued together. */
#define LIMIT 10001

static unsigned char covered[LIMIT];
static unsigned char joined[LIMIT];

int main(void)
{
    int n;
    if (scanf("%d", &n) != 1)
        return 1;
    memset(covered, 0, sizeof covered);
    memset(joined, 0, sizeof joined);
    for (int i = 0; i < n; ++i) {
        int s, e;
        if (scanf("%d%d", &s, &e) != 2)
            return 1;
        for (int x = s; x <= e; ++x) {
            covered[x] = 1;
            if (x < e)
                joined[x] = 1;
        }
    }

    int pieces = 0;
    int starts[LIMIT], ends[LIMIT];
    int x = 0;
    while (x < LIMIT) {
        if (!covered[x]) { ++x; continue; }
        int begin = x;
        while (x < LIMIT - 1 && joined[x]) ++x;
        starts[pieces] = begin;
        ends[pieces] = x;
        ++pieces;
        ++x;
    }
    printf("%d\n", pieces);
    for (int i = 0; i < pieces; ++i)
        printf("%d %d\n", starts[i], ends[i]);
    return 0;
}
