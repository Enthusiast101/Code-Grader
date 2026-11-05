#include <stdio.h>
#include <stdlib.h>

#define MAX 300

static char map[MAX][MAX + 1];
static int queue_r[MAX * MAX];
static int queue_c[MAX * MAX];

/* Breadth-first search from every unvisited land cell. */
static void bfs(int sr, int sc, int R, int C)
{
    int head = 0, tail = 0;
    queue_r[tail] = sr;
    queue_c[tail] = sc;
    tail++;
    map[sr][sc] = '0';
    while (head < tail) {
        int r = queue_r[head];
        int c = queue_c[head];
        head++;
        if (r > 0 && map[r - 1][c] == '1') { map[r - 1][c] = '0'; queue_r[tail] = r - 1; queue_c[tail] = c; tail++; }
        if (r + 1 < R && map[r + 1][c] == '1') { map[r + 1][c] = '0'; queue_r[tail] = r + 1; queue_c[tail] = c; tail++; }
        if (c > 0 && map[r][c - 1] == '1') { map[r][c - 1] = '0'; queue_r[tail] = r; queue_c[tail] = c - 1; tail++; }
        if (c + 1 < C && map[r][c + 1] == '1') { map[r][c + 1] = '0'; queue_r[tail] = r; queue_c[tail] = c + 1; tail++; }
    }
}

int main(void)
{
    int R, C;
    if (scanf("%d %d", &R, &C) != 2 || R < 1 || C < 1 || R > MAX || C > MAX)
        return 1;
    for (int r = 0; r < R; r++)
        if (scanf("%300s", map[r]) != 1)
            return 1;
    int islands = 0;
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            if (map[r][c] == '1') {
                islands++;
                bfs(r, c, R, C);
            }
    printf("%d\n", islands);
    return 0;
}
