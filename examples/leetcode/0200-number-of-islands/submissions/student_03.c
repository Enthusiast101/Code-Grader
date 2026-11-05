#include <stdio.h>
#include <stdlib.h>

#define LIMIT 300

static char board[LIMIT][LIMIT + 1];
static int pending_row[LIMIT * LIMIT];
static int pending_col[LIMIT * LIMIT];

// visit every cell of one island level by level
static void flood(int startRow, int startCol, int height, int width)
{
    int front = 0, back = 0;
    pending_row[back] = startRow;
    pending_col[back] = startCol;
    back++;
    board[startRow][startCol] = '0';
    while (front < back) {
        int y = pending_row[front];
        int x = pending_col[front];
        front++;
        if (y > 0 && board[y - 1][x] == '1') { board[y - 1][x] = '0'; pending_row[back] = y - 1; pending_col[back] = x; back++; }
        if (y + 1 < height && board[y + 1][x] == '1') { board[y + 1][x] = '0'; pending_row[back] = y + 1; pending_col[back] = x; back++; }
        if (x > 0 && board[y][x - 1] == '1') { board[y][x - 1] = '0'; pending_row[back] = y; pending_col[back] = x - 1; back++; }
        if (x + 1 < width && board[y][x + 1] == '1') { board[y][x + 1] = '0'; pending_row[back] = y; pending_col[back] = x + 1; back++; }
    }
}

int main(void)
{
    int height, width;
    if (scanf("%d %d", &height, &width) != 2 || height < 1 || width < 1 || height > LIMIT || width > LIMIT)
        return 1;
    for (int y = 0; y < height; y++)
        if (scanf("%300s", board[y]) != 1)
            return 1;
    int count = 0;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            if (board[y][x] == '1') {
                count++;
                flood(y, x, height, width);
            }
    printf("%d\n", count);
    return 0;
}
