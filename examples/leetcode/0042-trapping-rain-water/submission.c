#include <stdio.h>
#include <stdlib.h>

/* Two-pointer scan: water above each bar is bounded by the lower of the
 * tallest bars seen so far from the left and from the right. */
static int trapped_water(const int *heights, int count)
{
    int left = 0;
    int right = count - 1;
    int left_max = 0;
    int right_max = 0;
    int total = 0;

    while (left < right) {
        if (heights[left] < heights[right]) {
            if (heights[left] >= left_max) {
                left_max = heights[left];
            } else {
                total += left_max - heights[left];
            }
            left++;
        } else {
            if (heights[right] >= right_max) {
                right_max = heights[right];
            } else {
                total += right_max - heights[right];
            }
            right--;
        }
    }
    return total;
}

int main(void)
{
    int count;
    if (scanf("%d", &count) != 1 || count < 1) {
        return 1;
    }
    int *heights = malloc((size_t)count * sizeof *heights);
    if (heights == NULL) {
        return 1;
    }
    for (int i = 0; i < count; i++) {
        if (scanf("%d", &heights[i]) != 1) {
            free(heights);
            return 1;
        }
    }
    printf("%d\n", trapped_water(heights, count));
    free(heights);
    return 0;
}
