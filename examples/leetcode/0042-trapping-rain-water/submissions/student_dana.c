#include <stdio.h>
#include <stdlib.h>

/* Straightforward version: for each bar look for the highest bar on both
 * sides. O(n^2) but easy to check by hand. */
int main(void)
{
    int n;
    scanf("%d", &n);
    long long *a = malloc(sizeof(long long) * n);
    for (int i = 0; i < n; i++) scanf("%lld", &a[i]);

    long long sum = 0;
    for (int i = 0; i < n; i++) {
        long long leftHigh = 0, rightHigh = 0;
        for (int j = i; j >= 0; j--) if (a[j] > leftHigh) leftHigh = a[j];
        for (int j = i; j < n; j++) if (a[j] > rightHigh) rightHigh = a[j];
        long long level = leftHigh < rightHigh ? leftHigh : rightHigh;
        sum += level - a[i];
    }
    printf("%lld\n", sum);
    free(a);
    return 0;
}
