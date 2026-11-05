#include <stdio.h>
#include <stdlib.h>

#define MAXN 20000

static long long height[MAXN];
static long long tallest_left[MAXN];
static long long tallest_right[MAXN];

static long long smaller(long long a, long long b) { return a < b ? a : b; }
static long long larger(long long a, long long b) { return a > b ? a : b; }

/* Precompute the tallest bar to the left and right of every position. */
static long long solve(int n)
{
    tallest_left[0] = height[0];
    for (int i = 1; i < n; i++)
        tallest_left[i] = larger(tallest_left[i - 1], height[i]);

    tallest_right[n - 1] = height[n - 1];
    for (int i = n - 2; i >= 0; i--)
        tallest_right[i] = larger(tallest_right[i + 1], height[i]);

    long long answer = 0;
    for (int i = 0; i < n; i++)
        answer += smaller(tallest_left[i], tallest_right[i]) - height[i];
    return answer;
}

int main(void)
{
    int n = 0;
    if (scanf("%d", &n) != 1 || n < 1 || n > MAXN)
        return 1;
    for (int i = 0; i < n; i++)
        if (scanf("%lld", &height[i]) != 1)
            return 1;
    printf("%lld\n", solve(n));
    return 0;
}
