#include <stdio.h>

int main(void) {
    int count;
    if (scanf("%d", &count) != 1 || count <= 0) {
        return 1;
    }

    long long sum = 0;
    for (int i = 0; i < count; ++i) {
        int value;
        if (scanf("%d", &value) != 1) {
            return 1;
        }
        sum += value;
    }

    double average = sum / count;
    printf("%.2f\n", average);
    return 0;
}
