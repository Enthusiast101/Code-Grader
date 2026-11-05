#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
    int count;
    if (scanf("%d", &count) != 1 || count < 1 || count > 1000) {
        return 1;
    }

    int64_t sum = 0;
    for (int i = 0; i < count; ++i) {
        int32_t value;
        if (scanf("%" SCNd32, &value) != 1) {
            return 1;
        }
        sum += value;
    }

    printf("%.2f\n", (double)sum / (double)count);
    return 0;
}