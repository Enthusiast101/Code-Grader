#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
    int32_t left;
    int32_t right;

    if (scanf("%" SCNd32 " %" SCNd32, &left, &right) != 2) {
        return 1;
    }

    int64_t sum = (int64_t)left + (int64_t)right;
    printf("%" PRId64 "\n", sum);
    return 0;
}
