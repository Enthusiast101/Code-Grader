#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int count;
    long long target;
    if (scanf("%d %lld", &count, &target) != 2 || count < 0 || count > 20000) {
        return 1;
    }

    long long *values = malloc((size_t)count * sizeof(*values));
    if (values == NULL && count > 0) {
        return 1;
    }

    for (int i = 0; i < count; ++i) {
        if (scanf("%lld", &values[i]) != 1) {
            free(values);
            return 1;
        }
    }

    long long triples = 0;
    for (int i = 0; i < count; ++i) {
        for (int j = i + 1; j < count; ++j) {
            for (int k = j + 1; k < count; ++k) {
                if (values[i] + values[j] + values[k] == target) {
                    ++triples;
                }
            }
        }
    }

    printf("%lld\n", triples);
    free(values);
    return 0;
}
