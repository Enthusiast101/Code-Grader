#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    int *values = malloc(64 * sizeof(*values));
    if (values == NULL) {
        return 1;
    }

    for (int i = 0; i < 64; ++i) {
        values[i] = i * 2;
    }

    char *message = malloc(32);
    if (message == NULL) {
        free(values);
        return 1;
    }
    strcpy(message, "grading sample");
    printf("%s\n", message);

    free(values);
    volatile int stale_value = values[20];
    (void)stale_value;

    return 0;
}