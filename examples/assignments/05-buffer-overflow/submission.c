#include <stdio.h>

int main(void) {
    char username[8];

    if (scanf("%s", username) != 1) {
        return 1;
    }

    printf("Hello %s\n", username);
    return 0;
}
