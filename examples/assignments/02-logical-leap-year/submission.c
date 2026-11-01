#include <stdio.h>

int is_leap_year(int year) {
    return year % 4 == 0;
}

int main(void) {
    int year;
    if (scanf("%d", &year) != 1) {
        return 1;
    }

    puts(is_leap_year(year) ? "LEAP" : "COMMON");
    return 0;
}
