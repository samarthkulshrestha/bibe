// A clean program: same functions as uaf.c but the use happens before the
// free, so AddressSanitizer stays quiet and the trace is labeled normal.

#include <stdio.h>
#include <stdlib.h>

char *allocate(void) { return (char *)malloc(16); }
void do_free(char *p) { free(p); }
char do_use(char *p) { return p[0]; }

void warmup(void) {
    volatile int x = 0;
    for (int i = 0; i < 3; i++) x += i;
}

int main(void) {
    warmup();
    char *p = allocate();
    p[0] = 'A';
    char c = do_use(p); // use before free -> fine
    do_free(p);
    printf("%c\n", c);
    return 0;
}
