// A program with a use-after-free. AddressSanitizer reports the access in
// `do_use` and the free in `do_free`; the free and use are named functions so
// the ASan-report -> event-index join is unambiguous.

#include <stdio.h>
#include <stdlib.h>

char *allocate(void) { return (char *)malloc(16); }
void do_free(char *p) { free(p); }
char do_use(char *p) { return p[0]; } // read after free -> the symptom

void warmup(void) {
    volatile int x = 0;
    for (int i = 0; i < 3; i++) x += i;
}

int main(void) {
    warmup();
    char *p = allocate();
    p[0] = 'A';
    do_free(p); // the cause
    warmup();
    char c = do_use(p); // use-after-free crash here
    printf("%c\n", c);
    return 0;
}
