// Instrumentation shim for capturing a function-call event stream.
//
// Linked against programs compiled with `-finstrument-functions`; the compiler
// calls these hooks on every function entry/exit. Each entry logs
// `E <function> <timestamp_us> <depth>` to the file named by $BIBE_TRACE.
//
// Compile WITHOUT -finstrument-functions (and the hooks are marked
// no_instrument_function) so the shim does not trace itself.

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static FILE *g_log = NULL;
static int g_depth = 0;

__attribute__((no_instrument_function))
static FILE *log_file(void) {
    if (!g_log) {
        const char *path = getenv("BIBE_TRACE");
        g_log = fopen(path ? path : "trace.log", "w");
    }
    return g_log;
}

__attribute__((no_instrument_function))
static uint64_t now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

__attribute__((no_instrument_function))
void __cyg_profile_func_enter(void *this_fn, void *call_site) {
    (void)call_site;
    FILE *f = log_file();
    if (!f) return;

    Dl_info info;
    const char *name = "unknown";
    if (dladdr(this_fn, &info) && info.dli_sname) {
        name = info.dli_sname;
    }
    fprintf(f, "E %s %llu %d\n", name, (unsigned long long)now_us(), g_depth);
    fflush(f);
    g_depth++;
}

__attribute__((no_instrument_function))
void __cyg_profile_func_exit(void *this_fn, void *call_site) {
    (void)this_fn;
    (void)call_site;
    if (g_depth > 0) g_depth--;
}
