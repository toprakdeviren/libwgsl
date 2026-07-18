/**
 * libFuzzer entry for wgsl_check.
 *
 * Built only via `make fuzz` with -fsanitize=fuzzer.  Not part of
 * the default test gate.
 */
#include "wgsl.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size == 0 || size > 65536) return 0;
    char *src = (char *)malloc(size + 1);
    if (!src) return 0;
    memcpy(src, data, size);
    src[size] = '\0';

    WGSLResult *r = wgsl_check(src);
    if (r) {
        (void)wgsl_ok(r);
        (void)wgsl_diagnostic_count(r);
        (void)wgsl_module_json(r);
        (void)wgsl_error(r);
        wgsl_free(r);
    }

    /* Also poke the formatter — must not crash on garbage. */
    char *fmt = wgsl_format_n(src, size);
    if (fmt) wgsl_free_string(fmt);

    free(src);
    return 0;
}
