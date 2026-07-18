/**
 * @file golden_ref.c — golden reference runner helpers.
 *
 * Shared logic for comparing `wgsl_interp` buffer dumps against a
 * structured expectation.  The public C API is thin; the bulk of the
 * suite lives in tests/15-golden/ as (shader, expect) fixtures plus
 * the native test harness.
 */
#include "wgsl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Extract the `"buffers":[...]` slice (through `"anomalies"`). */
static int buffers_slice(const char *json, const char **out, size_t *out_len) {
    if (!json) return 0;
    const char *b = strstr(json, "\"buffers\"");
    if (!b) return 0;
    const char *e = strstr(b, "\"anomalies\"");
    *out = b;
    *out_len = e ? (size_t)(e - b) : strlen(b);
    return 1;
}

/**
 * Run interp and verify:
 *   - ok matches expect_ok (if expect_ok >= 0)
 *   - every needle appears in the JSON
 *   - two runs produce identical buffer slices (determinism)
 *
 * needles: NULL-terminated array of substrings (may be NULL).
 * Returns 1 on pass, 0 on fail (prints to stderr when label != NULL).
 */
int wgsl_golden_run(const char *label,
                    const char *src, const char *entry,
                    unsigned len,
                    int expect_ok,
                    const char *const *needles)
{
    char *j1 = wgsl_interp(src, entry, 0, 0, 0, len, NULL);
    if (!j1) {
        if (label) fprintf(stderr, "FAIL  %s: interp NULL\n", label);
        return 0;
    }
    int ok = 1;
    if (expect_ok == 1 && !strstr(j1, "\"ok\":true")) {
        if (label) fprintf(stderr, "FAIL  %s: expected ok:true\n", label);
        ok = 0;
    }
    if (expect_ok == 0 && strstr(j1, "\"ok\":true")) {
        if (label) fprintf(stderr, "FAIL  %s: expected ok:false\n", label);
        ok = 0;
    }
    if (needles) {
        for (int i = 0; needles[i]; i++) {
            if (!strstr(j1, needles[i])) {
                if (label)
                    fprintf(stderr, "FAIL  %s: missing needle %s\n",
                            label, needles[i]);
                ok = 0;
            }
        }
    }
    char *j2 = wgsl_interp(src, entry, 0, 0, 0, len, NULL);
    if (!j2) {
        if (label) fprintf(stderr, "FAIL  %s: second interp NULL\n", label);
        ok = 0;
    } else {
        const char *b1 = NULL, *b2 = NULL;
        size_t l1 = 0, l2 = 0;
        if (!buffers_slice(j1, &b1, &l1) || !buffers_slice(j2, &b2, &l2) ||
            l1 != l2 || memcmp(b1, b2, l1) != 0) {
            if (label)
                fprintf(stderr, "FAIL  %s: non-deterministic buffers\n", label);
            ok = 0;
        }
        wgsl_free_string(j2);
    }
    wgsl_free_string(j1);
    return ok;
}
