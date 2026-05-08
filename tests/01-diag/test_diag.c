/**
 * Phase 1 — diagnostic bag.
 *
 * Covers:
 *   - emit + count + at + has_error progression
 *   - severity tagging + error counting
 *   - line/column translation via WGSLSource
 *   - rule strings (filterable + unconditional)
 *   - long messages exceeding the stack scratch
 *   - destroy invalidates all pointers (sanity: re-init works)
 *   - end-of-file span (offset == length)
 */
#include "internal/diag.h"
#include "internal/source.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int fail = 0;

#define CHECK(cond, msg) do {                                          \
    if (!(cond)) {                                                     \
        fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);  \
        fail += 1;                                                     \
    }                                                                  \
} while (0)

int main(void) {
    /* Source that gives multiple lines so line/col translation is real. */
    const char *src_text =
        "fn main() {\n"          /* line 1, 0..11    LF at 11 → next is 12 */
        "  let x = 1;\n"         /* line 2, 12..24                          */
        "  let y = z;\n";        /* line 3, 25..37                          */
    WGSLSource src; wgsl_source_init(&src, src_text, strlen(src_text));

    WGSLDiagBag b; wgsl_diag_init(&b);

    /* Empty bag */
    CHECK(wgsl_diag_count(&b)     == 0,  "fresh: count 0");
    CHECK(wgsl_diag_has_error(&b) == 0,  "fresh: no error");
    CHECK(wgsl_diag_at(&b, 0)     == NULL, "fresh: at(0) is NULL");

    /* Emit one error spanning `z` on line 3.
     * "  let y = z;" — `z` is at column 11 (1-based).
     * Byte offset of `z`: line 3 starts at 25, plus column 11 - 1 = 10 → offset 35. */
    uint32_t z_offset = 35;
    int ok = wgsl_diag_emit_at(
        &b, &src, WGSL_DIAG_ERROR, z_offset, 1,
        NULL, "use of undeclared identifier '%s'", "z");
    CHECK(ok == 1,                          "emit returns 1");
    CHECK(wgsl_diag_count(&b)     == 1,    "count after emit");
    CHECK(wgsl_diag_has_error(&b) == 1,    "has_error true");

    const WGSLDiagnostic *d0 = wgsl_diag_at(&b, 0);
    CHECK(d0 != NULL,                              "at(0) non-null");
    CHECK(d0->severity   == WGSL_DIAG_ERROR,        "d0 severity");
    CHECK(d0->line       == 3,                      "d0 line");
    CHECK(d0->column     == 11,                     "d0 column");
    CHECK(d0->end_line   == 3,                      "d0 end_line");
    CHECK(d0->end_column == 12,                     "d0 end_column (exclusive)");
    CHECK(d0->rule       != NULL && *d0->rule == 0, "d0 unconditional rule = empty");
    CHECK(d0->message    != NULL,                   "d0 message non-null");
    CHECK(strcmp(d0->message, "use of undeclared identifier 'z'") == 0,
          "d0 message rendered");

    /* Emit a warning with a filterable rule. */
    ok = wgsl_diag_emit_at(
        &b, &src, WGSL_DIAG_WARNING, 0, 0,
        "derivative_uniformity",
        "potentially non-uniform control flow");
    CHECK(ok == 1, "emit warning ok");
    const WGSLDiagnostic *d1 = wgsl_diag_at(&b, 1);
    CHECK(d1->severity == WGSL_DIAG_WARNING,                 "d1 severity");
    CHECK(strcmp(d1->rule, "derivative_uniformity") == 0,    "d1 rule");
    CHECK(wgsl_diag_has_error(&b) == 1, "still has error from earlier d0");

    /* Long message exceeds the 256-byte stack scratch — force the
     * heap path inside emit_at. */
    char long_msg_input[600];
    memset(long_msg_input, 'x', sizeof long_msg_input - 1);
    long_msg_input[sizeof long_msg_input - 1] = '\0';
    ok = wgsl_diag_emit_at(
        &b, &src, WGSL_DIAG_INFO, 0, 0, NULL, "%s", long_msg_input);
    CHECK(ok == 1, "long message emit ok");
    const WGSLDiagnostic *d2 = wgsl_diag_at(&b, 2);
    CHECK(d2 != NULL && strlen(d2->message) == 599, "long message length preserved");
    CHECK(d2->message[598] == 'x' && d2->message[599] == '\0',
          "long message tail intact");

    /* End-of-file span (offset == length).  Should not fail. */
    ok = wgsl_diag_emit_at(
        &b, &src, WGSL_DIAG_HINT,
        (uint32_t)src.length, 0, NULL,
        "EOF reached");
    CHECK(ok == 1, "EOF span emit ok");
    const WGSLDiagnostic *d3 = wgsl_diag_at(&b, 3);
    CHECK(d3 != NULL,                "d3 exists");
    CHECK(d3->severity == WGSL_DIAG_HINT, "d3 severity");
    /* Should land at line N, col == bytes-past-line-start + 1 */
    CHECK(d3->line >= 1, "EOF line valid");

    /* Capacity growth — emit > initial capacity (8) entries to force realloc.
     *
     * The contract: `WGSLDiagnostic` records live in a heap-realloc-able
     * array, so pointers to records (`d0`, `d1`, ...) become invalid when
     * the array grows.  But the **strings** records point at live in a
     * separate arena that never realloc-invalidates — so the message and
     * rule pointers we captured before the growth must remain readable.
     *
     * Save the string pointers *now*, then trigger growth, then re-validate. */
    const char *d0_msg_before  = d0->message;
    const char *d1_rule_before = d1->rule;

    for (int i = 0; i < 16; i++) {
        ok = wgsl_diag_emit_at(
            &b, &src, WGSL_DIAG_INFO, 0, 1, NULL, "synthetic %d", i);
        CHECK(ok == 1, "growth emit ok");
    }
    CHECK(wgsl_diag_count(&b) == 4 + 16, "count after growth");

    /* Saved string pointers (arena-owned) survive the items-array realloc. */
    CHECK(strcmp(d0_msg_before, "use of undeclared identifier 'z'") == 0,
          "d0 message string stable across items realloc");
    CHECK(strcmp(d1_rule_before, "derivative_uniformity") == 0,
          "d1 rule string stable across items realloc");

    /* Re-fetch records — the new pointers must reflect the same content. */
    const WGSLDiagnostic *d0_again = wgsl_diag_at(&b, 0);
    const WGSLDiagnostic *d1_again = wgsl_diag_at(&b, 1);
    CHECK(d0_again->severity == WGSL_DIAG_ERROR &&
          d0_again->line == 3 && d0_again->column == 11,
          "d0 record consistent after re-fetch");
    CHECK(strcmp(d1_again->rule, "derivative_uniformity") == 0,
          "d1 rule consistent after re-fetch");

    /* Destroy + reuse. */
    wgsl_diag_destroy(&b);
    CHECK(wgsl_diag_count(&b) == 0, "post-destroy: 0");
    CHECK(wgsl_diag_has_error(&b) == 0, "post-destroy: no error");

    /* Re-init and emit one. */
    wgsl_diag_init(&b);
    ok = wgsl_diag_emit_at(&b, &src, WGSL_DIAG_ERROR, 0, 1, NULL, "again");
    CHECK(ok == 1, "post-destroy reinit emit");
    CHECK(wgsl_diag_count(&b) == 1, "post-destroy count");
    wgsl_diag_destroy(&b);

    wgsl_source_destroy(&src);

    if (fail == 0) {
        printf("PASS  test_diag\n");
        return 0;
    }
    fprintf(stderr, "FAIL  test_diag  %d check(s) failed\n", fail);
    return 1;
}
