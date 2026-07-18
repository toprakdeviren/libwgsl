/**
 * §5.3 — exact diagnostic assertions (line / column / severity / message).
 *
 * Replaces vacuous `!has_error_with` style checks with precise positions
 * for a small, stable set of violations.
 */
#include "wgsl.h"

#include <stdio.h>
#include <string.h>

static int fail = 0;
#define CHECK(cond, msg) do {                                          \
    if (!(cond)) {                                                     \
        fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);  \
        fail += 1;                                                     \
    }                                                                  \
} while (0)

typedef struct {
    const char      *label;
    const char      *src;
    WGSLDiagSeverity severity;
    uint32_t         line;
    uint32_t         column;
    const char      *msg_sub;   /* required substring; NULL = any message */
} ExactCase;

static int find_exact(const WGSLResult *r, const ExactCase *c) {
    int n = wgsl_diagnostic_count(r);
    for (int i = 0; i < n; i++) {
        const WGSLDiagnostic *d = wgsl_diagnostic(r, i);
        if (!d) continue;
        if (d->severity != c->severity) continue;
        if (d->line != c->line) continue;
        if (d->column != c->column) continue;
        if (c->msg_sub && (!d->message || !strstr(d->message, c->msg_sub)))
            continue;
        return 1;
    }
    return 0;
}

static void dump_diags(const WGSLResult *r) {
    int n = wgsl_diagnostic_count(r);
    for (int i = 0; i < n; i++) {
        const WGSLDiagnostic *d = wgsl_diagnostic(r, i);
        if (!d) continue;
        fprintf(stderr, "  diag[%d] sev=%d L%u:%u %s\n",
                i, (int)d->severity, d->line, d->column,
                d->message ? d->message : "?");
    }
}

int main(void) {
    /* Positions are 1-based (LSP).  Sources are crafted so the first
     * error lands on a known line/column. */
    static const ExactCase cases[] = {
        {
            "parse: bare fn",
            "fn { }\n",
            WGSL_DIAG_ERROR, 1, 4,
            "expected",
        },
        {
            "type: i32 = true",
            "fn f() {\n  var x: i32 = true;\n}\n",
            WGSL_DIAG_ERROR, 2, 16,
            "cannot be converted",
        },
        {
            "type: i32 + u32",
            "fn f() {\n  let x = 1i + 1u;\n}\n",
            WGSL_DIAG_ERROR, 2, 11,
            "no implicit conversion",
        },
        {
            "resolve: undeclared",
            "fn f() {\n  _ = ghost;\n}\n",
            WGSL_DIAG_ERROR, 2, 7,
            "undeclared",
        },
        {
            "range: u32 = -1",
            "fn f() {\n  let x: u32 = -1;\n}\n",
            WGSL_DIAG_ERROR, 2, 16,
            "out of range",
        },
    };

    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        const ExactCase *c = &cases[i];
        WGSLResult *r = wgsl_check(c->src);
        CHECK(!wgsl_ok(r), c->label);
        int found = find_exact(r, c);
        CHECK(found, c->label);
        if (!found) {
            fprintf(stderr, "  wanted sev=%d L%u:%u ~'%s'\n",
                    (int)c->severity, c->line, c->column,
                    c->msg_sub ? c->msg_sub : "*");
            dump_diags(r);
        }
        wgsl_free(r);
    }

    /* Positive control: clean shader has zero diagnostics. */
    {
        WGSLResult *r = wgsl_check("fn f() { let x: i32 = 1; _ = x; }\n");
        CHECK(wgsl_ok(r), "clean: ok");
        CHECK(wgsl_diagnostic_count(r) == 0, "clean: 0 diags");
        wgsl_free(r);
    }

    if (fail == 0) {
        printf("PASS  test_exact_diagnostics  (0 failures)\n");
        return 0;
    }
    fprintf(stderr, "FAIL  test_exact_diagnostics  (%d failures)\n", fail);
    return 1;
}
