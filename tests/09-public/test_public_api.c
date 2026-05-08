/**
 * Phase 9 Iter A — public C API foundation.
 *
 * Covers:
 *   - process lifecycle: wgsl_init / wgsl_shutdown / wgsl_spec_pin /
 *     wgsl_unicode_version
 *   - wgsl_check / wgsl_check_n on:
 *       - empty source       → ok = 1, no diags
 *       - simple shader      → ok = 1, no diags
 *       - parse error        → ok = 0, ≥1 error diag
 *       - resolve error      → ok = 0, "undeclared" diag visible via
 *                              wgsl_error()
 *   - NULL safety: wgsl_ok(NULL), wgsl_diagnostic_count(NULL),
 *                  wgsl_diagnostic(NULL, 0), wgsl_free(NULL),
 *                  wgsl_error(NULL)
 *   - wgsl_diagnostic returns a stable pointer with LSP-shape
 *     line/column/end_line/end_column
 *   - wgsl_lex / wgsl_lex_free emit raw tokens with kind classification
 *   - wgsl_module_json / wgsl_hover_at / wgsl_definition_at stubs
 *     return defined "absent" values (Iter B / C will fill in)
 */
#include "wgsl.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail = 0;
#define CHECK(cond, msg) do {                                          \
    if (!(cond)) {                                                     \
        fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);  \
        fail += 1;                                                     \
    }                                                                  \
} while (0)

int main(void) {
    /* ── Process lifecycle ────────────────────────────────────── */
    wgsl_init();
    CHECK(wgsl_spec_pin() != NULL, "spec pin non-NULL");
    CHECK(strcmp(wgsl_spec_pin(), WGSL_SPEC_PIN) == 0, "spec pin matches macro");
    CHECK(wgsl_unicode_version() != NULL, "unicode version non-NULL");
    CHECK(strcmp(wgsl_unicode_version(), "17.0.0") == 0,
          "unicode version is 17.0.0");

    /* ── NULL safety ───────────────────────────────────────────── */
    CHECK(wgsl_ok(NULL) == 0, "wgsl_ok(NULL) → 0");
    CHECK(wgsl_diagnostic_count(NULL) == 0, "diag_count(NULL) → 0");
    CHECK(wgsl_diagnostic(NULL, 0) == NULL, "diagnostic(NULL,0) → NULL");
    CHECK(wgsl_error(NULL) != NULL, "error(NULL) → non-NULL (empty)");
    CHECK(wgsl_error(NULL)[0] == '\0', "error(NULL) is empty string");
    wgsl_free(NULL);  /* must not crash */

    /* ── Empty source ─────────────────────────────────────────── */
    {
        WGSLResult *r = wgsl_check("");
        CHECK(r != NULL, "empty src: non-NULL result");
        CHECK(wgsl_ok(r) == 1, "empty src: ok = 1");
        CHECK(wgsl_diagnostic_count(r) == 0, "empty src: 0 diags");
        CHECK(wgsl_error(r)[0] == '\0', "empty src: error empty");
        wgsl_free(r);
    }

    /* ── NULL source ──────────────────────────────────────────── */
    {
        WGSLResult *r = wgsl_check(NULL);
        CHECK(r != NULL, "NULL src: non-NULL result");
        CHECK(wgsl_ok(r) == 1, "NULL src: ok = 1 (treated as empty)");
        wgsl_free(r);
    }

    /* ── Simple valid shader ──────────────────────────────────── */
    {
        const char *src =
            "@compute @workgroup_size(64)\n"
            "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
            "  let x: i32 = 1 + 2;\n"
            "  _ = x;\n"
            "}\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r) == 1, "simple shader: ok");
        CHECK(wgsl_diagnostic_count(r) == 0, "simple shader: 0 diags");
        wgsl_free(r);
    }

    /* ── Parse error ──────────────────────────────────────────── */
    {
        WGSLResult *r = wgsl_check("fn { }");
        CHECK(wgsl_ok(r) == 0, "parse error: ok = 0");
        CHECK(wgsl_diagnostic_count(r) >= 1, "parse error: ≥1 diag");
        CHECK(wgsl_error(r)[0] != '\0', "parse error: error msg non-empty");
        const WGSLDiagnostic *d = wgsl_diagnostic(r, 0);
        CHECK(d != NULL, "diagnostic[0] non-NULL");
        if (d) {
            CHECK(d->severity == WGSL_DIAG_ERROR, "diag[0] is error");
            CHECK(d->line >= 1 && d->column >= 1, "1-based line/col");
            CHECK(d->message != NULL && d->message[0] != '\0',
                  "diag has message");
        }
        wgsl_free(r);
    }

    /* ── Resolve error ────────────────────────────────────────── */
    {
        WGSLResult *r = wgsl_check("fn f() { _ = ghost; }\n");
        CHECK(wgsl_ok(r) == 0, "resolve err: ok = 0");
        CHECK(strstr(wgsl_error(r), "undeclared") != NULL,
              "resolve err: 'undeclared' in error msg");
        wgsl_free(r);
    }

    /* ── Diagnostic spans are LSP-shape (1-based, end exclusive) — */
    {
        const char *src = "fn f() {\n  let x = ghost;\n}\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r) == 0, "ghost: ok = 0");
        const WGSLDiagnostic *d = NULL;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            const WGSLDiagnostic *cand = wgsl_diagnostic(r, i);
            if (cand && strstr(cand->message, "undeclared")) { d = cand; break; }
        }
        CHECK(d != NULL, "found undeclared diag");
        if (d) {
            /* "ghost" starts at line 2, column 11 (after "  let x = "). */
            CHECK(d->line == 2,        "ghost line = 2");
            CHECK(d->column == 11,     "ghost column = 11");
            CHECK(d->end_line == 2,    "ghost end_line = 2");
            CHECK(d->end_column == 16, "ghost end_column = 16 (exclusive)");
        }
        wgsl_free(r);
    }

    /* ── wgsl_lex emits classified tokens ─────────────────────── */
    {
        const char *src = "let x = 1;";
        WGSLLexToken *tokens = NULL;
        int count = 0;
        int rc = wgsl_lex(src, &tokens, &count);
        CHECK(rc == 1, "wgsl_lex: rc = 1");
        CHECK(count > 0, "wgsl_lex: count > 0");
        /* First non-whitespace token is `let`. */
        int saw_let = 0, saw_int = 0;
        for (int i = 0; i < count; i++) {
            if (tokens[i].kind == WGSL_LEX_KEYWORD &&
                tokens[i].length == 3 &&
                memcmp(src + tokens[i].offset, "let", 3) == 0)
            {
                saw_let = 1;
            }
            if (tokens[i].kind == WGSL_LEX_NUMBER) saw_int = 1;
        }
        CHECK(saw_let, "wgsl_lex: saw `let` as KEYWORD");
        CHECK(saw_int, "wgsl_lex: saw NUMBER token");
        wgsl_lex_free(tokens);
    }

    /* ── wgsl_module_json / hover / goto-def absent-result paths — */
    {
        WGSLResult *r = wgsl_check("fn f() {}\n");
        CHECK(wgsl_module_json(r) != NULL, "module_json non-NULL");
        /* `f()` is not an entry point, no resources, no structs, no
         * overrides — JSON has empty arrays but is still well-formed. */
        CHECK(strstr(wgsl_module_json(r), "\"entry_points\":[]") != NULL,
              "module_json: empty entry_points");
        WGSLHover h = wgsl_hover_at(r, 0);
        CHECK(h.present == 0, "hover @0: absent");
        WGSLDefinition d = wgsl_definition_at(r, 0);
        CHECK(d.present == 0, "definition @0: absent");
        wgsl_free(r);
    }

    /* ── wgsl_check_n (non-NUL-terminated) ───────────────────── */
    {
        const char src_buf[] = "fn x()" "GARBAGE";
        WGSLResult *r = wgsl_check_n(src_buf, 6);   /* "fn x()" */
        /* Has a missing-body parse error but exercises the byte-range entry. */
        (void)wgsl_ok(r);
        CHECK(wgsl_diagnostic_count(r) >= 1, "check_n: missing-body diag");
        wgsl_free(r);
    }

    wgsl_shutdown();

    fprintf(stderr, "%s  test_public_api  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
