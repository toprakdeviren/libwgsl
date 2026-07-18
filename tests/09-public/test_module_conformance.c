/**
 * §2 WGSL Module — conformance smoke tests.
 *
 * Pins the testable claims from `docs/wgsl/en/02_wgsl_module.md`
 * against the public API.  Spec rules the library does NOT enforce
 * today are not asserted here — see `docs/wgsl/missing.md` for the
 * gap list (range `@diagnostic`, filter-conflict rejection, severity
 * levels beyond `error`, diagnostic-rule name table).
 *
 * What this file pins:
 *
 *   §2 lead         — module grammar: directives + global decls +
 *                     global asserts + empty `;` separators in any
 *                     legal order
 *   §2 lead         — directives must precede declarations
 *   §2 lead         — `global_assert` (module-scope `const_assert`)
 *                     accepts trivially-true and rejects
 *                     trivially-false predicates
 *   §2.2 Errors     — a shader-generation error keeps the module from
 *                     being "incorporated into a pipeline": ok=0,
 *                     diagnostic count ≥ 1, module_json empty
 *   §2.3 Diagnostics— global `diagnostic(...)` filter directive accepts
 *                     the §2.3.2 filterable rules
 *                     (`derivative_uniformity`, `subgroup_uniformity`)
 *                     and accepts two non-conflicting globals at once
 *   §2.4 Limits     — implementations MUST accept shaders that meet the
 *                     minimum supported values.  Pins each of the
 *                     source-shape limits (struct members, fn params,
 *                     brace nesting, switch selectors, array ctor
 *                     elements, composite nesting) at exactly the
 *                     spec minimum.
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

static void dump_on_fail(const WGSLResult *r, const char *label) {
    if (wgsl_ok(r)) return;
    fprintf(stderr, "  [%s] unexpected diagnostics:\n", label);
    int n = wgsl_diagnostic_count(r);
    for (int i = 0; i < n && i < 5; i++) {
        const WGSLDiagnostic *d = wgsl_diagnostic(r, i);
        fprintf(stderr, "    L%u:%u %s\n",
                d ? d->line : 0, d ? d->column : 0,
                d && d->message ? d->message : "(null)");
    }
}

/* Helper for limit tests: run wgsl_check, expect ok=1, dump on fail. */
static void expect_ok(const char *src, const char *label) {
    WGSLResult *r = wgsl_check(src);
    dump_on_fail(r, label);
    CHECK(wgsl_ok(r), label /* must accept */);
    CHECK(wgsl_diagnostic_count(r) == 0, label /* zero diagnostics */);
    wgsl_free(r);
}

int main(void) {
    /* §2 lead - module grammar: directives + decls + global.
     *               asserts + empty `;` separators                */
    {
        const char *src =
            "diagnostic(off, derivative_uniformity);\n"
            "enable f16;\n"
            ";\n"
            "const A : i32 = 1;\n"
            ";\n"
            "alias Idx = u32;\n"
            "struct S { x : i32 }\n"
            "const_assert 2 > 1;\n"
            "fn f() -> i32 { return A; }\n"
            ";\n";
        WGSLResult *r = wgsl_check(src);
        dump_on_fail(r, "module grammar");
        CHECK(wgsl_ok(r),
              "§2 module grammar: directives + decls + asserts + ';' "
              "separators accepted");
        wgsl_free(r);
    }

    /* §2 lead - directives must precede declarations. */
    {
        const char *src =
            "const X : i32 = 1;\n"
            "enable f16;\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(!wgsl_ok(r),
              "§2: directive after declaration must be rejected");
        CHECK(wgsl_diagnostic_count(r) >= 1,
              "directive-after-decl: ≥1 diagnostic");
        wgsl_free(r);
    }

    /* §2 lead - global_assert. */
    {
        WGSLResult *r = wgsl_check("const_assert 2 > 1;\n");
        dump_on_fail(r, "global_assert true");
        CHECK(wgsl_ok(r), "global_assert (2 > 1): accepted");
        wgsl_free(r);
    }
    {
        WGSLResult *r = wgsl_check("const_assert 1 > 2;\n");
        CHECK(!wgsl_ok(r),
              "global_assert (1 > 2): rejected (trivially false)");
        CHECK(wgsl_diagnostic_count(r) >= 1,
              "false global_assert: ≥1 diagnostic");
        wgsl_free(r);
    }
    /* And: const_assert is also valid inside a function. */
    {
        const char *src =
            "const_assert 2 > 1;\n"
            "fn f() { const_assert 3 > 2; }\n";
        WGSLResult *r = wgsl_check(src);
        dump_on_fail(r, "module+fn const_assert");
        CHECK(wgsl_ok(r),
              "const_assert at module and fn scope both valid");
        wgsl_free(r);
    }

    /* §2.2 - shader-generation error: module is not "incorporated.
     *           into a pipeline".  Pin the public-API contract:   *
     *           ok=0, ≥1 diagnostic, module_json empty.           */
    {
        /* Resolve-time error (undeclared identifier). */
        WGSLResult *r = wgsl_check("fn f() { return zzz; }\n");
        CHECK(!wgsl_ok(r),
              "§2.2 shader-generation error: ok=0 on undeclared ident");
        CHECK(wgsl_diagnostic_count(r) >= 1,
              "§2.2: ≥1 diagnostic raised");
        const WGSLDiagnostic *d = wgsl_diagnostic(r, 0);
        CHECK(d && d->severity == WGSL_DIAG_ERROR,
              "§2.3 severity: first diagnostic is error");
        CHECK(wgsl_module_json(r)[0] == '\0',
              "§2.2: module_json empty when shader-generation error "
              "(would-be pipeline rejected)");
        wgsl_free(r);
    }

    /* §2.3 - global diagnostic filter accepts the §2.3.2.
     *           filterable rules                                  */
    {
        WGSLResult *r = wgsl_check(
            "diagnostic(off, derivative_uniformity);\n"
            "@vertex fn vs() -> @builtin(position) vec4f {\n"
            "  return vec4f(0);\n"
            "}\n");
        dump_on_fail(r, "global filter: derivative_uniformity");
        CHECK(wgsl_ok(r),
              "§2.3.2: diagnostic(off, derivative_uniformity) accepted");
        wgsl_free(r);
    }
    {
        WGSLResult *r = wgsl_check(
            "diagnostic(off, subgroup_uniformity);\n");
        dump_on_fail(r, "global filter: subgroup_uniformity");
        CHECK(wgsl_ok(r),
              "§2.3.2: diagnostic(off, subgroup_uniformity) accepted");
        wgsl_free(r);
    }
    {
        /* Two non-conflicting global filters: different rules + */
        /* different severities → must coexist (§2.3.3 note).    */
        WGSLResult *r = wgsl_check(
            "diagnostic(off,     derivative_uniformity);\n"
            "diagnostic(warning, subgroup_uniformity);\n");
        dump_on_fail(r, "two non-conflicting global filters");
        CHECK(wgsl_ok(r),
              "§2.3.3: two non-conflicting global filters accepted");
        wgsl_free(r);
    }

    /* §2.4 Limits - implementations MUST accept shaders that.
     *                   satisfy the minimum supported value.      *
     *                                                             *
     *   The lib doesn't impose tighter limits today, so each      *
     *   spec-minimum source must compile cleanly.  These pins     *
     *   catch a future regression that imposes a too-low cap.     */

    /* L1: struct with 1023 members. */
    {
        size_t cap = 1u << 20;          /* 1 MiB scratch is plenty.    */
        char *s = malloc(cap);
        int p = snprintf(s, cap, "struct Big {\n");
        for (size_t i = 0; i < 1023; i++)
            p += snprintf(s + p, cap - p, "  m%zu : i32,\n", i);
        p += snprintf(s + p, cap - p, "}\n");
        expect_ok(s, "§2.4 limit: struct with 1023 members");
        free(s);
    }

    /* L2: function with 255 parameters. */
    {
        size_t cap = 1u << 18;
        char *s = malloc(cap);
        int p = snprintf(s, cap, "fn many(");
        for (size_t i = 0; i < 255; i++)
            p += snprintf(s + p, cap - p, "%sp%zu : i32",
                          i ? ", " : "", i);
        p += snprintf(s + p, cap - p, ") -> i32 { return p0; }\n");
        expect_ok(s, "§2.4 limit: function with 255 parameters");
        free(s);
    }

    /* L3: brace nesting depth 127 in a function body. */
    {
        size_t cap = 1u << 14;
        char *s = malloc(cap);
        int p = snprintf(s, cap, "fn deep() {\n");
        for (size_t i = 0; i < 127; i++)
            p += snprintf(s + p, cap - p, "{\n");
        for (size_t i = 0; i < 127; i++)
            p += snprintf(s + p, cap - p, "}\n");
        p += snprintf(s + p, cap - p, "}\n");
        expect_ok(s, "§2.4 limit: brace nesting depth 127");
        free(s);
    }

    /* L4: switch with 1023 case selector values (1022 cases +     *
     *     1 default).                                             */
    {
        size_t cap = 1u << 17;
        char *s = malloc(cap);
        int p = snprintf(s, cap,
                         "fn sw(x : i32) -> i32 {\n"
                         "  switch x {\n");
        for (size_t i = 0; i < 1022; i++)
            p += snprintf(s + p, cap - p,
                          "    case %zu: { return %zu; }\n", i, i);
        p += snprintf(s + p, cap - p,
                      "    default: { return -1; }\n"
                      "  }\n"
                      "}\n");
        expect_ok(s, "§2.4 limit: switch with 1023 case-selector values");
        free(s);
    }

    /* L5: array constructor with 2047 elements. */
    {
        size_t cap = 1u << 16;
        char *s = malloc(cap);
        int p = snprintf(s, cap,
                         "fn f() { let a = array<i32, 2047>(");
        for (size_t i = 0; i < 2047; i++)
            p += snprintf(s + p, cap - p, "%s0", i ? "," : "");
        p += snprintf(s + p, cap - p, "); }\n");
        expect_ok(s, "§2.4 limit: array constructor with 2047 elements");
        free(s);
    }

    /* L6: composite type nesting depth 15 (alias chain).          *
     *                                                             *
     *   Each `array<L_{k-1}, 2>` adds one level of composite      *
     *   nesting, so chain length 15 hits the minimum.             */
    {
        size_t cap = 1u << 14;
        char *s = malloc(cap);
        int p = snprintf(s, cap, "alias L0 = i32;\n");
        for (size_t i = 1; i <= 15; i++)
            p += snprintf(s + p, cap - p,
                          "alias L%zu = array<L%zu, 2>;\n", i, i - 1);
        expect_ok(s, "§2.4 limit: composite nesting depth 15");
        free(s);
    }

    fprintf(stderr, "%s  test_module_conformance  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
