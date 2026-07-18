/**
 * §4 Directives — conformance smoke tests.
 *
 * Pins the testable claims from `docs/wgsl/en/04_directives.md`
 * against the public API.  This file covers directive syntax, name
 * tables, placement, and diagnostic filters; feature behavior itself is
 * covered in the dedicated §4 extension conformance tests.
 *
 * What this file pins:
 *
 *   §4 lead       — a module of directives only is valid; directives
 *                   precede declarations and const-asserts; a
 *                   declaration before an `enable` / `requires`
 *                   directive is a shader-generation error
 *   §4.1.1 enable — single / list / trailing-comma / redundant /
 *                   separate-directive forms accepted; every
 *                   spec-listed extension name compiles
 *   §4.1.2 require — single / list / trailing-comma / all 11
 *                   spec-listed language extension names compile
 *   §4.2 diagnostic — every severity-control name (off/warning/
 *                   error/info) against every filterable rule
 *                   (derivative_uniformity, subgroup_uniformity)
 *                   accepted; all three directive kinds coexist in
 *                   any order before declarations
 */
#include "wgsl.h"

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

static int has_diag_with(const WGSLResult *r, const char *needle) {
    int n = wgsl_diagnostic_count(r);
    for (int i = 0; i < n; i++) {
        const WGSLDiagnostic *d = wgsl_diagnostic(r, i);
        if (d && d->message && strstr(d->message, needle)) return 1;
    }
    return 0;
}

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

static void expect_ok(const char *src, const char *label) {
    WGSLResult *r = wgsl_check(src);
    dump_on_fail(r, label);
    CHECK(wgsl_ok(r), label);
    CHECK(wgsl_diagnostic_count(r) == 0, label);
    wgsl_free(r);
}

static void expect_reject(const char *src, const char *needle,
                          const char *label) {
    WGSLResult *r = wgsl_check(src);
    CHECK(!wgsl_ok(r), label);
    CHECK(wgsl_diagnostic_count(r) >= 1, label);
    if (needle) CHECK(has_diag_with(r, needle), label);
    wgsl_free(r);
}

int main(void) {
    /* §4 lead - placement and presence. */
    expect_ok("enable f16;\n",
              "§4: directives-only module is valid");
    expect_ok("enable f16;\nconst X = 1;\n",
              "§4: directive followed by declaration");
    expect_reject("const X = 1;\nenable f16;\n",
                  "before any declarations",
                  "§4: declaration before `enable` → shader-generation error");
    expect_reject("const_assert true; enable f16;", "enable",
                  "§4: const_assert before `enable` → shader-generation error");
    expect_reject("struct S { x: i32 } requires readonly_and_readwrite_storage_textures;", "requires",
                  "§4: declaration before `requires` → shader-generation error");

    /* §4.1.1 - enable directive forms. */
    expect_ok("enable f16;\n",
              "§4.1.1: enable f16 single");
    expect_ok("enable f16, clip_distances;\n",
              "§4.1.1: enable list with two names");
    expect_ok("enable f16,\n"
              "       clip_distances,\n"
              "       dual_source_blending,\n"
              "       subgroups,\n"
              "       primitive_index;\n",
              "§4.1.1: every spec-listed enable extension compiles");
    expect_ok("enable f16,;\n",
              "§4.1.1: enable list trailing comma allowed");
    expect_ok("enable f16, f16;\n",
              "§4.1.1: redundant enable name within one directive");
    expect_ok("enable f16;\nenable f16;\n",
              "§4.1.1: redundant enable across two directives (per spec example)");
    expect_ok("enable f16;\nenable clip_distances;\n",
              "§4.1.1: separate enable directives");
    expect_reject("enable f16\n",
                  ";",
                  "§4.1.1: missing semicolon after enable list rejected");

    /* §4.1.2 - requires directive forms. */
    expect_ok("requires readonly_and_readwrite_storage_textures;\n",
              "§4.1.2: requires single");
    expect_ok("requires readonly_and_readwrite_storage_textures,"
              "         packed_4x8_integer_dot_product;\n",
              "§4.1.2: requires list with two names");
    /* All 11 spec-listed language extension names in one directive. */
    expect_ok(
        "requires readonly_and_readwrite_storage_textures,\n"
        "         packed_4x8_integer_dot_product,\n"
        "         unrestricted_pointer_parameters,\n"
        "         pointer_composite_access,\n"
        "         uniform_buffer_standard_layout,\n"
        "         subgroup_id,\n"
        "         subgroup_uniformity,\n"
        "         texture_and_sampler_let,\n"
        "         texture_formats_tier1,\n"
        "         linear_indexing,\n"
        "         immediate_data;\n",
        "§4.1.2: every spec-listed language extension name compiles");
    expect_ok("requires texture_formats_tier1,;\n",
              "§4.1.2: requires list trailing comma allowed");
    expect_ok("requires texture_formats_tier1;\n"
              "requires texture_formats_tier1;\n",
              "§4.1.2: redundant requires directives accepted");

    /* §4.2 - global diagnostic filter directive.
     *                                                             *
     * Every (severity_control × filterable_rule) pair compiles    *
     * at module scope.  §4.2 says the diagnostic directive is     *
     * spelled like the attribute form, without `@` and with `;`.  */
    {
        const char *severities[] = { "off", "warning", "error", "info" };
        const char *rules[] = { "derivative_uniformity", "subgroup_uniformity" };
        for (size_t s = 0; s < sizeof severities / sizeof severities[0]; s++) {
            for (size_t k = 0; k < sizeof rules / sizeof rules[0]; k++) {
                char src[128];
                snprintf(src, sizeof src,
                         "diagnostic(%s, %s);\n", severities[s], rules[k]);
                char label[128];
                snprintf(label, sizeof label,
                         "§4.2: diagnostic(%s, %s) accepted",
                         severities[s], rules[k]);
                expect_ok(src, label);
            }
        }
    }

    /* Mixed directives - all three kinds coexist before decls. */
    expect_ok(
        "diagnostic(off, derivative_uniformity);\n"
        "enable f16;\n"
        "requires texture_formats_tier1;\n"
        "const X : i32 = 1;\n",
        "§4: diagnostic + enable + requires + decl in spec order");
    expect_ok(
        "enable f16;\n"
        "diagnostic(off, derivative_uniformity);\n"
        "requires texture_formats_tier1;\n"
        "const X : i32 = 1;\n",
        "§4: directives may appear in any order among themselves");

    fprintf(stderr, "%s  test_directives_conformance  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
