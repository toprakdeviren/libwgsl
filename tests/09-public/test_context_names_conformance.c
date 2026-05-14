/**
 * §3.8 Context-Dependent Names — name-table conformance.
 *
 * §3.8 splits context-dependent names into 8 sub-sections, each
 * listing the closed set of valid spellings.  This file pins the
 * name-table enforcement: every spec-listed name compiles, and any
 * unrecognized single-token name in the same context is rejected.
 *
 * Scope:
 *   §3.8.3 — diagnostic rule names (`derivative_uniformity`,
 *            `subgroup_uniformity`)
 *   §3.8.5 — enable / requires extension names (5 enable + 11
 *            language extensions)
 *   §3.8.6 — interpolation type names (3 entries)
 *   §3.8.7 — interpolation sampling names (5 entries)
 *
 * Out of scope here:
 *   §3.8.2 — builtin value names: the name table is enforced
 *            (`@builtin(garbage)` is rejected), but the full
 *            §13.3.1.1 stage/direction/type/extension table is
 *            a separate spec-conformance bucket (see missing.md).
 *   §3.8.1 — attribute names: covered by the `unknown attribute`
 *            path in validate.c and existing parser tests.
 *   §3.8.4 — severity-control names: already covered in
 *            test_module_conformance.c and test_directives_conformance.c.
 *   §3.8.8 — swizzle names: validated by the lexer / type-check
 *            paths (mixed swizzle is a separate documented gap).
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

static void expect_ok_with_diag(const char *src, const char *needle,
                                const char *label) {
    WGSLResult *r = wgsl_check(src);
    dump_on_fail(r, label);
    CHECK(wgsl_ok(r), label);
    CHECK(wgsl_diagnostic_count(r) >= 1, label);
    if (needle) CHECK(has_diag_with(r, needle), label);
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
    /* ── §3.8.5 — enable extension names ───────────────────────── */
    {
        const char *names[] = {
            "f16",
            "clip_distances",
            "dual_source_blending",
            "subgroups",
            "primitive_index",
        };
        for (size_t i = 0; i < sizeof names / sizeof names[0]; i++) {
            char src[64], label[96];
            snprintf(src, sizeof src, "enable %s;\n", names[i]);
            snprintf(label, sizeof label,
                     "§3.8.5: enable %s accepted", names[i]);
            expect_ok(src, label);
        }
    }
    expect_reject("enable bogus;\n",
                  "unknown extension",
                  "§3.8.5: `enable bogus;` rejected with 'unknown extension'");
    expect_reject("enable f17;\n",
                  "unknown extension",
                  "§3.8.5: `enable f17;` rejected (off-by-one near 'f16')");

    /* ── §3.8.5 — language extension names ─────────────────────── */
    {
        const char *names[] = {
            "readonly_and_readwrite_storage_textures",
            "packed_4x8_integer_dot_product",
            "unrestricted_pointer_parameters",
            "pointer_composite_access",
            "uniform_buffer_standard_layout",
            "subgroup_id",
            "subgroup_uniformity",
            "texture_and_sampler_let",
            "texture_formats_tier1",
            "linear_indexing",
            "immediate_data",
        };
        for (size_t i = 0; i < sizeof names / sizeof names[0]; i++) {
            char src[96], label[128];
            snprintf(src, sizeof src, "requires %s;\n", names[i]);
            snprintf(label, sizeof label,
                     "§3.8.5: requires %s accepted", names[i]);
            expect_ok(src, label);
        }
    }
    expect_reject("requires bogus;\n",
                  "unknown language feature",
                  "§3.8.5: `requires bogus;` rejected");
    expect_reject("requires immediate_dat;\n",
                  "unknown language feature",
                  "§3.8.5: typo near 'immediate_data' rejected");

    /* ── §3.8.3 — diagnostic rule names ────────────────────────── */
    expect_ok("diagnostic(off, derivative_uniformity);\n",
              "§3.8.3: diagnostic(_, derivative_uniformity) accepted");
    expect_ok("diagnostic(off, subgroup_uniformity);\n",
              "§3.8.3: diagnostic(_, subgroup_uniformity) accepted");
    expect_ok_with_diag("diagnostic(off, garbage);\n",
                        "unknown diagnostic rule",
                        "§3.8.3: unrecognized single-token diagnostic rule accepted with warning");
    expect_ok_with_diag("diagnostic(off, derivative_uniformitt);\n",
                        "unknown diagnostic rule",
                        "§3.8.3: typo in diagnostic rule accepted with warning");
    /* §2.3.2: unrecognized multi-token rule "may" trigger a
     * diagnostic — the lib silently accepts.  Pin this so a future
     * stricter check doesn't regress without intent. */
    expect_ok("diagnostic(off, vendor.foo);\n",
              "§3.8.3: unrecognized multi-token rule silently accepted");

    /* ── §3.8.6 — interpolation type names ─────────────────────── *
     *                                                             *
     * `@interpolate` lives on entry-point I/O.  We exercise the    *
     * names on a fragment input slot — that path doesn't pull in   *
     * the §11.4 "vertex must return @builtin(position)" rule.      */
    {
        const char *types[] = { "perspective", "linear", "flat" };
        for (size_t i = 0; i < sizeof types / sizeof types[0]; i++) {
            char src[256], label[128];
            snprintf(src, sizeof src,
                "@fragment fn fs(@location(0) @interpolate(%s) v: f32) -> "
                "@location(0) vec4<f32> { return vec4<f32>(0.0); }\n",
                types[i]);
            snprintf(label, sizeof label,
                     "§3.8.6: @interpolate(%s) accepted", types[i]);
            expect_ok(src, label);
        }
    }
    expect_reject(
        "@fragment fn fs(@location(0) @interpolate(garbage) v: f32) -> "
        "@location(0) vec4<f32> { return vec4<f32>(0.0); }\n",
        "unknown interpolation type",
        "§3.8.6: @interpolate(garbage) rejected");
    expect_reject(
        "@fragment fn fs(@location(0) @interpolate(prespective) v: f32) -> "
        "@location(0) vec4<f32> { return vec4<f32>(0.0); }\n",
        "unknown interpolation type",
        "§3.8.6: typo @interpolate(prespective) rejected");

    /* ── §3.8.7 — interpolation sampling names ─────────────────── */
    {
        /* `sample` requires a perspective/linear interp type for
         * meaningful semantics; `first`/`either` go with `flat`.
         * Pin every name in a context the lib accepts. */
        struct { const char *itype; const char *samp; } cases[] = {
            { "perspective", "center"   },
            { "perspective", "centroid" },
            { "perspective", "sample"   },
            { "linear",      "center"   },
            { "linear",      "centroid" },
            { "linear",      "sample"   },
            { "flat",        "first"    },
            { "flat",        "either"   },
        };
        for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
            char src[256], label[128];
            snprintf(src, sizeof src,
                "@fragment fn fs(@location(0) @interpolate(%s, %s) v: f32) "
                "-> @location(0) vec4<f32> { return vec4<f32>(0.0); }\n",
                cases[i].itype, cases[i].samp);
            snprintf(label, sizeof label,
                     "§3.8.7: @interpolate(%s, %s) accepted",
                     cases[i].itype, cases[i].samp);
            expect_ok(src, label);
        }
    }
    expect_reject(
        "@fragment fn fs(@location(0) @interpolate(perspective, garbage) "
        "v: f32) -> @location(0) vec4<f32> { return vec4<f32>(0.0); }\n",
        "unknown interpolation sampling",
        "§3.8.7: @interpolate(perspective, garbage) rejected");
    expect_reject(
        "@fragment fn fs(@location(0) @interpolate(flat, frist) v: f32) -> "
        "@location(0) vec4<f32> { return vec4<f32>(0.0); }\n",
        "unknown interpolation sampling",
        "§3.8.7: typo near 'first' rejected");

    /* ── §3.8.2 — built-in value names (already enforced) ──────── *
     *                                                             *
     * `@builtin(garbage)` is rejected today (the per-spelling     *
     * name table is in validate.c `kBuiltinTable`).  Pin the      *
     * happy-path name and the unknown-name rejection so this      *
     * basic gate doesn't regress.  The §13.3.1.1                  *
     * stage/direction/type/extension matrix is a separate gap.    */
    expect_ok(
        "@vertex fn vs() -> @builtin(position) vec4<f32> "
        "{ return vec4<f32>(0); }\n",
        "§3.8.2: @builtin(position) accepted");
    expect_reject(
        "@vertex fn vs(@builtin(garbage) v : u32) -> "
        "@builtin(position) vec4<f32> { return vec4<f32>(0); }\n",
        "unknown builtin",
        "§3.8.2: @builtin(garbage) rejected");

    fprintf(stderr, "%s  test_context_names_conformance  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
