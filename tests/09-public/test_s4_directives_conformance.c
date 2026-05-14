/**
 * §4 Directives — extension gating + diagnostic filter conformance.
 *
 * Pins spec-conformance items from `docs/wgsl/missing.md` §4 that were
 * previously marked ❌ and are now closed:
 *
 *   §4.1.1  Enable Extensions
 *     - Enable-extension name table validation (5 names)
 *     - `clip_distances` builtin variable gating
 *     - `subgroups` gating for subgroup/quad builtin functions
 *     - `primitive_index` builtin variable gating
 *
 *   §4.1.2  Language Extensions (`requires`)
 *     - Language-extension name table validation (11 entries)
 *     - `immediate_data` — `var<immediate>` resolves (no longer
 *       "undeclared identifier")
 *
 *   §4.2  Global Diagnostic Filter
 *     - Rule name enumerant validated on directive form
 *     - Rule name enumerant validated on `@diagnostic` attribute form
 *
 * Out of scope here (covered by test_s4_lang_ext_conformance):
 *   - `readonly_and_readwrite_storage_textures` supported-feature behavior
 *   - `texture_formats_tier1` extra texel formats
 *   - `texture_and_sampler_let` supported-feature behavior
 *   - `pointer_composite_access` (depends on ptr typing)
 *   - `unrestricted_pointer_parameters` supported-feature behavior
 *   - `linear_indexing` supported-feature behavior
 *   - Diagnostic filter downstream application (tc_diag callers)
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
    /* ═══════════════════════════════════════════════════════════════
     *  §4.1.1 — Enable Extensions: builtin gating
     * ═══════════════════════════════════════════════════════════════ */

    /* ── clip_distances builtin gated behind `enable clip_distances;` ── *
     * `clip_distances` is a vertex output per §13.3.1.1, so we exercise *
     * the enable gate through a struct return that also carries        *
     * `@builtin(position)` (mandatory on a vertex entry point).        */
    expect_ok(
        "enable clip_distances;\n"
        "struct VsOut {\n"
        "  @builtin(position) p: vec4<f32>,\n"
        "  @builtin(clip_distances) c: array<f32, 1>,\n"
        "}\n"
        "@vertex fn vs() -> VsOut {\n"
        "  return VsOut(vec4<f32>(0.0), array<f32, 1>(0.0));\n"
        "}\n",
        "§4.1.1: @builtin(clip_distances) accepted with enable");
    expect_reject(
        "struct VsOut {\n"
        "  @builtin(position) p: vec4<f32>,\n"
        "  @builtin(clip_distances) c: array<f32, 1>,\n"
        "}\n"
        "@vertex fn vs() -> VsOut {\n"
        "  return VsOut(vec4<f32>(0.0), array<f32, 1>(0.0));\n"
        "}\n",
        "requires 'enable clip_distances;'",
        "§4.1.1: @builtin(clip_distances) rejected without enable");

    /* ── primitive_index builtin gated behind `enable primitive_index;` ── */
    expect_ok(
        "enable primitive_index;\n"
        "@fragment fn fs(@builtin(primitive_index) pidx: u32) -> "
        "@location(0) vec4<f32> {\n"
        "  return vec4<f32>(0);\n"
        "}\n",
        "§4.1.1: @builtin(primitive_index) accepted with enable");
    expect_reject(
        "@fragment fn fs(@builtin(primitive_index) pidx: u32) -> "
        "@location(0) vec4<f32> {\n"
        "  return vec4<f32>(0);\n"
        "}\n",
        "requires 'enable primitive_index;'",
        "§4.1.1: @builtin(primitive_index) rejected without enable");

    /* ── subgroup builtins gated behind `enable subgroups;` ── */
    expect_ok(
        "enable subgroups;\n"
        "@fragment fn fs(@builtin(subgroup_invocation_id) sid: u32) -> "
        "@location(0) vec4<f32> {\n"
        "  return vec4<f32>(0);\n"
        "}\n",
        "§4.1.1: @builtin(subgroup_invocation_id) accepted with enable");
    expect_reject(
        "@fragment fn fs(@builtin(subgroup_invocation_id) sid: u32) -> "
        "@location(0) vec4<f32> {\n"
        "  return vec4<f32>(0);\n"
        "}\n",
        "requires 'enable subgroups;'",
        "§4.1.1: @builtin(subgroup_invocation_id) rejected without enable");

    /* ── subgroup FUNCTIONS gated behind `enable subgroups;` ── */
    expect_ok(
        "enable subgroups;\n"
        "@compute @workgroup_size(64)\n"
        "fn cs() {\n"
        "  let x = subgroupAdd(1u);\n"
        "}\n",
        "§4.1.1: subgroupAdd() accepted with enable subgroups");
    expect_reject(
        "@compute @workgroup_size(64)\n"
        "fn cs() {\n"
        "  let x = subgroupAdd(1u);\n"
        "}\n",
        "requires 'enable subgroups;'",
        "§4.1.1: subgroupAdd() rejected without enable subgroups");

    /* Other subgroup fns */
    expect_reject(
        "@compute @workgroup_size(64)\n"
        "fn cs() {\n"
        "  let b = subgroupBallot(true);\n"
        "}\n",
        "requires 'enable subgroups;'",
        "§4.1.1: subgroupBallot() rejected without enable subgroups");
    expect_reject(
        "@compute @workgroup_size(64)\n"
        "fn cs() {\n"
        "  let e = subgroupElect();\n"
        "}\n",
        "requires 'enable subgroups;'",
        "§4.1.1: subgroupElect() rejected without enable subgroups");

    /* ── quad functions gated behind `enable subgroups;` ── */
    expect_ok(
        "enable subgroups;\n"
        "@compute @workgroup_size(64)\n"
        "fn cs() {\n"
        "  let x = quadSwapX(1u);\n"
        "}\n",
        "§4.1.1: quadSwapX() accepted with enable subgroups");
    expect_reject(
        "@compute @workgroup_size(64)\n"
        "fn cs() {\n"
        "  let x = quadSwapX(1u);\n"
        "}\n",
        "requires 'enable subgroups;'",
        "§4.1.1: quadSwapX() rejected without enable subgroups");
    expect_reject(
        "@compute @workgroup_size(64)\n"
        "fn cs() {\n"
        "  let x = quadBroadcast(1u, 0u);\n"
        "}\n",
        "requires 'enable subgroups;'",
        "§4.1.1: quadBroadcast() rejected without enable subgroups");

    /* All subgroup functions accepted with enable subgroups. */
    expect_ok(
        "enable subgroups;\n"
        "@compute @workgroup_size(64)\n"
        "fn cs() {\n"
        "  let a = subgroupAdd(1u);\n"
        "  let b = subgroupMul(1u);\n"
        "  let c = subgroupMax(1u);\n"
        "  let d = subgroupMin(1u);\n"
        "  let e = subgroupAnd(1u);\n"
        "  let f = subgroupOr(1u);\n"
        "  let g = subgroupXor(1u);\n"
        "  let h = subgroupAll(true);\n"
        "  let i = subgroupAny(true);\n"
        "  let j = subgroupElect();\n"
        "  let k = subgroupBallot(true);\n"
        "  let l = subgroupBroadcast(1u, 0u);\n"
        "  let m = subgroupBroadcastFirst(1u);\n"
        "  let n = subgroupShuffle(1u, 0u);\n"
        "  let o = subgroupShuffleDown(1u, 0u);\n"
        "  let p = subgroupShuffleUp(1u, 0u);\n"
        "  let q = subgroupShuffleXor(1u, 0u);\n"
        "  let r = subgroupExclusiveAdd(1u);\n"
        "  let s = subgroupExclusiveMul(1u);\n"
        "  let t = subgroupInclusiveAdd(1u);\n"
        "  let u = subgroupInclusiveMul(1u);\n"
        "  let v = quadBroadcast(1u, 0u);\n"
        "  let w = quadSwapX(1u);\n"
        "  let x = quadSwapY(1u);\n"
        "  let y = quadSwapDiagonal(1u);\n"
        "}\n",
        "§4.1.1: all subgroup+quad fns accepted with enable subgroups");

    /* ═══════════════════════════════════════════════════════════════
     *  §4.1.2 — Language Extensions: `immediate_data`
     * ═══════════════════════════════════════════════════════════════ */

    /* `var<immediate>` should now resolve (address space is predeclared) */
    expect_ok(
        "requires immediate_data;\n"
        "var<immediate> x: u32;\n",
        "§4.1.2: var<immediate> resolves with `requires immediate_data;`");

    /* Without `requires`, it should still resolve (address space is
     * predeclared unconditionally, like all other AS names).  The
     * `requires` gating check is a separate gap. */
    expect_ok(
        "var<immediate> x: u32;\n",
        "§4.1.2: var<immediate> resolves (AS is predeclared)");

    /* ═══════════════════════════════════════════════════════════════
     *  §4.2 — Global Diagnostic Filter: rule name validation
     * ═══════════════════════════════════════════════════════════════ */

    /* Directive form — known good (already tested in
     * test_context_names_conformance.c, re-pin here for coverage). */
    expect_ok("diagnostic(off, derivative_uniformity);\n",
              "§4.2: directive diagnostic(off, derivative_uniformity) accepted");
    expect_ok_with_diag("diagnostic(off, garbage);\n",
                        "unknown diagnostic rule",
                        "§4.2: directive diagnostic(off, garbage) accepted with warning");

    /* @diagnostic attribute form on function — known good rule names */
    expect_ok(
        "@diagnostic(off, derivative_uniformity)\n"
        "@fragment fn fs() -> @location(0) vec4<f32> {\n"
        "  return vec4<f32>(0);\n"
        "}\n",
        "§4.2: @diagnostic(off, derivative_uniformity) on fn accepted");
    expect_ok(
        "@diagnostic(off, subgroup_uniformity)\n"
        "@fragment fn fs() -> @location(0) vec4<f32> {\n"
        "  return vec4<f32>(0);\n"
        "}\n",
        "§4.2: @diagnostic(off, subgroup_uniformity) on fn accepted");

    /* @diagnostic attribute form — unknown single-token rule warns, but does not fail */
    expect_ok_with_diag(
        "@diagnostic(off, garbage)\n"
        "@fragment fn fs() -> @location(0) vec4<f32> {\n"
        "  return vec4<f32>(0);\n"
        "}\n",
        "unknown diagnostic rule",
        "§4.2: @diagnostic(off, garbage) on fn accepted with warning");

    /* @diagnostic attribute form — all four severities work */
    {
        const char *sevs[] = { "off", "info", "warning", "error" };
        for (size_t s = 0; s < sizeof sevs / sizeof sevs[0]; s++) {
            char src[256], label[128];
            snprintf(src, sizeof src,
                "@diagnostic(%s, derivative_uniformity)\n"
                "@fragment fn fs() -> @location(0) vec4<f32> {\n"
                "  return vec4<f32>(0);\n"
                "}\n", sevs[s]);
            snprintf(label, sizeof label,
                "§4.2: @diagnostic(%s, derivative_uniformity) accepted", sevs[s]);
            expect_ok(src, label);
        }
    }

    /* ═══════════════════════════════════════════════════════════════
     *  §4.2 — Global Diagnostic Filter: application to diagnostics
     * ═══════════════════════════════════════════════════════════════ */

    /* A divergent workgroupBarrier() emits a derivative_uniformity diagnostic
     * by default (warning). */
    expect_reject(
        "var<private> cond: bool;\n"
        "@compute @workgroup_size(64)\n"
        "fn cs() {\n"
        "  if (cond) {\n"
        "    workgroupBarrier();\n"
        "  }\n"
        "}\n",
        "must be called in uniform control flow",
        "§4.2 filter: workgroupBarrier in divergent flow rejected by default");

    /* A global `diagnostic(off, ...)` suppresses it. */
    expect_ok(
        "diagnostic(off, derivative_uniformity);\n"
        "var<private> cond: bool;\n"
        "@compute @workgroup_size(64)\n"
        "fn cs() {\n"
        "  if (cond) {\n"
        "    workgroupBarrier();\n"
        "  }\n"
        "}\n",
        "§4.2 filter: global diagnostic(off) suppresses divergent barrier diag");

    /* A per-function `@diagnostic(off, ...)` suppresses it. */
    expect_ok(
        "var<private> cond: bool;\n"
        "@diagnostic(off, derivative_uniformity)\n"
        "@compute @workgroup_size(64)\n"
        "fn cs() {\n"
        "  if (cond) {\n"
        "    workgroupBarrier();\n"
        "  }\n"
        "}\n",
        "§4.2 filter: @diagnostic(off) suppresses divergent barrier diag");

    /* Statement-range @diagnostic before a control statement suppresses
     * only diagnostics in that statement's affected range. */
    expect_ok(
        "var<private> cond: bool;\n"
        "@compute @workgroup_size(64)\n"
        "fn cs() {\n"
        "  @diagnostic(off, derivative_uniformity) if (cond) {\n"
        "    workgroupBarrier();\n"
        "  }\n"
        "}\n",
        "§4.2 filter: statement @diagnostic(off) before if suppresses barrier diag");

    /* Attribute lists between the condition and compound body also form
     * a diagnostic range for the body. */
    expect_ok(
        "var<private> cond: bool;\n"
        "@compute @workgroup_size(64)\n"
        "fn cs() {\n"
        "  if (cond) @diagnostic(off, derivative_uniformity) {\n"
        "    workgroupBarrier();\n"
        "  }\n"
        "}\n",
        "§4.2 filter: body @diagnostic(off) suppresses barrier diag");

    /* Scoping test: per-function @diagnostic overrides global diagnostic. */
    expect_reject(
        "diagnostic(off, derivative_uniformity);\n"
        "var<private> cond: bool;\n"
        "@diagnostic(error, derivative_uniformity)\n"
        "@compute @workgroup_size(64)\n"
        "fn cs() {\n"
        "  if (cond) {\n"
        "    workgroupBarrier();\n"
        "  }\n"
        "}\n",
        "must be called in uniform control flow",
        "§4.2 filter: fn @diagnostic(error) overrides global diagnostic(off)");

    /* If a callee's own derivative-required operation is covered by an
     * outer off filter, a later statement-local error filter at the call
     * site must not resurrect a transitive requirement. */
    expect_ok(
        "diagnostic(off, derivative_uniformity);\n"
        "fn foo() { _ = textureSample(t, s, 0.0); }\n"
        "@group(0) @binding(0) var t: texture_1d<f32>;\n"
        "@group(0) @binding(1) var s: sampler;\n"
        "var<private> cond: bool;\n"
        "@fragment fn main() {\n"
        "  @diagnostic(error, derivative_uniformity) if (cond) {\n"
        "    foo();\n"
        "  }\n"
        "}\n",
        "§4.2 filter: callee off filter suppresses transitive call requirement");

    /* Nested statement scope overrides and then restores the outer scope. */
    expect_reject(
        "var<private> cond: bool;\n"
        "@diagnostic(error, derivative_uniformity)\n"
        "@compute @workgroup_size(64)\n"
        "fn cs() {\n"
        "  if (cond) @diagnostic(off, derivative_uniformity) {\n"
        "    workgroupBarrier();\n"
        "  }\n"
        "  if (cond) {\n"
        "    workgroupBarrier();\n"
        "  }\n"
        "}\n",
        "must be called in uniform control flow",
        "§4.2 filter: inner statement scope restores function scope after pop");

    fprintf(stderr, "%s  test_s4_directives_conformance  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
