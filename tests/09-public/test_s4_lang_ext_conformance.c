/**
 * §4.1.2 Language Extensions (`requires`) — conformance tests.
 *
 * Language extensions are automatically available when the
 * implementation supports them.  A `requires` directive documents the
 * dependency and checks support; it does not enable the feature.
 *
 * Covered:
 *   - `readonly_and_readwrite_storage_textures` — read/read_write on
 *     storage textures works with or without a documenting `requires`.
 *   - `texture_formats_tier1` — tier1 texel formats work with or
 *     without a documenting `requires`.
 *   - `texture_and_sampler_let` — `let` with texture/sampler type works
 *     with or without a documenting `requires`.
 *   - `linear_indexing` — `@builtin(global_invocation_index)` and
 *     `@builtin(workgroup_index)` work with or without `requires`.
 *   - `pointer_composite_access` / `unrestricted_pointer_parameters` —
 *     tests deferred (depends on ptr typing).
 */
#include "wgsl.h"

#include <stdint.h>
#include <stdio.h>

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

static void expect_ok(const char *src, const char *label) {
    WGSLResult *r = wgsl_check(src);
    dump_on_fail(r, label);
    CHECK(wgsl_ok(r), label);
    CHECK(wgsl_diagnostic_count(r) == 0, label);
    wgsl_free(r);
}

int main(void) {
    /* ═══════════════════════════════════════════════════════════════
     *  §4.1.2 — readonly_and_readwrite_storage_textures
     * ═══════════════════════════════════════════════════════════════ */

    /* Core: write access always available */
    expect_ok(
        "@group(0) @binding(0)\n"
        "var t: texture_storage_2d<rgba8unorm, write>;\n",
        "§4.1.2 RO/RW: write access on storage texture (core, no requires)");

    /* read access is available because the implementation supports the
     * language feature; `requires` is documentary. */
    expect_ok(
        "requires readonly_and_readwrite_storage_textures;\n"
        "@group(0) @binding(0)\n"
        "var t: texture_storage_2d<rgba8unorm, read>;\n",
        "§4.1.2 RO/RW: read access accepted with requires");
    expect_ok(
        "@group(0) @binding(0)\n"
        "var t: texture_storage_2d<rgba8unorm, read>;\n",
        "§4.1.2 RO/RW: read access accepted without requires");

    /* read_write access follows the same supported-feature model. */
    expect_ok(
        "requires readonly_and_readwrite_storage_textures;\n"
        "@group(0) @binding(0)\n"
        "var t: texture_storage_2d<rgba8unorm, read_write>;\n",
        "§4.1.2 RO/RW: read_write access accepted with requires");
    expect_ok(
        "@group(0) @binding(0)\n"
        "var t: texture_storage_2d<rgba8unorm, read_write>;\n",
        "§4.1.2 RO/RW: read_write access accepted without requires");

    /* ═══════════════════════════════════════════════════════════════
     *  §4.1.2 — texture_formats_tier1
     * ═══════════════════════════════════════════════════════════════ */

    /* Core format — always works */
    expect_ok(
        "@group(0) @binding(0)\n"
        "var t: texture_storage_2d<rgba32float, write>;\n",
        "§4.1.2 T1: core format rgba32float (no requires needed)");
    expect_ok(
        "@group(0) @binding(0)\n"
        "var t: texture_storage_2d<r32float, write>;\n",
        "§4.1.2 T1: core format r32float (no requires needed)");
    expect_ok(
        "@group(0) @binding(0)\n"
        "var t: texture_storage_2d<bgra8unorm, write>;\n",
        "§4.1.2 T1: core format bgra8unorm (no requires needed)");

    /* Tier1 formats are available because the implementation supports
     * the language feature; `requires` is documentary. */
    expect_ok(
        "@group(0) @binding(0)\n"
        "var t: texture_storage_2d<r8unorm, write>;\n",
        "§4.1.2 T1: r8unorm accepted without requires");
    expect_ok(
        "requires texture_formats_tier1;\n"
        "@group(0) @binding(0)\n"
        "var t: texture_storage_2d<r8unorm, write>;\n",
        "§4.1.2 T1: r8unorm accepted with requires");

    /* Tier1: r16unorm (newly added format) */
    expect_ok(
        "@group(0) @binding(0)\n"
        "var t: texture_storage_2d<r16unorm, write>;\n",
        "§4.1.2 T1: r16unorm accepted without requires");
    expect_ok(
        "requires texture_formats_tier1;\n"
        "@group(0) @binding(0)\n"
        "var t: texture_storage_2d<r16unorm, write>;\n",
        "§4.1.2 T1: r16unorm accepted with requires");

    /* Tier1: rg8unorm */
    expect_ok(
        "@group(0) @binding(0)\n"
        "var t: texture_storage_2d<rg8unorm, write>;\n",
        "§4.1.2 T1: rg8unorm accepted without requires");
    expect_ok(
        "requires texture_formats_tier1;\n"
        "@group(0) @binding(0)\n"
        "var t: texture_storage_2d<rg8unorm, write>;\n",
        "§4.1.2 T1: rg8unorm accepted with requires");

    /* Tier1: rgba16unorm (newly added) */
    expect_ok(
        "@group(0) @binding(0)\n"
        "var t: texture_storage_2d<rgba16unorm, write>;\n",
        "§4.1.2 T1: rgba16unorm accepted without requires");
    expect_ok(
        "requires texture_formats_tier1;\n"
        "@group(0) @binding(0)\n"
        "var t: texture_storage_2d<rgba16unorm, write>;\n",
        "§4.1.2 T1: rgba16unorm accepted with requires");

    /* Tier1: rgb10a2uint (newly added) */
    expect_ok(
        "@group(0) @binding(0)\n"
        "var t: texture_storage_2d<rgb10a2uint, write>;\n",
        "§4.1.2 T1: rgb10a2uint accepted without requires");
    expect_ok(
        "requires texture_formats_tier1;\n"
        "@group(0) @binding(0)\n"
        "var t: texture_storage_2d<rgb10a2uint, write>;\n",
        "§4.1.2 T1: rgb10a2uint accepted with requires");

    /* Tier1: rg11b10ufloat (newly added — spec name) */
    expect_ok(
        "@group(0) @binding(0)\n"
        "var t: texture_storage_2d<rg11b10ufloat, write>;\n",
        "§4.1.2 T1: rg11b10ufloat accepted without requires");
    expect_ok(
        "requires texture_formats_tier1;\n"
        "@group(0) @binding(0)\n"
        "var t: texture_storage_2d<rg11b10ufloat, write>;\n",
        "§4.1.2 T1: rg11b10ufloat accepted with requires");

    /* ═══════════════════════════════════════════════════════════════
     *  §4.1.2 — texture_and_sampler_let
     * ═══════════════════════════════════════════════════════════════ */

    /* let with sampler type — available with implementation support. */
    expect_ok(
        "requires texture_and_sampler_let;\n"
        "@group(0) @binding(0) var s: sampler;\n"
        "@fragment fn fs() -> @location(0) vec4<f32> {\n"
        "  let my_s: sampler = s;\n"
        "  return vec4<f32>(0);\n"
        "}\n",
        "§4.1.2 TSL: let sampler accepted with requires");
    expect_ok(
        "@group(0) @binding(0) var s: sampler;\n"
        "@fragment fn fs() -> @location(0) vec4<f32> {\n"
        "  let my_s: sampler = s;\n"
        "  return vec4<f32>(0);\n"
        "}\n",
        "§4.1.2 TSL: let sampler accepted without requires");

    /* let with texture type — same supported-feature model. */
    expect_ok(
        "requires texture_and_sampler_let;\n"
        "@group(0) @binding(0) var t: texture_2d<f32>;\n"
        "@fragment fn fs() -> @location(0) vec4<f32> {\n"
        "  let my_t: texture_2d<f32> = t;\n"
        "  return vec4<f32>(0);\n"
        "}\n",
        "§4.1.2 TSL: let texture accepted with requires");
    expect_ok(
        "@group(0) @binding(0) var t: texture_2d<f32>;\n"
        "@fragment fn fs() -> @location(0) vec4<f32> {\n"
        "  let my_t: texture_2d<f32> = t;\n"
        "  return vec4<f32>(0);\n"
        "}\n",
        "§4.1.2 TSL: let texture accepted without requires");

    /* let with a non-texture/sampler type — always OK */
    expect_ok(
        "@fragment fn fs() -> @location(0) vec4<f32> {\n"
        "  let x: f32 = 1.0;\n"
        "  return vec4<f32>(x);\n"
        "}\n",
        "§4.1.2 TSL: let f32 always OK (no feature needed)");

    /* ═══════════════════════════════════════════════════════════════
     *  §4.1.2 — linear_indexing
     * ═══════════════════════════════════════════════════════════════ */

    /* @builtin(global_invocation_index) — available with implementation
     * support; `requires` is documentary. */
    expect_ok(
        "requires linear_indexing;\n"
        "@compute @workgroup_size(64)\n"
        "fn cs(@builtin(global_invocation_index) gii: u32) {\n"
        "  _ = gii;\n"
        "}\n",
        "§4.1.2 LI: @builtin(global_invocation_index) accepted with requires");
    expect_ok(
        "@compute @workgroup_size(64)\n"
        "fn cs(@builtin(global_invocation_index) gii: u32) {\n"
        "  _ = gii;\n"
        "}\n",
        "§4.1.2 LI: @builtin(global_invocation_index) accepted without requires");

    /* @builtin(workgroup_index) — same supported-feature model. */
    expect_ok(
        "requires linear_indexing;\n"
        "@compute @workgroup_size(64)\n"
        "fn cs(@builtin(workgroup_index) wi: u32) {\n"
        "  _ = wi;\n"
        "}\n",
        "§4.1.2 LI: @builtin(workgroup_index) accepted with requires");
    expect_ok(
        "@compute @workgroup_size(64)\n"
        "fn cs(@builtin(workgroup_index) wi: u32) {\n"
        "  _ = wi;\n"
        "}\n",
        "§4.1.2 LI: @builtin(workgroup_index) accepted without requires");

    /* Existing builtins still work without any requires */
    expect_ok(
        "@compute @workgroup_size(64)\n"
        "fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
        "  _ = gid;\n"
        "}\n",
        "§4.1.2 LI: @builtin(global_invocation_id) — core, always OK");
    expect_ok(
        "@compute @workgroup_size(64)\n"
        "fn cs(@builtin(workgroup_id) wid: vec3<u32>) {\n"
        "  _ = wid;\n"
        "}\n",
        "§4.1.2 LI: @builtin(workgroup_id) — core, always OK");

    fprintf(stderr, "%s  test_s4_lang_ext_conformance  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
