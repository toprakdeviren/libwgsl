/**
 * §1 Introduction — conformance smoke tests.
 *
 * Pins the concrete, testable claims made in
 * `docs/wgsl/en/01_introduction.md` against the public API so the
 * intro example and the intro's high-level rules cannot regress
 * silently.  Each block cites the paragraph it anchors:
 *
 *   - §1.1 lead example          — lighting fragment shader compiles
 *   - §1.1 note                  — "A WGSL program does not require an
 *                                  entry point" (module_json shows
 *                                  empty entry_points)
 *   - §1.1 module organisation   — directive + alias + const + override
 *                                  + struct + var + fn coexist
 *   - §1.1 statement list        — every statement form from the bullet
 *                                  list inside one fn body
 *   - §1.1 concrete-conversion   — "WGSL does not have implicit
 *                                  conversions or promotions from
 *                                  concrete types"
 *   - §1.1 explicit conversion   — three forms: value constructor,
 *                                  conversion-style constructor,
 *                                  bitcast reinterpretation
 *   - §1.1 scalar→vector promote — "limited facility to promote scalar
 *                                  types to vector types"
 *   - §1.3 math notation         — transpose / floor / ceil / trunc
 *                                  builtins exist and type-check
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

static int contains(const char *hay, const char *needle) {
    return hay && strstr(hay, needle) != NULL;
}

static int has_diag_with(const WGSLResult *r, const char *needle) {
    int n = wgsl_diagnostic_count(r);
    for (int i = 0; i < n; i++) {
        const WGSLDiagnostic *d = wgsl_diagnostic(r, i);
        if (d && d->message && strstr(d->message, needle)) return 1;
    }
    return 0;
}

/* Dump diagnostics on unexpected failure so regressions point at the
 * right place instead of just "ok=0". */
static void dump_on_fail(const WGSLResult *r, const char *label) {
    if (wgsl_ok(r)) return;
    fprintf(stderr, "  [%s] unexpected diagnostics:\n", label);
    int n = wgsl_diagnostic_count(r);
    for (int i = 0; i < n; i++) {
        const WGSLDiagnostic *d = wgsl_diagnostic(r, i);
        fprintf(stderr, "    L%u:%u %s\n",
                d ? d->line : 0, d ? d->column : 0,
                d && d->message ? d->message : "(null)");
    }
}

int main(void) {
    /* ── §1.1 — lead example: lighting fragment shader ─────────── */
    {
        const char *src =
            "// A fragment shader which lights textured geometry with point lights.\n"
            "\n"
            "struct PointLight {\n"
            "  position : vec3f,\n"
            "  color : vec3f,\n"
            "}\n"
            "\n"
            "struct LightStorage {\n"
            "  pointCount : u32,\n"
            "  point : array<PointLight>,\n"
            "}\n"
            "@group(0) @binding(0) var<storage> lights : LightStorage;\n"
            "\n"
            "@group(1) @binding(0) var baseColorSampler : sampler;\n"
            "@group(1) @binding(1) var baseColorTexture : texture_2d<f32>;\n"
            "\n"
            "@fragment\n"
            "fn fragmentMain(@location(0) worldPos : vec3f,\n"
            "                @location(1) normal : vec3f,\n"
            "                @location(2) uv : vec2f) -> @location(0) vec4f {\n"
            "  let baseColor = textureSample(baseColorTexture, baseColorSampler, uv);\n"
            "\n"
            "  let N = normalize(normal);\n"
            "  var surfaceColor = vec3f(0);\n"
            "\n"
            "  for (var i = 0u; i < lights.pointCount; i++) {\n"
            "    let worldToLight = lights.point[i].position - worldPos;\n"
            "    let dist = length(worldToLight);\n"
            "    let dir = normalize(worldToLight);\n"
            "\n"
            "    let radiance = lights.point[i].color * (1 / pow(dist, 2));\n"
            "    let nDotL = max(dot(N, dir), 0);\n"
            "\n"
            "    surfaceColor += baseColor.rgb * radiance * nDotL;\n"
            "  }\n"
            "\n"
            "  return vec4(surfaceColor, baseColor.a);\n"
            "}\n";
        WGSLResult *r = wgsl_check(src);
        dump_on_fail(r, "lead example");
        CHECK(wgsl_ok(r), "intro lead example: ok");
        CHECK(wgsl_diagnostic_count(r) == 0,
              "intro lead example: 0 diagnostics");
        const char *j = wgsl_module_json(r);
        CHECK(contains(j, "\"name\":\"fragmentMain\""),
              "intro example: entry point fragmentMain in module_json");
        CHECK(contains(j, "\"stage\":\"fragment\""),
              "intro example: fragment stage in module_json");
        CHECK(contains(j, "\"name\":\"PointLight\""),
              "intro example: PointLight struct in module_json");
        CHECK(contains(j, "\"name\":\"LightStorage\""),
              "intro example: LightStorage struct in module_json");
        CHECK(contains(j, "\"name\":\"lights\""),
              "intro example: lights resource in module_json");
        CHECK(contains(j, "\"name\":\"baseColorSampler\""),
              "intro example: sampler resource in module_json");
        CHECK(contains(j, "\"name\":\"baseColorTexture\""),
              "intro example: texture resource in module_json");
        wgsl_free(r);
    }

    /* ── §1.1 note — module without entry point is valid ───────── *
     *
     * "A WGSL program does not require an entry point; however, such
     *  a program cannot be executed by the API because an entry point
     *  is required to generate a `GPUProgrammableStage`."
     *
     * Pin: a module that only declares types, constants, and helper
     * functions compiles (ok=1) and surfaces zero entry points in the
     * module summary so the host knows it cannot dispatch.
     */
    {
        const char *src =
            "const PI : f32 = 3.14159;\n"
            "struct Vertex { pos : vec3<f32>, uv : vec2<f32> }\n"
            "alias Index = u32;\n"
            "fn double_it(v : f32) -> f32 { return v * 2.0; }\n";
        WGSLResult *r = wgsl_check(src);
        dump_on_fail(r, "no-entry");
        CHECK(wgsl_ok(r), "no entry point: ok (decls-only module is valid)");
        CHECK(wgsl_diagnostic_count(r) == 0, "no entry point: 0 diags");
        const char *j = wgsl_module_json(r);
        CHECK(contains(j, "\"entry_points\":[]"),
              "no entry point: module_json entry_points is empty");
        CHECK(contains(j, "\"name\":\"Vertex\""),
              "no entry point: Vertex struct still appears");
        wgsl_free(r);
    }

    /* ── §1.1 module organisation roll-call ────────────────────── *
     *
     * "A WGSL program is organized into: Directives, Functions,
     *  Statements, Literals, Constants, Variables, Expressions,
     *  Types, Attributes."
     *
     * Pin: one module that exercises every category at once
     * (directive, alias, const, override, struct, module-scope var,
     * attribute, expression, statement).  If any category regresses,
     * this single source breaks.
     */
    {
        const char *src =
            "diagnostic(off, derivative_uniformity);\n"
            "alias Index = u32;\n"
            "const PI : f32 = 3.14159;\n"
            "override SCALE : f32 = 1.0;\n"
            "struct Params { count : Index, factor : f32 }\n"
            "@group(0) @binding(0) var<storage,read_write> params : Params;\n"
            "\n"
            "@compute @workgroup_size(64)\n"
            "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
            "  let idx : Index = gid.x;\n"
            "  if (idx < params.count) {\n"
            "    params.factor = params.factor * PI * SCALE;\n"
            "  }\n"
            "}\n";
        WGSLResult *r = wgsl_check(src);
        dump_on_fail(r, "module organisation");
        CHECK(wgsl_ok(r), "module organisation roll-call: ok");
        const char *j = wgsl_module_json(r);
        CHECK(contains(j, "\"overrides\":[") &&
              contains(j, "\"name\":\"SCALE\""),
              "override SCALE surfaces in module_json");
        CHECK(contains(j, "\"workgroup_size\":[64,1,1]"),
              "@workgroup_size(64) attribute parsed");
        wgsl_free(r);
    }

    /* ── §1.1 statement category roll-call ─────────────────────── *
     *
     * Every statement form enumerated in §1.1 inside one entry point:
     *   - declarations (let, var, const_assert)
     *   - assignment (simple, compound)
     *   - if / else-if / else
     *   - switch
     *   - loop with continuing and break-if
     *   - while
     *   - for
     *   - break, continue
     *   - function call (helper())
     *   - return
     */
    {
        const char *src =
            "fn helper(v : i32) -> i32 { return v + 1; }\n"
            "@compute @workgroup_size(1)\n"
            "fn main() {\n"
            "  let a : i32 = 1;\n"
            "  var b : i32 = a;\n"
            "  const_assert 2 > 1;\n"
            "  b = b + a;\n"
            "  b += 1;\n"
            "  if (b > 0) { b = 1; }\n"
            "  else if (b < 0) { b = -1; }\n"
            "  else { b = 0; }\n"
            "  switch b {\n"
            "    case 0: { b = 10; }\n"
            "    case 1, 2: { b = 20; }\n"
            "    default: { b = -1; }\n"
            "  }\n"
            "  loop {\n"
            "    if (b > 100) { break; }\n"
            "    b = b + 1;\n"
            "    continuing { break if b > 1000; }\n"
            "  }\n"
            "  while (b < 0) { b = b + 1; }\n"
            "  for (var i = 0; i < 3; i = i + 1) {\n"
            "    b = b + helper(i);\n"
            "    if (b > 50) { continue; }\n"
            "  }\n"
            "}\n";
        WGSLResult *r = wgsl_check(src);
        dump_on_fail(r, "statement roll-call");
        CHECK(wgsl_ok(r), "statement category roll-call: ok");
        CHECK(wgsl_diagnostic_count(r) == 0,
              "statement roll-call: 0 diags");
        wgsl_free(r);
    }

    /* ── §1.1 — no implicit conversion between concrete types ──── *
     *
     * "Converting a value from one concrete numeric or boolean type
     *  to another requires an explicit conversion, value constructor,
     *  or reinterpretation of bits."
     *
     * Pin every concrete-pair binary op the diagnostic mentions
     * (existing tests only cover i32+u32).
     */
    {
        struct { const char *label; const char *src; const char *needle; } cases[] = {
            { "i32+u32",  "fn f() { let x = 1i + 1u; }\n",  "no implicit conversion" },
            { "i32+f32",  "fn f() { let x = 1i + 1.0f; }\n", "no implicit conversion" },
            { "u32+f32",  "fn f() { let x = 1u + 1.0f; }\n", "no implicit conversion" },
            { "bool+i32", "fn f() { let x = true + 1i; }\n", "no implicit conversion" },
        };
        for (size_t k = 0; k < sizeof cases / sizeof cases[0]; k++) {
            WGSLResult *r = wgsl_check(cases[k].src);
            CHECK(!wgsl_ok(r), cases[k].label /* must error */);
            CHECK(has_diag_with(r, cases[k].needle),
                  cases[k].label /* needle in diagnostics */);
            wgsl_free(r);
        }
    }

    /* ── §1.1 — three forms of explicit concrete conversion ────── *
     *
     * The intro names them as: "explicit conversion, value
     * constructor, or reinterpretation of bits".  Pin each form.
     */
    {
        /* Value constructor: f32(int_val) builds an f32. */
        WGSLResult *r = wgsl_check(
            "fn f() { let a : i32 = 1; let b : f32 = f32(a); }\n");
        dump_on_fail(r, "explicit: constructor");
        CHECK(wgsl_ok(r), "explicit conversion via constructor f32(i32): ok");
        wgsl_free(r);
    }
    {
        /* Conversion-style constructor on a vector. */
        WGSLResult *r = wgsl_check(
            "fn f() {\n"
            "  let a : vec3<i32> = vec3<i32>(1, 2, 3);\n"
            "  let b : vec3<f32> = vec3<f32>(a);\n"
            "}\n");
        dump_on_fail(r, "explicit: vector convert");
        CHECK(wgsl_ok(r),
              "explicit conversion via vec3<f32>(vec3<i32>): ok");
        wgsl_free(r);
    }
    {
        /* Bit reinterpretation. */
        WGSLResult *r = wgsl_check(
            "fn f() { let a : i32 = 1; let b : u32 = bitcast<u32>(a); }\n");
        dump_on_fail(r, "explicit: bitcast");
        CHECK(wgsl_ok(r), "explicit reinterpretation via bitcast<u32>: ok");
        wgsl_free(r);
    }

    /* ── §1.1 — scalar→vector promotion ────────────────────────── *
     *
     * "WGSL does provide some limited facility to promote scalar
     *  types to vector types."
     *
     * Pin: scalar argument fills every component in a vec constructor,
     * and a scalar broadcasts in arithmetic against a vector operand.
     */
    {
        WGSLResult *r = wgsl_check(
            "fn f() {\n"
            "  let v3 : vec3<f32> = vec3<f32>(7.0);\n"
            "  let v4 : vec4<f32> = vec4<f32>(0.0);\n"
            "}\n");
        dump_on_fail(r, "scalar→vec constructor");
        CHECK(wgsl_ok(r), "scalar→vec promotion via constructor: ok");
        wgsl_free(r);
    }
    {
        WGSLResult *r = wgsl_check(
            "fn f() {\n"
            "  let a = vec3<f32>(1.0, 2.0, 3.0);\n"
            "  let b = a + 5.0;\n"
            "  let c = 2.0 * a;\n"
            "}\n");
        dump_on_fail(r, "scalar→vec arithmetic");
        CHECK(wgsl_ok(r), "scalar↔vec broadcast in arithmetic: ok");
        wgsl_free(r);
    }

    /* ── §1.3 mathematical notation — transpose / floor / ceil / trunc ── *
     *
     * §1.3 defines transpose(A) and the floor/ceiling/truncate
     * operations.  Pin that the matching builtins are present and
     * type-check on representative inputs.
     */
    {
        WGSLResult *r = wgsl_check(
            "fn f() {\n"
            "  let m : mat2x3<f32> = mat2x3<f32>(\n"
            "    vec3<f32>(1.0, 2.0, 3.0),\n"
            "    vec3<f32>(4.0, 5.0, 6.0));\n"
            "  let t : mat3x2<f32> = transpose(m);\n"
            "}\n");
        dump_on_fail(r, "transpose");
        CHECK(wgsl_ok(r),
              "§1.3 transpose: mat2x3<f32> → mat3x2<f32> type-checks");
        wgsl_free(r);
    }
    {
        WGSLResult *r = wgsl_check(
            "fn f() {\n"
            "  let a : f32 = floor(1.7);\n"
            "  let b : f32 = ceil(1.2);\n"
            "  let c : f32 = trunc(1.7);\n"
            "}\n");
        dump_on_fail(r, "floor/ceil/trunc");
        CHECK(wgsl_ok(r),
              "§1.3 floor / ceil / trunc builtins type-check on f32");
        wgsl_free(r);
    }

    fprintf(stderr, "%s  test_intro_conformance  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
