/**
 * WGSL to MSL emitter smoke tests.
 * Locks UNSUPPORTED_TYPE and PARTIAL_IO through typed diagnostic codes.
 */
#include "wgsl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail;
#define CHECK(c, m) do { if (!(c)) { \
  fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, m); fail++; } } while (0)

/* True if emit carries at least one diagnostic with the given code enum.
 * Control-flow signal is `code`; never scrape message text. */
static int emit_has_code(const WGSLMslEmit *e, int code) {
    if (!e) return 0;
    for (int i = 0; i < wgsl_msl_emit_diag_count(e); i++) {
        const WGSLMslDiag *d = wgsl_msl_emit_diag(e, i);
        if (d && d->code == code) return 1;
    }
    return 0;
}

int main(void) {
    /* Compute kernel + storage buffer + builtins
     * Buffer is a kernel param, not a file-scope device global. */
    {
        const char *src =
            "@group(0) @binding(0) var<storage, read_write> B: array<u32>;\n"
            "@compute @workgroup_size(64)\n"
            "fn cs(@builtin(global_invocation_id) gid: vec3u) {\n"
            "  B[gid.x] = gid.x + 1u;\n"
            "}\n";
        char *msl = wgsl_to_msl(src);
        CHECK(msl != NULL, "compute: non-null");
        CHECK(strstr(msl, "kernel") != NULL, "compute: kernel qualifier");
        CHECK(strstr(msl, "thread_position_in_grid") != NULL,
              "compute: global_invocation_id → thread_position_in_grid");
        CHECK(strstr(msl, "device") != NULL && strstr(msl, "[[buffer(0)]]") != NULL,
              "compute resource lowering: device *B [[buffer(0)]]");
        CHECK(strstr(msl, "device uint *B") != NULL ||
              strstr(msl, "device /*array*/") == NULL,
              "compute resource lowering: array storage → element pointer");
        /* Silent hole closed: no file-scope `device … B [[buffer]]` global. */
        CHECK(strstr(msl, "device uint *B [[buffer(0)]];") == NULL,
              "compute resource lowering: buffer not a module-scope global");
        CHECK(strstr(msl, "metal_stdlib") != NULL, "compute: includes metal_stdlib");
        CHECK(strstr(msl, "uint3") != NULL || strstr(msl, "vec3") != NULL ||
              strstr(msl, "uint") != NULL, "compute: types mapped");
        wgsl_free_string(msl);
    }

    /* resource lowering texture + sampler as entry params + textureSample → .sample */
    {
        const char *src =
            "@group(0) @binding(0) var t: texture_2d<f32>;\n"
            "@group(0) @binding(1) var s: sampler;\n"
            "@fragment\n"
            "fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {\n"
            "  return textureSample(t, s, uv);\n"
            "}\n";
        WGSLMslEmit *e = wgsl_msl_emit(src);
        CHECK(e != NULL, "tex resource lowering: emit non-null");
        const char *msl = wgsl_msl_emit_text(e);
        CHECK(strstr(msl, "texture2d<float> t [[texture(0)]]") != NULL,
              "tex resource lowering: texture param + [[texture(N)]]");
        CHECK(strstr(msl, "sampler s [[sampler(1)]]") != NULL,
              "tex resource lowering: sampler param + [[sampler(N)]]");
        CHECK(strstr(msl, ".sample(") != NULL,
              "tex resource lowering: textureSample → .sample");
        CHECK(strstr(msl, "textureSample") == NULL,
              "tex resource lowering: no raw textureSample call left");
        /* No file-scope texture/sampler globals. */
        CHECK(strstr(msl, "texture2d<float> t;\n") == NULL,
              "tex resource lowering: texture not module-scope bare global");
        CHECK(wgsl_msl_emit_ok(e), "tex resource lowering: emit ok (mapped types)");
        wgsl_msl_emit_free(e);
    }

    /* resource lowering workgroup + barrier map */
    {
        const char *src =
            "var<workgroup> tile : array<u32, 4>;\n"
            "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
            "@compute @workgroup_size(4)\n"
            "fn cs(@builtin(local_invocation_id) l : vec3u) {\n"
            "  tile[l.x] = l.x; workgroupBarrier(); o[l.x] = tile[0];\n"
            "}\n";
        char *msl = wgsl_to_msl(src);
        CHECK(msl != NULL, "wg resource lowering: non-null");
        CHECK(strstr(msl, "threadgroup uint *tile [[threadgroup(") != NULL,
              "wg resource lowering: workgroup → threadgroup param");
        CHECK(strstr(msl, "threadgroup_barrier(mem_flags::mem_threadgroup)") != NULL,
              "wg resource lowering: workgroupBarrier mapped");
        wgsl_free_string(msl);
    }

    /* remaining MSL: fixed arrays are real Metal array<T,N>, f16 maps to half. */
    {
        const char *src =
            "enable f16;\n"
            "struct S { xs: array<vec2h, 4>, }\n"
            "fn f() {\n"
            "  var a: array<u32, 4>;\n"
            "  _ = a[0];\n"
            "}\n";
        WGSLMslEmit *e = wgsl_msl_emit(src);
        CHECK(e != NULL, "remaining MSL array/f16: emit non-null");
        CHECK(wgsl_msl_emit_ok(e), "remaining MSL array/f16: complete");
        const char *t = wgsl_msl_emit_text(e);
        CHECK(strstr(t, "array<half2, 4> xs") != NULL,
              "remaining MSL array/f16: struct array<vec2h,4> → array<half2,4>");
        CHECK(strstr(t, "array<uint, 4> a") != NULL,
              "remaining MSL array/f16: local array<u32,4> → array<uint,4>");
        CHECK(strstr(t, "/*array*/") == NULL,
              "remaining MSL array/f16: no array comment fallback");
        wgsl_msl_emit_free(e);
    }

    /* Vertex + position builtin */
    {
        const char *src =
            "struct VOut {\n"
            "  @builtin(position) pos: vec4f,\n"
            "}\n"
            "@vertex\n"
            "fn vs(@builtin(vertex_index) vid: u32) -> VOut {\n"
            "  var o: VOut;\n"
            "  o.pos = vec4f(0.0, 0.0, 0.0, 1.0);\n"
            "  return o;\n"
            "}\n";
        char *msl = wgsl_to_msl(src);
        CHECK(msl != NULL, "vertex: non-null");
        CHECK(strstr(msl, "vertex ") != NULL, "vertex: qualifier");
        CHECK(strstr(msl, "[[position]]") != NULL, "vertex: position attr");
        CHECK(strstr(msl, "vertex_id") != NULL, "vertex: vertex_index → vertex_id");
        CHECK(strstr(msl, "float4") != NULL, "vertex: vec4f → float4");
        CHECK(strstr(msl, "struct VOut") != NULL, "vertex: struct kept");
        wgsl_free_string(msl);
    }

    /* Fragment + location — stage_in lowering load-bearing: stage_in + body rewrite */
    {
        const char *src =
            "@fragment\n"
            "fn fs(@location(0) color: vec4f) -> @location(0) vec4f {\n"
            "  return color;\n"
            "}\n";
        char *msl = wgsl_to_msl(src);
        CHECK(msl != NULL, "frag: non-null");
        CHECK(strstr(msl, "fragment ") != NULL, "frag: qualifier");
        CHECK(strstr(msl, "stage_in") != NULL, "frag stage_in lowering: [[stage_in]]");
        CHECK(strstr(msl, "__wgsl_in_fs") != NULL ||
              strstr(msl, "struct __wgsl_in_") != NULL,
              "frag stage_in lowering: synthetic stage_in struct");
        CHECK(strstr(msl, "user(locn0)") != NULL, "frag stage_in lowering: user(locn0) on member");
        CHECK(strstr(msl, "in.color") != NULL, "frag stage_in lowering: body rewrite color→in.color");
        CHECK(strstr(msl, "return color") == NULL,
              "frag stage_in lowering: bare param name not left in return");
        wgsl_free_string(msl);
    }

    /* stage_in lowering: vertex free builtins + implicit position return */
    {
        const char *src =
            "@vertex\n"
            "fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {\n"
            "  return vec4f(0.0, 0.0, 0.0, 1.0);\n"
            "}\n";
        char *msl = wgsl_to_msl(src);
        CHECK(msl != NULL, "vs stage_in lowering: non-null");
        CHECK(strstr(msl, "vertex ") != NULL, "vs stage_in lowering: qualifier");
        CHECK(strstr(msl, "vertex_id") != NULL, "vs stage_in lowering: vertex_index→vertex_id");
        CHECK(strstr(msl, "stage_in") == NULL, "vs stage_in lowering: no stage_in for builtin-only");
        wgsl_free_string(msl);
    }

    /* Types + math */
    {
        const char *src =
            "fn f(a: f32, b: vec3f) -> f32 {\n"
            "  let x = a * b.x + sin(a);\n"
            "  return x;\n"
            "}\n";
        char *msl = wgsl_to_msl(src);
        CHECK(msl != NULL, "math: non-null");
        CHECK(strstr(msl, "float") != NULL, "math: f32 → float");
        CHECK(strstr(msl, "float3") != NULL, "math: vec3f → float3");
        CHECK(strstr(msl, "sin") != NULL, "math: sin kept");
        wgsl_free_string(msl);
    }

    /* let → const in MSL body */
    {
        const char *src = "fn f() { let x: i32 = 1; _ = x; }\n";
        char *msl = wgsl_to_msl(src);
        CHECK(msl != NULL, "let: non-null");
        CHECK(strstr(msl, "const ") != NULL || strstr(msl, "int ") != NULL,
              "let: typed local");
        CHECK(strstr(msl, "(void)") != NULL, "let: phony → (void)");
        wgsl_free_string(msl);
    }

    /* via result API */
    {
        WGSLResult *r = wgsl_check("fn g() {}\n");
        char *msl = wgsl_to_msl_result(r);
        CHECK(msl != NULL && strstr(msl, "g") != NULL, "result API");
        wgsl_free_string(msl);
        wgsl_free(r);
    }

    /* Structured incomplete / unsupported diagnostics (typed codes). */
    {
        /* Supported graphics path: no false-positive incomplete. */
        const char *src =
            "@fragment\n"
            "fn fs(@location(0) color: vec4f) -> @location(0) vec4f {\n"
            "  return color;\n"
            "}\n";
        WGSLMslEmit *e = wgsl_msl_emit(src);
        CHECK(e != NULL, "typed diagnostics ok path: emit non-null");
        CHECK(wgsl_msl_emit_ok(e), "typed diagnostics ok path: complete (no false positive)");
        CHECK(wgsl_msl_emit_incomplete(e) == 0, "typed diagnostics ok path: incomplete=0");
        CHECK(wgsl_msl_emit_diag_count(e) == 0, "typed diagnostics ok path: zero diags");
        CHECK(strstr(wgsl_msl_emit_text(e), "stage_in") != NULL,
              "typed diagnostics ok path: still emits stage_in MSL");
        wgsl_msl_emit_free(e);
    }
    {
        /* control-flow lowering: for is supported — real MSL, no UNSUPPORTED_STMT. */
        const char *src =
            "@compute @workgroup_size(1)\n"
            "fn main() {\n"
            "  for (var i = 0; i < 3; i = i + 1) { }\n"
            "}\n";
        WGSLMslEmit *e = wgsl_msl_emit(src);
        CHECK(e != NULL, "control-flow lowering for: emit non-null");
        CHECK(wgsl_msl_emit_ok(e), "control-flow lowering for: complete (no UNSUPPORTED)");
        CHECK(strstr(wgsl_msl_emit_text(e), "for (") != NULL, "control-flow lowering for: real for emit");
        CHECK(strstr(wgsl_msl_emit_text(e), "/*wgsl for") == NULL,
              "control-flow lowering for: not comment fallback");
        wgsl_msl_emit_free(e);
    }
    {
        /* control-flow lowering: switch supported — no fallthrough (explicit break). */
        const char *src =
            "fn f(x: i32) -> i32 {\n"
            "  switch (x) {\n"
            "    case 0, 1: { return 1; }\n"
            "    default: { return 0; }\n"
            "  }\n"
            "}\n";
        WGSLMslEmit *e = wgsl_msl_emit(src);
        CHECK(e != NULL, "control-flow lowering switch: emit non-null");
        CHECK(wgsl_msl_emit_ok(e), "control-flow lowering switch: complete");
        const char *t = wgsl_msl_emit_text(e);
        CHECK(strstr(t, "switch (") != NULL, "control-flow lowering switch: real switch");
        CHECK(strstr(t, "case ") != NULL, "control-flow lowering switch: case labels");
        CHECK(strstr(t, "break;") != NULL, "control-flow lowering switch: explicit break (no fallthrough)");
        wgsl_msl_emit_free(e);
    }
    {
        /* control-flow lowering: loop + continuing + break-if */
        const char *src =
            "@compute @workgroup_size(1)\n"
            "fn main() {\n"
            "  var i = 0;\n"
            "  loop {\n"
            "    if (i > 10) { break; }\n"
            "    continuing {\n"
            "      i = i + 1;\n"
            "      break if i >= 3;\n"
            "    }\n"
            "  }\n"
            "}\n";
        WGSLMslEmit *e = wgsl_msl_emit(src);
        CHECK(e != NULL, "control-flow lowering loop: emit non-null");
        CHECK(wgsl_msl_emit_ok(e), "control-flow lowering loop: complete");
        const char *t = wgsl_msl_emit_text(e);
        CHECK(strstr(t, "for (;;)") != NULL, "control-flow lowering loop: for(;;) lowering");
        CHECK(strstr(t, "__wgsl_brk_") != NULL, "control-flow lowering loop: continuing wrapper");
        CHECK(strstr(t, "if (") != NULL && strstr(t, "break") != NULL,
              "control-flow lowering loop: break-if → if (…) break");
        wgsl_msl_emit_free(e);
    }
    {
        /* control-flow lowering compose stage_in lowering/typed diagnostics: stage_in rewrite inside for body. */
        const char *src =
            "@fragment\n"
            "fn fs(@location(0) color: vec4f) -> @location(0) vec4f {\n"
            "  var acc = color;\n"
            "  for (var i = 0; i < 1; i = i + 1) {\n"
            "    acc = color;\n"
            "  }\n"
            "  return acc;\n"
            "}\n";
        WGSLMslEmit *e = wgsl_msl_emit(src);
        CHECK(e != NULL && wgsl_msl_emit_ok(e), "control-flow lowering compose: ok");
        CHECK(strstr(wgsl_msl_emit_text(e), "in.color") != NULL,
              "control-flow lowering compose: stage_in rewrite inside for");
        CHECK(strstr(wgsl_msl_emit_text(e), "for (") != NULL, "control-flow lowering compose: for present");
        wgsl_msl_emit_free(e);
    }
    {
        /* control-flow lowering compose: loop-local shadows stage_in param → SHADOWED_REWRITE */
        const char *src =
            "@fragment\n"
            "fn fs(@location(0) color: vec4f) -> @location(0) vec4f {\n"
            "  for (var i = 0; i < 1; i = i + 1) {\n"
            "    let color = vec4f(0.0);\n"
            "    discard;\n"
            "  }\n"
            "  return vec4f(1.0);\n"
            "}\n";
        WGSLMslEmit *e = wgsl_msl_emit(src);
        CHECK(e != NULL, "control-flow lowering shadow-in-for: emit");
        CHECK(wgsl_msl_emit_incomplete(e), "control-flow lowering shadow-in-for: incomplete");
        int found = 0;
        for (int i = 0; i < wgsl_msl_emit_diag_count(e); i++) {
            const WGSLMslDiag *d = wgsl_msl_emit_diag(e, i);
            if (d && d->code == WGSL_MSL_DIAG_SHADOWED_REWRITE) found = 1;
        }
        CHECK(found, "control-flow lowering shadow-in-for: SHADOWED_REWRITE");
        wgsl_msl_emit_free(e);
    }
    {
        /* Shadowed stage_in param: local let same name → SHADOWED_REWRITE,
         * and must NOT rewrite the local binding into in.color. */
        const char *src =
            "@fragment\n"
            "fn fs(@location(0) color: vec4f) -> @location(0) vec4f {\n"
            "  let color = vec4f(0.0);\n"
            "  return color;\n"
            "}\n";
        WGSLMslEmit *e = wgsl_msl_emit(src);
        CHECK(e != NULL, "typed diagnostics shadow: emit non-null");
        CHECK(wgsl_msl_emit_incomplete(e), "typed diagnostics shadow: incomplete");
        int found = 0;
        for (int i = 0; i < wgsl_msl_emit_diag_count(e); i++) {
            const WGSLMslDiag *d = wgsl_msl_emit_diag(e, i);
            if (d && d->code == WGSL_MSL_DIAG_SHADOWED_REWRITE) found = 1;
        }
        CHECK(found, "typed diagnostics shadow: SHADOWED_REWRITE code");
        /* Mis-rewrite would produce `return in.color` for the local. */
        CHECK(strstr(wgsl_msl_emit_text(e), "return in.color") == NULL,
              "typed diagnostics shadow: does not rewrite shadowed local to in.color");
        wgsl_msl_emit_free(e);
    }

    /* Resource subtype coverage and remaining PARTIAL_IO cases. */
    {
        /* Storage texture type + textureStore are complete for common formats. */
        const char *src =
            "@group(0) @binding(0) var t: texture_storage_2d<rgba8unorm, write>;\n"
            "@compute @workgroup_size(1)\n"
            "fn cs() { textureStore(t, vec2u(0u, 0u), vec4f(1.0)); }\n";
        WGSLMslEmit *e = wgsl_msl_emit(src);
        CHECK(e != NULL, "storage texture: emit non-null");
        CHECK(wgsl_msl_emit_ok(e), "storage texture: complete");
        const char *t = wgsl_msl_emit_text(e);
        CHECK(strstr(t, "texture2d<float, access::write> t [[texture(0)]]") != NULL,
              "storage texture: type/access/binding mapped");
        CHECK(strstr(t, ".write(") != NULL && strstr(t, "textureStore") == NULL,
              "storage texture: textureStore -> .write");
        CHECK(!emit_has_code(e, WGSL_MSL_DIAG_UNSUPPORTED_TYPE),
              "storage texture: no unsupported type code");
        wgsl_msl_emit_free(e);
    }
    {
        /* textureSample* overload family maps to MSL sample options. */
        const char *src =
            "@group(0) @binding(0) var t: texture_2d<f32>;\n"
            "@group(0) @binding(1) var s: sampler;\n"
            "@group(0) @binding(2) var d: texture_depth_2d;\n"
            "@group(0) @binding(3) var sc: sampler_comparison;\n"
            "@fragment\n"
            "fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {\n"
            "  let a = textureSampleLevel(t, s, uv, 0.0);\n"
            "  let b = textureSampleBias(t, s, uv, 0.0);\n"
            "  let c = textureSampleCompare(d, sc, uv, 0.5);\n"
            "  return a + b + vec4f(c);\n"
            "}\n";
        WGSLMslEmit *e = wgsl_msl_emit(src);
        CHECK(e != NULL, "textureSample*: emit non-null");
        CHECK(wgsl_msl_emit_ok(e), "textureSample*: complete");
        const char *t = wgsl_msl_emit_text(e);
        CHECK(strstr(t, ".sample(") != NULL, "textureSample*: sample call");
        CHECK(strstr(t, "level(") != NULL, "textureSampleLevel: level()");
        CHECK(strstr(t, "bias(") != NULL, "textureSampleBias: bias()");
        CHECK(strstr(t, ".sample_compare(") != NULL,
              "textureSampleCompare: sample_compare");
        CHECK(strstr(t, "textureSample") == NULL,
              "textureSample*: no raw WGSL sample names");
        wgsl_msl_emit_free(e);
    }
    {
        /* PARTIAL_IO: struct-typed entry parameter is not flattened yet. */
        const char *src =
            "struct In { @location(0) c: vec4f }\n"
            "@fragment\n"
            "fn fs(i: In) -> @location(0) vec4f { return i.c; }\n";
        WGSLMslEmit *e = wgsl_msl_emit(src);
        CHECK(e != NULL, "PARTIAL_IO struct-param: emit non-null");
        CHECK(wgsl_msl_emit_incomplete(e),
              "remaining MSL PARTIAL_IO struct-param: incomplete");
        CHECK(emit_has_code(e, WGSL_MSL_DIAG_PARTIAL_IO),
              "remaining MSL PARTIAL_IO struct-param: code enum present");
        wgsl_msl_emit_free(e);
    }
    {
        /* PARTIAL_IO: fragment @location(N≠0) return (stage_in lowering out-struct gap). */
        const char *src =
            "@fragment\n"
            "fn fs(@location(0) c: vec4f) -> @location(1) vec4f { return c; }\n";
        WGSLMslEmit *e = wgsl_msl_emit(src);
        CHECK(e != NULL, "remaining MSL PARTIAL_IO loc1-ret: emit non-null");
        CHECK(wgsl_msl_emit_incomplete(e),
              "remaining MSL PARTIAL_IO loc1-ret: incomplete");
        CHECK(emit_has_code(e, WGSL_MSL_DIAG_PARTIAL_IO),
              "remaining MSL PARTIAL_IO loc1-ret: code enum present");
        wgsl_msl_emit_free(e);
    }
    {
        /* Supported paths must not false-positive incomplete diags. */
        const char *compute =
            "@group(0) @binding(0) var<storage, read_write> B: array<u32>;\n"
            "@compute @workgroup_size(1)\n"
            "fn cs(@builtin(global_invocation_id) g: vec3u) { B[g.x] = 1u; }\n";
        WGSLMslEmit *ec = wgsl_msl_emit(compute);
        CHECK(ec != NULL, "remaining MSL ok compute: emit non-null");
        CHECK(wgsl_msl_emit_ok(ec), "remaining MSL ok compute: ok");
        CHECK(wgsl_msl_emit_incomplete(ec) == 0,
              "remaining MSL ok compute: incomplete=0");
        CHECK(wgsl_msl_emit_diag_count(ec) == 0,
              "remaining MSL ok compute: zero diags");
        CHECK(!emit_has_code(ec, WGSL_MSL_DIAG_UNSUPPORTED_TYPE) &&
              !emit_has_code(ec, WGSL_MSL_DIAG_PARTIAL_IO),
              "remaining MSL ok compute: no TYPE/PARTIAL codes");
        wgsl_msl_emit_free(ec);

        const char *fs =
            "@fragment\n"
            "fn fs(@location(0) color: vec4f) -> @location(0) vec4f {\n"
            "  return color;\n"
            "}\n";
        WGSLMslEmit *ef = wgsl_msl_emit(fs);
        CHECK(ef != NULL, "remaining MSL ok stage_in: emit non-null");
        CHECK(wgsl_msl_emit_ok(ef), "remaining MSL ok stage_in: ok");
        CHECK(wgsl_msl_emit_incomplete(ef) == 0,
              "remaining MSL ok stage_in: incomplete=0");
        CHECK(wgsl_msl_emit_diag_count(ef) == 0,
              "remaining MSL ok stage_in: zero diags");
        CHECK(strstr(wgsl_msl_emit_text(ef), "stage_in") != NULL,
              "remaining MSL ok stage_in: still emits stage_in");
        CHECK(!emit_has_code(ef, WGSL_MSL_DIAG_UNSUPPORTED_TYPE) &&
              !emit_has_code(ef, WGSL_MSL_DIAG_PARTIAL_IO),
              "remaining MSL ok stage_in: no TYPE/PARTIAL codes");
        wgsl_msl_emit_free(ef);
    }

    /* control-flow lowering correctness: break → nearest enclosing loop-or-switch only.
     * Structural lock (Metal does not execute semantics): count of
     * `__wgsl_brk_0 = true` must equal true outer-loop breaks only. */
    {
        /* Repro 1: switch-break must NOT set flag; only outer if-break. */
        const char *src =
            "@compute @workgroup_size(1)\n"
            "fn cs(@builtin(local_invocation_index) li: u32) {\n"
            "  var x = 0;\n"
            "  loop {\n"
            "    switch (li) {\n"
            "      case 0u: { break; }\n"
            "      default: { }\n"
            "    }\n"
            "    x = x + 10;\n"
            "    if (x > 100) { break; }\n"
            "    continuing { x = x + 1; }\n"
            "  }\n"
            "}\n";
        WGSLMslEmit *e = wgsl_msl_emit(src);
        CHECK(e != NULL && wgsl_msl_emit_ok(e), "control-flow lowering nest-switch: ok");
        const char *t = wgsl_msl_emit_text(e);
        {
            int n = 0;
            const char *needle = "__wgsl_brk_0 = true";
            for (const char *p = t; (p = strstr(p, needle)) != NULL;
                 p += strlen(needle))
                n++;
            CHECK(n == 1,
                  "control-flow lowering nest-switch: exactly one outer-loop flag-break");
        }
        wgsl_msl_emit_free(e);
    }
    {
        /* Repro 2: nested for-break must NOT set outer flag. */
        const char *src =
            "@compute @workgroup_size(1)\n"
            "fn cs() {\n"
            "  var x = 0;\n"
            "  loop {\n"
            "    for (var i = 0; i < 3; i = i + 1) {\n"
            "      if (i == 1) { break; }\n"
            "    }\n"
            "    x = x + 1000;\n"
            "    if (x > 5000) { break; }\n"
            "    continuing { x = x + 1; }\n"
            "  }\n"
            "}\n";
        WGSLMslEmit *e = wgsl_msl_emit(src);
        CHECK(e != NULL && wgsl_msl_emit_ok(e), "control-flow lowering nest-for: ok");
        const char *t = wgsl_msl_emit_text(e);
        {
            int n = 0;
            const char *needle = "__wgsl_brk_0 = true";
            for (const char *p = t; (p = strstr(p, needle)) != NULL;
                 p += strlen(needle))
                n++;
            CHECK(n == 1, "control-flow lowering nest-for: exactly one outer-loop flag-break");
        }
        wgsl_msl_emit_free(e);
    }

    fprintf(stderr, "%s  test_to_msl  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
