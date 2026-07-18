/**
 * ML-kernel analysis smoke.
 */
#include "wgsl.h"

#include <stdio.h>
#include <string.h>

static int fail;
#define CHECK(c, m) do { if (!(c)) { \
  fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,m); fail++; } } while(0)

int main(void) {
    /* Softmax-like kernel */
    {
        const char *src =
            "@group(0) @binding(0) var<storage, read> x: array<f32>;\n"
            "@group(0) @binding(1) var<storage, read_write> y: array<f32>;\n"
            "@compute @workgroup_size(64)\n"
            "fn softmax_kernel(@builtin(global_invocation_id) g: vec3u) {\n"
            "  let i = g.x;\n"
            "  var s = 0.0;\n"
            "  for (var j = 0u; j < 4u; j++) { s = s + exp(x[j]); }\n"
            "  y[i] = exp(x[i]) / s;\n"
            "}\n";
        char *j = wgsl_ml_analyze_json_src(src);
        CHECK(j != NULL, "sm: json");
        CHECK(strstr(j, "\"softmax\"") != NULL, "sm: pattern");
        CHECK(strstr(j, "\"is_compute\":true") != NULL, "sm: compute");
        CHECK(strstr(j, "\"bindings\"") != NULL, "sm: bindings");
        CHECK(strstr(j, "workgroup_hints") != NULL, "sm: hints");
        CHECK(strstr(j, "\"suggested_x\"") != NULL, "sm: suggested");
        wgsl_free_string(j);
    }

    /* Attention-named + dot */
    {
        const char *src =
            "@group(0) @binding(0) var<storage, read> Q: array<vec4f, 8>;\n"
            "@group(0) @binding(1) var<storage, read> K: array<vec4f, 8>;\n"
            "@group(0) @binding(2) var<storage, read> V: array<vec4f, 8>;\n"
            "@group(0) @binding(3) var<storage, read_write> O: array<vec4f, 8>;\n"
            "@compute @workgroup_size(32)\n"
            "fn attention_fwd(@builtin(local_invocation_id) l: vec3u) {\n"
            "  var denom = 0.0;\n"
            "  for (var j = 0u; j < 8u; j++) {\n"
            "    denom = denom + exp(dot(Q[l.x], K[j]));\n"
            "  }\n"
            "  let w = exp(dot(Q[l.x], K[0u])) / denom;\n"
            "  O[l.x] = V[0u] * w;\n"
            "}\n";
        char *j = wgsl_ml_analyze_json_src(src);
        CHECK(j != NULL, "attn: json");
        CHECK(strstr(j, "\"attention\"") != NULL, "attn: pattern tag");
        CHECK(strstr(j, "\"version\":1") != NULL, "attn: schema version");
        CHECK(strstr(j, "\"shape_flow\"") != NULL, "attn: shape flow");
        CHECK(strstr(j, "\"head_dim\":4") != NULL, "attn: head_dim");
        CHECK(strstr(j, "\"seq_len\":8") != NULL, "attn: seq_len");
        CHECK(strstr(j, "\"softmax_axis\":0") != NULL, "attn: softmax axis");
        CHECK(strstr(j, "\"rank\":2") != NULL, "attn: rank");
        CHECK(strstr(j, "\"shape\":[8,4]") != NULL, "attn: binding shape");
        wgsl_free_string(j);
    }

    /* Matmul dimensions from storage tensors */
    {
        const char *src =
            "@group(0) @binding(0) var<storage, read> A: array<vec4f, 8>;\n"
            "@group(0) @binding(1) var<storage, read> B: array<vec2f, 4>;\n"
            "@group(0) @binding(2) var<storage, read_write> C: array<vec2f, 8>;\n"
            "fn mm_row(a: vec4f) -> vec2f {\n"
            "  let b0 = B[0u]; let b1 = B[1u];\n"
            "  let b2 = B[2u]; let b3 = B[3u];\n"
            "  return vec2f(dot(a, vec4f(b0.x, b1.x, b2.x, b3.x)),\n"
            "               dot(a, vec4f(b0.y, b1.y, b2.y, b3.y)));\n"
            "}\n"
            "@compute @workgroup_size(8)\n"
            "fn matmul_main(@builtin(global_invocation_id) g: vec3u) {\n"
            "  C[g.x] = mm_row(A[g.x]);\n"
            "}\n";
        char *j = wgsl_ml_analyze_json_src(src);
        CHECK(j != NULL, "mm: json");
        CHECK(strstr(j, "\"matmul\"") != NULL, "mm: pattern");
        CHECK(strstr(j, "\"access\":\"read_write\"") != NULL, "mm: access");
        CHECK(strstr(j, "\"matmul\":{\"m\":8,\"n\":2,\"k\":4,\"broadcast\":false") != NULL,
              "mm: dims");
        CHECK(strstr(j, "\"autotune_protocol\":{\"version\":1") != NULL,
              "mm: autotune protocol");
        CHECK(strstr(j, "\"result_fields\":[\"workgroup_x\",\"ok\",\"elapsed_ns\"]") != NULL,
              "mm: protocol fields");
        wgsl_free_string(j);
    }

    /* Layer-norm signals */
    {
        const char *src =
            "fn layernorm(v: vec4f) -> vec4f {\n"
            "  let m = (v.x+v.y+v.z+v.w) * 0.25;\n"
            "  let inv = inverseSqrt(0.01);\n"
            "  return (v - vec4f(m,m,m,m)) * inv;\n"
            "}\n"
            "@compute @workgroup_size(1)\n"
            "fn main() { _ = layernorm(vec4f(1.0,2.0,3.0,4.0)); }\n";
        char *j = wgsl_ml_analyze_json_src(src);
        CHECK(j != NULL, "ln: json");
        CHECK(strstr(j, "layernorm") != NULL || strstr(j, "rsqrt") != NULL,
              "ln: layernorm or rsqrt signal");
        wgsl_free_string(j);
    }

    /* Binding layout extraction */
    {
        const char *src =
            "@group(1) @binding(2) var<uniform> U: vec4f;\n"
            "@group(0) @binding(0) var<storage, read_write> O: array<f32, 128>;\n"
            "@compute @workgroup_size(128)\n"
            "fn main() { O[0] = U.x; }\n";
        char *j = wgsl_ml_analyze_json_src(src);
        CHECK(j != NULL, "bind: json");
        CHECK(strstr(j, "\"group\":1") != NULL || strstr(j, "\"group\": 1") != NULL ||
              strstr(j, "\"binding\":2") != NULL, "bind: group/binding present");
        CHECK(strstr(j, "uniform") != NULL, "bind: uniform AS");
        CHECK(strstr(j, "storage") != NULL, "bind: storage AS");
        CHECK(strstr(j, "array_len") != NULL, "bind: shape");
        wgsl_free_string(j);
    }

    /* Elementwise / activation name */
    {
        const char *src =
            "fn gelu_act(x: f32) -> f32 { return x * 0.5; }\n"
            "@compute @workgroup_size(256)\n"
            "fn main(@builtin(global_invocation_id) g: vec3u) {\n"
            "  _ = gelu_act(f32(g.x));\n"
            "}\n";
        char *j = wgsl_ml_analyze_json_src(src);
        CHECK(j != NULL, "gelu: json");
        CHECK(strstr(j, "gelu") != NULL, "gelu: pattern");
        wgsl_free_string(j);
    }

    fprintf(stderr, "%s  test_ml_kernel  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
