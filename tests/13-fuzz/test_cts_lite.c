/**
 * CTS-lite fixture harness.
 *
 * A curated set of WGSL snippets modelled on shapes that WebGPU CTS /
 * real shaders stress (compute entry, IO attrs, arrays, atomics
 * surface, bad syntax).  Each case declares an expected verdict:
 *   expect_ok  — pipeline must succeed
 *   expect_err — pipeline must fail with ≥1 error
 *   expect_any — crash-free only (ok or err)
 *
 * External CTS trees can be bulk-imported via tools/import-cts-wgsl.sh
 * into .build/cts-import/ and exercised with `make test-cts-import`
 * (optional; not part of the default gate).
 */
#include "wgsl.h"

#include <stdio.h>
#include <string.h>

static int fail;
#define CHECK(cond, msg) do {                                          \
    if (!(cond)) {                                                     \
        fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);  \
        fail += 1;                                                     \
    }                                                                  \
} while (0)

typedef enum {
    EXP_OK = 0,
    EXP_ERR,
    EXP_ANY,
} Expect;

typedef struct {
    const char *name;
    Expect      exp;
    const char *src;
} Case;

static const Case k_cases[] = {
    /* accept. */
    {
        "empty_module", EXP_OK,
        "",
    },
    {
        "trivial_fn", EXP_OK,
        "fn f() {}\n",
    },
    {
        "compute_entry", EXP_OK,
        "@compute @workgroup_size(1)\n"
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
        "  _ = gid.x;\n"
        "}\n",
    },
    {
        "struct_io_fragment", EXP_OK,
        "struct FragIn {\n"
        "  @location(0) color: vec4f,\n"
        "}\n"
        "@fragment\n"
        "fn fs(input: FragIn) -> @location(0) vec4f {\n"
        "  return input.color;\n"
        "}\n",
    },
    {
        "const_array", EXP_OK,
        "const N: u32 = 4u;\n"
        "fn f() {\n"
        "  var a: array<f32, N>;\n"
        "  a[0] = 1.0;\n"
        "  _ = a[0];\n"
        "}\n",
    },
    {
        "control_flow", EXP_OK,
        "fn f(x: i32) -> i32 {\n"
        "  var y = x;\n"
        "  if (y > 0) { y = y - 1; } else { y = y + 1; }\n"
        "  loop {\n"
        "    if (y <= 0) { break; }\n"
        "    y = y - 1;\n"
        "    continuing { y = y; }\n"
        "  }\n"
        "  return y;\n"
        "}\n",
    },
    {
        "vec_swizzle", EXP_OK,
        "fn f() {\n"
        "  let v = vec4f(1.0, 2.0, 3.0, 4.0);\n"
        "  let s = v.xy;\n"
        "  _ = s.x + s.y;\n"
        "}\n",
    },
    {
        "math_builtins", EXP_OK,
        "fn f(a: f32, b: f32) -> f32 {\n"
        "  return abs(a) + max(a, b) * min(a, b);\n"
        "}\n",
    },
    {
        "enable_f16", EXP_OK,
        "enable f16;\n"
        "fn f(x: f16) -> f16 { return x + 1.0h; }\n",
    },
    {
        "alias_and_const", EXP_OK,
        "alias V3 = vec3<f32>;\n"
        "const O: V3 = V3(0.0, 0.0, 0.0);\n"
        "fn f() { _ = O.x; }\n",
    },
    {
        "phony_assign", EXP_OK,
        "fn f() { _ = 1; _ = vec2i(1, 2); }\n",
    },

    /* reject. */
    {
        "bad_brace", EXP_ERR,
        "fn f( { }\n",
    },
    {
        "type_mismatch", EXP_ERR,
        "fn f() { let x: i32 = 1u; }\n",
    },
    {
        "undeclared", EXP_ERR,
        "fn f() { _ = ghost; }\n",
    },
    {
        "div_zero_const", EXP_ERR,
        "const X: i32 = 1 / 0;\n",
    },
    {
        "missing_location", EXP_ERR,
        "@fragment\n"
        "fn fs(color: vec4f) -> vec4f { return color; }\n",
    },
    {
        "call_entry_point", EXP_ERR,
        "@compute @workgroup_size(1)\n"
        "fn main() {}\n"
        "fn g() { main(); }\n",
    },

    /* any (crash-free; may ok or err depending on front-end depth). */
    {
        "nested_templates_noise", EXP_ANY,
        "fn f() { let x = array<array<array<i32, 2>, 2>, 2>(); _ = x; }\n",
    },
    {
        "deep_parens", EXP_ANY,
        "fn f() { let x = (((((((((1))))))))); _ = x; }\n",
    },
    {
        "unicode_ident_comment", EXP_ANY,
        "// café ☕\n"
        "fn f() { let x: i32 = 1; _ = x; }\n",
    },
};

static int run_case(const Case *c) {
    WGSLResult *r = wgsl_check(c->src);
    if (!r) {
        fprintf(stderr, "FAIL  %s: wgsl_check returned NULL\n", c->name);
        return 0;
    }
    int ok = wgsl_ok(r);
    int pass = 0;
    switch (c->exp) {
    case EXP_OK:
        pass = ok;
        if (!pass) {
            fprintf(stderr, "FAIL  %s: expected ok, got errors: %s\n",
                    c->name, wgsl_error(r));
        }
        break;
    case EXP_ERR:
        pass = !ok;
        if (!pass) {
            fprintf(stderr, "FAIL  %s: expected error, got ok\n", c->name);
        }
        break;
    case EXP_ANY:
        pass = 1; /* mere survival */
        break;
    }
    /* format must not crash either */
    char *fmt = wgsl_format(c->src);
    if (fmt) wgsl_free_string(fmt);
    wgsl_free(r);
    return pass;
}

int main(void) {
    int n = (int)(sizeof k_cases / sizeof k_cases[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (run_case(&k_cases[i])) passed++;
        else fail++;
    }
    fprintf(stderr, "%s  test_cts_lite  (%d/%d cases, %d failure%s)\n",
            fail ? "FAIL" : "PASS", passed, n,
            fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
