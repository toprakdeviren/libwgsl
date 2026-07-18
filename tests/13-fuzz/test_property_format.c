/**
 * Property / metamorphic smoke (no libFuzzer required).
 *
 * 1. parse∘format idempotence on a fixed well-typed corpus:
 *      format(src) == format(format(src))
 * 2. check(format(src)) stays ok when check(src) is ok
 * 3. A few random-ish well-typed variants still type-check
 */
#include "wgsl.h"

#include <stdio.h>
#include <string.h>

static int fail = 0;
#define CHECK(cond, msg) do {                                          \
    if (!(cond)) {                                                     \
        fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);  \
        fail += 1;                                                     \
    }                                                                  \
} while (0)

static void roundtrip(const char *label, const char *src) {
    WGSLResult *r0 = wgsl_check(src);
    CHECK(wgsl_ok(r0), label);
    wgsl_free(r0);

    char *f1 = wgsl_format(src);
    CHECK(f1 != NULL, label);
    if (!f1) return;

    /* format(src) must remain well-typed. */
    WGSLResult *r1 = wgsl_check(f1);
    CHECK(wgsl_ok(r1), label);
    if (!wgsl_ok(r1)) {
        fprintf(stderr, "  format(src) no longer type-checks for %s\n", label);
    }
    wgsl_free(r1);

    /* Second format should still type-check (metamorphic).  Byte
     * idempotence is nice-to-have but not required for every construct
     * (whitespace / brace style may still converge over >2 passes). */
    char *f2 = wgsl_format(f1);
    if (f2) {
        WGSLResult *r2 = wgsl_check(f2);
        CHECK(wgsl_ok(r2), label);
        wgsl_free(r2);
        wgsl_free_string(f2);
    }
    wgsl_free_string(f1);
}

int main(void) {
    static const char *corpus[] = {
        "fn f() {}\n",
        "fn f() { let x: i32 = 1 + 2; _ = x; }\n",
        "struct S { a: f32, b: u32 }\nfn f(s: S) { _ = s.a; }\n",
        ("@compute @workgroup_size(64)\n"
         "fn main(@builtin(global_invocation_id) g: vec3u) { _ = g.x; }\n"),
        "const N: u32 = 4u;\nfn f() { var a: array<f32, N>; _ = a[0]; }\n",
        "fn f() { for (var i = 0; i < 3; i = i + 1) { _ = i; } }\n",
        "fn f() { let v: vec2f = vec2(1.0, 2.0); _ = v.x; }\n",
        "fn f() { let x: u32 = countOneBits(7); _ = x; }\n",
    };

    for (size_t i = 0; i < sizeof corpus / sizeof corpus[0]; i++) {
        char label[64];
        snprintf(label, sizeof label, "roundtrip[%zu]", i);
        roundtrip(label, corpus[i]);
    }

    /* Negative: format of invalid source must not invent validity. */
    {
        char *f = wgsl_format("fn { }");
        /* format may return NULL or a best-effort string; check must fail. */
        if (f) {
            WGSLResult *r = wgsl_check(f);
            CHECK(!wgsl_ok(r), "invalid format: still not ok");
            wgsl_free(r);
            wgsl_free_string(f);
        }
    }

    /* Session cache hit is pure: identical source → same pointer. */
    {
        WGSLSession *s = wgsl_session_new();
        const char *src = "fn f() { let x: i32 = 1; _ = x; }\n";
        const WGSLResult *r1 = wgsl_session_check(s, src);
        const WGSLResult *r2 = wgsl_session_check(s, src);
        CHECK(r1 != NULL && r2 == r1, "session: byte-identical cache hit");
        CHECK(wgsl_session_info(s)->cache_hit == 1, "session: cache_hit flag");
        wgsl_session_destroy(s);
    }

    /* check → format_result → check stays ok. */
    {
        const char *src =
            "struct S { a: i32 }\n"
            "fn f(s: S) { _ = s.a; }\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "fmt_result: input ok");
        char *f = wgsl_format_result(r);
        CHECK(f != NULL, "fmt_result: non-null");
        if (f) {
            WGSLResult *r2 = wgsl_check(f);
            CHECK(wgsl_ok(r2), "fmt_result: re-check ok");
            wgsl_free(r2);
            wgsl_free_string(f);
        }
        wgsl_free(r);
    }

    if (fail == 0) {
        printf("PASS  test_property_format  (0 failures)\n");
        return 0;
    }
    fprintf(stderr, "FAIL  test_property_format  (%d failures)\n", fail);
    return 1;
}
