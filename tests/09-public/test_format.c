/**
 * §3.3 — pretty-printer: parse∘format idempotence + re-check.
 */
#include "wgsl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail;
#define CHECK(c, m) do { if (!(c)) { fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,m); fail++; } } while(0)

int main(void) {
    /* Simple function */
    {
        const char *src = "fn add(a:i32,b:i32)->i32{return a+b;}";
        char *f1 = wgsl_format(src);
        CHECK(f1 != NULL, "format non-null");
        CHECK(strstr(f1, "fn add") != NULL, "has fn add");
        CHECK(strstr(f1, "return") != NULL, "has return");
        char *f2 = wgsl_format(f1);
        CHECK(f2 != NULL, "format2");
        CHECK(strcmp(f1, f2) == 0, "idempotent format∘format");
        WGSLResult *r = wgsl_check(f1);
        CHECK(wgsl_ok(r), "formatted source still checks");
        wgsl_free(r);
        free(f1); free(f2);
    }

    /* Struct + attrs */
    {
        const char *src =
            "@group(0)@binding(0)var<storage,read_write>B:array<f32>;\n"
            "struct S{a:i32,b:f32}\n"
            "@compute@workgroup_size(64)fn main(@builtin(global_invocation_id)gid:vec3<u32>){B[gid.x]=1.0;}";
        char *f = wgsl_format(src);
        CHECK(f != NULL, "format compute");
        CHECK(strstr(f, "struct S") != NULL, "struct");
        CHECK(strstr(f, "@compute") != NULL || strstr(f, "compute") != NULL, "compute attr");
        WGSLResult *r = wgsl_check(f);
        CHECK(wgsl_ok(r), "formatted compute ok");
        char *fr = wgsl_format_result(r);
        CHECK(fr != NULL && strlen(fr) > 0, "format_result");
        free(fr);
        wgsl_free(r);
        free(f);
    }

    /* enable */
    {
        char *f = wgsl_format("enable f16;fn f()->f16{return 1.0h;}");
        CHECK(f && strstr(f, "enable f16"), "enable kept");
        free(f);
    }

    /* Canonical blank-line policy: exactly one blank line between top-levels. */
    {
        char *f = wgsl_format("enable f16;const A:i32=1;fn f(){}fn g(){}");
        CHECK(f != NULL, "blank policy: format");
        CHECK(strstr(f, "enable f16;\n\nconst A: i32 = 1;\n\nfn f()") != NULL,
              "blank policy: one blank line before const/fn");
        CHECK(strstr(f, "\n\n\n") == NULL, "blank policy: no triple newline");
        free(f);
    }

    /* Selection/range formatting rewrites only the touched top-level unit. */
    {
        const char *src =
            "fn a(){let x=1;}\n"
            "fn b(){let y=2;}\n";
        const char *needle = strstr(src, "y=2");
        uint32_t es = 0, ee = 0;
        char *f = wgsl_format_range_n(src, strlen(src),
                                      (uint32_t)(needle - src),
                                      (uint32_t)(needle - src),
                                      &es, &ee);
        CHECK(f != NULL, "range format: non-null");
        CHECK(es == (uint32_t)(strstr(src, "fn b") - src),
              "range format: starts at touched decl");
        CHECK(ee == (uint32_t)strlen(src) - 1,
              "range format: ends at touched decl");
        CHECK(strstr(f, "fn b()") != NULL, "range format: contains b");
        CHECK(strstr(f, "fn a") == NULL, "range format: excludes a");
        CHECK(strstr(f, "let y = 2;") != NULL, "range format: formats body");
        free(f);
    }

    /* empty */
    {
        char *f = wgsl_format("");
        CHECK(f != NULL, "empty formats");
        free(f);
    }

    /* Trivia: leading + trailing line comments survive. */
    {
        const char *src =
            "// file header\n"
            "fn add(a: i32, b: i32) -> i32 {\n"
            "    return a + b; // sum\n"
            "}\n"
            "/* trailer */\n";
        char *f = wgsl_format(src);
        CHECK(f != NULL, "trivia: non-null");
        CHECK(strstr(f, "// file header") != NULL, "trivia: leading comment");
        CHECK(strstr(f, "// sum") != NULL, "trivia: trailing comment");
        CHECK(strstr(f, "/* trailer */") != NULL, "trivia: trailing block");
        /* Idempotent even with comments. */
        char *f2 = wgsl_format(f);
        CHECK(f2 != NULL && strcmp(f, f2) == 0, "trivia: format∘format");
        WGSLResult *r = wgsl_check(f);
        CHECK(wgsl_ok(r), "trivia: still checks");
        wgsl_free(r);
        free(f); free(f2);
    }

    /* Trivia: comments between top-level decls. */
    {
        const char *src =
            "const A: i32 = 1;\n"
            "// mid\n"
            "fn f() {}\n";
        char *f = wgsl_format(src);
        CHECK(f != NULL && strstr(f, "// mid") != NULL, "trivia: mid comment");
        free(f);
    }

    /* Trivia: block comment before struct member. */
    {
        const char *src =
            "struct S {\n"
            "    /* x coord */\n"
            "    x: f32,\n"
            "}\n";
        char *f = wgsl_format(src);
        CHECK(f != NULL && strstr(f, "/* x coord */") != NULL,
              "trivia: member block comment");
        free(f);
    }

    fprintf(stderr, "%s  test_format  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
