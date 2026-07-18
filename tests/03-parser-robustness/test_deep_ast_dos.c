/**
 * Stack-overflow DoS guard for adversarially deep ASTs.
 *
 * The parser caps its OWN recursion, but left-leaning chains built
 * iteratively (`a+a+…`, `a[0][0]…`, `a.b.b…`, `a()()…`) produce an
 * arbitrarily deep AST with no parser recursion.  Every downstream pass
 * (resolver, const-eval, type-checker, validator, ast-dump) then walks
 * that depth recursively and would overflow the call stack.
 *
 * Each such pass now carries a WGSL_MAX_AST_DEPTH guard.  This test drives
 * the FULL public pipeline (`wgsl_check`) on each vector and asserts it is
 * rejected with a "too deeply" diagnostic — and, simply by running to
 * completion, that it does NOT crash.  Run under ASan to prove the deep
 * inputs are memory-safe (no stack overflow).
 */
#include "wgsl.h"

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

/* Build  prefix + rep*open + mid + rep*close + suffix  (heap; caller frees). */
static char *build_nest(const char *prefix, const char *open, const char *mid,
                        const char *close, int rep, const char *suffix) {
    size_t lp = strlen(prefix), lo = strlen(open), lm = strlen(mid);
    size_t lc = strlen(close),  ls = strlen(suffix);
    size_t total = lp + (size_t)rep * lo + lm + (size_t)rep * lc + ls + 1;
    char *b = (char *)malloc(total);
    if (!b) { fprintf(stderr, "OOM\n"); exit(2); }
    char *w = b;
    memcpy(w, prefix, lp); w += lp;
    for (int i = 0; i < rep; i++) { memcpy(w, open, lo); w += lo; }
    memcpy(w, mid, lm); w += lm;
    for (int i = 0; i < rep; i++) { memcpy(w, close, lc); w += lc; }
    memcpy(w, suffix, ls); w += ls;
    *w = '\0';
    return b;
}

static int has_diag(WGSLResult *r, const char *needle) {
    int n = wgsl_diagnostic_count(r);
    for (int i = 0; i < n; i++) {
        const WGSLDiagnostic *d = wgsl_diagnostic(r, i);
        if (d && d->message && strstr(d->message, needle)) return 1;
    }
    return 0;
}

/* Rejected cleanly (no crash) with a depth diagnostic; frees src. */
static void deep_rejected(const char *label, char *src) {
    WGSLResult *r = wgsl_check(src);
    CHECK(r != NULL, label);
    if (r) {
        CHECK(!wgsl_ok(r), label);              /* must reject */
        CHECK(has_diag(r, "too deeply"), label); /* with a depth diagnostic */
        wgsl_free(r);
    }
    free(src);
}

int main(void) {
    /* Depth chosen well past WGSL_MAX_AST_DEPTH (512) so every downstream
     * walker would recurse past its guard; small enough to stay fast. */
    const int N = 4000;

    /* Left-leaning binary chain — resolver / const-eval / type-check /
     * validate all recurse on the spine. */
    deep_rejected("binary chain rejected",
        build_nest("const c : i32 = 1", "+1", "", "", N, ";"));

    /* Postfix index chain — a[0][0][0]… */
    deep_rejected("index chain rejected",
        build_nest("const c = a", "[0]", "", "", N, ";"));

    /* Postfix call chain — a()()()… */
    deep_rejected("call chain rejected",
        build_nest("const c = a", "()", "", "", N, ";"));

    /* Postfix member chain — a.b.b.b… */
    deep_rejected("member chain rejected",
        build_nest("const c = a", ".b", "", "", N, ";"));

    /* Deep parenthesised expression inside a function body. */
    deep_rejected("paren chain (fn) rejected",
        build_nest("fn m() { let x = ", "(", "1", ")", N, "; }"));

    /* Deep nested blocks in a function body. */
    deep_rejected("nested blocks rejected",
        build_nest("fn f() ", "{", "", "}", N, ""));

    /* Deeply nested statement (if) — bounded by the parser cap. */
    deep_rejected("nested if rejected",
        build_nest("fn f() { ", "if (true) {", "", "}", N, " }"));

    /* Deeply nested type specifier — bounded by the §3.9 template cap. */
    deep_rejected("nested type-spec rejected",
        build_nest("alias A = ", "array<", "i32", ">", N, ";"));

    /* Public LSP entry points also walk the AST — a deep chain reached
     * through wgsl_semantic_tokens / wgsl_hover_at / wgsl_definition_at
     * must not overflow the stack.  Running to completion is the test. */
    {
        char *src = build_nest("const c = a", "+a", "", "", N, ";");

        WGSLLexToken *toks = NULL; int ntok = 0;
        wgsl_semantic_tokens(src, &toks, &ntok);   /* must not crash */
        CHECK(ntok >= 0, "semantic_tokens: no crash on deep chain");
        wgsl_lex_free(toks);

        WGSLResult *r = wgsl_check(src);
        CHECK(r != NULL, "deep chain: result for LSP queries");
        if (r) {
            /* Offset near the end lands deep in the left-leaning spine. */
            uint32_t off = (uint32_t)strlen(src) - 2;
            WGSLHover h = wgsl_hover_at(r, off);       /* must not crash */
            (void)h;
            WGSLDefinition d = wgsl_definition_at(r, off); /* must not crash */
            (void)d;
            CHECK(1, "hover/definition: no crash on deep chain");
            wgsl_free(r);
        }
        free(src);
    }

    /* Deep TYPE-GRAPH chain: `struct S1{a:S0} struct S2{a:S1} …` — many
     * shallow declarations form a resolved-type chain of depth N with no
     * deep AST and no template `<…>` nesting.  The type-predicate walkers
     * (is_host_shareable_deep, contains_runtime_array, …) recurse on it and
     * must not overflow the stack.  Running to completion is the test. */
    {
        int Ns = 1500;   /* well past WGSL_MAX_AST_DEPTH (512) */
        /* Each line "struct Sk { a: Sk-1 }\n" is < 40 bytes. */
        size_t cap = (size_t)Ns * 48 + 256;
        char *src = (char *)malloc(cap);
        CHECK(src != NULL, "type-chain: alloc");
        if (src) {
            size_t w = 0;
            w += (size_t)snprintf(src + w, cap - w, "struct S0 { a: u32 }\n");
            for (int i = 1; i < Ns; i++) {
                w += (size_t)snprintf(src + w, cap - w,
                                      "struct S%d { a: S%d }\n", i, i - 1);
            }
            w += (size_t)snprintf(src + w, cap - w,
                "@group(0) @binding(0) var<storage> b: S%d;\n"
                "@compute @workgroup_size(1) fn main() {}\n", Ns - 1);

            WGSLResult *r = wgsl_check(src);   /* must not crash */
            CHECK(r != NULL, "type-chain: result");
            if (r) {
                /* A 1500-deep storage struct is not host-shareable at that
                 * depth → rejected; the point is it did not crash. */
                CHECK(!wgsl_ok(r), "deep struct type-chain rejected (no crash)");
                wgsl_free(r);
            }
            free(src);
        }
    }

    /* Deep chain inside a `continuing` block — exercises the check pass's
     * ad-hoc walkers (collect_continuing_uses / scan_continues_before), which
     * do NOT route through the type-expr chokepoint.  The pass-entry AST-depth
     * probe must bound them.  Must not crash. */
    {
        char *src = build_nest(
            "fn f() { var i = 0; loop { if (i > 0) { break; } continuing { i = i",
            "+i", "", "", N, "; } } }");
        WGSLResult *r = wgsl_check(src);
        CHECK(r != NULL, "continuing-chain: result");
        if (r) {
            CHECK(!wgsl_ok(r), "continuing-block chain rejected (no crash)");
            wgsl_free(r);
        }
        free(src);
    }

    /* Deep struct type-chain used in a var<uniform> with an @size'd wrapper —
     * exercises the layout walkers (wgsl_layout_size_of / align_of / struct),
     * which recurse on the resolved type graph.  Must not crash *or hang*
     * (they were O(2^depth) before computing member align+size in one pass). */
    {
        int Ns = 1500;
        size_t cap = (size_t)Ns * 48 + 256;
        char *src = (char *)malloc(cap);
        CHECK(src != NULL, "layout-chain: alloc");
        if (src) {
            size_t w = 0;
            w += (size_t)snprintf(src + w, cap - w, "struct S0 { x: i32 }\n");
            for (int i = 1; i < Ns; i++)
                w += (size_t)snprintf(src + w, cap - w,
                                      "struct S%d { m: S%d }\n", i, i - 1);
            w += (size_t)snprintf(src + w, cap - w,
                "struct Top { @size(65536) m: S%d }\n"
                "@group(0) @binding(0) var<uniform> u: Top;\n"
                "@compute @workgroup_size(1) fn main() {}\n", Ns - 1);
            WGSLResult *r = wgsl_check(src);   /* must not crash or hang */
            CHECK(r != NULL, "layout-chain: result");
            if (r) {
                CHECK(!wgsl_ok(r), "deep @size layout chain rejected (no crash/hang)");
                wgsl_free(r);
            }
            free(src);
        }
    }

    /* Control: a normal program with modest nesting still validates. */
    {
        WGSLResult *r = wgsl_check(
            "const K : i32 = 1 + 2 * (3 - 4);\n"
            "fn f() -> i32 {\n"
            "  var s = 0;\n"
            "  for (var i = 0; i < 10; i = i + 1) {\n"
            "    if (i > 5) { s = s + i; } else { s = s - i; }\n"
            "  }\n"
            "  return s;\n"
            "}\n");
        CHECK(r != NULL, "control: result");
        if (r) {
            CHECK(wgsl_ok(r), "control: modest program still validates");
            wgsl_free(r);
        }
    }

    if (fail == 0) {
        printf("PASS  test_deep_ast_dos\n");
        return 0;
    }
    fprintf(stderr, "FAIL  test_deep_ast_dos  %d check(s) failed\n", fail);
    return 1;
}
