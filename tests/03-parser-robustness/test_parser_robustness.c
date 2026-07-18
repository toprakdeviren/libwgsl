/**
 * Parser/lexer robustness against adversarial input.
 *
 * Covers parser recursion-depth guards, directive ordering, and
 * template-list nesting overflow.  Pathological nesting must produce a
 * clean diagnostic, never a stack-overflow crash.  Run this binary under
 * ASan/UBSan to prove memory-safety on the deep inputs.
 */
#include "internal/parser.h"
#include "internal/lexer.h"
#include "internal/diag.h"
#include "internal/utf8.h"

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

typedef struct {
    WGSLArena      arena;
    WGSLSource     src;
    WGSLDiagBag    diag;
    WGSLLexResult  lex;
    WGSLAst        ast;
    int            ok;
} P;

/* Drives lex + parse only (enough for all three items: 0.6e is emitted
 * during tokenize, 0.1 and 0.6a during parse). */
static void run(P *p, const char *text) {
    memset(p, 0, sizeof *p);
    wgsl_arena_init(&p->arena);
    wgsl_diag_init(&p->diag);
    wgsl_source_init(&p->src, text, strlen(text));
    int lex_ok   = wgsl_tokenize(&p->src, &p->arena, &p->diag, &p->lex);
    int parse_ok = wgsl_parse(&p->lex, &p->src, &p->arena, &p->diag, &p->ast);
    p->ok = lex_ok && parse_ok && !wgsl_diag_has_error(&p->diag);
}
static void done(P *p) {
    wgsl_diag_destroy(&p->diag);
    wgsl_source_destroy(&p->src);
    wgsl_arena_destroy(&p->arena);
}

static int has_diag(P *p, const char *needle) {
    size_t n = wgsl_diag_count(&p->diag);
    for (size_t i = 0; i < n; i++) {
        const WGSLDiagnostic *d = wgsl_diag_at(&p->diag, i);
        if (d && d->message && strstr(d->message, needle)) return 1;
    }
    return 0;
}

/* Build  prefix + rep*open + mid + rep*close + suffix  (heap; caller frees). */
static char *build_nest(const char *prefix, const char *open, const char *mid,
                        const char *close, int rep, const char *suffix) {
    size_t lp = strlen(prefix), lo = strlen(open), lm = strlen(mid);
    size_t lc = strlen(close),  ls = strlen(suffix);
    size_t total = lp + (size_t)rep * lo + lm + (size_t)rep * lc + ls + 1;
    char *b = (char *)malloc(total);
    if (!b) { fprintf(stderr, "OOM building test input\n"); exit(2); }
    char *w = b;
    memcpy(w, prefix, lp); w += lp;
    for (int i = 0; i < rep; i++) { memcpy(w, open, lo); w += lo; }
    memcpy(w, mid, lm); w += lm;
    for (int i = 0; i < rep; i++) { memcpy(w, close, lc); w += lc; }
    memcpy(w, suffix, ls); w += ls;
    *w = '\0';
    return b;
}

/* Deep pathological input: must reject cleanly (no crash) with the guard
 * message.  Running to completion under ASan is itself the crash test. */
static void expect_too_deep(const char *label, char *text, const char *needle) {
    P p; run(&p, text);
    CHECK(!p.ok, label);
    CHECK(has_diag(&p, needle), label);
    done(&p);
    free(text);
}

int main(void) {
    wgsl_utf8_init();

    /* Parser recursion-depth guard. */

    /* Nested parentheses: (((…1…))) */
    expect_too_deep("nested parens rejected",
        build_nest("const x = ", "(", "1", ")", 5000, ";"),
        "expression nested too deeply");

    /* Prefix operator chain: !!!!!true.  (Note: a run of `-` collapses
     * into `--` tokens and never recurses, so use `!`, which does not
     * combine.) */
    expect_too_deep("prefix chain rejected",
        build_nest("const x = ", "!", "true", "", 5000, ";"),
        "expression nested too deeply");

    /* Call nesting: f(f(f(…1…))) */
    expect_too_deep("call nesting rejected",
        build_nest("const x = ", "f(", "1", ")", 5000, ";"),
        "expression nested too deeply");

    /* Index nesting: a[a[a[…0…]]] */
    expect_too_deep("index nesting rejected",
        build_nest("const x = a", "[a", "[0]", "]", 5000, ";"),
        "expression nested too deeply");

    /* Nested blocks in a function body: {{{{…}}}} */
    expect_too_deep("nested blocks rejected",
        build_nest("fn f() ", "{", "", "}", 5000, ""),
        "statement nested too deeply");

    /* Control: an expression nested well under the cap still parses. */
    {
        char *ok_text = build_nest("const x = ", "(", "1", ")", 100, ";");
        P p; run(&p, ok_text);
        CHECK(p.ok, "under-cap parens still accepted");
        done(&p);
        free(ok_text);
    }

    /* Diagnostic directive ordering. */
    {
        P p; run(&p, "fn f() {}\ndiagnostic(off, derivative_uniformity);");
        CHECK(!p.ok, "diagnostic-after-decl rejected");
        CHECK(has_diag(&p, "'diagnostic' directive must appear before any declarations"),
              "diagnostic-after-decl message");
        done(&p);
    }
    {
        /* Control: diagnostic before any declaration is still accepted. */
        P p; run(&p, "diagnostic(off, derivative_uniformity);\nfn f() {}");
        CHECK(p.ok, "diagnostic-before-decl accepted");
        done(&p);
    }
    {
        /* Control: enable/requires ordering unchanged (regression guard). */
        P p; run(&p, "fn f() {}\nenable f16;");
        CHECK(!p.ok, "enable-after-decl still rejected");
        CHECK(has_diag(&p, "'enable' directive must appear before any declarations"),
              "enable-after-decl message");
        done(&p);
    }

    /* Template-list nesting overflow. */
    {
        /* 300 nested `vec4<…>` exceeds the §3.9 stack limit of 256. */
        char *deep = build_nest("alias A = ", "vec4<", "f32", ">", 300, ";");
        P p; run(&p, deep);
        CHECK(!p.ok, "deep template rejected");
        CHECK(has_diag(&p, "template list too deeply nested"),
              "deep template message");
        done(&p);
        free(deep);
    }
    {
        /* Control: a modestly nested template still parses. */
        P p; run(&p, "alias A = array<array<array<i32, 2>, 2>, 2>;");
        CHECK(p.ok, "0.6e modest template accepted");
        done(&p);
    }

    if (fail == 0) {
        printf("PASS  test_parser_robustness\n");
        return 0;
    }
    fprintf(stderr, "FAIL  test_parser_robustness  %d check(s) failed\n", fail);
    return 1;
}
