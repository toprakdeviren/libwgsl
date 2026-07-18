/**
 * Token stream coverage for the WebGPU hello triangle example.
 *
 * This test reads `examples/hello-triangle.wgsl`, runs the lexer on
 * it, and verifies:
 *
 *   - lex returns ok (no error diagnostics)
 *   - the file produces a recognizable token sequence (keywords,
 *     identifiers, attributes, operators in the right places)
 *   - line/column numbers track correctly (the `@vertex` attribute
 *     starts on line 5, column 1)
 *   - the trailing EOF token is present
 *
 * Template-list discovery is wired into `wgsl_tokenize`, so the
 * `<` / `>` around `array<vec2f, 3>` come out as
 * `TEMPLATE_START` / `TEMPLATE_END`.
 */
#include "internal/lexer.h"
#include "internal/utf8.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HELLO_PATH "examples/hello-triangle.wgsl"

static int fail = 0;

#define CHECK(cond, msg) do {                                          \
    if (!(cond)) {                                                     \
        fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);  \
        fail += 1;                                                     \
    }                                                                  \
} while (0)

static char *slurp(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "FAIL  cannot open %s (run from wgsl/ root?)\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[got] = '\0';
    *out_len = got;
    return buf;
}

/* Find the n-th non-trivia token (0-based). */
static const WGSLToken *nth_value(const WGSLLexResult *r, size_t n) {
    size_t seen = 0;
    for (size_t i = 0; i < r->count; i++) {
        WGSLTokenKind k = r->tokens[i].kind;
        if (k == WGSL_TOK_BLANKSPACE) continue;
        if (k == WGSL_TOK_LINE_COMMENT || k == WGSL_TOK_BLOCK_COMMENT) continue;
        if (seen == n) return &r->tokens[i];
        seen += 1;
    }
    return NULL;
}

static int count_kind(const WGSLLexResult *r, WGSLTokenKind k) {
    int n = 0;
    for (size_t i = 0; i < r->count; i++)
        if (r->tokens[i].kind == k) n += 1;
    return n;
}

int main(void) {
    wgsl_utf8_init();

    size_t len = 0;
    char  *src = slurp(HELLO_PATH, &len);
    if (!src) return 1;

    WGSLArena arena;     wgsl_arena_init(&arena);
    WGSLDiagBag diag;    wgsl_diag_init(&diag);
    WGSLSource source;   wgsl_source_init(&source, src, len);
    WGSLLexResult r = {0};

    int ok = wgsl_tokenize(&source, &arena, &diag, &r);

    CHECK(ok == 1,                    "lex returns ok");
    CHECK(!wgsl_diag_has_error(&diag), "no error diagnostic");
    CHECK(r.count > 0,                 "non-empty token stream");
    CHECK(r.tokens[r.count - 1].kind == WGSL_TOK_EOF, "ends with EOF");

    /* — Specific landmarks (non-trivia indices) — */
    const WGSLToken *t0 = nth_value(&r, 0);
    CHECK(t0 != NULL,                                   "tok[0] exists");
    CHECK(t0 && t0->kind == WGSL_TOK_AT,                "tok[0] is `@`");
    CHECK(t0 && t0->line == 5 && t0->column == 1,       "tok[0] @ 5:1");

    const WGSLToken *t1 = nth_value(&r, 1);
    CHECK(t1 && t1->kind == WGSL_TOK_IDENT,             "tok[1] is `vertex`");
    /* `vertex` is a context-dep attribute name, NOT a keyword. */

    const WGSLToken *t2 = nth_value(&r, 2);
    CHECK(t2 && t2->kind == WGSL_TOK_KW_FN,             "tok[2] is `fn`");

    const WGSLToken *t3 = nth_value(&r, 3);
    CHECK(t3 && t3->kind == WGSL_TOK_IDENT,             "tok[3] is `vs_main`");
    CHECK(t3 && t3->span.length == 7,                   "tok[3] length 7");

    /* The `let pos = array<vec2f, 3>(...)` line.  Track the LET keyword. */
    int saw_let = 0, saw_array = 0, saw_tpl_start = 0, saw_tpl_end = 0;
    int saw_less = 0, saw_greater = 0;
    int saw_arrow = 0, saw_return = 0, saw_lbracket = 0, saw_at = 0;
    for (size_t i = 0; i < r.count; i++) {
        WGSLTokenKind k = r.tokens[i].kind;
        if (k == WGSL_TOK_KW_LET)         saw_let       += 1;
        if (k == WGSL_TOK_LESS)           saw_less      += 1;
        if (k == WGSL_TOK_GREATER)        saw_greater   += 1;
        if (k == WGSL_TOK_TEMPLATE_START) saw_tpl_start += 1;
        if (k == WGSL_TOK_TEMPLATE_END)   saw_tpl_end   += 1;
        if (k == WGSL_TOK_ARROW)          saw_arrow     += 1;
        if (k == WGSL_TOK_KW_RETURN)      saw_return    += 1;
        if (k == WGSL_TOK_LBRACKET)       saw_lbracket  += 1;
        if (k == WGSL_TOK_AT)             saw_at        += 1;
        if (k == WGSL_TOK_IDENT && r.tokens[i].span.length == 5 &&
            memcmp(src + r.tokens[i].span.offset, "array", 5) == 0)
            saw_array += 1;
    }
    CHECK(saw_let == 1,         "exactly one `let`");
    CHECK(saw_array >= 1,       "at least one `array` ident");
    CHECK(saw_less == 0,        "no bare `<` after template discovery");
    CHECK(saw_greater == 0,     "no bare `>` after template discovery");
    CHECK(saw_tpl_start == 1,   "one TEMPLATE_START (`array<…>`)");
    CHECK(saw_tpl_end == 1,     "one TEMPLATE_END");
    CHECK(saw_arrow == 2,       "two `->` (one per fn header)");
    CHECK(saw_return == 2,      "two return statements");
    CHECK(saw_lbracket == 1,    "one `[` (pos[vid])");
    /* @vertex, @builtin, @builtin, @fragment, @location → 5 */
    CHECK(saw_at == 5,          "five `@` attribute introducers");

    /* All fn-call parens balance: count LPAREN == count RPAREN. */
    int lp = count_kind(&r, WGSL_TOK_LPAREN);
    int rp = count_kind(&r, WGSL_TOK_RPAREN);
    CHECK(lp == rp,          "( and ) balance");
    int lb = count_kind(&r, WGSL_TOK_LBRACE);
    int rb = count_kind(&r, WGSL_TOK_RBRACE);
    CHECK(lb == rb,          "{ and } balance");

    int n_floats   = count_kind(&r, WGSL_TOK_FLOAT_LIT);
    int n_ints     = count_kind(&r, WGSL_TOK_INT_LIT);
    CHECK(n_floats >= 10,    "at least 10 float literals");
    CHECK(n_ints >= 1,       "at least 1 int literal (the array-size 3)");

    if (fail == 0) {
        printf("PASS  test_lexer_hello  "
               "(file=%zu B, tokens=%zu, ints=%d, floats=%d, "
               "@=%d, ()=%d {}=%d)\n",
               len, r.count, n_ints, n_floats, saw_at, lp, lb);
    }

    wgsl_diag_destroy(&diag);
    wgsl_source_destroy(&source);
    wgsl_arena_destroy(&arena);
    free(src);

    if (fail == 0) return 0;
    fprintf(stderr, "FAIL  test_lexer_hello  %d check(s) failed\n", fail);
    return 1;
}
