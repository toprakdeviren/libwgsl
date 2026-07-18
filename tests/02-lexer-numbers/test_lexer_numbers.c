/**
 * Numeric literal lexer tests.
 *
 * Covers (per WGSL §3.5.2):
 *   decimal_int     0   123   0i   0u   42i   42u
 *   decimal_float   0.5   1.0   1.   .5   1e2   1.5e+2   2.0f   3.0h   1f   2h   0e1
 *   hex_int         0x0   0xFF   0xff   0xFFu   0xCAFEi
 *   hex_float       0x1.0p10   0x1.fp-3f   0xAp4   0x1p0h
 *   suffix flags    abstract / i / u / f / h, plus HEX bit
 *   leading-zero rule  012  →  error
 *   bad cases       0xZ  0xFF.   1e   0x1p
 */
#include "internal/lexer.h"
#include "internal/utf8.h"

#include <stdint.h>
#include <stdio.h>
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
    WGSLLexResult  result;
    int            ok;
} Lex;

static void lex_run(Lex *l, const char *text) {
    memset(l, 0, sizeof *l);
    wgsl_arena_init(&l->arena);
    wgsl_diag_init(&l->diag);
    wgsl_source_init(&l->src, text, strlen(text));
    l->ok = wgsl_tokenize(&l->src, &l->arena, &l->diag, &l->result);
}
static void lex_done(Lex *l) {
    wgsl_diag_destroy(&l->diag);
    wgsl_source_destroy(&l->src);
    wgsl_arena_destroy(&l->arena);
}

/* Find the first non-trivia, non-EOF token after a fresh lex. */
static const WGSLToken *first_value(const WGSLLexResult *r) {
    for (size_t i = 0; i < r->count; i++) {
        WGSLTokenKind k = r->tokens[i].kind;
        if (k == WGSL_TOK_BLANKSPACE) continue;
        if (k == WGSL_TOK_LINE_COMMENT || k == WGSL_TOK_BLOCK_COMMENT) continue;
        if (k == WGSL_TOK_EOF) return NULL;
        return &r->tokens[i];
    }
    return NULL;
}

static void expect_one(const char *src,
                       WGSLTokenKind want_kind,
                       uint32_t want_payload,
                       uint32_t want_length,
                       const char *tag)
{
    Lex l; lex_run(&l, src);
    if (!l.ok) {
        fprintf(stderr, "FAIL  %s: lex returned 0 for \"%s\"\n", tag, src);
        fail += 1;
        lex_done(&l);
        return;
    }
    const WGSLToken *t = first_value(&l.result);
    if (!t) {
        fprintf(stderr, "FAIL  %s: no value token in \"%s\"\n", tag, src);
        fail += 1;
        lex_done(&l);
        return;
    }
    if (t->kind != want_kind) {
        fprintf(stderr, "FAIL  %s: kind %s, want %s\n",
                tag, wgsl_token_kind_name(t->kind),
                wgsl_token_kind_name(want_kind));
        fail += 1;
    }
    if (t->payload != want_payload) {
        fprintf(stderr, "FAIL  %s: payload 0x%x, want 0x%x (\"%s\")\n",
                tag, t->payload, want_payload, src);
        fail += 1;
    }
    if (t->span.length != want_length) {
        fprintf(stderr, "FAIL  %s: length %u, want %u\n",
                tag, t->span.length, want_length);
        fail += 1;
    }
    lex_done(&l);
}

static void expect_lex_error(const char *src, const char *tag) {
    Lex l; lex_run(&l, src);
    if (l.ok != 0) {
        fprintf(stderr, "FAIL  %s: expected lex error on \"%s\"\n", tag, src);
        fail += 1;
    }
    if (!wgsl_diag_has_error(&l.diag)) {
        fprintf(stderr, "FAIL  %s: expected diag error on \"%s\"\n", tag, src);
        fail += 1;
    }
    lex_done(&l);
}

int main(void) {
    wgsl_utf8_init();

    /* — Decimal integer — */
    expect_one("0",    WGSL_TOK_INT_LIT, WGSL_NUM_ABSTRACT, 1, "0");
    expect_one("0i",   WGSL_TOK_INT_LIT, WGSL_NUM_SUFFIX_I, 2, "0i");
    expect_one("0u",   WGSL_TOK_INT_LIT, WGSL_NUM_SUFFIX_U, 2, "0u");
    expect_one("123",  WGSL_TOK_INT_LIT, WGSL_NUM_ABSTRACT, 3, "123");
    expect_one("42i",  WGSL_TOK_INT_LIT, WGSL_NUM_SUFFIX_I, 3, "42i");
    expect_one("42u",  WGSL_TOK_INT_LIT, WGSL_NUM_SUFFIX_U, 3, "42u");

    /* — Decimal float — */
    expect_one("0.5",     WGSL_TOK_FLOAT_LIT, WGSL_NUM_ABSTRACT, 3, "0.5");
    expect_one("1.0",     WGSL_TOK_FLOAT_LIT, WGSL_NUM_ABSTRACT, 3, "1.0");
    expect_one("1.",      WGSL_TOK_FLOAT_LIT, WGSL_NUM_ABSTRACT, 2, "1.");
    expect_one(".5",      WGSL_TOK_FLOAT_LIT, WGSL_NUM_ABSTRACT, 2, ".5");
    expect_one("1e2",     WGSL_TOK_FLOAT_LIT, WGSL_NUM_ABSTRACT, 3, "1e2");
    expect_one("1.5e+2",  WGSL_TOK_FLOAT_LIT, WGSL_NUM_ABSTRACT, 6, "1.5e+2");
    expect_one("2.0f",    WGSL_TOK_FLOAT_LIT, WGSL_NUM_SUFFIX_F, 4, "2.0f");
    expect_one("3.0h",    WGSL_TOK_FLOAT_LIT, WGSL_NUM_SUFFIX_H, 4, "3.0h");
    expect_one("1f",      WGSL_TOK_FLOAT_LIT, WGSL_NUM_SUFFIX_F, 2, "1f");
    expect_one("2h",      WGSL_TOK_FLOAT_LIT, WGSL_NUM_SUFFIX_H, 2, "2h");
    expect_one("0e1",     WGSL_TOK_FLOAT_LIT, WGSL_NUM_ABSTRACT, 3, "0e1");
    expect_one("0E1",     WGSL_TOK_FLOAT_LIT, WGSL_NUM_ABSTRACT, 3, "0E1");
    expect_one("1.5E-3f", WGSL_TOK_FLOAT_LIT, WGSL_NUM_SUFFIX_F, 7, "1.5E-3f");

    /* — Hex integer — */
    expect_one("0x0",     WGSL_TOK_INT_LIT,
               WGSL_NUM_ABSTRACT | WGSL_NUM_HEX, 3, "0x0");
    expect_one("0xFF",    WGSL_TOK_INT_LIT,
               WGSL_NUM_ABSTRACT | WGSL_NUM_HEX, 4, "0xFF");
    expect_one("0xff",    WGSL_TOK_INT_LIT,
               WGSL_NUM_ABSTRACT | WGSL_NUM_HEX, 4, "0xff");
    expect_one("0XFFu",   WGSL_TOK_INT_LIT,
               WGSL_NUM_SUFFIX_U | WGSL_NUM_HEX, 5, "0XFFu");
    expect_one("0xCAFEi", WGSL_TOK_INT_LIT,
               WGSL_NUM_SUFFIX_I | WGSL_NUM_HEX, 7, "0xCAFEi");

    /* — Hex float — */
    expect_one("0x1.0p10",  WGSL_TOK_FLOAT_LIT,
               WGSL_NUM_ABSTRACT | WGSL_NUM_HEX, 8, "0x1.0p10");
    expect_one("0x1.fp-3f", WGSL_TOK_FLOAT_LIT,
               WGSL_NUM_SUFFIX_F | WGSL_NUM_HEX, 9, "0x1.fp-3f");
    expect_one("0xAp4",     WGSL_TOK_FLOAT_LIT,
               WGSL_NUM_ABSTRACT | WGSL_NUM_HEX, 5, "0xAp4");
    expect_one("0x1p0h",    WGSL_TOK_FLOAT_LIT,
               WGSL_NUM_SUFFIX_H | WGSL_NUM_HEX, 6, "0x1p0h");
    /* hex_float with `.` but no exponent is also legal per spec rule 1+2 */
    expect_one("0x1.0",     WGSL_TOK_FLOAT_LIT,
               WGSL_NUM_ABSTRACT | WGSL_NUM_HEX, 5, "0x1.0");

    /* — Bad cases — */
    expect_lex_error("012",    "leading-zero non-zero int");
    expect_lex_error("0x",     "hex int without digits");
    expect_lex_error("1e",     "decimal float exponent without digits");
    expect_lex_error("0x1p",   "hex float exponent without digits");
    expect_lex_error("0x3h",   "hex int with h suffix");
    expect_lex_error("0xf.h",  "hex float trailing dot without exponent");

    if (fail == 0) {
        printf("PASS  test_lexer_numbers\n");
        return 0;
    }
    fprintf(stderr, "FAIL  test_lexer_numbers  %d check(s) failed\n", fail);
    return 1;
}
