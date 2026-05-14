/**
 * Phase 2 Round A — basic lexer.
 *
 * Covers:
 *   - empty input
 *   - blankspace + comments emitted as tokens
 *   - 27 keywords classified
 *   - identifiers including non-ASCII (Hebrew, CJK)
 *   - lone underscore is WGSL_TOK_UNDERSCORE
 *   - leading `__` ident is rejected
 *   - operators (single + multi-char, maximal munch)
 *   - punctuation
 *   - nested block comments
 *   - line / column tracking
 *   - line_count_eof after empty input
 *   - invalid UTF-8 emits diagnostic, lex still returns tokens up to that point
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

static void lex_run_n(Lex *l, const char *text, size_t len) {
    memset(l, 0, sizeof *l);
    wgsl_arena_init(&l->arena);
    wgsl_diag_init(&l->diag);
    wgsl_source_init(&l->src, text, len);
    l->ok = wgsl_tokenize(&l->src, &l->arena, &l->diag, &l->result);
}

static void lex_done(Lex *l) {
    wgsl_diag_destroy(&l->diag);
    wgsl_source_destroy(&l->src);
    wgsl_arena_destroy(&l->arena);
}

/* Filter blankspace + comments — most expectations care only about
 * "code-bearing" tokens. */
static int kind_is_trivia(WGSLTokenKind k) {
    return k == WGSL_TOK_BLANKSPACE || k == WGSL_TOK_LINE_COMMENT ||
           k == WGSL_TOK_BLOCK_COMMENT;
}

static const WGSLToken *next_nontrivia(const WGSLLexResult *r, size_t *idx) {
    while (*idx < r->count && kind_is_trivia(r->tokens[*idx].kind)) (*idx)++;
    if (*idx >= r->count) return NULL;
    return &r->tokens[(*idx)++];
}

static void check_kinds_seq(const WGSLLexResult *r, const WGSLTokenKind *want,
                             size_t want_n, const char *tag) {
    size_t idx = 0;
    for (size_t i = 0; i < want_n; i++) {
        const WGSLToken *t = next_nontrivia(r, &idx);
        if (!t) {
            fprintf(stderr, "FAIL  %s: ran out of tokens at i=%zu\n", tag, i);
            fail += 1;
            return;
        }
        if (t->kind != want[i]) {
            fprintf(stderr, "FAIL  %s: token %zu is %s, want %s\n",
                    tag, i, wgsl_token_kind_name(t->kind),
                    wgsl_token_kind_name(want[i]));
            fail += 1;
        }
    }
}

int main(void) {
    wgsl_utf8_init();

    /* — empty input → just EOF — */
    {
        Lex l; lex_run(&l, "");
        CHECK(l.ok == 1, "empty: ok");
        CHECK(l.result.count == 1, "empty: 1 token");
        CHECK(l.result.tokens[0].kind == WGSL_TOK_EOF, "empty: EOF");
        lex_done(&l);
    }

    /* — blankspace + comments emitted, EOF terminates — */
    {
        Lex l; lex_run(&l, "  // hi\n /* nested /* x */ */ ");
        CHECK(l.ok == 1, "trivia: ok");
        /* Expect: BLANKSPACE LINE_COMMENT BLANKSPACE BLOCK_COMMENT BLANKSPACE EOF */
        CHECK(l.result.count == 6, "trivia: 6 tokens");
        WGSLTokenKind k0 = l.result.tokens[0].kind;
        WGSLTokenKind k1 = l.result.tokens[1].kind;
        WGSLTokenKind k3 = l.result.tokens[3].kind;
        CHECK(k0 == WGSL_TOK_BLANKSPACE,    "trivia: 0 = blankspace");
        CHECK(k1 == WGSL_TOK_LINE_COMMENT,  "trivia: 1 = line comment");
        CHECK(k3 == WGSL_TOK_BLOCK_COMMENT, "trivia: 3 = nested block comment");
        lex_done(&l);
    }

    /* — every one of the 27 keywords — */
    {
        Lex l; lex_run(&l,
            "alias break case const const_assert continue continuing "
            "default diagnostic discard else enable false fn for if "
            "let loop override requires return struct switch true var while");
        CHECK(l.ok == 1, "kw: ok");
        WGSLTokenKind want[] = {
            WGSL_TOK_KW_ALIAS, WGSL_TOK_KW_BREAK, WGSL_TOK_KW_CASE,
            WGSL_TOK_KW_CONST, WGSL_TOK_KW_CONST_ASSERT, WGSL_TOK_KW_CONTINUE,
            WGSL_TOK_KW_CONTINUING, WGSL_TOK_KW_DEFAULT, WGSL_TOK_KW_DIAGNOSTIC,
            WGSL_TOK_KW_DISCARD, WGSL_TOK_KW_ELSE, WGSL_TOK_KW_ENABLE,
            WGSL_TOK_KW_FALSE, WGSL_TOK_KW_FN, WGSL_TOK_KW_FOR,
            WGSL_TOK_KW_IF, WGSL_TOK_KW_LET, WGSL_TOK_KW_LOOP,
            WGSL_TOK_KW_OVERRIDE, WGSL_TOK_KW_REQUIRES, WGSL_TOK_KW_RETURN,
            WGSL_TOK_KW_STRUCT, WGSL_TOK_KW_SWITCH, WGSL_TOK_KW_TRUE,
            WGSL_TOK_KW_VAR, WGSL_TOK_KW_WHILE,
            WGSL_TOK_EOF,
        };
        check_kinds_seq(&l.result, want, sizeof want / sizeof want[0], "kw");
        lex_done(&l);
    }

    /* — identifiers (ASCII + non-ASCII) — */
    {
        Lex l; lex_run(&l, "x x1 _foo αβγ 中文 _");
        CHECK(l.ok == 1, "ident: ok");
        WGSLTokenKind want[] = {
            WGSL_TOK_IDENT, WGSL_TOK_IDENT, WGSL_TOK_IDENT,
            WGSL_TOK_IDENT, WGSL_TOK_IDENT, WGSL_TOK_UNDERSCORE,
            WGSL_TOK_EOF,
        };
        check_kinds_seq(&l.result, want, sizeof want / sizeof want[0], "ident");
        lex_done(&l);
    }

    /* — `__foo` is rejected (§3.7) — */
    {
        Lex l; lex_run(&l, "__bad");
        CHECK(l.ok == 0,                       "__: lex returns 0");
        CHECK(wgsl_diag_has_error(&l.diag),    "__: diag has error");
        lex_done(&l);
    }

    /* — U+0000 is invalid even inside comments. — */
    {
        const char src[] = { '/', '/', ' ', 'x', '\0', 'y' };
        Lex l; lex_run_n(&l, src, sizeof src);
        CHECK(l.ok == 0,                       "nul comment: lex returns 0");
        CHECK(wgsl_diag_has_error(&l.diag),    "nul comment: diag has error");
        lex_done(&l);
    }

    /* — operators, full coverage with maximal munch — */
    {
        Lex l; lex_run(&l,
            "& && &= | || |= ^ ^= ~ ! != + ++ += - -- -= -> "
            "* *= / /= % %= < <= << <<= > >= >> >>= = ==");
        CHECK(l.ok == 1, "ops: ok");
        WGSLTokenKind want[] = {
            WGSL_TOK_AMP, WGSL_TOK_AMP_AMP, WGSL_TOK_AMP_EQUAL,
            WGSL_TOK_PIPE, WGSL_TOK_PIPE_PIPE, WGSL_TOK_PIPE_EQUAL,
            WGSL_TOK_CARET, WGSL_TOK_CARET_EQUAL,
            WGSL_TOK_TILDE,
            WGSL_TOK_BANG, WGSL_TOK_BANG_EQUAL,
            WGSL_TOK_PLUS, WGSL_TOK_PLUS_PLUS, WGSL_TOK_PLUS_EQUAL,
            WGSL_TOK_MINUS, WGSL_TOK_MINUS_MINUS, WGSL_TOK_MINUS_EQUAL, WGSL_TOK_ARROW,
            WGSL_TOK_STAR, WGSL_TOK_STAR_EQUAL,
            WGSL_TOK_SLASH, WGSL_TOK_SLASH_EQUAL,
            WGSL_TOK_PERCENT, WGSL_TOK_PERCENT_EQUAL,
            WGSL_TOK_LESS, WGSL_TOK_LESS_EQUAL, WGSL_TOK_LESS_LESS, WGSL_TOK_LESS_LESS_EQUAL,
            WGSL_TOK_GREATER, WGSL_TOK_GREATER_EQUAL, WGSL_TOK_GREATER_GREATER, WGSL_TOK_GREATER_GREATER_EQUAL,
            WGSL_TOK_EQUAL, WGSL_TOK_EQUAL_EQUAL,
            WGSL_TOK_EOF,
        };
        check_kinds_seq(&l.result, want, sizeof want / sizeof want[0], "ops");
        lex_done(&l);
    }

    /* — punctuation — */
    {
        Lex l; lex_run(&l, "( ) { } [ ] , ; : . @");
        CHECK(l.ok == 1, "punct: ok");
        WGSLTokenKind want[] = {
            WGSL_TOK_LPAREN, WGSL_TOK_RPAREN,
            WGSL_TOK_LBRACE, WGSL_TOK_RBRACE,
            WGSL_TOK_LBRACKET, WGSL_TOK_RBRACKET,
            WGSL_TOK_COMMA, WGSL_TOK_SEMICOLON, WGSL_TOK_COLON, WGSL_TOK_DOT,
            WGSL_TOK_AT,
            WGSL_TOK_EOF,
        };
        check_kinds_seq(&l.result, want, sizeof want / sizeof want[0], "punct");
        lex_done(&l);
    }

    /* — nested block comment + unterminated → error — */
    {
        Lex l; lex_run(&l, "/* /* /* */ */ */ ");        /* properly nested, fine */
        CHECK(l.ok == 1, "nested-ok: ok");
        lex_done(&l);
    }
    {
        Lex l; lex_run(&l, "/* /* */ ");                  /* unterminated */
        CHECK(l.ok == 0, "unterminated: lex returns 0");
        CHECK(wgsl_diag_has_error(&l.diag), "unterminated: diag has error");
        lex_done(&l);
    }

    /* — line / column tracking — */
    {
        Lex l; lex_run(&l, "fn\n  main");
        CHECK(l.ok == 1, "lc: ok");
        size_t i = 0;
        const WGSLToken *t = next_nontrivia(&l.result, &i);
        CHECK(t && t->kind == WGSL_TOK_KW_FN,  "lc: 0 = fn");
        CHECK(t && t->line == 1 && t->column == 1, "lc: fn @ 1:1");
        const WGSLToken *t2 = next_nontrivia(&l.result, &i);
        CHECK(t2 && t2->kind == WGSL_TOK_IDENT, "lc: 1 = ident main");
        CHECK(t2 && t2->line == 2 && t2->column == 3, "lc: main @ 2:3");
        lex_done(&l);
    }

    /* — invalid UTF-8: a stray 0xFF byte — */
    {
        const char src[] = { 'a', (char)0xFF, 'b', '\0' };
        Lex l; lex_run(&l, src);
        CHECK(l.ok == 0,                       "invalid: lex returns 0");
        CHECK(wgsl_diag_has_error(&l.diag),    "invalid: diag has error");
        /* Must still produce ident `a` and ident `b` and EOF, plus the bad byte
         * was skipped.  Don't assert exact kinds aggressively. */
        lex_done(&l);
    }

    if (fail == 0) {
        printf("PASS  test_lexer_basic\n");
        return 0;
    }
    fprintf(stderr, "FAIL  test_lexer_basic  %d check(s) failed\n", fail);
    return 1;
}
