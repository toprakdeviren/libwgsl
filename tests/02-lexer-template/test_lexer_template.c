/**
 * WGSL §3.9 template-list discovery tests.
 *
 * Validates that ambiguous `<` / `>` characters are retagged
 * `TEMPLATE_START` / `TEMPLATE_END` exactly when they delimit a
 * template list, and left as comparison / shift operators otherwise.
 *
 * Coverage map:
 *
 *   1. `vec3<f32>`                 — basic template
 *   2. `array<vec2f, 3>`           — type + non-type arg
 *   3. `array<vec4<f32>>`          — `>>` split into TEMPLATE_END + GREATER
 *                                    where the second `>` also closes
 *   4. `var<storage, read_write>`  — keyword-prefixed template
 *   5. `a < b`                     — bare comparison: no template
 *   6. `a < b;`                    — `;` clears stack: no template
 *   7. `a < b > c`                 — ambiguity wins for template
 *   8. `a >= b`                    — `>=` stays a comparison op
 *   9. `vec<f32> = expr`           — template + assignment
 *  10. `if (a < b) { … }`          — `<` inside parens, no `>` to close
 *
 *   plus: `>>=` and `>=` splitting cases
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

static int count_kind(const WGSLLexResult *r, WGSLTokenKind k) {
    int n = 0;
    for (size_t i = 0; i < r->count; i++) if (r->tokens[i].kind == k) n += 1;
    return n;
}

/* Match an exact non-trivia kind sequence. */
static void expect_seq(const char *src,
                       const WGSLTokenKind *want, size_t want_n,
                       const char *tag)
{
    Lex l; lex_run(&l, src);
    CHECK(l.ok == 1, "lex ok (during expect_seq)");

    size_t out_idx = 0;
    for (size_t i = 0; i < l.result.count; i++) {
        WGSLTokenKind k = l.result.tokens[i].kind;
        if (k == WGSL_TOK_BLANKSPACE || k == WGSL_TOK_LINE_COMMENT ||
            k == WGSL_TOK_BLOCK_COMMENT) continue;
        if (out_idx >= want_n) {
            fprintf(stderr,
                    "FAIL  %s: extra token at non-trivia idx %zu = %s (src=\"%s\")\n",
                    tag, out_idx, wgsl_token_kind_name(k), src);
            fail += 1;
            lex_done(&l);
            return;
        }
        if (k != want[out_idx]) {
            fprintf(stderr,
                    "FAIL  %s: token %zu is %s, want %s (src=\"%s\")\n",
                    tag, out_idx, wgsl_token_kind_name(k),
                    wgsl_token_kind_name(want[out_idx]), src);
            fail += 1;
            lex_done(&l);
            return;
        }
        out_idx += 1;
    }
    if (out_idx != want_n) {
        fprintf(stderr, "FAIL  %s: missing tokens (got %zu, want %zu) (src=\"%s\")\n",
                tag, out_idx, want_n, src);
        fail += 1;
    }
    lex_done(&l);
}

#define E(...) (WGSLTokenKind[]){ __VA_ARGS__ }
#define EN(...) (sizeof(E(__VA_ARGS__)) / sizeof(WGSLTokenKind))

int main(void) {
    wgsl_utf8_init();

    /* — 1. `vec3<f32>` — */
    expect_seq("vec3<f32>",
        E(WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_START,
          WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_END, WGSL_TOK_EOF),
        EN(WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_START,
           WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_END, WGSL_TOK_EOF),
        "vec3<f32>");

    /* — 2. `array<vec2f, 3>` — */
    expect_seq("array<vec2f, 3>",
        E(WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_START,
          WGSL_TOK_IDENT, WGSL_TOK_COMMA, WGSL_TOK_INT_LIT,
          WGSL_TOK_TEMPLATE_END, WGSL_TOK_EOF),
        EN(WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_START,
           WGSL_TOK_IDENT, WGSL_TOK_COMMA, WGSL_TOK_INT_LIT,
           WGSL_TOK_TEMPLATE_END, WGSL_TOK_EOF),
        "array<vec2f,3>");

    /* — 3. `array<vec4<f32>>` — `>>` split — */
    expect_seq("array<vec4<f32>>",
        E(WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_START,
          WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_START,
          WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_END,
          WGSL_TOK_TEMPLATE_END, WGSL_TOK_EOF),
        EN(WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_START,
           WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_START,
           WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_END,
           WGSL_TOK_TEMPLATE_END, WGSL_TOK_EOF),
        "array<vec4<f32>>");

    /* — 4. `var<storage, read_write>` — keyword-prefixed — */
    expect_seq("var<storage, read_write>",
        E(WGSL_TOK_KW_VAR, WGSL_TOK_TEMPLATE_START,
          WGSL_TOK_IDENT, WGSL_TOK_COMMA, WGSL_TOK_IDENT,
          WGSL_TOK_TEMPLATE_END, WGSL_TOK_EOF),
        EN(WGSL_TOK_KW_VAR, WGSL_TOK_TEMPLATE_START,
           WGSL_TOK_IDENT, WGSL_TOK_COMMA, WGSL_TOK_IDENT,
           WGSL_TOK_TEMPLATE_END, WGSL_TOK_EOF),
        "var<storage,read_write>");

    /* — 5. `a < b` — bare comparison: no template — */
    expect_seq("a < b",
        E(WGSL_TOK_IDENT, WGSL_TOK_LESS, WGSL_TOK_IDENT, WGSL_TOK_EOF),
        EN(WGSL_TOK_IDENT, WGSL_TOK_LESS, WGSL_TOK_IDENT, WGSL_TOK_EOF),
        "a<b (no template)");

    /* — 6. `a < b;` — semicolon clears — */
    expect_seq("a < b;",
        E(WGSL_TOK_IDENT, WGSL_TOK_LESS, WGSL_TOK_IDENT,
          WGSL_TOK_SEMICOLON, WGSL_TOK_EOF),
        EN(WGSL_TOK_IDENT, WGSL_TOK_LESS, WGSL_TOK_IDENT,
           WGSL_TOK_SEMICOLON, WGSL_TOK_EOF),
        "a<b; (cleared)");

    /* — 7. `a < b > c` — ambiguity wins for template per §3.9 — */
    expect_seq("a < b > c",
        E(WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_START,
          WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_END, WGSL_TOK_IDENT,
          WGSL_TOK_EOF),
        EN(WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_START,
           WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_END, WGSL_TOK_IDENT,
           WGSL_TOK_EOF),
        "a<b>c (template)");

    /* — 8. `a >= b` — comparison stays — */
    expect_seq("a >= b",
        E(WGSL_TOK_IDENT, WGSL_TOK_GREATER_EQUAL, WGSL_TOK_IDENT,
          WGSL_TOK_EOF),
        EN(WGSL_TOK_IDENT, WGSL_TOK_GREATER_EQUAL, WGSL_TOK_IDENT,
           WGSL_TOK_EOF),
        "a>=b");

    /* — 9. `vec<f32> = expr` — template + lone `=` clears stack — */
    expect_seq("vec<f32> = expr",
        E(WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_START,
          WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_END,
          WGSL_TOK_EQUAL, WGSL_TOK_IDENT, WGSL_TOK_EOF),
        EN(WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_START,
           WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_END,
           WGSL_TOK_EQUAL, WGSL_TOK_IDENT, WGSL_TOK_EOF),
        "vec<f32>=expr");

    /* — 10. `if (a < b) { x }` — `<` inside parens, no `>` → no template — */
    expect_seq("if (a < b) { x }",
        E(WGSL_TOK_KW_IF, WGSL_TOK_LPAREN,
          WGSL_TOK_IDENT, WGSL_TOK_LESS, WGSL_TOK_IDENT,
          WGSL_TOK_RPAREN, WGSL_TOK_LBRACE,
          WGSL_TOK_IDENT, WGSL_TOK_RBRACE, WGSL_TOK_EOF),
        EN(WGSL_TOK_KW_IF, WGSL_TOK_LPAREN,
           WGSL_TOK_IDENT, WGSL_TOK_LESS, WGSL_TOK_IDENT,
           WGSL_TOK_RPAREN, WGSL_TOK_LBRACE,
           WGSL_TOK_IDENT, WGSL_TOK_RBRACE, WGSL_TOK_EOF),
        "if(a<b){x}");

    /* — 11. `<<` (shift) never becomes a template even after ident — */
    expect_seq("a << 2",
        E(WGSL_TOK_IDENT, WGSL_TOK_LESS_LESS, WGSL_TOK_INT_LIT,
          WGSL_TOK_EOF),
        EN(WGSL_TOK_IDENT, WGSL_TOK_LESS_LESS, WGSL_TOK_INT_LIT,
           WGSL_TOK_EOF),
        "a<<2");

    /* — 12. `<=` is comparison, never opens a template — */
    expect_seq("a <= b",
        E(WGSL_TOK_IDENT, WGSL_TOK_LESS_EQUAL, WGSL_TOK_IDENT,
          WGSL_TOK_EOF),
        EN(WGSL_TOK_IDENT, WGSL_TOK_LESS_EQUAL, WGSL_TOK_IDENT,
           WGSL_TOK_EOF),
        "a<=b");

    /* — 13. `>>=` split: TEMPLATE_END + GREATER_EQUAL inside one template — */
    expect_seq("vec<f32>>=x",
        E(WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_START,
          WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_END,
          WGSL_TOK_GREATER_EQUAL, WGSL_TOK_IDENT, WGSL_TOK_EOF),
        EN(WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_START,
           WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_END,
           WGSL_TOK_GREATER_EQUAL, WGSL_TOK_IDENT, WGSL_TOK_EOF),
        "vec<f32>>=x");

    /* — 14. `>=` split: TEMPLATE_END + EQUAL — */
    expect_seq("vec<f32>=y",
        E(WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_START,
          WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_END,
          WGSL_TOK_EQUAL, WGSL_TOK_IDENT, WGSL_TOK_EOF),
        EN(WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_START,
           WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_END,
           WGSL_TOK_EQUAL, WGSL_TOK_IDENT, WGSL_TOK_EOF),
        "vec<f32>=y (>= split)");

    /* — 14b. `>>=` split closing TWO nested templates — §3.9                 *
     *                                                                        *
     * `array<vec<f32>>=x` — the lexer produces ident, `<`, ident, `<`,       *
     * ident, `>>=`, ident.  Discovery must close BOTH pending templates      *
     * with the leading `>>` and treat the trailing `=` as an assignment.     */
    expect_seq("array<vec<f32>>=x",
        E(WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_START,
          WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_START,
          WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_END,
          WGSL_TOK_TEMPLATE_END, WGSL_TOK_EQUAL,
          WGSL_TOK_IDENT, WGSL_TOK_EOF),
        EN(WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_START,
           WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_START,
           WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_END,
           WGSL_TOK_TEMPLATE_END, WGSL_TOK_EQUAL,
           WGSL_TOK_IDENT, WGSL_TOK_EOF),
        ">>= closes two pending templates then `=`");

    /* — 14c. `>>=` with only ONE pending template stays a comparison — §3.9 *
     *                                                                       *
     * `a<b>>=c` (one pending) splits into TEMPLATE_END + GREATER_EQUAL.     *
     * The trailing `>=` is a comparison, NOT an assignment.                  */
    expect_seq("a<b>>=c",
        E(WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_START,
          WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_END,
          WGSL_TOK_GREATER_EQUAL, WGSL_TOK_IDENT, WGSL_TOK_EOF),
        EN(WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_START,
           WGSL_TOK_IDENT, WGSL_TOK_TEMPLATE_END,
           WGSL_TOK_GREATER_EQUAL, WGSL_TOK_IDENT, WGSL_TOK_EOF),
        ">>= closes ONE pending then trailing `>=` stays comparison");

    /* — 15. Sanity: a complex line — */
    {
        Lex l; lex_run(&l,
            "fn f(p: ptr<storage, array<vec4<f32>, 3>, read>) -> vec3<f32> {}");
        CHECK(l.ok == 1, "complex: lex ok");
        /* Expect 5 templates: `ptr<…>`, `array<…>`, `vec4<…>`, the
         * final `vec3<…>`, and the inner `<f32>` inside vec4. */
        int starts = count_kind(&l.result, WGSL_TOK_TEMPLATE_START);
        int ends   = count_kind(&l.result, WGSL_TOK_TEMPLATE_END);
        CHECK(starts == 4 && ends == 4,
              "complex: 4 TEMPLATE_START / 4 TEMPLATE_END (ptr, array, vec4, vec3)");
        /* And no bare LESS/GREATER tokens leaked through. */
        CHECK(count_kind(&l.result, WGSL_TOK_LESS) == 0,
              "complex: no bare LESS");
        CHECK(count_kind(&l.result, WGSL_TOK_GREATER) == 0,
              "complex: no bare GREATER");
        lex_done(&l);
    }

    if (fail == 0) {
        printf("PASS  test_lexer_template\n");
        return 0;
    }
    fprintf(stderr, "FAIL  test_lexer_template  %d check(s) failed\n", fail);
    return 1;
}
