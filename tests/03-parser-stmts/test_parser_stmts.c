/**
 * Function-scope declaration and assignment parser tests.
 *
 * Covers:
 *   - var x: i32 = 1;          (typed + initialized)
 *   - var x = 1;               (initialized, type inferred)
 *   - var x: i32;              (typed, uninitialized)
 *   - const k: i32 = 42;
 *   - const_assert true;
 *   - simple assignment        x = 1;
 *   - compound assignment      x += 2;  x <<= 3;  x %= y;
 *   - phony assignment         _ = expr;
 *   - increment / decrement    x++;  x--;
 *   - expression-statement (call)  foo(1, 2);
 */
#include "internal/parser.h"
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
    WGSLLexResult  lex;
    WGSLAst        ast;
    int            ok;
} P;

static void run(P *p, const char *text) {
    memset(p, 0, sizeof *p);
    wgsl_arena_init(&p->arena);
    wgsl_diag_init(&p->diag);
    wgsl_source_init(&p->src, text, strlen(text));
    int lex_ok = wgsl_tokenize(&p->src, &p->arena, &p->diag, &p->lex);
    int parse_ok = wgsl_parse(&p->lex, &p->src, &p->arena, &p->diag, &p->ast);
    p->ok = lex_ok && parse_ok && !wgsl_diag_has_error(&p->diag);
}
static void done(P *p) {
    wgsl_diag_destroy(&p->diag);
    wgsl_source_destroy(&p->src);
    wgsl_arena_destroy(&p->arena);
}

static int name_eq(const WGSLNode *n, const WGSLSource *s, const char *want) {
    uint32_t off = wgsl_node_name_span(n).offset;
    uint32_t len = wgsl_node_name_span(n).length;
    size_t wl = strlen(want);
    if (len != wl) return 0;
    return memcmp(s->bytes + off, want, wl) == 0;
}

/* Walk root.children[0] (the only fn) and return its first body statement. */
static const WGSLNode *first_body_stmt(const WGSLAst *a) {
    const WGSLNode *fn = a->root->children[0];
    const WGSLNode *body = fn->children[fn->child_count - 1];
    return body->children[0];
}

int main(void) {
    wgsl_utf8_init();

    /* — var x: i32 = 1;  — */
    {
        P p; run(&p, "fn t() { var x: i32 = 1; }");
        CHECK(p.ok, "var: ok");
        const WGSLNode *s = first_body_stmt(&p.ast);
        CHECK(s->kind == WGSL_NODE_DECL_VAR, "var: kind");
        CHECK(name_eq(s, &p.src, "x"), "var: name = x");
        uint32_t has_type = (uint32_t)wgsl_varlike_has_type(s);
        uint32_t has_init = (uint32_t)wgsl_varlike_has_init(s);
        CHECK(has_type == 1 && has_init == 1, "var: has_type + has_init");
        CHECK(s->child_count == 2, "var: 2 children (type, init)");
        done(&p);
    }
    /* — var x = 1;  — */
    {
        P p; run(&p, "fn t() { var x = 1; }");
        CHECK(p.ok, "var-init-only: ok");
        const WGSLNode *s = first_body_stmt(&p.ast);
        uint32_t has_type = (uint32_t)wgsl_varlike_has_type(s);
        uint32_t has_init = (uint32_t)wgsl_varlike_has_init(s);
        CHECK(has_type == 0 && has_init == 1, "var-init-only: flags");
        CHECK(s->child_count == 1, "var-init-only: 1 child (init)");
        done(&p);
    }
    /* — var x: i32;  — */
    {
        P p; run(&p, "fn t() { var x: i32; }");
        CHECK(p.ok, "var-type-only: ok");
        const WGSLNode *s = first_body_stmt(&p.ast);
        uint32_t has_type = (uint32_t)wgsl_varlike_has_type(s);
        uint32_t has_init = (uint32_t)wgsl_varlike_has_init(s);
        CHECK(has_type == 1 && has_init == 0, "var-type-only: flags");
        CHECK(s->child_count == 1, "var-type-only: 1 child (type)");
        done(&p);
    }

    /* — const k: i32 = 42; — */
    {
        P p; run(&p, "fn t() { const k: i32 = 42; }");
        CHECK(p.ok, "const: ok");
        const WGSLNode *s = first_body_stmt(&p.ast);
        CHECK(s->kind == WGSL_NODE_DECL_CONST, "const: kind");
        CHECK(name_eq(s, &p.src, "k"), "const: name = k");
        CHECK(s->child_count == 2, "const: 2 children (type, init)");
        done(&p);
    }

    /* — const_assert true; — */
    {
        P p; run(&p, "fn t() { const_assert true; }");
        CHECK(p.ok, "const_assert: ok");
        const WGSLNode *s = first_body_stmt(&p.ast);
        CHECK(s->kind == WGSL_NODE_DECL_CONST_ASSERT, "const_assert: kind");
        CHECK(s->child_count == 1, "const_assert: 1 child (expr)");
        done(&p);
    }

    /* — simple assignment x = 1; — */
    {
        P p; run(&p, "fn t() { x = 1; }");
        CHECK(p.ok, "assign: ok");
        const WGSLNode *s = first_body_stmt(&p.ast);
        CHECK(s->kind == WGSL_NODE_STMT_ASSIGN, "assign: kind");
        CHECK(s->child_count == 2, "assign: 2 children (lhs, rhs)");
        WGSLTokenKind op = (WGSLTokenKind)wgsl_node_op_kind_u32(s);
        CHECK(op == WGSL_TOK_EQUAL, "assign: op = =");
        done(&p);
    }

    /* — compound: x += 2;  x <<= 3;  x %= y; — */
    {
        struct { const char *src; WGSLTokenKind want_op; } cases[] = {
            { "fn t() { x += 2; }",   WGSL_TOK_PLUS_EQUAL },
            { "fn t() { x -= 2; }",   WGSL_TOK_MINUS_EQUAL },
            { "fn t() { x *= 2; }",   WGSL_TOK_STAR_EQUAL },
            { "fn t() { x /= 2; }",   WGSL_TOK_SLASH_EQUAL },
            { "fn t() { x %= y; }",   WGSL_TOK_PERCENT_EQUAL },
            { "fn t() { x &= 1; }",   WGSL_TOK_AMP_EQUAL },
            { "fn t() { x |= 1; }",   WGSL_TOK_PIPE_EQUAL },
            { "fn t() { x ^= 1; }",   WGSL_TOK_CARET_EQUAL },
            { "fn t() { x <<= 1; }",  WGSL_TOK_LESS_LESS_EQUAL },
            { "fn t() { x >>= 1; }",  WGSL_TOK_GREATER_GREATER_EQUAL },
        };
        for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
            P p; run(&p, cases[i].src);
            CHECK(p.ok, "compound: ok");
            const WGSLNode *s = first_body_stmt(&p.ast);
            CHECK(s->kind == WGSL_NODE_STMT_COMPOUND_ASSIGN, "compound: kind");
            WGSLTokenKind op = (WGSLTokenKind)wgsl_node_op_kind_u32(s);
            CHECK(op == cases[i].want_op, "compound: op match");
            done(&p);
        }
    }

    /* — phony  _ = expr;  — */
    {
        P p; run(&p, "fn t() { _ = foo(1); }");
        CHECK(p.ok, "phony: ok");
        const WGSLNode *s = first_body_stmt(&p.ast);
        CHECK(s->kind == WGSL_NODE_STMT_PHONY_ASSIGN, "phony: kind");
        CHECK(s->child_count == 1, "phony: 1 child (rhs)");
        done(&p);
    }

    /* — increment / decrement — */
    {
        P p; run(&p, "fn t() { x++; }");
        CHECK(p.ok, "inc: ok");
        CHECK(first_body_stmt(&p.ast)->kind == WGSL_NODE_STMT_INCREMENT, "inc: kind");
        done(&p);
    }
    {
        P p; run(&p, "fn t() { x--; }");
        CHECK(p.ok, "dec: ok");
        CHECK(first_body_stmt(&p.ast)->kind == WGSL_NODE_STMT_DECREMENT, "dec: kind");
        done(&p);
    }

    /* — expression-statement (call) — */
    {
        P p; run(&p, "fn t() { foo(1, 2); }");
        CHECK(p.ok, "call-stmt: ok");
        const WGSLNode *s = first_body_stmt(&p.ast);
        CHECK(s->kind == WGSL_NODE_STMT_FN_CALL, "call-stmt: kind");
        CHECK(s->children[0]->kind == WGSL_NODE_EXPR_CALL, "call-stmt: inner = call");
        done(&p);
    }

    if (fail == 0) {
        printf("PASS  test_parser_stmts\n");
        return 0;
    }
    fprintf(stderr, "FAIL  test_parser_stmts  %d check(s) failed\n", fail);
    return 1;
}
