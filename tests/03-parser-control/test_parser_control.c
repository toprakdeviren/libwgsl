/**
 * Phase 3 Round B Iter 1 — control flow.
 *
 * Covers:
 *   - if + else if + else
 *   - while
 *   - for (init; cond; update)  with all four flag combinations
 *   - loop { … continuing { … break if … } }
 *   - switch + case (multi-selector) + default
 *   - break ; continue ; discard ;
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

static const WGSLNode *first_body_stmt(const WGSLAst *a) {
    const WGSLNode *fn = a->root->children[0];
    const WGSLNode *body = fn->children[fn->child_count - 1];
    return body->children[0];
}
static int kind_count(const WGSLNode *n, WGSLNodeKind k) {
    int c = 0;
    for (uint32_t i = 0; i < n->child_count; i++)
        if (n->children[i]->kind == (uint16_t)k) c += 1;
    return c;
}

int main(void) {
    wgsl_utf8_init();

    /* — if + else if + else — */
    {
        P p; run(&p,
            "fn t() {"
            "  if a { }"
            "  else if b { }"
            "  else if c { }"
            "  else { }"
            "}");
        CHECK(p.ok, "if-chain: ok");
        const WGSLNode *s = first_body_stmt(&p.ast);
        CHECK(s->kind == WGSL_NODE_STMT_IF, "if-chain: kind");
        /* children: [cond, then-body, else-if, else-if, else] */
        CHECK(s->child_count == 5, "if-chain: 5 children");
        CHECK(s->children[0]->kind == WGSL_NODE_EXPR_IDENT, "if-chain: cond");
        CHECK(s->children[1]->kind == WGSL_NODE_STMT_COMPOUND, "if-chain: then");
        CHECK(s->children[2]->kind == WGSL_NODE_STMT_ELSE_IF, "if-chain: else-if 1");
        CHECK(s->children[3]->kind == WGSL_NODE_STMT_ELSE_IF, "if-chain: else-if 2");
        CHECK(s->children[4]->kind == WGSL_NODE_STMT_ELSE, "if-chain: else");
        done(&p);
    }

    /* — while — */
    {
        P p; run(&p, "fn t() { while x < 10 { x++; } }");
        CHECK(p.ok, "while: ok");
        const WGSLNode *s = first_body_stmt(&p.ast);
        CHECK(s->kind == WGSL_NODE_STMT_WHILE, "while: kind");
        CHECK(s->child_count == 2, "while: cond + body");
        CHECK(s->children[1]->kind == WGSL_NODE_STMT_COMPOUND, "while: body compound");
        const WGSLNode *body = s->children[1];
        CHECK(body->child_count == 1 && body->children[0]->kind == WGSL_NODE_STMT_INCREMENT,
              "while: body has x++");
        done(&p);
    }

    /* — for: full (var-init / cond / update / body) — */
    {
        P p; run(&p, "fn t() { for (var i = 0; i < 10; i++) { } }");
        CHECK(p.ok, "for-full: ok");
        const WGSLNode *s = first_body_stmt(&p.ast);
        CHECK(s->kind == WGSL_NODE_STMT_FOR, "for-full: kind");
        uint32_t flags = (uint32_t)s->payload[0];
        CHECK(flags == 0x7, "for-full: bitmap = 0b111");
        CHECK(s->child_count == 4, "for-full: 4 children (init, cond, update, body)");
        CHECK(s->children[0]->kind == WGSL_NODE_DECL_VAR,         "for-full: init = var");
        CHECK(s->children[1]->kind == WGSL_NODE_EXPR_BINARY,      "for-full: cond = binary");
        CHECK(s->children[2]->kind == WGSL_NODE_STMT_INCREMENT,   "for-full: update = ++");
        CHECK(s->children[3]->kind == WGSL_NODE_STMT_COMPOUND,    "for-full: body");
        done(&p);
    }
    /* — for: empty header — */
    {
        P p; run(&p, "fn t() { for (;;) { } }");
        CHECK(p.ok, "for-empty: ok");
        const WGSLNode *s = first_body_stmt(&p.ast);
        uint32_t flags = (uint32_t)s->payload[0];
        CHECK(flags == 0, "for-empty: bitmap = 0");
        CHECK(s->child_count == 1, "for-empty: 1 child (body)");
        done(&p);
    }
    /* — for: cond only — */
    {
        P p; run(&p, "fn t() { for (; x < 10;) { } }");
        CHECK(p.ok, "for-cond: ok");
        const WGSLNode *s = first_body_stmt(&p.ast);
        uint32_t flags = (uint32_t)s->payload[0];
        CHECK(flags == 0x2, "for-cond: bitmap = 0b010");
        CHECK(s->child_count == 2, "for-cond: 2 children (cond, body)");
        CHECK(s->children[0]->kind == WGSL_NODE_EXPR_BINARY, "for-cond: cond first");
        done(&p);
    }
    /* — for: assignment update (lhs = expr) — */
    {
        P p; run(&p, "fn t() { for (var i: i32 = 0; i < n; i = i + 1) { } }");
        CHECK(p.ok, "for-assign: ok");
        const WGSLNode *s = first_body_stmt(&p.ast);
        CHECK(s->children[2]->kind == WGSL_NODE_STMT_ASSIGN, "for-assign: update is assign");
        done(&p);
    }

    /* — loop with continuing + break_if — */
    {
        P p; run(&p,
            "fn t() {"
            "  loop {"
            "    x++;"
            "    continuing {"
            "      y -= 1;"
            "      break if x >= 10;"
            "    }"
            "  }"
            "}");
        CHECK(p.ok, "loop: ok");
        const WGSLNode *s = first_body_stmt(&p.ast);
        CHECK(s->kind == WGSL_NODE_STMT_LOOP, "loop: kind");
        /* children: [stmt, ..., continuing] — last is continuing */
        const WGSLNode *cont = s->children[s->child_count - 1];
        CHECK(cont->kind == WGSL_NODE_STMT_CONTINUING, "loop: last child = continuing");
        /* continuing children: [stmt, ..., break_if] — last is break_if */
        const WGSLNode *bi = cont->children[cont->child_count - 1];
        CHECK(bi->kind == WGSL_NODE_STMT_BREAK_IF, "loop: tail of continuing = break_if");
        CHECK(bi->child_count == 1, "break_if: 1 child (cond)");
        done(&p);
    }

    /* — switch with case + default — */
    {
        P p; run(&p,
            "fn t() {"
            "  switch n {"
            "    case 0, 1: { x = 1; }"
            "    case 2:    { x = 2; }"
            "    default:   { x = 99; }"
            "  }"
            "}");
        CHECK(p.ok, "switch: ok");
        const WGSLNode *s = first_body_stmt(&p.ast);
        CHECK(s->kind == WGSL_NODE_STMT_SWITCH, "switch: kind");
        /* children: [selector, case, case, default-clause] */
        CHECK(s->child_count == 4, "switch: 4 children (sel + 3 clauses)");
        CHECK(s->children[0]->kind == WGSL_NODE_EXPR_IDENT, "switch: selector");
        int clauses = kind_count(s, WGSL_NODE_STMT_CASE_CLAUSE);
        CHECK(clauses == 3, "switch: 3 case_clauses (incl. default)");

        const WGSLNode *c0 = s->children[1];
        uint32_t c0_count = (uint32_t)(c0->payload[0] & 0xFFFFFFFFu);
        uint32_t c0_default = (uint32_t)(c0->payload[0] >> 32);
        CHECK(c0_count == 2 && c0_default == 0,
              "switch: clause 0 has 2 selectors, not default");

        const WGSLNode *c2 = s->children[3];
        uint32_t c2_count   = (uint32_t)(c2->payload[0] & 0xFFFFFFFFu);
        uint32_t c2_default = (uint32_t)(c2->payload[0] >> 32);
        CHECK(c2_count == 0 && c2_default == 1,
              "switch: clause 2 is default-alone");

        done(&p);
    }

    /* — break / continue / discard — */
    {
        P p; run(&p, "fn t() { while true { break; continue; discard; } }");
        CHECK(p.ok, "leaf-stmts: ok");
        const WGSLNode *while_s = first_body_stmt(&p.ast);
        const WGSLNode *body = while_s->children[1];
        CHECK(body->child_count == 3, "leaf-stmts: 3 stmts in body");
        CHECK(body->children[0]->kind == WGSL_NODE_STMT_BREAK,    "leaf: break");
        CHECK(body->children[1]->kind == WGSL_NODE_STMT_CONTINUE, "leaf: continue");
        CHECK(body->children[2]->kind == WGSL_NODE_STMT_DISCARD,  "leaf: discard");
        done(&p);
    }

    if (fail == 0) {
        printf("PASS  test_parser_control\n");
        return 0;
    }
    fprintf(stderr, "FAIL  test_parser_control  %d check(s) failed\n", fail);
    return 1;
}
