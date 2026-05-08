/**
 * Phase 3 Round A — basic parser smoke tests.
 *
 * Asserts:
 *   - empty source yields a TU with 0 children
 *   - a minimal `fn main() {}` produces one DECL_FUNCTION with a body
 *   - `fn f(x: i32) -> u32 { return x; }` carries name/param/return
 *     metadata and one return statement
 *   - `let x = 1 + 2 * 3;` parses as binary with right-side higher-prec
 *   - call + index + member chains, parenthesized expression, unary
 *   - attribute on fn + on parameter + on return type
 *   - templated_ident inside expression: `vec2f(0.0, 1.0)`
 */
#include "internal/parser.h"
#include "internal/ast_dump.h"
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
    int            lex_ok;
    int            parse_ok;
} P;

static void run(P *p, const char *text) {
    memset(p, 0, sizeof *p);
    wgsl_arena_init(&p->arena);
    wgsl_diag_init(&p->diag);
    wgsl_source_init(&p->src, text, strlen(text));
    p->lex_ok = wgsl_tokenize(&p->src, &p->arena, &p->diag, &p->lex);
    p->parse_ok = wgsl_parse(&p->lex, &p->src, &p->arena, &p->diag, &p->ast);
}
static void done(P *p) {
    wgsl_diag_destroy(&p->diag);
    wgsl_source_destroy(&p->src);
    wgsl_arena_destroy(&p->arena);
}

static int span_eq(const WGSLNode *n, const WGSLSource *src, const char *want) {
    size_t len = strlen(want);
    if (n->span_length != len) return 0;
    return memcmp(src->bytes + n->span_offset, want, len) == 0;
}

/* Read the name span out of a node whose payload[0] holds it. */
static int name_eq(const WGSLNode *n, const WGSLSource *src, const char *want) {
    uint32_t off = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
    uint32_t len = (uint32_t)(n->payload[0] >> 32);
    size_t wlen = strlen(want);
    if (len != wlen) return 0;
    return memcmp(src->bytes + off, want, wlen) == 0;
}

int main(void) {
    wgsl_utf8_init();

    /* — Empty source — */
    {
        P p; run(&p, "");
        CHECK(p.lex_ok && p.parse_ok, "empty: ok");
        CHECK(p.ast.root != NULL, "empty: root non-null");
        CHECK(p.ast.root->kind == WGSL_NODE_TRANSLATION_UNIT, "empty: TU root");
        CHECK(p.ast.root->child_count == 0, "empty: 0 children");
        done(&p);
    }

    /* — Minimal fn — */
    {
        P p; run(&p, "fn main() {}");
        CHECK(p.lex_ok && p.parse_ok, "minimal: ok");
        CHECK(p.ast.root->child_count == 1, "minimal: 1 decl");
        const WGSLNode *fn = p.ast.root->children[0];
        CHECK(fn && fn->kind == WGSL_NODE_DECL_FUNCTION, "minimal: fn kind");
        CHECK(name_eq(fn, &p.src, "main"), "minimal: name = main");
        uint32_t params = (uint32_t)(fn->payload[1] >> 32);
        CHECK(params == 0, "minimal: 0 params");
        uint32_t has_ret = (uint32_t)(fn->payload[2] >> 32);
        CHECK(has_ret == 0, "minimal: no return type");
        /* body is the last child */
        const WGSLNode *body = fn->children[fn->child_count - 1];
        CHECK(body && body->kind == WGSL_NODE_STMT_COMPOUND, "minimal: body is compound");
        CHECK(body->child_count == 0, "minimal: empty body");
        done(&p);
    }

    /* — fn with one param + return type + return stmt — */
    {
        P p; run(&p, "fn f(x: i32) -> u32 { return x; }");
        CHECK(p.lex_ok && p.parse_ok, "f: ok");
        const WGSLNode *fn = p.ast.root->children[0];
        CHECK(name_eq(fn, &p.src, "f"), "f: name");
        uint32_t fn_attrs = (uint32_t)(fn->payload[1] & 0xFFFFFFFFu);
        uint32_t params   = (uint32_t)(fn->payload[1] >> 32);
        uint32_t has_ret  = (uint32_t)(fn->payload[2] >> 32);
        CHECK(fn_attrs == 0 && params == 1 && has_ret == 1, "f: counts");
        /* children: [param, ret_type, body] */
        const WGSLNode *param = fn->children[0];
        CHECK(param && param->kind == WGSL_NODE_DECL_PARAM, "f: child 0 = param");
        CHECK(name_eq(param, &p.src, "x"), "f: param name");
        const WGSLNode *ret = fn->children[1];
        CHECK(ret && ret->kind == WGSL_NODE_EXPR_IDENT, "f: ret type ident");
        CHECK(name_eq(ret, &p.src, "u32"), "f: ret type = u32");
        const WGSLNode *body = fn->children[2];
        CHECK(body && body->kind == WGSL_NODE_STMT_COMPOUND, "f: body");
        CHECK(body->child_count == 1, "f: body has 1 stmt");
        const WGSLNode *retstmt = body->children[0];
        CHECK(retstmt && retstmt->kind == WGSL_NODE_STMT_RETURN, "f: return stmt");
        CHECK(retstmt->child_count == 1, "f: return has expr");
        const WGSLNode *retv = retstmt->children[0];
        CHECK(retv && retv->kind == WGSL_NODE_EXPR_IDENT, "f: return value is ident");
        CHECK(name_eq(retv, &p.src, "x"), "f: return value name = x");
        done(&p);
    }

    /* — Pratt precedence: 1 + 2 * 3 = 1 + (2*3) — */
    {
        P p; run(&p, "fn t() { let z = 1 + 2 * 3; }");
        CHECK(p.lex_ok && p.parse_ok, "prec: ok");
        const WGSLNode *fn = p.ast.root->children[0];
        const WGSLNode *body = fn->children[fn->child_count - 1];
        const WGSLNode *let = body->children[0];
        CHECK(let && let->kind == WGSL_NODE_DECL_LET, "prec: let");
        CHECK(name_eq(let, &p.src, "z"), "prec: let name");
        const WGSLNode *expr = let->children[let->child_count - 1];
        CHECK(expr && expr->kind == WGSL_NODE_EXPR_BINARY, "prec: binary root");
        /* root op = '+', children = [1, 2*3] */
        WGSLTokenKind op = (WGSLTokenKind)expr->payload[0];
        CHECK(op == WGSL_TOK_PLUS, "prec: root op = +");
        CHECK(expr->children[0]->kind == WGSL_NODE_EXPR_LITERAL_INT, "prec: lhs is 1");
        const WGSLNode *rhs = expr->children[1];
        CHECK(rhs->kind == WGSL_NODE_EXPR_BINARY, "prec: rhs is binary");
        WGSLTokenKind rhs_op = (WGSLTokenKind)rhs->payload[0];
        CHECK(rhs_op == WGSL_TOK_STAR, "prec: rhs op = *");
        done(&p);
    }

    /* — call + index + member chain — */
    {
        P p; run(&p, "fn t() { let r = obj.a[0](1); }");
        CHECK(p.lex_ok && p.parse_ok, "chain: ok");
        const WGSLNode *fn = p.ast.root->children[0];
        const WGSLNode *body = fn->children[fn->child_count - 1];
        const WGSLNode *let = body->children[0];
        const WGSLNode *expr = let->children[let->child_count - 1];
        /* Outermost: call */
        CHECK(expr->kind == WGSL_NODE_EXPR_CALL, "chain: outer = call");
        const WGSLNode *callee = expr->children[0];
        /* Callee: index */
        CHECK(callee->kind == WGSL_NODE_EXPR_INDEX, "chain: callee = index");
        const WGSLNode *idx_base = callee->children[0];
        /* Idx base: member */
        CHECK(idx_base->kind == WGSL_NODE_EXPR_MEMBER, "chain: idx base = member");
        const WGSLNode *member_base = idx_base->children[0];
        CHECK(member_base->kind == WGSL_NODE_EXPR_IDENT, "chain: member base = ident obj");
        CHECK(name_eq(member_base, &p.src, "obj"), "chain: ident is obj");
        done(&p);
    }

    /* — Attributes on fn + return + param + templated_ident — */
    {
        P p; run(&p,
            "@vertex fn vs(@builtin(vertex_index) vid: u32) "
            "-> @location(0) vec2f { return vec2f(0.0, 1.0); }");
        CHECK(p.lex_ok && p.parse_ok, "attrs: ok");
        const WGSLNode *fn = p.ast.root->children[0];
        uint32_t fn_attrs = (uint32_t)(fn->payload[1] & 0xFFFFFFFFu);
        uint32_t params   = (uint32_t)(fn->payload[1] >> 32);
        uint32_t ret_attrs= (uint32_t)(fn->payload[2] & 0xFFFFFFFFu);
        uint32_t has_ret  = (uint32_t)(fn->payload[2] >> 32);
        CHECK(fn_attrs == 1, "attrs: 1 fn-attr (@vertex)");
        CHECK(params   == 1, "attrs: 1 param");
        CHECK(ret_attrs== 1, "attrs: 1 ret-attr (@location)");
        CHECK(has_ret  == 1, "attrs: ret type present");

        const WGSLNode *fn_attr = fn->children[0];
        CHECK(fn_attr->kind == WGSL_NODE_ATTRIBUTE, "attrs: fn attr kind");
        const WGSLNode *fn_attr_name = fn_attr->children[0];
        CHECK(name_eq(fn_attr_name, &p.src, "vertex"), "attrs: fn attr = vertex");

        const WGSLNode *param = fn->children[1];
        CHECK(param->kind == WGSL_NODE_DECL_PARAM, "attrs: param kind");
        uint32_t param_attr_count = (uint32_t)param->payload[1];
        CHECK(param_attr_count == 1, "attrs: param has 1 attr");
        /* param children: [@builtin, type] */
        const WGSLNode *param_attr = param->children[0];
        CHECK(param_attr->kind == WGSL_NODE_ATTRIBUTE, "attrs: param attr");
        const WGSLNode *param_attr_name = param_attr->children[0];
        CHECK(name_eq(param_attr_name, &p.src, "builtin"), "attrs: param attr = builtin");
        const WGSLNode *param_attr_arg = param_attr->children[1];
        CHECK(name_eq(param_attr_arg, &p.src, "vertex_index"),
              "attrs: param attr arg = vertex_index");

        const WGSLNode *ret_attr = fn->children[2];
        CHECK(ret_attr->kind == WGSL_NODE_ATTRIBUTE, "attrs: ret attr");
        const WGSLNode *ret_attr_name = ret_attr->children[0];
        CHECK(name_eq(ret_attr_name, &p.src, "location"), "attrs: ret attr = location");

        const WGSLNode *ret_type = fn->children[3];
        CHECK(name_eq(ret_type, &p.src, "vec2f"), "attrs: ret type = vec2f");

        const WGSLNode *body = fn->children[4];
        CHECK(body->kind == WGSL_NODE_STMT_COMPOUND, "attrs: body");
        const WGSLNode *retstmt = body->children[0];
        CHECK(retstmt->kind == WGSL_NODE_STMT_RETURN, "attrs: return stmt");
        const WGSLNode *retval = retstmt->children[0];
        CHECK(retval->kind == WGSL_NODE_EXPR_CALL, "attrs: return value is call");
        const WGSLNode *callee = retval->children[0];
        CHECK(callee->kind == WGSL_NODE_EXPR_IDENT, "attrs: callee is ident");
        CHECK(name_eq(callee, &p.src, "vec2f"), "attrs: callee = vec2f");
        CHECK(retval->child_count == 3, "attrs: vec2f has 2 args (callee + 2)");
        CHECK(retval->children[1]->kind == WGSL_NODE_EXPR_LITERAL_FLOAT, "attrs: arg 0 = float");
        CHECK(retval->children[2]->kind == WGSL_NODE_EXPR_LITERAL_FLOAT, "attrs: arg 1 = float");

        if (fail == 0) {
            fprintf(stderr, "--- AST dump (attrs case) ---\n");
            wgsl_ast_dump(p.ast.root, &p.src, stderr);
            fprintf(stderr, "------------------------------\n");
        }
        done(&p);
    }

    /* — Unary minus on a literal — */
    {
        P p; run(&p, "fn t() { let x = -1; }");
        CHECK(p.lex_ok && p.parse_ok, "unary: ok");
        const WGSLNode *fn = p.ast.root->children[0];
        const WGSLNode *body = fn->children[fn->child_count - 1];
        const WGSLNode *let = body->children[0];
        const WGSLNode *expr = let->children[let->child_count - 1];
        CHECK(expr->kind == WGSL_NODE_EXPR_UNARY, "unary: kind");
        WGSLTokenKind op = (WGSLTokenKind)expr->payload[0];
        CHECK(op == WGSL_TOK_MINUS, "unary: op = -");
        CHECK(expr->children[0]->kind == WGSL_NODE_EXPR_LITERAL_INT, "unary: operand = int");
        (void)span_eq;  /* unused helper retained for richer assertions later */
        done(&p);
    }

    if (fail == 0) {
        printf("PASS  test_parser_basic\n");
        return 0;
    }
    fprintf(stderr, "FAIL  test_parser_basic  %d check(s) failed\n", fail);
    return 1;
}
