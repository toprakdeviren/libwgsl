/**
 * @file parser.c — WGSL recursive-descent + Pratt parser, Round A.
 *
 * See internal/parser.h for the supported grammar subset.
 *
 * Conventions in this file:
 *   - `NL` is a heap-backed `WGSLNode**` list used to accumulate
 *      children before they get copied into the arena.
 *   - Every parse_* function returns a non-NULL node on success; on a
 *      hard error (bad token at a sync point), it emits a diagnostic
 *      via `parse_error` and may still return a placeholder node so
 *      callers do not need to NULL-check on every recursive descent.
 *   - Recovery is "skip to next `;` or `}` or EOF".
 *
 * Function decl payload layout (`WGSL_NODE_DECL_FUNCTION`):
 *   payload[0]  =  name_offset (low 32) | name_length (high 32)
 *   payload[1]  =  fn_attr_count (low 32) | param_count (high 32)
 *   payload[2]  =  ret_attr_count (low 32) | has_return_type (high 32)
 *
 * Function-decl children layout (source order):
 *   [fn_attrs...] [params...] [ret_attrs...] [optional return_type] [body]
 */
#include "internal/parser.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal/token.h"

/* ── State + helpers ─────────────────────────────────────────────── */

typedef struct {
    const WGSLToken *tokens;
    size_t           count;
    size_t           pos;

    const WGSLSource *src;
    WGSLArena        *arena;
    WGSLDiagBag      *diag;

    int               had_error;
} P;

typedef struct {
    WGSLNode **items;
    size_t     count;
    size_t     capacity;
} NL;

static void nl_init(NL *nl) { nl->items = NULL; nl->count = 0; nl->capacity = 0; }
static void nl_destroy(NL *nl) { free(nl->items); nl->items = NULL; nl->count = 0; nl->capacity = 0; }
static int  nl_push(NL *nl, WGSLNode *n) {
    if (!n) return 0;
    if (nl->count == nl->capacity) {
        size_t cap = nl->capacity ? nl->capacity * 2 : 8;
        WGSLNode **g = (WGSLNode **)realloc(nl->items, cap * sizeof(*g));
        if (!g) return 0;
        nl->items = g;
        nl->capacity = cap;
    }
    nl->items[nl->count++] = n;
    return 1;
}

static int is_trivia_kind(WGSLTokenKind k) {
    return k == WGSL_TOK_BLANKSPACE
        || k == WGSL_TOK_LINE_COMMENT
        || k == WGSL_TOK_BLOCK_COMMENT;
}
static void skip_trivia(P *p) {
    while (p->pos < p->count && is_trivia_kind(p->tokens[p->pos].kind)) p->pos++;
}
static const WGSLToken *peek_n(P *p, size_t n) {
    size_t pos = p->pos;
    while (pos < p->count) {
        if (!is_trivia_kind(p->tokens[pos].kind)) {
            if (n == 0) return &p->tokens[pos];
            n -= 1;
        }
        pos += 1;
    }
    return &p->tokens[p->count - 1];   /* EOF guaranteed last */
}
static const WGSLToken *peek(P *p)              { return peek_n(p, 0); }
static const WGSLToken *advance(P *p) {
    skip_trivia(p);
    if (p->pos >= p->count) return &p->tokens[p->count - 1];
    return &p->tokens[p->pos++];
}
static int at(P *p, WGSLTokenKind k)            { return peek(p)->kind == k; }
static int match(P *p, WGSLTokenKind k)         { if (at(p, k)) { advance(p); return 1; } return 0; }

static void parse_error(P *p, uint32_t off, uint32_t len, const char *fmt, ...) {
    char buf[256];
    va_list ap; va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    wgsl_diag_emit_at(p->diag, p->src, WGSL_DIAG_ERROR, off, len, NULL, "%s", buf);
    p->had_error = 1;
}

static const WGSLToken *expect(P *p, WGSLTokenKind k, const char *what) {
    if (at(p, k)) return advance(p);
    const WGSLToken *got = peek(p);
    parse_error(p, got->span.offset, got->span.length,
                "expected %s, got %s", what, wgsl_token_kind_name(got->kind));
    return got;
}

/* Skip tokens until we hit a likely sync point or EOF. */
static void recover_to_sync(P *p) {
    while (!at(p, WGSL_TOK_EOF)) {
        WGSLTokenKind k = peek(p)->kind;
        if (k == WGSL_TOK_SEMICOLON) { advance(p); return; }
        if (k == WGSL_TOK_RBRACE)    { return; }
        advance(p);
    }
}

/* ── Node construction ───────────────────────────────────────────── */

static WGSLNode *make_node(P *p, WGSLNodeKind kind, WGSLSpan span) {
    WGSLNode *n = (WGSLNode *)wgsl_arena_calloc(p->arena, 1, sizeof *n);
    if (!n) {
        p->had_error = 1;
        return NULL;
    }
    n->kind        = (uint16_t)kind;
    n->span_offset = span.offset;
    n->span_length = span.length;
    return n;
}

static int finalize_children(P *p, WGSLNode *parent, NL *list) {
    if (!parent) return 0;
    if (list->count == 0) {
        parent->children    = NULL;
        parent->child_count = 0;
        return 1;
    }
    WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(
        p->arena, list->count * sizeof(WGSLNode *));
    if (!arr) {
        p->had_error = 1;
        return 0;
    }
    memcpy(arr, list->items, list->count * sizeof(WGSLNode *));
    parent->children    = arr;
    parent->child_count = (uint32_t)list->count;
    return 1;
}

/* ── Expressions (Pratt) ─────────────────────────────────────────── */

/* Forward declarations */
static WGSLNode *parse_expression(P *p);
static WGSLNode *parse_primary(P *p);
static WGSLNode *parse_unary(P *p);
static WGSLNode *parse_postfix(P *p, WGSLNode *base);
static WGSLNode *parse_template_arg_list(P *p, WGSLNode *ident);
static WGSLNode *parse_type_specifier(P *p);

/* §8.19 abridged precedence table for Round A.  Higher = tighter.
 * Returns 0 for non-binary tokens. */
static int prec_of(WGSLTokenKind k) {
    switch (k) {
    case WGSL_TOK_PIPE_PIPE:        return 1;
    case WGSL_TOK_AMP_AMP:          return 2;
    case WGSL_TOK_PIPE:             return 3;
    case WGSL_TOK_CARET:            return 4;
    case WGSL_TOK_AMP:              return 5;
    case WGSL_TOK_EQUAL_EQUAL:
    case WGSL_TOK_BANG_EQUAL:       return 6;
    case WGSL_TOK_LESS:
    case WGSL_TOK_LESS_EQUAL:
    case WGSL_TOK_GREATER:
    case WGSL_TOK_GREATER_EQUAL:    return 7;
    case WGSL_TOK_LESS_LESS:
    case WGSL_TOK_GREATER_GREATER:  return 8;
    case WGSL_TOK_PLUS:
    case WGSL_TOK_MINUS:            return 9;
    case WGSL_TOK_STAR:
    case WGSL_TOK_SLASH:
    case WGSL_TOK_PERCENT:          return 10;
    default:                        return 0;
    }
}

static WGSLNode *parse_binary_rhs(P *p, WGSLNode *lhs, int min_prec) {
    while (1) {
        const WGSLToken *op = peek(p);
        int prec = prec_of(op->kind);
        if (prec == 0 || prec < min_prec) return lhs;

        WGSLTokenKind op_kind = op->kind;
        WGSLSpan op_span = op->span;
        advance(p);

        WGSLNode *rhs = parse_unary(p);
        if (!rhs) return lhs;

        /* Left-assoc: parse right side with higher min_prec. */
        rhs = parse_binary_rhs(p, rhs, prec + 1);

        WGSLSpan span = wgsl_span_make(
            lhs->span_offset,
            rhs->span_offset + rhs->span_length - lhs->span_offset);

        WGSLNode *bin = make_node(p, WGSL_NODE_EXPR_BINARY, span);
        if (!bin) return lhs;

        bin->payload[0] = (uint64_t)op_kind;
        bin->payload[1] = ((uint64_t)op_span.offset) | ((uint64_t)op_span.length << 32);

        WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, 2 * sizeof(WGSLNode *));
        if (!arr) { p->had_error = 1; return lhs; }
        arr[0] = lhs; arr[1] = rhs;
        bin->children    = arr;
        bin->child_count = 2;

        lhs = bin;
    }
}

static WGSLNode *parse_expression(P *p) {
    WGSLNode *lhs = parse_unary(p);
    if (!lhs) return NULL;
    return parse_binary_rhs(p, lhs, 1);
}

static WGSLNode *parse_unary(P *p) {
    const WGSLToken *t = peek(p);
    WGSLTokenKind k = t->kind;
    if (k == WGSL_TOK_MINUS || k == WGSL_TOK_BANG ||
        k == WGSL_TOK_TILDE || k == WGSL_TOK_STAR  ||
        k == WGSL_TOK_AMP)
    {
        WGSLSpan op_span = t->span;
        WGSLTokenKind op_kind = k;
        advance(p);
        WGSLNode *operand = parse_unary(p);
        if (!operand) return NULL;
        WGSLSpan span = wgsl_span_make(op_span.offset,
            operand->span_offset + operand->span_length - op_span.offset);
        WGSLNodeKind nk;
        switch (op_kind) {
        case WGSL_TOK_STAR: nk = WGSL_NODE_EXPR_INDIRECTION; break;
        case WGSL_TOK_AMP:  nk = WGSL_NODE_EXPR_ADDR_OF;     break;
        default:            nk = WGSL_NODE_EXPR_UNARY;       break;
        }
        WGSLNode *u = make_node(p, nk, span);
        if (!u) return NULL;
        u->payload[0] = (uint64_t)op_kind;
        u->payload[1] = ((uint64_t)op_span.offset) | ((uint64_t)op_span.length << 32);
        WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, sizeof(WGSLNode *));
        if (!arr) { p->had_error = 1; return NULL; }
        arr[0] = operand;
        u->children    = arr;
        u->child_count = 1;
        return u;
    }
    return parse_postfix(p, parse_primary(p));
}

/* `vec3<f32>` → emit a TEMPLATED_IDENT with children: ident node + arg expressions.
 * Caller has already consumed the IDENT.  This consumes from TEMPLATE_START
 * to TEMPLATE_END.  Returns a node spanning ident-through-TEMPLATE_END. */
static WGSLNode *parse_template_arg_list(P *p, WGSLNode *ident) {
    /* current token is TEMPLATE_START */
    expect(p, WGSL_TOK_TEMPLATE_START, "template arg list opener");

    NL kids; nl_init(&kids);
    nl_push(&kids, ident);

    if (!at(p, WGSL_TOK_TEMPLATE_END)) {
        do {
            if (at(p, WGSL_TOK_TEMPLATE_END)) break;   /* trailing comma */
            WGSLNode *arg = parse_expression(p);
            if (arg) nl_push(&kids, arg);
        } while (match(p, WGSL_TOK_COMMA));
    }
    const WGSLToken *end = peek(p);
    expect(p, WGSL_TOK_TEMPLATE_END, "'>' (template list end)");

    WGSLSpan span = wgsl_span_make(
        ident->span_offset,
        end->span.offset + end->span.length - ident->span_offset);

    WGSLNode *node = make_node(p, WGSL_NODE_EXPR_TEMPLATED_IDENT, span);
    if (!node) { nl_destroy(&kids); return ident; }
    finalize_children(p, node, &kids);
    nl_destroy(&kids);
    return node;
}

static WGSLNode *parse_postfix(P *p, WGSLNode *base) {
    if (!base) return NULL;
    while (1) {
        WGSLTokenKind k = peek(p)->kind;
        if (k == WGSL_TOK_LPAREN) {
            advance(p);
            NL kids; nl_init(&kids);
            nl_push(&kids, base);
            if (!at(p, WGSL_TOK_RPAREN)) {
                do {
                    if (at(p, WGSL_TOK_RPAREN)) break;
                    WGSLNode *arg = parse_expression(p);
                    if (arg) nl_push(&kids, arg);
                } while (match(p, WGSL_TOK_COMMA));
            }
            const WGSLToken *rp = peek(p);
            expect(p, WGSL_TOK_RPAREN, "')'");
            WGSLSpan span = wgsl_span_make(
                base->span_offset,
                rp->span.offset + rp->span.length - base->span_offset);
            WGSLNode *call = make_node(p, WGSL_NODE_EXPR_CALL, span);
            if (call) {
                finalize_children(p, call, &kids);
                base = call;
            }
            nl_destroy(&kids);
            continue;
        }
        if (k == WGSL_TOK_LBRACKET) {
            advance(p);
            WGSLNode *idx = parse_expression(p);
            const WGSLToken *rb = peek(p);
            expect(p, WGSL_TOK_RBRACKET, "']'");
            WGSLSpan span = wgsl_span_make(
                base->span_offset,
                rb->span.offset + rb->span.length - base->span_offset);
            WGSLNode *node = make_node(p, WGSL_NODE_EXPR_INDEX, span);
            if (node && idx) {
                WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, 2 * sizeof(WGSLNode *));
                if (arr) {
                    arr[0] = base; arr[1] = idx;
                    node->children    = arr;
                    node->child_count = 2;
                }
                base = node;
            }
            continue;
        }
        if (k == WGSL_TOK_DOT) {
            advance(p);
            const WGSLToken *member = expect(p, WGSL_TOK_IDENT, "member name");
            WGSLSpan member_span = member->span;
            WGSLSpan span = wgsl_span_make(
                base->span_offset,
                member_span.offset + member_span.length - base->span_offset);
            WGSLNode *node = make_node(p, WGSL_NODE_EXPR_MEMBER, span);
            if (node) {
                WGSLNode *member_node = make_node(p, WGSL_NODE_EXPR_IDENT, member_span);
                if (member_node) {
                    member_node->payload[0] =
                        ((uint64_t)member_span.offset) | ((uint64_t)member_span.length << 32);
                    WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, 2 * sizeof(WGSLNode *));
                    if (arr) {
                        arr[0] = base; arr[1] = member_node;
                        node->children    = arr;
                        node->child_count = 2;
                    }
                }
                base = node;
            }
            continue;
        }
        return base;
    }
}

static WGSLNode *parse_primary(P *p) {
    const WGSLToken *t = peek(p);
    WGSLSpan span = t->span;

    switch (t->kind) {
    case WGSL_TOK_INT_LIT: {
        advance(p);
        WGSLNode *n = make_node(p, WGSL_NODE_EXPR_LITERAL_INT, span);
        if (n) n->payload[0] = (uint64_t)t->payload;
        return n;
    }
    case WGSL_TOK_FLOAT_LIT: {
        advance(p);
        WGSLNode *n = make_node(p, WGSL_NODE_EXPR_LITERAL_FLOAT, span);
        if (n) n->payload[0] = (uint64_t)t->payload;
        return n;
    }
    case WGSL_TOK_KW_TRUE:
    case WGSL_TOK_KW_FALSE: {
        advance(p);
        WGSLNode *n = make_node(p, WGSL_NODE_EXPR_LITERAL_BOOL, span);
        if (n) n->payload[0] = (t->kind == WGSL_TOK_KW_TRUE) ? 1u : 0u;
        return n;
    }
    case WGSL_TOK_LPAREN: {
        advance(p);
        WGSLNode *inner = parse_expression(p);
        const WGSLToken *rp = peek(p);
        expect(p, WGSL_TOK_RPAREN, "')'");
        WGSLSpan pspan = wgsl_span_make(
            span.offset, rp->span.offset + rp->span.length - span.offset);
        WGSLNode *n = make_node(p, WGSL_NODE_EXPR_PAREN, pspan);
        if (n && inner) {
            WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, sizeof(WGSLNode *));
            if (arr) { arr[0] = inner; n->children = arr; n->child_count = 1; }
        }
        return n;
    }
    case WGSL_TOK_IDENT: {
        advance(p);
        WGSLNode *id = make_node(p, WGSL_NODE_EXPR_IDENT, span);
        if (id) id->payload[0] = ((uint64_t)span.offset) | ((uint64_t)span.length << 32);
        if (at(p, WGSL_TOK_TEMPLATE_START)) {
            return parse_template_arg_list(p, id);
        }
        return id;
    }
    default:
        parse_error(p, t->span.offset, t->span.length,
                    "expected expression, got %s",
                    wgsl_token_kind_name(t->kind));
        advance(p);
        return make_node(p, WGSL_NODE_INVALID, t->span);
    }
}

/* type_specifier := template_elaborated_ident.  An IDENT with optional
 * `<...>`.  We synthesize the same shape as parse_primary's IDENT path. */
static WGSLNode *parse_type_specifier(P *p) {
    if (!at(p, WGSL_TOK_IDENT) && !at(p, WGSL_TOK_KW_VAR)) {
        const WGSLToken *t = peek(p);
        parse_error(p, t->span.offset, t->span.length,
                    "expected type specifier, got %s",
                    wgsl_token_kind_name(t->kind));
        advance(p);
        return make_node(p, WGSL_NODE_INVALID, t->span);
    }
    const WGSLToken *t = advance(p);
    WGSLNode *id = make_node(p, WGSL_NODE_EXPR_IDENT, t->span);
    if (id) id->payload[0] = ((uint64_t)t->span.offset) | ((uint64_t)t->span.length << 32);
    if (at(p, WGSL_TOK_TEMPLATE_START)) {
        return parse_template_arg_list(p, id);
    }
    return id;
}

/* ── Attributes ─────────────────────────────────────────────────── */

static WGSLNode *parse_attribute(P *p) {
    const WGSLToken *at_tok = expect(p, WGSL_TOK_AT, "'@'");
    /* Attribute names are context-dependent; lexer emits them as IDENT. */
    if (!at(p, WGSL_TOK_IDENT)) {
        const WGSLToken *t = peek(p);
        parse_error(p, t->span.offset, t->span.length,
                    "expected attribute name, got %s",
                    wgsl_token_kind_name(t->kind));
        return make_node(p, WGSL_NODE_INVALID, at_tok->span);
    }
    const WGSLToken *name = advance(p);

    NL kids; nl_init(&kids);
    WGSLNode *name_node = make_node(p, WGSL_NODE_EXPR_IDENT, name->span);
    if (name_node) {
        name_node->payload[0] =
            ((uint64_t)name->span.offset) | ((uint64_t)name->span.length << 32);
        nl_push(&kids, name_node);
    }

    uint32_t end = name->span.offset + name->span.length;
    if (match(p, WGSL_TOK_LPAREN)) {
        if (!at(p, WGSL_TOK_RPAREN)) {
            do {
                if (at(p, WGSL_TOK_RPAREN)) break;
                WGSLNode *arg = parse_expression(p);
                if (arg) nl_push(&kids, arg);
            } while (match(p, WGSL_TOK_COMMA));
        }
        const WGSLToken *rp = peek(p);
        if (rp->kind == WGSL_TOK_RPAREN) {
            end = rp->span.offset + rp->span.length;
            advance(p);
        } else {
            expect(p, WGSL_TOK_RPAREN, "')'");
        }
    }

    WGSLSpan span = wgsl_span_make(at_tok->span.offset, end - at_tok->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_ATTRIBUTE, span);
    if (!node) { nl_destroy(&kids); return NULL; }
    finalize_children(p, node, &kids);
    nl_destroy(&kids);
    return node;
}

static int parse_attributes(P *p, NL *out) {
    while (at(p, WGSL_TOK_AT)) {
        WGSLNode *a = parse_attribute(p);
        if (a) nl_push(out, a);
    }
    return 1;
}

/* ── Statements ─────────────────────────────────────────────────── */

static WGSLNode *parse_statement(P *p);

static WGSLNode *parse_compound_statement(P *p) {
    const WGSLToken *open = expect(p, WGSL_TOK_LBRACE, "'{'");

    NL kids; nl_init(&kids);
    while (!at(p, WGSL_TOK_RBRACE) && !at(p, WGSL_TOK_EOF)) {
        WGSLNode *s = parse_statement(p);
        if (s && s->kind != WGSL_NODE_INVALID) {
            nl_push(&kids, s);
        } else {
            recover_to_sync(p);
        }
    }
    const WGSLToken *close = peek(p);
    expect(p, WGSL_TOK_RBRACE, "'}'");

    WGSLSpan span = wgsl_span_make(
        open->span.offset, close->span.offset + close->span.length - open->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_STMT_COMPOUND, span);
    if (node) finalize_children(p, node, &kids);
    nl_destroy(&kids);
    return node;
}

/* Helper: optionally consume `;` and compute the right span end.
 * If `as_statement == 0` (used by for_init / for_update), skip the `;`.
 * Returns the desired end offset for the span. */
static uint32_t finish_stmt_or_inline(P *p, uint32_t fallback_end, int as_statement) {
    if (!as_statement) return fallback_end;
    const WGSLToken *semi = peek(p);
    expect(p, WGSL_TOK_SEMICOLON, "';'");
    return semi->span.offset + semi->span.length;
}

/* Parse `let IDENT (':' type)? '=' expr` (with or without trailing `;`). */
static WGSLNode *parse_let_decl(P *p, int as_statement) {
    const WGSLToken *kw   = expect(p, WGSL_TOK_KW_LET, "'let'");
    const WGSLToken *name = expect(p, WGSL_TOK_IDENT, "identifier");

    NL kids; nl_init(&kids);
    uint64_t name_pld =
        ((uint64_t)name->span.offset) | ((uint64_t)name->span.length << 32);

    int has_type = 0;
    if (match(p, WGSL_TOK_COLON)) {
        WGSLNode *t = parse_type_specifier(p);
        if (t) { nl_push(&kids, t); has_type = 1; }
    }
    expect(p, WGSL_TOK_EQUAL, "'='");
    WGSLNode *init = parse_expression(p);
    if (init) nl_push(&kids, init);

    uint32_t fallback = init ? init->span_offset + init->span_length
                             : name->span.offset + name->span.length;
    uint32_t end = finish_stmt_or_inline(p, fallback, as_statement);

    WGSLSpan span = wgsl_span_make(kw->span.offset, end - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_DECL_LET, span);
    if (node) {
        node->payload[0] = name_pld;
        node->payload[1] = (uint64_t)(has_type ? 1u : 0u);
        finalize_children(p, node, &kids);
    }
    nl_destroy(&kids);
    return node;
}

/* Parse `var ('<' tpl, … '>')? IDENT (':' type)? ('=' expr)?`.
 *
 * Both function-scope and module-scope use this parser; the caller
 * controls whether attributes (`attrs_or_null`) and a template list
 * (`allow_template`) are accepted.  Function-scope passes `NULL, 0`;
 * module-scope passes the collected `&attrs, 1`.
 *
 * DECL_VAR payload encoding (used by both contexts):
 *   payload[0]: name span (offset | length << 32)
 *   payload[1]: attr_count (low 32) | tpl_count (high 32)
 *   payload[2]: has_type (low 32) | has_init (high 32)
 *
 * Children layout: [attrs…][tpl_args…][type?][init?]. */
static WGSLNode *parse_var_decl(
    P *p, int as_statement, int allow_template, NL *attrs_or_null)
{
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_VAR, "'var'");

    NL kids; nl_init(&kids);
    uint32_t attr_count = 0;
    if (attrs_or_null) {
        for (size_t i = 0; i < attrs_or_null->count; i++)
            nl_push(&kids, attrs_or_null->items[i]);
        attr_count = (uint32_t)attrs_or_null->count;
    }

    /* Optional template list (module-scope only): <addr_space>
     * or <addr_space, access>. */
    uint32_t tpl_count = 0;
    if (allow_template && at(p, WGSL_TOK_TEMPLATE_START)) {
        advance(p);
        do {
            if (at(p, WGSL_TOK_TEMPLATE_END)) break;
            WGSLNode *e = parse_expression(p);
            if (e) { nl_push(&kids, e); tpl_count += 1; }
        } while (match(p, WGSL_TOK_COMMA));
        expect(p, WGSL_TOK_TEMPLATE_END, "'>' (template list end)");
    }

    const WGSLToken *name = expect(p, WGSL_TOK_IDENT, "variable name");
    uint64_t name_pld =
        ((uint64_t)name->span.offset) | ((uint64_t)name->span.length << 32);

    int has_type = 0;
    if (match(p, WGSL_TOK_COLON)) {
        WGSLNode *t = parse_type_specifier(p);
        if (t) { nl_push(&kids, t); has_type = 1; }
    }
    int has_init = 0;
    if (match(p, WGSL_TOK_EQUAL)) {
        WGSLNode *init = parse_expression(p);
        if (init) { nl_push(&kids, init); has_init = 1; }
    }

    uint32_t fallback = (kids.count > 0)
        ? kids.items[kids.count - 1]->span_offset
            + kids.items[kids.count - 1]->span_length
        : name->span.offset + name->span.length;
    uint32_t end = finish_stmt_or_inline(p, fallback, as_statement);

    uint32_t start = (attr_count > 0)
        ? attrs_or_null->items[0]->span_offset
        : kw->span.offset;
    WGSLSpan span = wgsl_span_make(start, end - start);
    WGSLNode *node = make_node(p, WGSL_NODE_DECL_VAR, span);
    if (node) {
        node->payload[0] = name_pld;
        node->payload[1] = ((uint64_t)attr_count) | ((uint64_t)tpl_count << 32);
        node->payload[2] =
            ((uint64_t)(has_type ? 1u : 0u)) |
            ((uint64_t)(has_init ? 1u : 0u) << 32);
        finalize_children(p, node, &kids);
    }
    nl_destroy(&kids);
    return node;
}

/* Parse `const IDENT (':' type)? '=' expr`.
 * (Module-scope `const` has the same shape but lives in TU; Round B
 * Iter 2 reuses this helper.) */
static WGSLNode *parse_const_decl(P *p, int as_statement) {
    const WGSLToken *kw   = expect(p, WGSL_TOK_KW_CONST, "'const'");
    const WGSLToken *name = expect(p, WGSL_TOK_IDENT, "name");

    NL kids; nl_init(&kids);
    uint64_t name_pld =
        ((uint64_t)name->span.offset) | ((uint64_t)name->span.length << 32);

    int has_type = 0;
    if (match(p, WGSL_TOK_COLON)) {
        WGSLNode *t = parse_type_specifier(p);
        if (t) { nl_push(&kids, t); has_type = 1; }
    }
    expect(p, WGSL_TOK_EQUAL, "'='");
    WGSLNode *init = parse_expression(p);
    if (init) nl_push(&kids, init);

    uint32_t fallback = init ? init->span_offset + init->span_length
                             : name->span.offset + name->span.length;
    uint32_t end = finish_stmt_or_inline(p, fallback, as_statement);

    WGSLSpan span = wgsl_span_make(kw->span.offset, end - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_DECL_CONST, span);
    if (node) {
        node->payload[0] = name_pld;
        node->payload[1] = (uint64_t)(has_type ? 1u : 0u);
        finalize_children(p, node, &kids);
    }
    nl_destroy(&kids);
    return node;
}

static WGSLNode *parse_const_assert_statement(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_CONST_ASSERT, "'const_assert'");
    WGSLNode *e = parse_expression(p);
    const WGSLToken *semi = peek(p);
    expect(p, WGSL_TOK_SEMICOLON, "';'");

    WGSLSpan span = wgsl_span_make(
        kw->span.offset, semi->span.offset + semi->span.length - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_DECL_CONST_ASSERT, span);
    if (node && e) {
        WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, sizeof(WGSLNode *));
        if (arr) { arr[0] = e; node->children = arr; node->child_count = 1; }
    }
    return node;
}

static WGSLNode *parse_return_statement(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_RETURN, "'return'");

    NL kids; nl_init(&kids);
    if (!at(p, WGSL_TOK_SEMICOLON)) {
        WGSLNode *e = parse_expression(p);
        if (e) nl_push(&kids, e);
    }
    const WGSLToken *semi = peek(p);
    expect(p, WGSL_TOK_SEMICOLON, "';'");

    WGSLSpan span = wgsl_span_make(
        kw->span.offset, semi->span.offset + semi->span.length - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_STMT_RETURN, span);
    if (node) finalize_children(p, node, &kids);
    nl_destroy(&kids);
    return node;
}

/* Default statement: handles assignment / compound assignment / phony /
 * increment / decrement / function-call statement.  Pass `as_statement=0`
 * for use as a for-init / for-update update.  Returns NULL if there is
 * nothing to parse. */
static WGSLNode *parse_default_statement(P *p, int as_statement) {
    /* Phony assignment: `_` `=` expr ';' */
    if (at(p, WGSL_TOK_UNDERSCORE) && peek_n(p, 1)->kind == WGSL_TOK_EQUAL) {
        const WGSLToken *under = advance(p);
        advance(p);   /* `=` */
        WGSLNode *rhs = parse_expression(p);
        uint32_t fallback = rhs ? rhs->span_offset + rhs->span_length
                                : under->span.offset + under->span.length;
        uint32_t end = finish_stmt_or_inline(p, fallback, as_statement);
        WGSLSpan span = wgsl_span_make(under->span.offset, end - under->span.offset);
        WGSLNode *node = make_node(p, WGSL_NODE_STMT_PHONY_ASSIGN, span);
        if (node && rhs) {
            WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, sizeof(WGSLNode *));
            if (arr) { arr[0] = rhs; node->children = arr; node->child_count = 1; }
        }
        return node;
    }

    WGSLNode *lhs = parse_expression(p);
    if (!lhs) return NULL;

    WGSLTokenKind k = peek(p)->kind;

    /* `++` / `--` */
    if (k == WGSL_TOK_PLUS_PLUS || k == WGSL_TOK_MINUS_MINUS) {
        advance(p);
        uint32_t fallback = lhs->span_offset + lhs->span_length + 2;
        uint32_t end = finish_stmt_or_inline(p, fallback, as_statement);
        WGSLSpan span = wgsl_span_make(lhs->span_offset, end - lhs->span_offset);
        WGSLNodeKind nk = (k == WGSL_TOK_PLUS_PLUS)
            ? WGSL_NODE_STMT_INCREMENT : WGSL_NODE_STMT_DECREMENT;
        WGSLNode *node = make_node(p, nk, span);
        if (node) {
            WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, sizeof(WGSLNode *));
            if (arr) { arr[0] = lhs; node->children = arr; node->child_count = 1; }
        }
        return node;
    }

    /* Assignment forms */
    int is_assign = 0, is_compound = 0;
    switch (k) {
    case WGSL_TOK_EQUAL:                  is_assign = 1; break;
    case WGSL_TOK_PLUS_EQUAL:
    case WGSL_TOK_MINUS_EQUAL:
    case WGSL_TOK_STAR_EQUAL:
    case WGSL_TOK_SLASH_EQUAL:
    case WGSL_TOK_PERCENT_EQUAL:
    case WGSL_TOK_AMP_EQUAL:
    case WGSL_TOK_PIPE_EQUAL:
    case WGSL_TOK_CARET_EQUAL:
    case WGSL_TOK_LESS_LESS_EQUAL:
    case WGSL_TOK_GREATER_GREATER_EQUAL:  is_compound = 1; is_assign = 1; break;
    default: break;
    }

    if (is_assign) {
        const WGSLToken *op = peek(p);
        WGSLTokenKind op_kind = op->kind;
        WGSLSpan op_span = op->span;
        advance(p);
        WGSLNode *rhs = parse_expression(p);
        uint32_t fallback = rhs ? rhs->span_offset + rhs->span_length
                                : op_span.offset + op_span.length;
        uint32_t end = finish_stmt_or_inline(p, fallback, as_statement);
        WGSLSpan span = wgsl_span_make(lhs->span_offset, end - lhs->span_offset);
        WGSLNodeKind nk = is_compound
            ? WGSL_NODE_STMT_COMPOUND_ASSIGN : WGSL_NODE_STMT_ASSIGN;
        WGSLNode *node = make_node(p, nk, span);
        if (node) {
            node->payload[0] = (uint64_t)op_kind;
            node->payload[1] =
                ((uint64_t)op_span.offset) | ((uint64_t)op_span.length << 32);
            WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, 2 * sizeof(WGSLNode *));
            if (arr && rhs) {
                arr[0] = lhs; arr[1] = rhs;
                node->children = arr; node->child_count = 2;
            }
        }
        return node;
    }

    /* Plain expression statement (call, indexing, etc.). */
    uint32_t fallback = lhs->span_offset + lhs->span_length;
    uint32_t end = finish_stmt_or_inline(p, fallback, as_statement);
    WGSLSpan span = wgsl_span_make(lhs->span_offset, end - lhs->span_offset);
    WGSLNode *node = make_node(p, WGSL_NODE_STMT_FN_CALL, span);
    if (node) {
        WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, sizeof(WGSLNode *));
        if (arr) { arr[0] = lhs; node->children = arr; node->child_count = 1; }
    }
    return node;
}

/* — Control flow — */

static WGSLNode *parse_if_statement(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_IF, "'if'");
    WGSLNode *cond  = parse_expression(p);
    WGSLNode *then_ = parse_compound_statement(p);

    NL kids; nl_init(&kids);
    if (cond)  nl_push(&kids, cond);
    if (then_) nl_push(&kids, then_);

    while (at(p, WGSL_TOK_KW_ELSE)) {
        const WGSLToken *else_kw = advance(p);
        if (at(p, WGSL_TOK_KW_IF)) {
            advance(p);
            WGSLNode *ec = parse_expression(p);
            WGSLNode *eb = parse_compound_statement(p);
            uint32_t end = eb ? eb->span_offset + eb->span_length
                              : else_kw->span.offset + else_kw->span.length;
            WGSLSpan espan = wgsl_span_make(else_kw->span.offset, end - else_kw->span.offset);
            WGSLNode *enode = make_node(p, WGSL_NODE_STMT_ELSE_IF, espan);
            if (enode) {
                NL ek; nl_init(&ek);
                if (ec) nl_push(&ek, ec);
                if (eb) nl_push(&ek, eb);
                finalize_children(p, enode, &ek);
                nl_destroy(&ek);
                nl_push(&kids, enode);
            }
        } else {
            WGSLNode *eb = parse_compound_statement(p);
            uint32_t end = eb ? eb->span_offset + eb->span_length
                              : else_kw->span.offset + else_kw->span.length;
            WGSLSpan espan = wgsl_span_make(else_kw->span.offset, end - else_kw->span.offset);
            WGSLNode *enode = make_node(p, WGSL_NODE_STMT_ELSE, espan);
            if (enode) {
                NL ek; nl_init(&ek);
                if (eb) nl_push(&ek, eb);
                finalize_children(p, enode, &ek);
                nl_destroy(&ek);
                nl_push(&kids, enode);
            }
            break;
        }
    }

    uint32_t end_off = (kids.count > 0)
        ? kids.items[kids.count - 1]->span_offset
            + kids.items[kids.count - 1]->span_length
        : kw->span.offset + kw->span.length;
    WGSLSpan span = wgsl_span_make(kw->span.offset, end_off - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_STMT_IF, span);
    if (node) finalize_children(p, node, &kids);
    nl_destroy(&kids);
    return node;
}

static WGSLNode *parse_while_statement(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_WHILE, "'while'");
    WGSLNode *cond = parse_expression(p);
    WGSLNode *body = parse_compound_statement(p);

    uint32_t end = body ? body->span_offset + body->span_length
                        : kw->span.offset + kw->span.length;
    WGSLSpan span = wgsl_span_make(kw->span.offset, end - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_STMT_WHILE, span);
    if (node) {
        NL kids; nl_init(&kids);
        if (cond) nl_push(&kids, cond);
        if (body) nl_push(&kids, body);
        finalize_children(p, node, &kids);
        nl_destroy(&kids);
    }
    return node;
}

static WGSLNode *parse_statement(P *p);
static WGSLNode *parse_continuing_statement(P *p);

/* `for` `(` for_init? `;` expression? `;` for_update? `)` compound
 * payload[0] low byte = bitmap (bit0=init, bit1=cond, bit2=update).
 * Children layout: [init?, cond?, update?, body] in present order. */
static WGSLNode *parse_for_init_or_update(P *p) {
    WGSLTokenKind k = peek(p)->kind;
    if (k == WGSL_TOK_KW_LET)          return parse_let_decl(p, /*as_statement=*/0);
    if (k == WGSL_TOK_KW_VAR)          return parse_var_decl(p, /*as_statement=*/0,
                                                              /*allow_template=*/0,
                                                              /*attrs=*/NULL);
    if (k == WGSL_TOK_KW_CONST)        return parse_const_decl(p, /*as_statement=*/0);
    return parse_default_statement(p, /*as_statement=*/0);
}

static WGSLNode *parse_for_statement(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_FOR, "'for'");
    expect(p, WGSL_TOK_LPAREN, "'('");

    NL kids; nl_init(&kids);
    uint32_t flags = 0;

    if (!at(p, WGSL_TOK_SEMICOLON)) {
        WGSLNode *init = parse_for_init_or_update(p);
        if (init) { nl_push(&kids, init); flags |= 1u; }
    }
    expect(p, WGSL_TOK_SEMICOLON, "';' (for header)");

    if (!at(p, WGSL_TOK_SEMICOLON)) {
        WGSLNode *cond = parse_expression(p);
        if (cond) { nl_push(&kids, cond); flags |= 2u; }
    }
    expect(p, WGSL_TOK_SEMICOLON, "';' (for header)");

    if (!at(p, WGSL_TOK_RPAREN)) {
        WGSLNode *upd = parse_for_init_or_update(p);
        if (upd) { nl_push(&kids, upd); flags |= 4u; }
    }
    expect(p, WGSL_TOK_RPAREN, "')'");

    WGSLNode *body = parse_compound_statement(p);
    if (body) nl_push(&kids, body);

    uint32_t end = body ? body->span_offset + body->span_length
                        : kw->span.offset + kw->span.length;
    WGSLSpan span = wgsl_span_make(kw->span.offset, end - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_STMT_FOR, span);
    if (node) {
        node->payload[0] = (uint64_t)flags;
        finalize_children(p, node, &kids);
    }
    nl_destroy(&kids);
    return node;
}

static WGSLNode *parse_loop_statement(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_LOOP, "'loop'");
    expect(p, WGSL_TOK_LBRACE, "'{'");

    NL kids; nl_init(&kids);
    while (!at(p, WGSL_TOK_RBRACE) && !at(p, WGSL_TOK_EOF)) {
        if (at(p, WGSL_TOK_KW_CONTINUING)) {
            WGSLNode *c = parse_continuing_statement(p);
            if (c) nl_push(&kids, c);
            break;     /* continuing must be the last statement */
        }
        WGSLNode *s = parse_statement(p);
        if (s && s->kind != WGSL_NODE_INVALID) nl_push(&kids, s);
        else recover_to_sync(p);
    }
    const WGSLToken *close = peek(p);
    expect(p, WGSL_TOK_RBRACE, "'}'");

    WGSLSpan span = wgsl_span_make(
        kw->span.offset, close->span.offset + close->span.length - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_STMT_LOOP, span);
    if (node) finalize_children(p, node, &kids);
    nl_destroy(&kids);
    return node;
}

static WGSLNode *parse_continuing_statement(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_CONTINUING, "'continuing'");
    expect(p, WGSL_TOK_LBRACE, "'{'");

    NL kids; nl_init(&kids);
    while (!at(p, WGSL_TOK_RBRACE) && !at(p, WGSL_TOK_EOF)) {
        /* `break if expr ;` is permitted — and only — at the tail. */
        if (at(p, WGSL_TOK_KW_BREAK) && peek_n(p, 1)->kind == WGSL_TOK_KW_IF) {
            const WGSLToken *bk = advance(p);
            advance(p);   /* if */
            WGSLNode *cond = parse_expression(p);
            const WGSLToken *semi = peek(p);
            expect(p, WGSL_TOK_SEMICOLON, "';'");
            WGSLSpan bspan = wgsl_span_make(
                bk->span.offset, semi->span.offset + semi->span.length - bk->span.offset);
            WGSLNode *bi = make_node(p, WGSL_NODE_STMT_BREAK_IF, bspan);
            if (bi && cond) {
                WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, sizeof(WGSLNode *));
                if (arr) { arr[0] = cond; bi->children = arr; bi->child_count = 1; }
            }
            if (bi) nl_push(&kids, bi);
            break;
        }
        WGSLNode *s = parse_statement(p);
        if (s && s->kind != WGSL_NODE_INVALID) nl_push(&kids, s);
        else recover_to_sync(p);
    }
    const WGSLToken *close = peek(p);
    expect(p, WGSL_TOK_RBRACE, "'}'");

    WGSLSpan span = wgsl_span_make(
        kw->span.offset, close->span.offset + close->span.length - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_STMT_CONTINUING, span);
    if (node) finalize_children(p, node, &kids);
    nl_destroy(&kids);
    return node;
}

static WGSLNode *parse_break_statement(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_BREAK, "'break'");
    /* `break if expr ;` is parsed at the tail of `continuing { … }` and
     * not here.  A bare `break ;` always becomes WGSL_NODE_STMT_BREAK. */
    const WGSLToken *semi = peek(p);
    expect(p, WGSL_TOK_SEMICOLON, "';'");
    WGSLSpan span = wgsl_span_make(
        kw->span.offset, semi->span.offset + semi->span.length - kw->span.offset);
    return make_node(p, WGSL_NODE_STMT_BREAK, span);
}

static WGSLNode *parse_continue_statement(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_CONTINUE, "'continue'");
    const WGSLToken *semi = peek(p);
    expect(p, WGSL_TOK_SEMICOLON, "';'");
    WGSLSpan span = wgsl_span_make(
        kw->span.offset, semi->span.offset + semi->span.length - kw->span.offset);
    return make_node(p, WGSL_NODE_STMT_CONTINUE, span);
}

static WGSLNode *parse_discard_statement(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_DISCARD, "'discard'");
    const WGSLToken *semi = peek(p);
    expect(p, WGSL_TOK_SEMICOLON, "';'");
    WGSLSpan span = wgsl_span_make(
        kw->span.offset, semi->span.offset + semi->span.length - kw->span.offset);
    return make_node(p, WGSL_NODE_STMT_DISCARD, span);
}

/* `case` selectors `:`? compound  |  `default` `:`? compound
 * payload[0] low 32 = selector_count
 * payload[0] high 32 = is_default_alone (0/1)
 * children: [selector_exprs..., body] for case;
 *           [body] for default-alone. */
static WGSLNode *parse_case_clause(P *p) {
    WGSLTokenKind k = peek(p)->kind;

    if (k == WGSL_TOK_KW_DEFAULT) {
        const WGSLToken *kw = advance(p);
        match(p, WGSL_TOK_COLON);
        WGSLNode *body = parse_compound_statement(p);
        uint32_t end = body ? body->span_offset + body->span_length
                            : kw->span.offset + kw->span.length;
        WGSLSpan span = wgsl_span_make(kw->span.offset, end - kw->span.offset);
        WGSLNode *node = make_node(p, WGSL_NODE_STMT_CASE_CLAUSE, span);
        if (node) {
            node->payload[0] = ((uint64_t)0) | ((uint64_t)1u << 32);
            if (body) {
                WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, sizeof(WGSLNode *));
                if (arr) { arr[0] = body; node->children = arr; node->child_count = 1; }
            }
        }
        return node;
    }

    /* `case` selectors */
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_CASE, "'case' or 'default'");
    NL sels; nl_init(&sels);
    do {
        if (at(p, WGSL_TOK_COLON) || at(p, WGSL_TOK_LBRACE)) break;
        if (at(p, WGSL_TOK_KW_DEFAULT)) {
            const WGSLToken *t = peek(p);
            parse_error(p, t->span.offset, t->span.length,
                        "Round B does not support `default` mid-case-list — "
                        "use a separate `default` clause");
            advance(p);
        } else {
            WGSLNode *e = parse_expression(p);
            if (e) nl_push(&sels, e);
        }
    } while (match(p, WGSL_TOK_COMMA));

    match(p, WGSL_TOK_COLON);
    WGSLNode *body = parse_compound_statement(p);

    uint32_t end = body ? body->span_offset + body->span_length
                        : kw->span.offset + kw->span.length;
    WGSLSpan span = wgsl_span_make(kw->span.offset, end - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_STMT_CASE_CLAUSE, span);
    if (node) {
        node->payload[0] = (uint64_t)sels.count;
        NL kids; nl_init(&kids);
        for (size_t i = 0; i < sels.count; i++) nl_push(&kids, sels.items[i]);
        if (body) nl_push(&kids, body);
        finalize_children(p, node, &kids);
        nl_destroy(&kids);
    }
    nl_destroy(&sels);
    return node;
}

static WGSLNode *parse_switch_statement(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_SWITCH, "'switch'");
    WGSLNode *selector = parse_expression(p);
    expect(p, WGSL_TOK_LBRACE, "'{' (switch body)");

    NL kids; nl_init(&kids);
    if (selector) nl_push(&kids, selector);

    while (!at(p, WGSL_TOK_RBRACE) && !at(p, WGSL_TOK_EOF)) {
        WGSLNode *clause = parse_case_clause(p);
        if (clause) nl_push(&kids, clause);
        else        recover_to_sync(p);
    }
    const WGSLToken *close = peek(p);
    expect(p, WGSL_TOK_RBRACE, "'}'");

    WGSLSpan span = wgsl_span_make(
        kw->span.offset, close->span.offset + close->span.length - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_STMT_SWITCH, span);
    if (node) finalize_children(p, node, &kids);
    nl_destroy(&kids);
    return node;
}

static WGSLNode *parse_statement(P *p) {
    WGSLTokenKind k = peek(p)->kind;
    switch (k) {
    case WGSL_TOK_LBRACE:           return parse_compound_statement(p);
    case WGSL_TOK_KW_LET:           return parse_let_decl(p, 1);
    case WGSL_TOK_KW_VAR:           return parse_var_decl(p, 1, 0, NULL);
    case WGSL_TOK_KW_CONST:         return parse_const_decl(p, 1);
    case WGSL_TOK_KW_CONST_ASSERT:  return parse_const_assert_statement(p);
    case WGSL_TOK_KW_RETURN:        return parse_return_statement(p);
    case WGSL_TOK_KW_IF:            return parse_if_statement(p);
    case WGSL_TOK_KW_WHILE:         return parse_while_statement(p);
    case WGSL_TOK_KW_FOR:           return parse_for_statement(p);
    case WGSL_TOK_KW_LOOP:          return parse_loop_statement(p);
    case WGSL_TOK_KW_SWITCH:        return parse_switch_statement(p);
    case WGSL_TOK_KW_BREAK:         return parse_break_statement(p);
    case WGSL_TOK_KW_CONTINUE:      return parse_continue_statement(p);
    case WGSL_TOK_KW_DISCARD:       return parse_discard_statement(p);
    case WGSL_TOK_SEMICOLON:
        advance(p);
        return make_node(p, WGSL_NODE_INVALID, wgsl_span_make(0, 0));
    default:
        return parse_default_statement(p, /*as_statement=*/1);
    }
}

/* ── Decls ──────────────────────────────────────────────────────── */

static WGSLNode *parse_param(P *p) {
    NL attrs; nl_init(&attrs);
    parse_attributes(p, &attrs);

    const WGSLToken *name = expect(p, WGSL_TOK_IDENT, "parameter name");
    expect(p, WGSL_TOK_COLON, "':'");
    WGSLNode *type = parse_type_specifier(p);

    uint32_t attr_count = (uint32_t)attrs.count;

    NL kids; nl_init(&kids);
    for (size_t i = 0; i < attrs.count; i++) nl_push(&kids, attrs.items[i]);
    if (type) nl_push(&kids, type);
    nl_destroy(&attrs);

    uint32_t end = type ? type->span_offset + type->span_length
                        : name->span.offset + name->span.length;
    uint32_t start = (kids.count > 0 && kids.items[0]->kind == WGSL_NODE_ATTRIBUTE)
                       ? kids.items[0]->span_offset
                       : name->span.offset;

    WGSLSpan span = wgsl_span_make(start, end - start);
    WGSLNode *node = make_node(p, WGSL_NODE_DECL_PARAM, span);
    if (node) {
        node->payload[0] = ((uint64_t)name->span.offset) | ((uint64_t)name->span.length << 32);
        node->payload[1] = (uint64_t)attr_count;
        finalize_children(p, node, &kids);
    }
    nl_destroy(&kids);
    return node;
}

static WGSLNode *parse_function_decl(P *p, NL *fn_attrs) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_FN, "'fn'");
    const WGSLToken *name = expect(p, WGSL_TOK_IDENT, "function name");
    expect(p, WGSL_TOK_LPAREN, "'('");

    NL kids; nl_init(&kids);

    /* fn attrs first, source order */
    for (size_t i = 0; i < fn_attrs->count; i++) nl_push(&kids, fn_attrs->items[i]);
    uint32_t fn_attr_count = (uint32_t)fn_attrs->count;

    /* params */
    uint32_t param_count = 0;
    if (!at(p, WGSL_TOK_RPAREN)) {
        do {
            if (at(p, WGSL_TOK_RPAREN)) break;
            WGSLNode *param = parse_param(p);
            if (param) { nl_push(&kids, param); param_count += 1; }
        } while (match(p, WGSL_TOK_COMMA));
    }
    expect(p, WGSL_TOK_RPAREN, "')'");

    /* return attrs + return type */
    uint32_t ret_attr_count = 0;
    uint32_t has_return_type = 0;
    if (match(p, WGSL_TOK_ARROW)) {
        NL ret_attrs; nl_init(&ret_attrs);
        parse_attributes(p, &ret_attrs);
        for (size_t i = 0; i < ret_attrs.count; i++) {
            nl_push(&kids, ret_attrs.items[i]);
            ret_attr_count += 1;
        }
        nl_destroy(&ret_attrs);

        WGSLNode *ret_type = parse_type_specifier(p);
        if (ret_type) { nl_push(&kids, ret_type); has_return_type = 1; }
    }

    /* body */
    WGSLNode *body = parse_compound_statement(p);
    if (body) nl_push(&kids, body);

    uint32_t start = fn_attr_count > 0 ? fn_attrs->items[0]->span_offset : kw->span.offset;
    uint32_t end   = body ? body->span_offset + body->span_length
                          : kw->span.offset + kw->span.length;

    WGSLSpan span = wgsl_span_make(start, end - start);
    WGSLNode *node = make_node(p, WGSL_NODE_DECL_FUNCTION, span);
    if (node) {
        node->payload[0] = ((uint64_t)name->span.offset) | ((uint64_t)name->span.length << 32);
        node->payload[1] = ((uint64_t)fn_attr_count) | ((uint64_t)param_count << 32);
        node->payload[2] = ((uint64_t)ret_attr_count) | ((uint64_t)has_return_type << 32);
        finalize_children(p, node, &kids);
    }
    nl_destroy(&kids);
    return node;
}

/* ── Module-scope decls ─────────────────────────────────────────── */

/* Helper: build a child IDENT node for a token's span. */
static WGSLNode *make_ident_from(P *p, const WGSLToken *t) {
    WGSLNode *n = make_node(p, WGSL_NODE_EXPR_IDENT, t->span);
    if (n) n->payload[0] =
        ((uint64_t)t->span.offset) | ((uint64_t)t->span.length << 32);
    return n;
}

/* `attribute* IDENT ':' type_specifier`. */
static WGSLNode *parse_struct_member(P *p) {
    NL attrs; nl_init(&attrs);
    parse_attributes(p, &attrs);
    uint32_t attr_count = (uint32_t)attrs.count;

    const WGSLToken *name = expect(p, WGSL_TOK_IDENT, "member name");
    expect(p, WGSL_TOK_COLON, "':'");
    WGSLNode *type = parse_type_specifier(p);

    NL kids; nl_init(&kids);
    for (size_t i = 0; i < attrs.count; i++) nl_push(&kids, attrs.items[i]);
    if (type) nl_push(&kids, type);
    nl_destroy(&attrs);

    uint32_t start = (attr_count > 0) ? kids.items[0]->span_offset
                                       : name->span.offset;
    uint32_t end = type ? type->span_offset + type->span_length
                        : name->span.offset + name->span.length;
    WGSLSpan span = wgsl_span_make(start, end - start);
    WGSLNode *node = make_node(p, WGSL_NODE_DECL_STRUCT_MEMBER, span);
    if (node) {
        node->payload[0] = ((uint64_t)name->span.offset) | ((uint64_t)name->span.length << 32);
        node->payload[1] = (uint64_t)attr_count;
        finalize_children(p, node, &kids);
    }
    nl_destroy(&kids);
    return node;
}

/* `'struct' IDENT '{' member (',' member)* ','? '}'`. */
static WGSLNode *parse_struct_decl(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_STRUCT, "'struct'");
    const WGSLToken *name = expect(p, WGSL_TOK_IDENT, "struct name");
    expect(p, WGSL_TOK_LBRACE, "'{'");

    NL kids; nl_init(&kids);
    while (!at(p, WGSL_TOK_RBRACE) && !at(p, WGSL_TOK_EOF)) {
        WGSLNode *m = parse_struct_member(p);
        if (m) nl_push(&kids, m);
        if (!match(p, WGSL_TOK_COMMA)) break;
    }
    const WGSLToken *close = peek(p);
    expect(p, WGSL_TOK_RBRACE, "'}'");

    WGSLSpan span = wgsl_span_make(
        kw->span.offset, close->span.offset + close->span.length - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_DECL_STRUCT, span);
    if (node) {
        node->payload[0] = ((uint64_t)name->span.offset) | ((uint64_t)name->span.length << 32);
        node->payload[1] = (uint64_t)kids.count;     /* member count */
        finalize_children(p, node, &kids);
    }
    nl_destroy(&kids);
    return node;
}

/* `'alias' IDENT '=' type_specifier ';'`. */
static WGSLNode *parse_alias_decl(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_ALIAS, "'alias'");
    const WGSLToken *name = expect(p, WGSL_TOK_IDENT, "alias name");
    expect(p, WGSL_TOK_EQUAL, "'='");
    WGSLNode *type = parse_type_specifier(p);
    const WGSLToken *semi = peek(p);
    expect(p, WGSL_TOK_SEMICOLON, "';'");

    WGSLSpan span = wgsl_span_make(
        kw->span.offset, semi->span.offset + semi->span.length - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_DECL_ALIAS, span);
    if (node) {
        node->payload[0] = ((uint64_t)name->span.offset) | ((uint64_t)name->span.length << 32);
        if (type) {
            WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, sizeof(WGSLNode *));
            if (arr) { arr[0] = type; node->children = arr; node->child_count = 1; }
        }
    }
    return node;
}

/* `attribute* 'override' IDENT (':' type)? ('=' expr)? ';'`.
 * Payload mirrors the var encoding so walkers can reuse logic.
 *   payload[0]: name span
 *   payload[1]: attr_count (low) | tpl_count = 0 (high)
 *   payload[2]: has_type | has_init << 32
 *   children: [attrs…][type?][init?] */
static WGSLNode *parse_override_decl(P *p, NL *attrs) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_OVERRIDE, "'override'");
    const WGSLToken *name = expect(p, WGSL_TOK_IDENT, "override name");

    NL kids; nl_init(&kids);
    for (size_t i = 0; i < attrs->count; i++) nl_push(&kids, attrs->items[i]);
    uint32_t attr_count = (uint32_t)attrs->count;

    int has_type = 0;
    if (match(p, WGSL_TOK_COLON)) {
        WGSLNode *t = parse_type_specifier(p);
        if (t) { nl_push(&kids, t); has_type = 1; }
    }
    int has_init = 0;
    if (match(p, WGSL_TOK_EQUAL)) {
        WGSLNode *init = parse_expression(p);
        if (init) { nl_push(&kids, init); has_init = 1; }
    }
    const WGSLToken *semi = peek(p);
    expect(p, WGSL_TOK_SEMICOLON, "';'");

    uint32_t start = (attr_count > 0) ? attrs->items[0]->span_offset : kw->span.offset;
    WGSLSpan span = wgsl_span_make(
        start, semi->span.offset + semi->span.length - start);
    WGSLNode *node = make_node(p, WGSL_NODE_DECL_OVERRIDE, span);
    if (node) {
        node->payload[0] = ((uint64_t)name->span.offset) | ((uint64_t)name->span.length << 32);
        node->payload[1] = (uint64_t)attr_count;     /* tpl_count = 0 */
        node->payload[2] =
            ((uint64_t)(has_type ? 1u : 0u)) |
            ((uint64_t)(has_init ? 1u : 0u) << 32);
        finalize_children(p, node, &kids);
    }
    nl_destroy(&kids);
    return node;
}

/* ── Directives ─────────────────────────────────────────────────── */

/* Generic comma-separated ident-list directive: `kw name (',' name)* ','? ';'`.
 * Each name becomes an EXPR_IDENT child. */
static WGSLNode *parse_ident_list_directive(
    P *p, WGSLTokenKind kw_kind, const char *kw_name, WGSLNodeKind node_kind)
{
    const WGSLToken *kw = expect(p, kw_kind, kw_name);
    NL names; nl_init(&names);
    do {
        if (at(p, WGSL_TOK_SEMICOLON)) break;          /* trailing comma */
        const WGSLToken *t = expect(p, WGSL_TOK_IDENT, "extension name");
        WGSLNode *n = make_ident_from(p, t);
        if (n) nl_push(&names, n);
    } while (match(p, WGSL_TOK_COMMA));
    const WGSLToken *semi = peek(p);
    expect(p, WGSL_TOK_SEMICOLON, "';'");

    WGSLSpan span = wgsl_span_make(
        kw->span.offset, semi->span.offset + semi->span.length - kw->span.offset);
    WGSLNode *node = make_node(p, node_kind, span);
    if (node) {
        node->payload[0] = (uint64_t)names.count;
        finalize_children(p, node, &names);
    }
    nl_destroy(&names);
    return node;
}

/* `'diagnostic' '(' severity_name ',' rule_name ('.' sub_name)? ')' ';'`
 *   payload[0] = rule-part-count (1 if `rule`, 2 if `rule.sub`)
 *   children:    [severity_ident, rule_ident, optional sub_ident] */
static WGSLNode *parse_diagnostic_directive(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_DIAGNOSTIC, "'diagnostic'");
    expect(p, WGSL_TOK_LPAREN, "'('");

    NL kids; nl_init(&kids);

    const WGSLToken *sev = expect(p, WGSL_TOK_IDENT, "severity name");
    nl_push(&kids, make_ident_from(p, sev));
    expect(p, WGSL_TOK_COMMA, "','");

    const WGSLToken *r1 = expect(p, WGSL_TOK_IDENT, "rule name");
    nl_push(&kids, make_ident_from(p, r1));
    uint32_t rule_parts = 1;
    if (match(p, WGSL_TOK_DOT)) {
        const WGSLToken *r2 = expect(p, WGSL_TOK_IDENT, "rule sub-name");
        nl_push(&kids, make_ident_from(p, r2));
        rule_parts = 2;
    }
    expect(p, WGSL_TOK_RPAREN, "')'");
    const WGSLToken *semi = peek(p);
    expect(p, WGSL_TOK_SEMICOLON, "';'");

    WGSLSpan span = wgsl_span_make(
        kw->span.offset, semi->span.offset + semi->span.length - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_DIR_DIAGNOSTIC, span);
    if (node) {
        node->payload[0] = (uint64_t)rule_parts;
        finalize_children(p, node, &kids);
    }
    nl_destroy(&kids);
    return node;
}

/* ── Global decl dispatch ───────────────────────────────────────── */

static int parse_global_decl(P *p, NL *out) {
    NL attrs; nl_init(&attrs);
    parse_attributes(p, &attrs);

    WGSLTokenKind k = peek(p)->kind;
    WGSLNode *decl = NULL;
    int consumed_attrs = 0;
    int produced = 0;

    switch (k) {
    case WGSL_TOK_KW_FN:
        decl = parse_function_decl(p, &attrs);
        consumed_attrs = 1;
        break;

    case WGSL_TOK_KW_STRUCT:
        decl = parse_struct_decl(p);
        break;

    case WGSL_TOK_KW_ALIAS:
        decl = parse_alias_decl(p);
        break;

    case WGSL_TOK_KW_CONST:
        decl = parse_const_decl(p, /*as_statement=*/1);
        break;

    case WGSL_TOK_KW_CONST_ASSERT:
        decl = parse_const_assert_statement(p);
        break;

    case WGSL_TOK_KW_OVERRIDE:
        decl = parse_override_decl(p, &attrs);
        consumed_attrs = 1;
        break;

    case WGSL_TOK_KW_VAR:
        decl = parse_var_decl(p, /*as_statement=*/1,
                              /*allow_template=*/1, &attrs);
        consumed_attrs = 1;
        break;

    case WGSL_TOK_KW_ENABLE:
        decl = parse_ident_list_directive(
            p, WGSL_TOK_KW_ENABLE, "'enable'", WGSL_NODE_DIR_ENABLE);
        break;

    case WGSL_TOK_KW_REQUIRES:
        decl = parse_ident_list_directive(
            p, WGSL_TOK_KW_REQUIRES, "'requires'", WGSL_NODE_DIR_REQUIRES);
        break;

    case WGSL_TOK_KW_DIAGNOSTIC:
        decl = parse_diagnostic_directive(p);
        break;

    default: {
        const WGSLToken *t = peek(p);
        parse_error(p, t->span.offset, t->span.length,
                    "expected a top-level declaration, got %s",
                    wgsl_token_kind_name(t->kind));
        break;
    }
    }

    /* Diagnose attributes that the declaration kind doesn't accept. */
    if (attrs.count > 0 && !consumed_attrs) {
        WGSLNode *first = attrs.items[0];
        parse_error(p, first->span_offset, first->span_length,
                    "attributes are not permitted on this declaration");
    }
    nl_destroy(&attrs);

    if (decl) produced = nl_push(out, decl);
    return produced;
}

/* ── Entry point ────────────────────────────────────────────────── */

int wgsl_parse(
    const WGSLLexResult *tokens,
    const WGSLSource    *source,
    WGSLArena           *arena,
    WGSLDiagBag         *diag,
    WGSLAst             *out)
{
    if (!tokens || !source || !arena || !diag || !out) return 0;

    P p;
    memset(&p, 0, sizeof p);
    p.tokens = tokens->tokens;
    p.count  = tokens->count;
    p.src    = source;
    p.arena  = arena;
    p.diag   = diag;

    WGSLNode *tu = make_node(&p, WGSL_NODE_TRANSLATION_UNIT,
                             wgsl_span_make(0, (uint32_t)source->length));
    if (!tu) return 0;

    NL list; nl_init(&list);
    while (!at(&p, WGSL_TOK_EOF)) {
        if (at(&p, WGSL_TOK_SEMICOLON)) { advance(&p); continue; }
        if (!parse_global_decl(&p, &list)) {
            recover_to_sync(&p);
        }
    }
    finalize_children(&p, tu, &list);
    nl_destroy(&list);

    out->root = tu;
    return p.had_error ? 0 : 1;
}
