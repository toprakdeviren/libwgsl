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
    int               has_seen_decl;
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
        if (nl->capacity > SIZE_MAX / 2) return 0;
        size_t cap = nl->capacity ? nl->capacity * 2 : 8;
        if (cap > SIZE_MAX / sizeof(WGSLNode *)) return 0;
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
    if (got->kind == WGSL_TOK_RESERVED) {
        uint32_t off = got->span.offset;
        uint32_t len = got->span.length;
        parse_error(p, off, len,
                    "'%.*s' is a reserved word and cannot be used in WGSL",
                    (int)len, p->src->bytes + off);
    } else {
        parse_error(p, got->span.offset, got->span.length,
                    "expected %s, got %s", what, wgsl_token_kind_name(got->kind));
    }
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
    if (list->items) {
        memcpy(arr, list->items, list->count * sizeof(WGSLNode *));
    }
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


#include "parser/expressions_attrs.inc"
#include "parser/statements.inc"
#include "parser/decls_entry.inc"
