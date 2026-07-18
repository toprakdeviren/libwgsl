/**
 * @file helpers.c — parser cursor, NL list, node construction.
 */
#include "internal/parser_priv.h"

void nl_init(NL *nl) { nl->items = NULL; nl->count = 0; nl->capacity = 0; }
void nl_destroy(NL *nl) { free(nl->items); nl->items = NULL; nl->count = 0; nl->capacity = 0; }
int  nl_push(NL *nl, WGSLNode *n) {
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

int is_trivia_kind(WGSLTokenKind k) {
    return k == WGSL_TOK_BLANKSPACE
        || k == WGSL_TOK_LINE_COMMENT
        || k == WGSL_TOK_BLOCK_COMMENT;
}
void skip_trivia(P *p) {
    while (p->pos < p->count && is_trivia_kind(p->tokens[p->pos].kind)) p->pos++;
}
const WGSLToken *peek_n(P *p, size_t n) {
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
const WGSLToken *peek(P *p)              { return peek_n(p, 0); }
const WGSLToken *advance(P *p) {
    skip_trivia(p);
    if (p->pos >= p->count) return &p->tokens[p->count - 1];
    return &p->tokens[p->pos++];
}
int at(P *p, WGSLTokenKind k)            { return peek(p)->kind == k; }
int match(P *p, WGSLTokenKind k)         { if (at(p, k)) { advance(p); return 1; } return 0; }

void parse_error(P *p, uint32_t off, uint32_t len, const char *fmt, ...) {
    char buf[256];
    va_list ap; va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    wgsl_diag_emit_at(p->diag, p->src, WGSL_DIAG_ERROR, off, len, NULL, "%s", buf);
    p->had_error = 1;
}

const WGSLToken *expect(P *p, WGSLTokenKind k, const char *what) {
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
void recover_to_sync(P *p) {
    while (!at(p, WGSL_TOK_EOF)) {
        WGSLTokenKind k = peek(p)->kind;
        if (k == WGSL_TOK_SEMICOLON) { advance(p); return; }
        if (k == WGSL_TOK_RBRACE)    { return; }
        advance(p);
    }
}

/* Node construction. */

WGSLNode *make_node(P *p, WGSLNodeKind kind, WGSLSpan span) {
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

int finalize_children(P *p, WGSLNode *parent, NL *list) {
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

