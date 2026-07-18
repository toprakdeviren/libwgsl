/**
 * @file format.c — canonical WGSL pretty-printer.
 *
 * Walks a checked (or parseable) AST and emits consistently indented
 * source.  Leaf text (idents, literals, operators) is taken from the
 * original source spans so spelling is preserved; layout (spaces,
 * newlines, indent) is canonical.  Line/block comments from the input
 * are reattached in source order (leading before decls/stmts, trailing
 * on the same line when possible).
 *
 * Goal: for well-formed modules, parse∘format is idempotent under a
 * second format pass (`format(format(src)) == format(src)`), and
 * comments survive a round-trip.
 */
#include "wgsl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal/ast.h"
#include "internal/token.h"

typedef struct {
    uint32_t off;
    uint32_t len;
} FmtCmt;

typedef struct {
    char  *buf;
    size_t len, cap;
    int    indent;
    int    at_line_start;
    int    oom;
    const char *src;
    size_t src_len;
    /* Trivia (comments), sorted by source offset; cursor advances. */
    const FmtCmt *cmts;
    int    ncmt;
    int    cmt_i;
} F;

static void f_grow(F *f, size_t n) {
    if (f->oom) return;
    if (f->len + n + 1 <= f->cap) return;
    size_t nc = f->cap ? f->cap : 256;
    while (nc < f->len + n + 1) {
        if (nc > SIZE_MAX / 2) { f->oom = 1; return; }
        nc *= 2;
    }
    char *g = (char *)realloc(f->buf, nc);
    if (!g) { f->oom = 1; return; }
    f->buf = g; f->cap = nc;
}

static void f_putn(F *f, const char *s, size_t n) {
    if (!s || !n || f->oom) return;
    f_grow(f, n);
    if (f->oom) return;
    memcpy(f->buf + f->len, s, n);
    f->len += n;
    f->buf[f->len] = 0;
    f->at_line_start = (s[n - 1] == '\n');
}

static void f_puts(F *f, const char *s) { if (s) f_putn(f, s, strlen(s)); }
static void f_putc(F *f, char c) { f_putn(f, &c, 1); }

static void f_nl(F *f) {
    f_putc(f, '\n');
    f->at_line_start = 1;
}

static void f_indent(F *f) {
    if (!f->at_line_start) return;
    for (int i = 0; i < f->indent; i++) f_puts(f, "    ");
    f->at_line_start = 0;
}

static void f_span(F *f, uint32_t off, uint32_t len) {
    if (!f->src || (size_t)off + (size_t)len > f->src_len) return;
    f_indent(f);
    f_putn(f, f->src + off, len);
}

/* Emit every unconsumed comment whose start is strictly before `limit`. */
static void f_comments_before(F *f, uint32_t limit) {
    if (!f->cmts || !f->src) return;
    while (f->cmt_i < f->ncmt && f->cmts[f->cmt_i].off < limit) {
        const FmtCmt *c = &f->cmts[f->cmt_i++];
        if ((size_t)c->off + (size_t)c->len > f->src_len) continue;
        f_indent(f);
        f_putn(f, f->src + c->off, c->len);
        /* Line comments exclude the terminating newline. */
        if (c->len == 0 || f->src[c->off + c->len - 1] != '\n')
            f_nl(f);
        else
            f->at_line_start = 1;
    }
}

/* Same-line trailing comment after original source position `after`. */
static void f_trailing_after(F *f, uint32_t after) {
    if (!f->cmts || !f->src || f->cmt_i >= f->ncmt) return;
    const FmtCmt *c = &f->cmts[f->cmt_i];
    if (c->off < after) return;
    if ((size_t)c->off + (size_t)c->len > f->src_len) return;
    for (uint32_t i = after; i < c->off; i++) {
        char ch = f->src[i];
        if (ch == '\n' || ch == '\r') return;
        if (ch != ' ' && ch != '\t' && ch != 0x0Bu && ch != 0x0Cu) return;
    }
    if (!f->at_line_start) f_putc(f, ' ');
    f_putn(f, f->src + c->off, c->len);
    f->cmt_i++;
}

static void f_end_semi(F *f, const WGSLNode *n) {
    f_putc(f, ';');
    if (n) {
        uint32_t end = n->span_offset;
        if (n->span_length > 0 &&
            n->span_offset <= UINT32_MAX - n->span_length)
            end = n->span_offset + n->span_length;
        f_trailing_after(f, end);
    }
    f_nl(f);
}

static void f_name(F *f, const WGSLNode *n) {
    if (!n) return;
    uint32_t o = wgsl_node_name_span(n).offset;
    uint32_t l = wgsl_node_name_span(n).length;
    /* For nodes that store name in payload[0] as span pair. */
    if (n->kind == WGSL_NODE_EXPR_IDENT ||
        n->kind == WGSL_NODE_DECL_FUNCTION ||
        n->kind == WGSL_NODE_DECL_PARAM ||
        n->kind == WGSL_NODE_DECL_STRUCT ||
        n->kind == WGSL_NODE_DECL_STRUCT_MEMBER ||
        n->kind == WGSL_NODE_DECL_CONST ||
        n->kind == WGSL_NODE_DECL_LET ||
        n->kind == WGSL_NODE_DECL_VAR ||
        n->kind == WGSL_NODE_DECL_OVERRIDE ||
        n->kind == WGSL_NODE_DECL_ALIAS ||
        n->kind == WGSL_NODE_ATTRIBUTE)
    {
        f_span(f, o, l);
        return;
    }
    /* fallback: full node span */
    f_span(f, n->span_offset, n->span_length);
}

static void emit_node(F *f, const WGSLNode *n, int depth);
static void emit_expr(F *f, const WGSLNode *n, int depth);
static void emit_stmt(F *f, const WGSLNode *n, int depth);
static void emit_type(F *f, const WGSLNode *n, int depth);

static void emit_attrs(F *f, const WGSLNode *const *kids, uint32_t first,
                       uint32_t count, int depth)
{
    for (uint32_t i = 0; i < count; i++) {
        const WGSLNode *a = kids[first + i];
        if (!a || a->kind != WGSL_NODE_ATTRIBUTE) continue;
        f_indent(f);
        f_putc(f, '@');
        /* attr name is children[0] ident or payload */
        if (a->child_count > 0 && a->children[0])
            f_name(f, a->children[0]);
        if (a->child_count > 1) {
            f_putc(f, '(');
            for (uint32_t k = 1; k < a->child_count; k++) {
                if (k > 1) f_puts(f, ", ");
                emit_expr(f, a->children[k], depth + 1);
            }
            f_putc(f, ')');
        }
        f_putc(f, ' ');
    }
}

static void emit_type(F *f, const WGSLNode *n, int depth) {
    if (!n) return;
    if (depth > WGSL_MAX_AST_DEPTH) { f_puts(f, "/*deep*/"); return; }
    /* Type specs are expressions (templated idents etc.). */
    emit_expr(f, n, depth);
}

static void emit_expr(F *f, const WGSLNode *n, int depth) {
    if (!n) return;
    if (depth > WGSL_MAX_AST_DEPTH) { f_puts(f, "/*deep*/"); return; }
    f_indent(f);
    switch (n->kind) {
    case WGSL_NODE_EXPR_IDENT:
        f_name(f, n);
        break;
    case WGSL_NODE_EXPR_LITERAL_BOOL:
        f_puts(f, wgsl_lit_bool(n) ? "true" : "false");
        break;
    case WGSL_NODE_EXPR_LITERAL_INT:
    case WGSL_NODE_EXPR_LITERAL_FLOAT:
        f_span(f, n->span_offset, n->span_length);
        break;
    case WGSL_NODE_EXPR_PAREN:
        f_putc(f, '(');
        if (n->child_count > 0) emit_expr(f, n->children[0], depth + 1);
        f_putc(f, ')');
        break;
    case WGSL_NODE_EXPR_UNARY: {
        WGSLTokenKind op = (WGSLTokenKind)wgsl_node_op_kind_u32(n);
        const char *ops = wgsl_token_kind_name(op);
        if (ops && ops[0] != '<') f_puts(f, ops);
        if (n->child_count > 0) emit_expr(f, n->children[0], depth + 1);
        break;
    }
    case WGSL_NODE_EXPR_BINARY: {
        WGSLTokenKind op = (WGSLTokenKind)wgsl_node_op_kind_u32(n);
        const char *ops = wgsl_token_kind_name(op);
        if (n->child_count > 0) emit_expr(f, n->children[0], depth + 1);
        f_putc(f, ' ');
        if (ops && ops[0] != '<') f_puts(f, ops);
        f_putc(f, ' ');
        if (n->child_count > 1) emit_expr(f, n->children[1], depth + 1);
        break;
    }
    case WGSL_NODE_EXPR_CALL: {
        if (n->child_count > 0) emit_expr(f, n->children[0], depth + 1);
        f_putc(f, '(');
        for (uint32_t i = 1; i < n->child_count; i++) {
            if (i > 1) f_puts(f, ", ");
            emit_expr(f, n->children[i], depth + 1);
        }
        f_putc(f, ')');
        break;
    }
    case WGSL_NODE_EXPR_INDEX:
        if (n->child_count > 0) emit_expr(f, n->children[0], depth + 1);
        f_putc(f, '[');
        if (n->child_count > 1) emit_expr(f, n->children[1], depth + 1);
        f_putc(f, ']');
        break;
    case WGSL_NODE_EXPR_MEMBER:
        if (n->child_count > 0) emit_expr(f, n->children[0], depth + 1);
        f_putc(f, '.');
        if (n->child_count > 1) {
            /* member name is often IDENT child */
            emit_expr(f, n->children[1], depth + 1);
        } else {
            uint32_t o = wgsl_node_name_span(n).offset;
            uint32_t l = wgsl_node_name_span(n).length;
            if (l) f_span(f, o, l);
        }
        break;
    case WGSL_NODE_EXPR_ADDR_OF:
        f_putc(f, '&');
        if (n->child_count > 0) emit_expr(f, n->children[0], depth + 1);
        break;
    case WGSL_NODE_EXPR_INDIRECTION:
        f_putc(f, '*');
        if (n->child_count > 0) emit_expr(f, n->children[0], depth + 1);
        break;
    case WGSL_NODE_EXPR_TEMPLATED_IDENT: {
        /* children[0]=base ident, rest = type args (or TEMPLATE_ARG_LIST) */
        if (n->child_count > 0) emit_expr(f, n->children[0], depth + 1);
        f_putc(f, '<');
        int first = 1;
        for (uint32_t i = 1; i < n->child_count; i++) {
            const WGSLNode *c = n->children[i];
            if (!c) continue;
            if (c->kind == WGSL_NODE_TEMPLATE_ARG_LIST) {
                for (uint32_t k = 0; k < c->child_count; k++) {
                    if (!first) f_puts(f, ", ");
                    first = 0;
                    emit_expr(f, c->children[k], depth + 1);
                }
            } else {
                if (!first) f_puts(f, ", ");
                first = 0;
                emit_expr(f, c, depth + 1);
            }
        }
        f_putc(f, '>');
        break;
    }
    default:
        /* fallback: original spelling */
        f_span(f, n->span_offset, n->span_length);
        break;
    }
}

static void emit_var_like(F *f, const WGSLNode *n, const char *kw, int depth) {
    /* let:    payload[1]=has_type; children [type?][init]
     * const:  payload[1]=has_type; children [type?][init]
     * var:    payload[1]=attr_count|tpl<<32; payload[2]=has_type|has_init<<32
     *         children [attrs…][tpl…][type?][init?]
     * override: payload[1]=attr_count; payload[2]=has_type|has_init<<32
     */
    f_indent(f);
    uint32_t attrs = 0, tpls = 0;
    int has_type = 0, has_init = 0;

    if (n->kind == WGSL_NODE_DECL_VAR) {
        attrs = wgsl_varlike_attr_count(n);
        tpls = wgsl_varlike_tpl_count(n);
        has_type = wgsl_varlike_has_type(n);
        has_init = wgsl_varlike_has_init(n);
    } else if (n->kind == WGSL_NODE_DECL_OVERRIDE) {
        attrs = wgsl_varlike_attr_count(n);
        has_type = wgsl_varlike_has_type(n);
        has_init = wgsl_varlike_has_init(n);
    } else if (n->kind == WGSL_NODE_DECL_LET || n->kind == WGSL_NODE_DECL_CONST) {
        has_type = wgsl_letlike_has_type(n);
        has_init = 1;
    }

    if (attrs)
        emit_attrs(f, (const WGSLNode *const *)n->children, 0, attrs, depth);

    f_puts(f, kw);
    uint32_t ci = attrs;
    if (tpls > 0) {
        f_putc(f, '<');
        for (uint32_t t = 0; t < tpls && ci < n->child_count; t++, ci++) {
            if (t) f_puts(f, ", ");
            emit_expr(f, n->children[ci], depth + 1);
        }
        f_putc(f, '>');
    }
    f_putc(f, ' ');
    uint32_t no = wgsl_node_name_span(n).offset;
    uint32_t nl = wgsl_node_name_span(n).length;
    f_span(f, no, nl);

    if (has_type && ci < n->child_count) {
        f_puts(f, ": ");
        emit_type(f, n->children[ci++], depth + 1);
    }
    if (has_init && ci < n->child_count) {
        f_puts(f, " = ");
        emit_expr(f, n->children[ci], depth + 1);
    }
    f_end_semi(f, n);
}

static void emit_compound(F *f, const WGSLNode *n, int depth) {
    f_indent(f);
    f_puts(f, "{\n");
    f->indent++;
    for (uint32_t i = 0; i < n->child_count; i++) {
        const WGSLNode *c = n->children[i];
        if (!c) continue;
        if (c->kind == WGSL_NODE_ATTRIBUTE) continue; /* stmt attrs rare */
        f_comments_before(f, c->span_offset);
        emit_stmt(f, c, depth + 1);
    }
    /* Comments before the closing brace. */
    if (n->span_length > 0)
        f_comments_before(f, n->span_offset + n->span_length - 1);
    f->indent--;
    f_indent(f);
    f_puts(f, "}");
}

static void emit_stmt(F *f, const WGSLNode *n, int depth) {
    if (!n) return;
    if (depth > WGSL_MAX_AST_DEPTH) return;
    switch (n->kind) {
    case WGSL_NODE_STMT_COMPOUND:
        emit_compound(f, n, depth);
        f_nl(f);
        break;
    case WGSL_NODE_DECL_LET:
        emit_var_like(f, n, "let", depth);
        break;
    case WGSL_NODE_DECL_CONST:
        emit_var_like(f, n, "const", depth);
        break;
    case WGSL_NODE_DECL_VAR:
        emit_var_like(f, n, "var", depth);
        break;
    case WGSL_NODE_DECL_OVERRIDE:
        emit_var_like(f, n, "override", depth);
        break;
    case WGSL_NODE_STMT_RETURN:
        f_indent(f);
        f_puts(f, "return");
        if (n->child_count > 0 && n->children[0]) {
            f_putc(f, ' ');
            emit_expr(f, n->children[0], depth + 1);
        }
        f_end_semi(f, n);
        break;
    case WGSL_NODE_STMT_DISCARD:
        f_indent(f); f_puts(f, "discard"); f_end_semi(f, n);
        break;
    case WGSL_NODE_STMT_BREAK:
        f_indent(f); f_puts(f, "break"); f_end_semi(f, n);
        break;
    case WGSL_NODE_STMT_CONTINUE:
        f_indent(f); f_puts(f, "continue"); f_end_semi(f, n);
        break;
    case WGSL_NODE_STMT_ASSIGN:
        f_indent(f);
        if (n->child_count > 0) emit_expr(f, n->children[0], depth + 1);
        f_puts(f, " = ");
        if (n->child_count > 1) emit_expr(f, n->children[1], depth + 1);
        f_end_semi(f, n);
        break;
    case WGSL_NODE_STMT_COMPOUND_ASSIGN: {
        f_indent(f);
        WGSLTokenKind op = (WGSLTokenKind)wgsl_node_op_kind_u32(n);
        const char *ops = wgsl_token_kind_name(op);
        if (n->child_count > 0) emit_expr(f, n->children[0], depth + 1);
        f_putc(f, ' ');
        if (ops) f_puts(f, ops);
        f_putc(f, ' ');
        if (n->child_count > 1) emit_expr(f, n->children[1], depth + 1);
        f_end_semi(f, n);
        break;
    }
    case WGSL_NODE_STMT_PHONY_ASSIGN:
        f_indent(f);
        f_puts(f, "_ = ");
        if (n->child_count > 0) emit_expr(f, n->children[0], depth + 1);
        f_end_semi(f, n);
        break;
    case WGSL_NODE_STMT_INCREMENT:
        f_indent(f);
        if (n->child_count > 0) emit_expr(f, n->children[0], depth + 1);
        f_puts(f, "++");
        f_end_semi(f, n);
        break;
    case WGSL_NODE_STMT_DECREMENT:
        f_indent(f);
        if (n->child_count > 0) emit_expr(f, n->children[0], depth + 1);
        f_puts(f, "--");
        f_end_semi(f, n);
        break;
    case WGSL_NODE_STMT_IF: {
        f_indent(f);
        f_puts(f, "if (");
        if (n->child_count > 0) emit_expr(f, n->children[0], depth + 1);
        f_puts(f, ") ");
        /* then body */
        int i = 1;
        if (i < (int)n->child_count && n->children[i] &&
            n->children[i]->kind == WGSL_NODE_STMT_COMPOUND) {
            emit_compound(f, n->children[i], depth + 1);
            i++;
        }
        while (i < (int)n->child_count) {
            const WGSLNode *c = n->children[i++];
            if (!c) continue;
            if (c->kind == WGSL_NODE_STMT_ELSE_IF) {
                f_puts(f, " else if (");
                if (c->child_count > 0) emit_expr(f, c->children[0], depth + 1);
                f_puts(f, ") ");
                if (c->child_count > 1) emit_compound(f, c->children[1], depth + 1);
            } else if (c->kind == WGSL_NODE_STMT_ELSE) {
                f_puts(f, " else ");
                if (c->child_count > 0) emit_compound(f, c->children[0], depth + 1);
            } else if (c->kind == WGSL_NODE_STMT_COMPOUND) {
                f_puts(f, " else ");
                emit_compound(f, c, depth + 1);
            }
        }
        f_nl(f);
        break;
    }
    case WGSL_NODE_STMT_WHILE:
        f_indent(f);
        f_puts(f, "while (");
        if (n->child_count > 0) emit_expr(f, n->children[0], depth + 1);
        f_puts(f, ") ");
        if (n->child_count > 1) emit_compound(f, n->children[1], depth + 1);
        f_nl(f);
        break;
    case WGSL_NODE_STMT_LOOP:
        f_indent(f);
        f_puts(f, "loop ");
        /* body is compound; may have continuing child */
        for (uint32_t i = 0; i < n->child_count; i++) {
            if (n->children[i] && n->children[i]->kind == WGSL_NODE_STMT_COMPOUND)
                emit_compound(f, n->children[i], depth + 1);
            else if (n->children[i] && n->children[i]->kind == WGSL_NODE_STMT_CONTINUING) {
                f_puts(f, " continuing ");
                if (n->children[i]->child_count > 0)
                    emit_compound(f, n->children[i]->children[0], depth + 1);
            }
        }
        f_nl(f);
        break;
    case WGSL_NODE_STMT_FOR: {
        /* Best-effort: reprint original span for complex for headers */
        f_indent(f);
        f_span(f, n->span_offset, n->span_length);
        f_nl(f);
        break;
    }
    case WGSL_NODE_STMT_SWITCH:
        f_indent(f);
        f_span(f, n->span_offset, n->span_length);
        f_nl(f);
        break;
    case WGSL_NODE_STMT_FN_CALL:
        f_indent(f);
        if (n->child_count > 0) emit_expr(f, n->children[0], depth + 1);
        f_end_semi(f, n);
        break;
    case WGSL_NODE_DECL_CONST_ASSERT:
        f_indent(f);
        f_puts(f, "const_assert ");
        if (n->child_count > 0) emit_expr(f, n->children[0], depth + 1);
        f_end_semi(f, n);
        break;
    default:
        f_indent(f);
        f_span(f, n->span_offset, n->span_length);
        f_nl(f);
        break;
    }
}

static void emit_fn(F *f, const WGSLNode *n, int depth) {
    uint32_t FA = wgsl_fn_attr_count(n);
    uint32_t P  = wgsl_fn_param_count(n);
    uint32_t RA = wgsl_fn_ret_attr_count(n);
    uint32_t HR = wgsl_fn_has_return_type(n);

    f_comments_before(f, n->span_offset);
    if (FA) emit_attrs(f, (const WGSLNode *const *)n->children, 0, FA, depth);

    f_indent(f);
    f_puts(f, "fn ");
    uint32_t no = wgsl_node_name_span(n).offset;
    uint32_t nl = wgsl_node_name_span(n).length;
    f_span(f, no, nl);
    f_putc(f, '(');
    for (uint32_t k = 0; k < P; k++) {
        if (k) f_puts(f, ", ");
        uint32_t ci = FA + k;
        if (ci >= n->child_count) break;
        const WGSLNode *p = n->children[ci];
        if (!p || p->kind != WGSL_NODE_DECL_PARAM) continue;
        uint32_t PA = wgsl_param_attr_count(p);
        if (PA) emit_attrs(f, (const WGSLNode *const *)p->children, 0, PA, depth);
        uint32_t pno = wgsl_node_name_span(p).offset;
        uint32_t pnl = wgsl_node_name_span(p).length;
        f_span(f, pno, pnl);
        f_puts(f, ": ");
        if (PA < p->child_count) emit_type(f, p->children[PA], depth + 1);
    }
    f_putc(f, ')');
    if (HR) {
        f_puts(f, " -> ");
        uint32_t base = FA + P;
        if (RA) emit_attrs(f, (const WGSLNode *const *)n->children, base, RA, depth);
        uint32_t rti = base + RA;
        if (rti < n->child_count) emit_type(f, n->children[rti], depth + 1);
    }
    f_putc(f, ' ');
    /* body: last compound */
    for (uint32_t i = 0; i < n->child_count; i++) {
        if (n->children[i] && n->children[i]->kind == WGSL_NODE_STMT_COMPOUND) {
            emit_compound(f, n->children[i], depth + 1);
            break;
        }
    }
    f_nl(f);
}

static void emit_struct(F *f, const WGSLNode *n, int depth) {
    f_comments_before(f, n->span_offset);
    f_indent(f);
    f_puts(f, "struct ");
    uint32_t no = wgsl_node_name_span(n).offset;
    uint32_t nl = wgsl_node_name_span(n).length;
    f_span(f, no, nl);
    f_puts(f, " {\n");
    f->indent++;
    for (uint32_t i = 0; i < n->child_count; i++) {
        const WGSLNode *m = n->children[i];
        if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
        f_comments_before(f, m->span_offset);
        uint32_t MA = wgsl_member_attr_count(m);
        f_indent(f);
        if (MA) emit_attrs(f, (const WGSLNode *const *)m->children, 0, MA, depth);
        uint32_t mno = wgsl_node_name_span(m).offset;
        uint32_t mnl = wgsl_node_name_span(m).length;
        f_span(f, mno, mnl);
        f_puts(f, ": ");
        if (MA < m->child_count) emit_type(f, m->children[MA], depth + 1);
        f_putc(f, ',');
        {
            uint32_t end = m->span_offset;
            if (m->span_length > 0 &&
                m->span_offset <= UINT32_MAX - m->span_length)
                end = m->span_offset + m->span_length;
            f_trailing_after(f, end);
        }
        f_nl(f);
    }
    if (n->span_length > 0)
        f_comments_before(f, n->span_offset + n->span_length - 1);
    f->indent--;
    f_indent(f);
    f_puts(f, "}\n");
}

static void emit_node(F *f, const WGSLNode *n, int depth) {
    if (!n) return;
    switch (n->kind) {
    case WGSL_NODE_TRANSLATION_UNIT:
        for (uint32_t i = 0; i < n->child_count; i++) {
            const WGSLNode *c = n->children[i];
            if (!c) continue;
            f_comments_before(f, c->span_offset);
            if (i && !f->at_line_start) f_nl(f);
            if (i && f->at_line_start) {
                /* Blank line between top-level units (canonical). */
                f_nl(f);
            }
            emit_node(f, c, depth + 1);
        }
        /* Trailing file comments. */
        f_comments_before(f, UINT32_MAX);
        break;
    case WGSL_NODE_DIR_ENABLE:
        f_comments_before(f, n->span_offset);
        f_indent(f); f_puts(f, "enable ");
        for (uint32_t i = 0; i < n->child_count; i++) {
            if (i) f_puts(f, ", ");
            if (n->children[i]) f_name(f, n->children[i]);
        }
        f_end_semi(f, n);
        break;
    case WGSL_NODE_DIR_REQUIRES:
        f_comments_before(f, n->span_offset);
        f_indent(f); f_puts(f, "requires ");
        for (uint32_t i = 0; i < n->child_count; i++) {
            if (i) f_puts(f, ", ");
            if (n->children[i]) f_name(f, n->children[i]);
        }
        f_end_semi(f, n);
        break;
    case WGSL_NODE_DIR_DIAGNOSTIC:
        f_comments_before(f, n->span_offset);
        f_indent(f);
        f_span(f, n->span_offset, n->span_length);
        f_trailing_after(f, n->span_offset + n->span_length);
        f_nl(f);
        break;
    case WGSL_NODE_DECL_FUNCTION:
        emit_fn(f, n, depth);
        break;
    case WGSL_NODE_DECL_STRUCT:
        emit_struct(f, n, depth);
        break;
    case WGSL_NODE_DECL_ALIAS:
        f_comments_before(f, n->span_offset);
        f_indent(f);
        f_puts(f, "alias ");
        {
            uint32_t o = wgsl_node_name_span(n).offset;
            uint32_t l = wgsl_node_name_span(n).length;
            f_span(f, o, l);
        }
        f_puts(f, " = ");
        if (n->child_count > 0) emit_type(f, n->children[n->child_count - 1], depth + 1);
        f_end_semi(f, n);
        break;
    case WGSL_NODE_DECL_CONST:
        f_comments_before(f, n->span_offset);
        emit_var_like(f, n, "const", depth);
        break;
    case WGSL_NODE_DECL_OVERRIDE:
        f_comments_before(f, n->span_offset);
        emit_var_like(f, n, "override", depth);
        break;
    case WGSL_NODE_DECL_VAR:
        f_comments_before(f, n->span_offset);
        emit_var_like(f, n, "var", depth);
        break;
    case WGSL_NODE_DECL_CONST_ASSERT:
        f_comments_before(f, n->span_offset);
        emit_stmt(f, n, depth);
        break;
    default:
        f_comments_before(f, n->span_offset);
        f_indent(f);
        f_span(f, n->span_offset, n->span_length);
        f_nl(f);
        break;
    }
}

#include "internal/arena.h"
#include "internal/diag.h"
#include "internal/lexer.h"
#include "internal/parser.h"
#include "internal/source.h"
#include "internal/utf8.h"

static int collect_comments(const WGSLLexResult *lex, FmtCmt **out, int *out_n) {
    *out = NULL;
    *out_n = 0;
    if (!lex || !lex->tokens) return 1;
    int n = 0, cap = 0;
    FmtCmt *v = NULL;
    for (size_t i = 0; i < lex->count; i++) {
        const WGSLToken *t = &lex->tokens[i];
        if (t->kind != WGSL_TOK_LINE_COMMENT &&
            t->kind != WGSL_TOK_BLOCK_COMMENT)
            continue;
        if (n + 1 > cap) {
            int nc = cap ? cap * 2 : 8;
            FmtCmt *g = (FmtCmt *)realloc(v, (size_t)nc * sizeof *g);
            if (!g) { free(v); return 0; }
            v = g; cap = nc;
        }
        v[n].off = t->span.offset;
        v[n].len = t->span.length;
        n++;
    }
    *out = v;
    *out_n = n;
    return 1;
}

static char *fmt_strndup(const char *s, size_t n) {
    char *out = (char *)malloc(n + 1);
    if (!out) return NULL;
    if (n) memcpy(out, s, n);
    out[n] = 0;
    return out;
}

static char *format_core(const WGSLNode *root, const char *src, size_t src_len,
                         const FmtCmt *cmts, int ncmt)
{
    F f;
    memset(&f, 0, sizeof f);
    f.src = src;
    f.src_len = src_len;
    f.at_line_start = 1;
    f.cmts = cmts;
    f.ncmt = ncmt;
    f.cmt_i = 0;
    if (root) emit_node(&f, root, 0);
    /* Any leftover comments (orphan). */
    f_comments_before(&f, UINT32_MAX);
    if (f.oom) { free(f.buf); return NULL; }
    if (!f.buf) {
        f.buf = (char *)malloc(1);
        if (f.buf) f.buf[0] = 0;
        return f.buf;
    }
    if (f.len == 0 || f.buf[f.len - 1] != '\n') f_nl(&f);
    return f.buf;
}

/* Shared entry: format from AST + source bytes (re-lexes for comments). */
char *wgsl_format_from_ast(const WGSLNode *root, const char *src, size_t src_len) {
    if (!src) src = "";
    wgsl_utf8_init();
    WGSLArena arena; wgsl_arena_init(&arena);
    WGSLDiagBag diag; wgsl_diag_init(&diag);
    WGSLSource source;
    FmtCmt *cmts = NULL;
    int ncmt = 0;
    char *out = NULL;
    if (wgsl_source_init(&source, src, src_len)) {
        WGSLLexResult lex; memset(&lex, 0, sizeof lex);
        (void)wgsl_tokenize(&source, &arena, &diag, &lex);
        (void)collect_comments(&lex, &cmts, &ncmt);
        wgsl_source_destroy(&source);
    }
    out = format_core(root, src, src_len, cmts, ncmt);
    free(cmts);
    wgsl_diag_destroy(&diag);
    wgsl_arena_destroy(&arena);
    return out;
}

char *wgsl_format_n(const char *src, size_t src_len) {
    wgsl_utf8_init();
    if (!src) { src = ""; src_len = 0; }
    WGSLArena arena; wgsl_arena_init(&arena);
    WGSLDiagBag diag; wgsl_diag_init(&diag);
    WGSLSource source;
    char *copy = (char *)malloc(src_len + 1);
    if (!copy) return NULL;
    if (src_len) memcpy(copy, src, src_len);
    copy[src_len] = 0;
    if (!wgsl_source_init(&source, copy, src_len)) {
        free(copy); wgsl_diag_destroy(&diag); wgsl_arena_destroy(&arena);
        return NULL;
    }
    WGSLLexResult lex; memset(&lex, 0, sizeof lex);
    WGSLAst ast; memset(&ast, 0, sizeof ast);
    (void)wgsl_tokenize(&source, &arena, &diag, &lex);
    (void)wgsl_parse(&lex, &source, &arena, &diag, &ast);
    FmtCmt *cmts = NULL;
    int ncmt = 0;
    (void)collect_comments(&lex, &cmts, &ncmt);
    char *out = NULL;
    if (ast.root)
        out = format_core(ast.root, copy, src_len, cmts, ncmt);
    else {
        out = (char *)malloc(1);
        if (out) out[0] = 0;
    }
    free(cmts);
    wgsl_source_destroy(&source);
    wgsl_diag_destroy(&diag);
    wgsl_arena_destroy(&arena);
    free(copy);
    return out;
}

char *wgsl_format(const char *src) {
    return wgsl_format_n(src, src ? strlen(src) : 0);
}

char *wgsl_format_range_n(const char *src, size_t src_len,
                          uint32_t start_offset, uint32_t end_offset,
                          uint32_t *edit_start, uint32_t *edit_end)
{
    wgsl_utf8_init();
    if (!src) { src = ""; src_len = 0; }
    if (start_offset > src_len) start_offset = (uint32_t)src_len;
    if (end_offset > src_len) end_offset = (uint32_t)src_len;
    if (end_offset < start_offset) {
        uint32_t t = start_offset;
        start_offset = end_offset;
        end_offset = t;
    }
    if (edit_start) *edit_start = start_offset;
    if (edit_end) *edit_end = end_offset;

    WGSLArena arena; wgsl_arena_init(&arena);
    WGSLDiagBag diag; wgsl_diag_init(&diag);
    WGSLSource source;
    char *copy = (char *)malloc(src_len + 1);
    if (!copy) {
        wgsl_diag_destroy(&diag);
        wgsl_arena_destroy(&arena);
        return NULL;
    }
    if (src_len) memcpy(copy, src, src_len);
    copy[src_len] = 0;
    if (!wgsl_source_init(&source, copy, src_len)) {
        char *fallback = fmt_strndup(src + start_offset, end_offset - start_offset);
        free(copy);
        wgsl_diag_destroy(&diag);
        wgsl_arena_destroy(&arena);
        return fallback;
    }

    WGSLLexResult lex; memset(&lex, 0, sizeof lex);
    WGSLAst ast; memset(&ast, 0, sizeof ast);
    (void)wgsl_tokenize(&source, &arena, &diag, &lex);
    (void)wgsl_parse(&lex, &source, &arena, &diag, &ast);

    uint32_t sel_start = start_offset;
    uint32_t sel_end = end_offset;
    if (sel_end == sel_start && sel_end < src_len) sel_end++;
    int found = 0;
    uint32_t out_start = start_offset;
    uint32_t out_end = end_offset;
    if (ast.root && ast.root->kind == WGSL_NODE_TRANSLATION_UNIT) {
        for (uint32_t i = 0; i < ast.root->child_count; i++) {
            const WGSLNode *c = ast.root->children[i];
            if (!c || c->span_length == 0) continue;
            uint32_t cs = c->span_offset;
            uint32_t ce = c->span_offset;
            if (c->span_offset <= UINT32_MAX - c->span_length)
                ce = c->span_offset + c->span_length;
            if (ce < cs) ce = cs;
            int touches = (sel_start < ce && sel_end > cs);
            if (!touches && start_offset == end_offset)
                touches = (start_offset >= cs && start_offset <= ce);
            if (!touches) continue;
            if (!found) {
                out_start = cs;
                out_end = ce;
            } else {
                if (cs < out_start) out_start = cs;
                if (ce > out_end) out_end = ce;
            }
            found = 1;
        }
    }

    char *out = NULL;
    if (found && out_end >= out_start && out_end <= src_len) {
        if (edit_start) *edit_start = out_start;
        if (edit_end) *edit_end = out_end;
        out = wgsl_format_n(copy + out_start, out_end - out_start);
    } else {
        out = fmt_strndup(copy + start_offset, end_offset - start_offset);
    }

    wgsl_source_destroy(&source);
    wgsl_diag_destroy(&diag);
    wgsl_arena_destroy(&arena);
    free(copy);
    return out;
}

char *wgsl_format_range(const char *src,
                        uint32_t start_offset, uint32_t end_offset,
                        uint32_t *edit_start, uint32_t *edit_end)
{
    return wgsl_format_range_n(src, src ? strlen(src) : 0,
                               start_offset, end_offset,
                               edit_start, edit_end);
}
