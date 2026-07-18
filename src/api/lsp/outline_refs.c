/**
 * @file outline_refs.c — document symbols, references/rename, folding
 */
#include "wgsl.h"
#include "internal/result.h"
#include "internal/api_priv.h"
#include "internal/lsp_priv.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal/token.h"
#include "internal/ast.h"
#include "internal/resolver.h"
#include "internal/types.h"
#include "internal/check.h"
#include "internal/consteval.h"


/* document symbols. */

int wgsl_document_symbols(const WGSLResult *r,
                          WGSLOutlineSymbol **out, int *out_count)
{
    if (out) *out = NULL;
    if (out_count) *out_count = 0;
    if (!r || !out || !out_count) return 0;
    if (!r->ast.root) return 1;

    OutlineBuf b = {0};
    const WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        const WGSLNode *n = tu->children[i];
        if (!n) continue;
        WGSLOutlineKind ok = 0;
        switch (n->kind) {
        case WGSL_NODE_DECL_FUNCTION:   ok = WGSL_OUTLINE_FUNCTION; break;
        case WGSL_NODE_DECL_STRUCT:     ok = WGSL_OUTLINE_STRUCT; break;
        case WGSL_NODE_DECL_CONST:      ok = WGSL_OUTLINE_CONSTANT; break;
        case WGSL_NODE_DECL_OVERRIDE:   ok = WGSL_OUTLINE_OVERRIDE; break;
        case WGSL_NODE_DECL_VAR:        ok = WGSL_OUTLINE_VARIABLE; break;
        case WGSL_NODE_DECL_ALIAS:      ok = WGSL_OUTLINE_TYPE_ALIAS; break;
        default: continue;
        }
        uint32_t no = wgsl_node_name_span(n).offset;
        uint32_t nl = wgsl_node_name_span(n).length;
        if (nl == 0) continue;
        int parent = b.n;
        WGSLOutlineSymbol s = {
            .kind = ok,
            .name_offset = no, .name_length = nl,
            .range_offset = n->span_offset, .range_length = n->span_length,
            .parent = -1,
        };
        if (!ob_push(&b, s)) { free(b.v); return 0; }

        if (n->kind == WGSL_NODE_DECL_STRUCT) {
            for (uint32_t k = 0; k < n->child_count; k++) {
                const WGSLNode *m = n->children[k];
                if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
                uint32_t mno = wgsl_node_name_span(m).offset;
                uint32_t mnl = wgsl_node_name_span(m).length;
                if (mnl == 0) continue;
                WGSLOutlineSymbol fs = {
                    .kind = WGSL_OUTLINE_FIELD,
                    .name_offset = mno, .name_length = mnl,
                    .range_offset = m->span_offset, .range_length = m->span_length,
                    .parent = parent,
                };
                if (!ob_push(&b, fs)) { free(b.v); return 0; }
            }
        } else if (n->kind == WGSL_NODE_DECL_FUNCTION) {
            uint32_t FA = wgsl_fn_attr_count(n);
            uint32_t P  = wgsl_fn_param_count(n);
            for (uint32_t k = 0; k < P; k++) {
                uint32_t ci = FA + k;
                if (ci >= n->child_count) break;
                const WGSLNode *p = n->children[ci];
                if (!p || p->kind != WGSL_NODE_DECL_PARAM) continue;
                uint32_t pno = wgsl_node_name_span(p).offset;
                uint32_t pnl = wgsl_node_name_span(p).length;
                if (pnl == 0) continue;
                WGSLOutlineSymbol ps = {
                    .kind = WGSL_OUTLINE_PARAMETER,
                    .name_offset = pno, .name_length = pnl,
                    .range_offset = p->span_offset, .range_length = p->span_length,
                    .parent = parent,
                };
                if (!ob_push(&b, ps)) { free(b.v); return 0; }
            }
        }
    }
    *out = b.v;
    *out_count = b.n;
    return 1;
}

void wgsl_outline_free(WGSLOutlineSymbol *syms) { free(syms); }

/* references / rename. */

static void collect_refs_walk(const WGSLNode *n, WGSLSymbol *target,
                              RefBuf *b, int depth)
{
    if (!n || depth > WGSL_MAX_AST_DEPTH) return;
    if (n->kind == WGSL_NODE_EXPR_IDENT) {
        WGSLSymbol *s = wgsl_node_resolved_symbol(n);
        if (s == target && n->span_length > 0)
            rb_push(b, n->span_offset, n->span_length, 0);
    }
    for (uint32_t i = 0; i < n->child_count; i++)
        collect_refs_walk(n->children[i], target, b, depth + 1);
}

static int ref_cmp(const void *a, const void *b) {
    const WGSLReference *x = (const WGSLReference *)a;
    const WGSLReference *y = (const WGSLReference *)b;
    if (x->offset != y->offset) return x->offset < y->offset ? -1 : 1;
    if (x->length != y->length) return x->length < y->length ? -1 : 1;
    return 0;
}

int wgsl_references_at(const WGSLResult *r, uint32_t offset,
                       WGSLReference **out, int *out_count)
{
    if (out) *out = NULL;
    if (out_count) *out_count = 0;
    if (!r || !out || !out_count) return 0;

    WGSLSymbol *sym = lsp_symbol_at(r, offset);
    if (!sym || !sym->ast) return 1;   /* predeclared / none → empty */

    RefBuf b = {0};
    uint32_t no, nl;
    decl_name_span(sym, &no, &nl);
    if (nl > 0) rb_push(&b, no, nl, 1);
    if (r->ast.root) collect_refs_walk(r->ast.root, sym, &b, 0);

    /* Dedup (decl name may also appear as a resolved use). Prefer is_definition. */
    if (b.n > 1) qsort(b.v, (size_t)b.n, sizeof *b.v, ref_cmp);
    int w = 0;
    for (int i = 0; i < b.n; i++) {
        if (w > 0 && b.v[w - 1].offset == b.v[i].offset &&
            b.v[w - 1].length == b.v[i].length)
        {
            if (b.v[i].is_definition) b.v[w - 1].is_definition = 1;
            continue;
        }
        b.v[w++] = b.v[i];
    }
    b.n = w;

    *out = b.v;
    *out_count = b.n;
    return 1;
}

void wgsl_references_free(WGSLReference *refs) { free(refs); }

int wgsl_prepare_rename(const WGSLResult *r, uint32_t offset,
                        uint32_t *name_offset, uint32_t *name_length)
{
    if (name_offset) *name_offset = 0;
    if (name_length) *name_length = 0;
    WGSLSymbol *sym = lsp_symbol_at(r, offset);
    if (!sym || !sym->ast) return 0;
    uint32_t no, nl;
    decl_name_span(sym, &no, &nl);
    if (nl == 0) return 0;
    if (name_offset) *name_offset = no;
    if (name_length) *name_length = nl;
    return 1;
}

/* folding ranges. */

static void fold_walk(const WGSLNode *n, FoldBuf *b, int depth) {
    if (!n || depth > WGSL_MAX_AST_DEPTH) return;
    int fold = 0;
    switch (n->kind) {
    case WGSL_NODE_DECL_FUNCTION:
    case WGSL_NODE_DECL_STRUCT:
    case WGSL_NODE_STMT_COMPOUND:
    case WGSL_NODE_STMT_IF:
    case WGSL_NODE_STMT_SWITCH:
    case WGSL_NODE_STMT_LOOP:
    case WGSL_NODE_STMT_FOR:
    case WGSL_NODE_STMT_WHILE:
    case WGSL_NODE_STMT_CONTINUING:
        fold = 1;
        break;
    default: break;
    }
    if (fold && n->span_length > 0)
        fb_push(b, n->span_offset, n->span_offset + n->span_length);
    for (uint32_t i = 0; i < n->child_count; i++)
        fold_walk(n->children[i], b, depth + 1);
}

int wgsl_folding_ranges(const WGSLResult *r,
                        WGSLFoldingRange **out, int *out_count)
{
    if (out) *out = NULL;
    if (out_count) *out_count = 0;
    if (!r || !out || !out_count) return 0;
    FoldBuf b = {0};
    if (r->ast.root) fold_walk(r->ast.root, &b, 0);
    *out = b.v;
    *out_count = b.n;
    return 1;
}

void wgsl_folding_free(WGSLFoldingRange *ranges) { free(ranges); }

