/**
 * @file access_static.c — static access tables + address-space stage gates
 */
#include "internal/validate_priv.h"
#include "internal/validate.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal/resolver.h"
#include "internal/types.h"
#include "internal/ast.h"

/* §14.5 — every var-identifier reference is classified as READ or WRITE
 * based on its parent context:
 *   - LHS of `=` / compound assign / `++` / `--`  → WRITE (compound also reads)
 *   - `&var` (address-of)                          → over-approximated as WRITE
 *     (the resulting pointer might be dereferenced for writing — without
 *     inter-procedural pointer-use tracking we can't tell, so be safe)
 *   - `atomic*(&var, …)` builtins (except atomicLoad) → WRITE on pointee
 *   - everything else (RHS, indices, conditions,
 *     non-atomic call arguments, …)                → READ
 *
 * The result feeds two parallel per-decl tables, each holding a
 * stage-bitset: `read_table[i]` and `write_table[i]`. */

typedef enum { ACC_READ, ACC_WRITE } AccessKind;

typedef struct {
    WGSLValidator *v;
    uint32_t       stage;
    uint32_t      *read_table;
    uint32_t      *write_table;
} ScanCtx;

static void scan_expr(ScanCtx *cx, WGSLNode *n, AccessKind k);
static void scan_stmt(ScanCtx *cx, WGSLNode *n);

static void record_ident(ScanCtx *cx, WGSLNode *n, AccessKind k) {
    WGSLSymbol *sym = wgsl_node_resolved_symbol(n);
    if (!sym) return;
    if (sym->kind != WGSL_SYM_VAR && sym->kind != WGSL_SYM_OVERRIDE) return;
    int idx = wgsl_val_sym_index(cx->v, sym);
    if (idx < 0) return;
    if (k == ACC_WRITE && cx->write_table) cx->write_table[idx] |= cx->stage;
    else if (cx->read_table)               cx->read_table[idx]  |= cx->stage;
}

/* True iff the call writes through its first pointer arg.  All atomic
 * builtins except `atomicLoad` mutate the pointee. */
static int atomic_call_writes(const char *nm, uint32_t len) {
    if (len < 6 || memcmp(nm, "atomic", 6) != 0) return 0;
    if (len == 10 && memcmp(nm, "atomicLoad", 10) == 0) return 0;
    return 1;
}

static WGSLAccessMode storage_texture_access(const WGSLTypeInfo *t) {
    if (!t || t->kind != WGSL_TYPE_TEXTURE) return WGSL_ACCESS_NONE;
    WGSLTextureDim dim = (WGSLTextureDim)t->width;
    if (dim < WGSL_TEX_DIM_STORAGE_1D || dim > WGSL_TEX_DIM_STORAGE_3D) {
        return WGSL_ACCESS_NONE;
    }
    return (WGSLAccessMode)t->rows;
}

static void scan_expr(ScanCtx *cx, WGSLNode *n, AccessKind k) {
    if (!n) return;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_IDENT:
        record_ident(cx, n, k);
        return;
    case WGSL_NODE_EXPR_MEMBER:
        if (n->child_count >= 1) scan_expr(cx, n->children[0], k);
        return;
    case WGSL_NODE_EXPR_INDEX:
        if (n->child_count >= 1) scan_expr(cx, n->children[0], k);
        if (n->child_count >= 2) scan_expr(cx, n->children[1], ACC_READ);
        return;
    case WGSL_NODE_EXPR_PAREN:
    case WGSL_NODE_EXPR_UNARY:
        if (n->child_count >= 1) scan_expr(cx, n->children[0], ACC_READ);
        return;
    case WGSL_NODE_EXPR_ADDR_OF:
        /* &x escapes a pointer to x — over-approximate as WRITE. */
        if (n->child_count >= 1) scan_expr(cx, n->children[0], ACC_WRITE);
        return;
    case WGSL_NODE_EXPR_INDIRECTION:
        /* `*p` reads through p; `k` describes use of the dereferenced
         * value but `p` itself is just read here. */
        if (n->child_count >= 1) scan_expr(cx, n->children[0], ACC_READ);
        return;
    case WGSL_NODE_EXPR_CALL: {
        if (n->child_count == 0) return;
        WGSLNode *fn = n->children[0];
        const char *nm = NULL; uint32_t len = 0;
        if (fn && fn->kind == WGSL_NODE_EXPR_IDENT) {
            uint32_t off = wgsl_node_name_span(fn).offset;
            len = wgsl_node_name_span(fn).length;
            nm = cx->v->src->bytes + off;
        }
        int is_atomic = nm && len >= 6 && memcmp(nm, "atomic", 6) == 0;
        int atomic_write = is_atomic && atomic_call_writes(nm, len);
        for (uint32_t i = 1; i < n->child_count; i++) {
            WGSLNode *arg = n->children[i];
            if (!arg) continue;
            /* For atomic builtins, the first arg is `ptr<…, atomic<T>, …>`
             * (typically `&var`).  Skip the conservative ADDR_OF→WRITE
             * default so atomicLoad's pointee correctly classifies as
             * READ; non-load atomics route the pointee through WRITE. */
            if (i == 1 && is_atomic &&
                arg->kind == WGSL_NODE_EXPR_ADDR_OF &&
                arg->child_count >= 1)
            {
                scan_expr(cx, arg->children[0],
                          atomic_write ? ACC_WRITE : ACC_READ);
                continue;
            }
            scan_expr(cx, arg, ACC_READ);
        }
        return;
    }
    default:
        for (uint32_t i = 0; i < n->child_count; i++) {
            scan_expr(cx, n->children[i], ACC_READ);
        }
        return;
    }
}

static void scan_stmt(ScanCtx *cx, WGSLNode *n) {
    if (!n) return;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_STMT_ASSIGN:
        if (n->child_count >= 1) scan_expr(cx, n->children[0], ACC_WRITE);
        if (n->child_count >= 2) scan_expr(cx, n->children[1], ACC_READ);
        return;
    case WGSL_NODE_STMT_COMPOUND_ASSIGN:
        if (n->child_count >= 1) {
            scan_expr(cx, n->children[0], ACC_READ);
            scan_expr(cx, n->children[0], ACC_WRITE);
        }
        if (n->child_count >= 2) scan_expr(cx, n->children[1], ACC_READ);
        return;
    case WGSL_NODE_STMT_INCREMENT:
    case WGSL_NODE_STMT_DECREMENT:
        if (n->child_count >= 1) {
            scan_expr(cx, n->children[0], ACC_READ);
            scan_expr(cx, n->children[0], ACC_WRITE);
        }
        return;
    case WGSL_NODE_STMT_PHONY_ASSIGN:
        if (n->child_count >= 1) scan_expr(cx, n->children[0], ACC_READ);
        return;
    case WGSL_NODE_DECL_LET:
    case WGSL_NODE_DECL_CONST:
    case WGSL_NODE_DECL_VAR:
        for (uint32_t i = 0; i < n->child_count; i++) {
            scan_expr(cx, n->children[i], ACC_READ);
        }
        return;
    case WGSL_NODE_STMT_FN_CALL:
        if (n->child_count >= 1) scan_expr(cx, n->children[0], ACC_READ);
        return;
    default:
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if (c->kind >= WGSL_NODE_EXPR_LITERAL_BOOL &&
                c->kind <= WGSL_NODE_EXPR_INDIRECTION)
            {
                scan_expr(cx, c, ACC_READ);
            } else {
                scan_stmt(cx, c);
            }
        }
        return;
    }
}

typedef struct {
    WGSLValidator *v;
    uint32_t       stage;
    uint8_t       *visited;
    uint32_t      *read_table;
    uint32_t      *write_table;
} AccessCtx;

static void access_edge_cb(WGSLValidator *v, const WGSLSymbol *to, const WGSLNode *at, void *ctx) {
    AccessCtx *cx = (void *)ctx;
    int idx = wgsl_val_sym_index(v, to);
    if (idx < 0 || cx->visited[idx]) return;
    cx->visited[idx] = 1;
    if (to->ast && to->ast->child_count > 0) {
        WGSLNode *body = to->ast->children[to->ast->child_count - 1];
        ScanCtx sc = { cx->v, cx->stage, cx->read_table, cx->write_table };
        scan_stmt(&sc, body);
    }
    wgsl_val_enum_call_edges(v, to, access_edge_cb, cx);
}

AccessTables analyze_static_access(WGSLValidator *v) {
    AccessTables out = {0};
    size_t n = v->res->all_decl_count;
    uint8_t  *visited     = (uint8_t  *)calloc(n, sizeof *visited);
    uint32_t *read_table  = (uint32_t *)calloc(n, sizeof *read_table);
    uint32_t *write_table = (uint32_t *)calloc(n, sizeof *write_table);
    if (!visited || !read_table || !write_table) {
        free(visited); free(read_table); free(write_table);
        return out;
    }
    for (size_t i = 0; i < n; i++) {
        WGSLSymbol *s = v->res->all_decls[i];
        if (s->kind != WGSL_SYM_FUNCTION) continue;
        uint32_t stage = 0;
        if (s->flags & WGSL_SYM_FLAG_VERTEX)   stage = STAGE_VX;
        if (s->flags & WGSL_SYM_FLAG_FRAGMENT) stage = STAGE_FR;
        if (s->flags & WGSL_SYM_FLAG_COMPUTE)  stage = STAGE_CO;
        if (stage == 0) continue;
        memset(visited, 0, n);
        AccessCtx cx = { v, stage, visited, read_table, write_table };
        if (s->ast && s->ast->child_count > 0) {
            WGSLNode *body = s->ast->children[s->ast->child_count - 1];
            ScanCtx sc = { v, stage, read_table, write_table };
            scan_stmt(&sc, body);
        }
        wgsl_val_enum_call_edges(v, s, access_edge_cb, &cx);
    }
    free(visited);
    out.read_table  = read_table;
    out.write_table = write_table;
    return out;
}

void validate_address_spaces(WGSLValidator *v, WGSLNode *tu,
                                    const uint32_t *read_table,
                                    const uint32_t *write_table)
{
    if (!tu) return;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n || n->kind != WGSL_NODE_DECL_VAR) continue;
        WGSLSymbol *sym = wgsl_val_find_symbol_for_ast(v, n);
        if (!sym) continue;
        int idx = wgsl_val_sym_index(v, sym);
        uint32_t reads  = (idx >= 0 && read_table)  ? read_table[idx]  : 0;
        uint32_t writes = (idx >= 0 && write_table) ? write_table[idx] : 0;
        uint32_t stages = reads | writes;
        if (sym->as == WGSL_AS_WORKGROUP && (stages & (STAGE_VX | STAGE_FR))) {
            wgsl_val_error(v, n, "workgroup variables can only be accessed by compute stage");
        }
        /* §14.3 — vertex-stage code may only access read-only storage.
         * This is keyed by the declared access mode, not by whether a
         * particular expression is a read or write operation. */
        if (sym->as == WGSL_AS_STORAGE &&
            (sym->am == WGSL_ACCESS_WRITE ||
             sym->am == WGSL_ACCESS_READ_WRITE) &&
            (stages & STAGE_VX))
        {
            wgsl_val_error(v, n,
                "vertex stage cannot access read_write storage variables "
                "(only read access permitted in @vertex)");
        }
        WGSLAccessMode tex_am = storage_texture_access(sym->type);
        if ((tex_am == WGSL_ACCESS_WRITE ||
             tex_am == WGSL_ACCESS_READ_WRITE) &&
            (stages & STAGE_VX))
        {
            wgsl_val_error(v, n,
                "vertex stage cannot access write or read_write "
                "storage textures");
        }
    }
}


