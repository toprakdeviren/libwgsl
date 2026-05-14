/**
 * @file access.c — §14.3 + §14.5 + §9.4.11.
 *
 * Static-access analysis: walks every entry point's call-graph and
 * fn body to build per-decl × stage bitsets (`read_table[]` and
 * `write_table[]`) classifying every var-identifier reference as a
 * READ or WRITE.  Drives:
 *   - §14.3 workgroup-vars-only-in-compute,
 *           storage write/read_write not-in-vertex address-space gates;
 *   - §9.4.11 discard-only-reachable-from-fragment.
 *
 * Co-extensive with WGSL §14.3 / §14.5 / §9.4.11.
 */
#include "internal/validate.h"
#include "internal/validate_priv.h"

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
            uint32_t off = (uint32_t)(fn->payload[0] & 0xFFFFFFFFu);
            len = (uint32_t)(fn->payload[0] >> 32);
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

/* §11.4.1 — pointer alias analysis.
 *
 * This is intentionally separate from the §14.5 static access scanner
 * above.  That scanner conservatively treats `&x` as a possible write
 * for shader-stage address-space rules.  Alias analysis needs the more
 * precise §11.4.1 model: compute root identifiers, summarize direct and
 * transitive module-var / pointer-param reads and writes per function,
 * then reject call sites whose pointer arguments alias a written view.
 *
 * The summary is conservative at root granularity.  It does not attempt
 * field/component disjointness; that keeps the implementation aligned
 * with the spec's root-identifier wording and avoids unsound partial
 * overlap reasoning. */

typedef struct {
    uint8_t *module_read;
    uint8_t *module_write;
    uint8_t *param_read;
    uint8_t *param_write;
} AliasSummary;

typedef struct {
    WGSLValidator *v;
    AliasSummary  *summary;
    AliasSummary  *summaries;
    size_t         nsyms;
    int            emit_errors;
} AliasCtx;

static void alias_scan_expr(AliasCtx *cx, WGSLNode *n, AccessKind k);
static void alias_scan_stmt(AliasCtx *cx, WGSLNode *n);

static int alias_summary_init(AliasSummary *s, size_t n) {
    s->module_read  = (uint8_t *)calloc(n, sizeof *s->module_read);
    s->module_write = (uint8_t *)calloc(n, sizeof *s->module_write);
    s->param_read   = (uint8_t *)calloc(n, sizeof *s->param_read);
    s->param_write  = (uint8_t *)calloc(n, sizeof *s->param_write);
    if (!s->module_read || !s->module_write || !s->param_read || !s->param_write) {
        free(s->module_read);  free(s->module_write);
        free(s->param_read);   free(s->param_write);
        memset(s, 0, sizeof *s);
        return 0;
    }
    return 1;
}

static void alias_summary_free(AliasSummary *s) {
    free(s->module_read);
    free(s->module_write);
    free(s->param_read);
    free(s->param_write);
    memset(s, 0, sizeof *s);
}

static void alias_summary_clear(AliasSummary *s, size_t n) {
    memset(s->module_read,  0, n * sizeof *s->module_read);
    memset(s->module_write, 0, n * sizeof *s->module_write);
    memset(s->param_read,   0, n * sizeof *s->param_read);
    memset(s->param_write,  0, n * sizeof *s->param_write);
}

static int alias_summary_equal(const AliasSummary *a, const AliasSummary *b, size_t n) {
    return memcmp(a->module_read,  b->module_read,  n) == 0 &&
           memcmp(a->module_write, b->module_write, n) == 0 &&
           memcmp(a->param_read,   b->param_read,   n) == 0 &&
           memcmp(a->param_write,  b->param_write,  n) == 0;
}

static void alias_summary_copy(AliasSummary *dst, const AliasSummary *src, size_t n) {
    memcpy(dst->module_read,  src->module_read,  n);
    memcpy(dst->module_write, src->module_write, n);
    memcpy(dst->param_read,   src->param_read,   n);
    memcpy(dst->param_write,  src->param_write,  n);
}

static void alias_summary_or_module(AliasSummary *dst, const AliasSummary *src, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst->module_read[i]  |= src->module_read[i];
        dst->module_write[i] |= src->module_write[i];
    }
}

static int alias_is_expr(const WGSLNode *n) {
    return n && n->kind >= WGSL_NODE_EXPR_LITERAL_BOOL &&
           n->kind <= WGSL_NODE_EXPR_INDIRECTION;
}

static int alias_is_module_var(const WGSLSymbol *s) {
    return s && s->kind == WGSL_SYM_VAR &&
           s->as != WGSL_AS_FUNCTION &&
           s->as != WGSL_AS_NONE;
}

static int alias_is_pointer_param(const WGSLSymbol *s) {
    return s && s->kind == WGSL_SYM_PARAM &&
           s->type && s->type->kind == WGSL_TYPE_PTR;
}

static WGSLNode *alias_decl_init_expr(const WGSLNode *decl) {
    if (!decl) return NULL;
    switch ((WGSLNodeKind)decl->kind) {
    case WGSL_NODE_DECL_LET:
    case WGSL_NODE_DECL_CONST: {
        int has_type = (int)(decl->payload[1] & 1u);
        uint32_t idx = has_type ? 1u : 0u;
        return idx < decl->child_count ? decl->children[idx] : NULL;
    }
    case WGSL_NODE_DECL_VAR: {
        uint32_t A   = (uint32_t)(decl->payload[1] & 0xFFFFFFFFu);
        uint32_t TPL = (uint32_t)(decl->payload[1] >> 32);
        int has_type = (int)(decl->payload[2] & 1u);
        int has_init = (int)(decl->payload[2] >> 32);
        if (!has_init) return NULL;
        uint32_t idx = A + TPL + (has_type ? 1u : 0u);
        return idx < decl->child_count ? decl->children[idx] : NULL;
    }
    case WGSL_NODE_DECL_OVERRIDE: {
        uint32_t A = (uint32_t)(decl->payload[1] & 0xFFFFFFFFu);
        int has_type = (int)(decl->payload[2] & 1u);
        int has_init = (int)(decl->payload[2] >> 32);
        if (!has_init) return NULL;
        uint32_t idx = A + (has_type ? 1u : 0u);
        return idx < decl->child_count ? decl->children[idx] : NULL;
    }
    default:
        return NULL;
    }
}

static WGSLSymbol *alias_root_symbol(WGSLValidator *v, WGSLNode *n, int depth) {
    (void)v;
    if (!n || depth > 32) return NULL;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_IDENT: {
        WGSLSymbol *sym = wgsl_node_resolved_symbol(n);
        if (!sym) return NULL;
        if (sym->kind == WGSL_SYM_VAR) return sym;
        if (alias_is_pointer_param(sym)) return sym;
        if (sym->kind == WGSL_SYM_LET) {
            WGSLNode *init = alias_decl_init_expr(sym->ast);
            return alias_root_symbol(v, init, depth + 1);
        }
        return NULL;
    }
    case WGSL_NODE_EXPR_TEMPLATED_IDENT:
        return n->child_count > 0
            ? alias_root_symbol(v, n->children[0], depth + 1)
            : NULL;
    case WGSL_NODE_EXPR_PAREN:
    case WGSL_NODE_EXPR_ADDR_OF:
    case WGSL_NODE_EXPR_INDIRECTION:
    case WGSL_NODE_EXPR_INDEX:
    case WGSL_NODE_EXPR_MEMBER:
        return n->child_count > 0
            ? alias_root_symbol(v, n->children[0], depth + 1)
            : NULL;
    default:
        return NULL;
    }
}

static void alias_record_root(AliasCtx *cx, WGSLSymbol *root, AccessKind k) {
    if (!root || !cx || !cx->summary) return;
    int idx = wgsl_val_sym_index(cx->v, root);
    if (idx < 0 || (size_t)idx >= cx->nsyms) return;

    if (alias_is_module_var(root)) {
        if (k == ACC_WRITE) cx->summary->module_write[idx] = 1;
        else                cx->summary->module_read[idx] = 1;
    } else if (alias_is_pointer_param(root)) {
        if (k == ACC_WRITE) cx->summary->param_write[idx] = 1;
        else                cx->summary->param_read[idx] = 1;
    }
}

static WGSLSymbol *alias_call_target(WGSLNode *call) {
    if (!call || call->kind != WGSL_NODE_EXPR_CALL || call->child_count == 0) {
        return NULL;
    }
    WGSLNode *head = call->children[0];
    if (head && head->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
        head->child_count > 0)
    {
        head = head->children[0];
    }
    if (!head || head->kind != WGSL_NODE_EXPR_IDENT) return NULL;
    return wgsl_node_resolved_symbol(head);
}

static WGSLSymbol *alias_fn_param_symbol(
    WGSLValidator *v, const WGSLSymbol *fn, uint32_t param_idx)
{
    if (!fn || !fn->ast || fn->ast->kind != WGSL_NODE_DECL_FUNCTION) return NULL;
    WGSLNode *decl = fn->ast;
    uint32_t FA = (uint32_t)(decl->payload[1] & 0xFFFFFFFFu);
    uint32_t P  = (uint32_t)(decl->payload[1] >> 32);
    if (param_idx >= P || FA + param_idx >= decl->child_count) return NULL;
    WGSLNode *param = decl->children[FA + param_idx];
    if (!param || param->kind != WGSL_NODE_DECL_PARAM) return NULL;
    return wgsl_val_find_symbol_for_ast(v, param);
}

static void alias_scan_memory_side(AliasCtx *cx, WGSLNode *n) {
    if (!n) return;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_PAREN:
    case WGSL_NODE_EXPR_ADDR_OF:
    case WGSL_NODE_EXPR_INDIRECTION:
    case WGSL_NODE_EXPR_MEMBER:
        if (n->child_count > 0) alias_scan_memory_side(cx, n->children[0]);
        return;
    case WGSL_NODE_EXPR_INDEX:
        if (n->child_count > 0) alias_scan_memory_side(cx, n->children[0]);
        if (n->child_count > 1) alias_scan_expr(cx, n->children[1], ACC_READ);
        return;
    default:
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if (alias_is_expr(c)) alias_scan_expr(cx, c, ACC_READ);
            else                  alias_scan_stmt(cx, c);
        }
        return;
    }
}

static int alias_atomic_call(const WGSLSymbol *callee) {
    if (!callee) return 0;
    return callee->name_len >= 6 && memcmp(callee->name, "atomic", 6) == 0;
}

static int alias_workgroup_uniform_load_call(const WGSLSymbol *callee) {
    return callee && callee->name_len == 20 &&
           memcmp(callee->name, "workgroupUniformLoad", 20) == 0;
}

static void alias_record_pointer_arg_access(
    AliasCtx *cx, WGSLNode *arg, AccessKind k)
{
    WGSLSymbol *root = alias_root_symbol(cx->v, arg, 0);
    alias_record_root(cx, root, k);
    alias_scan_memory_side(cx, arg);
}

static void alias_record_pointer_arg_root(
    AliasCtx *cx, WGSLNode *arg, AccessKind k)
{
    WGSLSymbol *root = alias_root_symbol(cx->v, arg, 0);
    alias_record_root(cx, root, k);
}

static void alias_emit_pair_conflicts(
    AliasCtx *cx, WGSLNode *call, const WGSLSymbol *callee)
{
    if (!cx->emit_errors || !callee) return;
    int callee_idx = wgsl_val_sym_index(cx->v, callee);
    if (callee_idx < 0 || (size_t)callee_idx >= cx->nsyms) return;
    AliasSummary *cs = &cx->summaries[callee_idx];
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;

    for (uint32_t a = 0; a < argc; a++) {
        WGSLSymbol *pa = alias_fn_param_symbol(cx->v, callee, a);
        if (!alias_is_pointer_param(pa)) continue;
        WGSLSymbol *ra = alias_root_symbol(cx->v, call->children[1 + a], 0);
        if (!ra) continue;
        int pai = wgsl_val_sym_index(cx->v, pa);
        if (pai < 0 || (size_t)pai >= cx->nsyms) continue;
        int ar = cs->param_read[pai] != 0;
        int aw = cs->param_write[pai] != 0;
        if (!ar && !aw) continue;

        for (uint32_t b = a + 1; b < argc; b++) {
            WGSLSymbol *pb = alias_fn_param_symbol(cx->v, callee, b);
            if (!alias_is_pointer_param(pb)) continue;
            WGSLSymbol *rb = alias_root_symbol(cx->v, call->children[1 + b], 0);
            if (ra != rb) continue;
            int pbi = wgsl_val_sym_index(cx->v, pb);
            if (pbi < 0 || (size_t)pbi >= cx->nsyms) continue;
            int br = cs->param_read[pbi] != 0;
            int bw = cs->param_write[pbi] != 0;
            if ((aw && (br || bw)) || (bw && (ar || aw))) {
                wgsl_val_error(cx->v, call,
                    "alias analysis: pointer arguments %u and %u to "
                    "'%.*s' share root '%.*s' while the callee may write "
                    "through one",
                    (unsigned)(a + 1), (unsigned)(b + 1),
                    (int)callee->name_len, callee->name,
                    (int)ra->name_len, ra->name);
            }
        }
    }
}

static void alias_emit_module_conflicts(
    AliasCtx *cx, WGSLNode *call, const WGSLSymbol *callee)
{
    if (!cx->emit_errors || !callee) return;
    int callee_idx = wgsl_val_sym_index(cx->v, callee);
    if (callee_idx < 0 || (size_t)callee_idx >= cx->nsyms) return;
    AliasSummary *cs = &cx->summaries[callee_idx];
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;

    for (uint32_t a = 0; a < argc; a++) {
        WGSLSymbol *pa = alias_fn_param_symbol(cx->v, callee, a);
        if (!alias_is_pointer_param(pa)) continue;
        WGSLSymbol *root = alias_root_symbol(cx->v, call->children[1 + a], 0);
        if (!alias_is_module_var(root)) continue;
        int pai = wgsl_val_sym_index(cx->v, pa);
        int ri  = wgsl_val_sym_index(cx->v, root);
        if (pai < 0 || ri < 0 ||
            (size_t)pai >= cx->nsyms || (size_t)ri >= cx->nsyms)
            continue;

        int pr = cs->param_read[pai] != 0;
        int pw = cs->param_write[pai] != 0;
        int mr = cs->module_read[ri] != 0;
        int mw = cs->module_write[ri] != 0;
        if ((pw && (mr || mw)) || (pr && mw)) {
            wgsl_val_error(cx->v, call,
                "alias analysis: pointer argument %u to '%.*s' aliases "
                "module variable '%.*s' also accessed by the callee",
                (unsigned)(a + 1),
                (int)callee->name_len, callee->name,
                (int)root->name_len, root->name);
        }
    }
}

static void alias_scan_user_call(
    AliasCtx *cx, WGSLNode *call, WGSLSymbol *callee)
{
    int callee_idx = wgsl_val_sym_index(cx->v, callee);
    AliasSummary *cs =
        (callee_idx >= 0 && (size_t)callee_idx < cx->nsyms)
            ? &cx->summaries[callee_idx]
            : NULL;
    if (cs) alias_summary_or_module(cx->summary, cs, cx->nsyms);

    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;
    for (uint32_t a = 0; a < argc; a++) {
        WGSLNode *arg = call->children[1 + a];
        WGSLSymbol *param = alias_fn_param_symbol(cx->v, callee, a);
        if (!arg) continue;
        if (!alias_is_pointer_param(param) || !cs) {
            alias_scan_expr(cx, arg, ACC_READ);
            continue;
        }

        int pi = wgsl_val_sym_index(cx->v, param);
        if (pi < 0 || (size_t)pi >= cx->nsyms) {
            alias_scan_memory_side(cx, arg);
            continue;
        }
        int reads = cs->param_read[pi] != 0;
        int writes = cs->param_write[pi] != 0;
        if (reads || writes) {
            alias_scan_memory_side(cx, arg);
            if (reads)  alias_record_pointer_arg_root(cx, arg, ACC_READ);
            if (writes) alias_record_pointer_arg_root(cx, arg, ACC_WRITE);
        } else {
            alias_scan_memory_side(cx, arg);
        }
    }

    alias_emit_pair_conflicts(cx, call, callee);
    alias_emit_module_conflicts(cx, call, callee);
}

static void alias_scan_call(AliasCtx *cx, WGSLNode *call) {
    WGSLSymbol *callee = alias_call_target(call);
    if (callee && callee->kind == WGSL_SYM_FUNCTION) {
        alias_scan_user_call(cx, call, callee);
        return;
    }

    int is_atomic = alias_atomic_call(callee);
    int is_wgul = alias_workgroup_uniform_load_call(callee);
    int atomic_write = is_atomic &&
        !(callee->name_len == 10 && memcmp(callee->name, "atomicLoad", 10) == 0);
    for (uint32_t i = 1; i < call->child_count; i++) {
        WGSLNode *arg = call->children[i];
        if (!arg) continue;
        if (i == 1 && is_atomic) {
            alias_record_pointer_arg_access(
                cx, arg, atomic_write ? ACC_WRITE : ACC_READ);
        } else if (i == 1 && is_wgul) {
            alias_record_pointer_arg_access(cx, arg, ACC_READ);
        } else {
            alias_scan_expr(cx, arg, ACC_READ);
        }
    }
}

static void alias_scan_expr(AliasCtx *cx, WGSLNode *n, AccessKind k) {
    if (!n) return;
    if (wgsl_typecheck_is_ref(cx->v->tc, n)) {
        WGSLSymbol *root = alias_root_symbol(cx->v, n, 0);
        alias_record_root(cx, root, k);
        alias_scan_memory_side(cx, n);
        return;
    }

    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_CALL:
        alias_scan_call(cx, n);
        return;
    case WGSL_NODE_EXPR_ADDR_OF:
        if (n->child_count > 0) alias_scan_memory_side(cx, n->children[0]);
        return;
    case WGSL_NODE_EXPR_INDEX:
        if (alias_is_pointer_param(alias_root_symbol(cx->v, n, 0))) {
            WGSLSymbol *root = alias_root_symbol(cx->v, n, 0);
            alias_record_root(cx, root, k);
            alias_scan_memory_side(cx, n);
            return;
        }
        if (n->child_count > 0) alias_scan_expr(cx, n->children[0], ACC_READ);
        if (n->child_count > 1) alias_scan_expr(cx, n->children[1], ACC_READ);
        return;
    case WGSL_NODE_EXPR_MEMBER:
        if (alias_is_pointer_param(alias_root_symbol(cx->v, n, 0))) {
            WGSLSymbol *root = alias_root_symbol(cx->v, n, 0);
            alias_record_root(cx, root, k);
            alias_scan_memory_side(cx, n);
            return;
        }
        if (n->child_count > 0) alias_scan_expr(cx, n->children[0], ACC_READ);
        return;
    case WGSL_NODE_EXPR_PAREN:
    case WGSL_NODE_EXPR_UNARY:
    case WGSL_NODE_EXPR_INDIRECTION:
        if (n->child_count > 0) alias_scan_expr(cx, n->children[0], ACC_READ);
        return;
    default:
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if (alias_is_expr(c)) alias_scan_expr(cx, c, ACC_READ);
            else                  alias_scan_stmt(cx, c);
        }
        return;
    }
}

static void alias_scan_decl_init(AliasCtx *cx, WGSLNode *n) {
    WGSLNode *init = alias_decl_init_expr(n);
    if (init) alias_scan_expr(cx, init, ACC_READ);
}

static void alias_scan_stmt(AliasCtx *cx, WGSLNode *n) {
    if (!n) return;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_STMT_ASSIGN:
        if (n->child_count > 0) alias_scan_expr(cx, n->children[0], ACC_WRITE);
        if (n->child_count > 1) alias_scan_expr(cx, n->children[1], ACC_READ);
        return;
    case WGSL_NODE_STMT_COMPOUND_ASSIGN:
        if (n->child_count > 0) {
            alias_scan_expr(cx, n->children[0], ACC_READ);
            alias_scan_expr(cx, n->children[0], ACC_WRITE);
        }
        if (n->child_count > 1) alias_scan_expr(cx, n->children[1], ACC_READ);
        return;
    case WGSL_NODE_STMT_INCREMENT:
    case WGSL_NODE_STMT_DECREMENT:
        if (n->child_count > 0) {
            alias_scan_expr(cx, n->children[0], ACC_READ);
            alias_scan_expr(cx, n->children[0], ACC_WRITE);
        }
        return;
    case WGSL_NODE_STMT_PHONY_ASSIGN:
    case WGSL_NODE_STMT_RETURN:
        if (n->child_count > 0) alias_scan_expr(cx, n->children[0], ACC_READ);
        return;
    case WGSL_NODE_STMT_FN_CALL:
        if (n->child_count > 0) alias_scan_expr(cx, n->children[0], ACC_READ);
        return;
    case WGSL_NODE_DECL_LET:
    case WGSL_NODE_DECL_CONST:
    case WGSL_NODE_DECL_OVERRIDE:
    case WGSL_NODE_DECL_VAR:
        alias_scan_decl_init(cx, n);
        return;
    default:
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if (alias_is_expr(c)) alias_scan_expr(cx, c, ACC_READ);
            else                  alias_scan_stmt(cx, c);
        }
        return;
    }
}

void validate_alias_analysis(WGSLValidator *v) {
    if (!v || !v->res) return;
    size_t n = v->res->all_decl_count;
    if (n == 0) return;

    AliasSummary *summaries = (AliasSummary *)calloc(n, sizeof *summaries);
    if (!summaries) return;
    AliasSummary scratch = {0};
    if (!alias_summary_init(&scratch, n)) {
        free(summaries);
        return;
    }
    for (size_t i = 0; i < n; i++) {
        if (!alias_summary_init(&summaries[i], n)) {
            for (size_t j = 0; j < i; j++) alias_summary_free(&summaries[j]);
            alias_summary_free(&scratch);
            free(summaries);
            return;
        }
    }

    for (size_t iter = 0; iter <= n; iter++) {
        int changed = 0;
        for (size_t i = 0; i < n; i++) {
            WGSLSymbol *s = v->res->all_decls[i];
            if (!s || s->kind != WGSL_SYM_FUNCTION ||
                !s->ast || s->ast->child_count == 0)
                continue;
            alias_summary_clear(&scratch, n);
            AliasCtx cx = { v, &scratch, summaries, n, 0 };
            WGSLNode *body = s->ast->children[s->ast->child_count - 1];
            alias_scan_stmt(&cx, body);
            if (!alias_summary_equal(&summaries[i], &scratch, n)) {
                alias_summary_copy(&summaries[i], &scratch, n);
                changed = 1;
            }
        }
        if (!changed) break;
    }

    for (size_t i = 0; i < n; i++) {
        WGSLSymbol *s = v->res->all_decls[i];
        if (!s || s->kind != WGSL_SYM_FUNCTION ||
            !s->ast || s->ast->child_count == 0)
            continue;
        alias_summary_clear(&scratch, n);
        AliasCtx cx = { v, &scratch, summaries, n, 1 };
        WGSLNode *body = s->ast->children[s->ast->child_count - 1];
        alias_scan_stmt(&cx, body);
    }

    for (size_t i = 0; i < n; i++) alias_summary_free(&summaries[i]);
    alias_summary_free(&scratch);
    free(summaries);
}

/* §9.4.11 — `discard` may only appear in code reachable from a
 * fragment entry point.  Walk each fn body for STMT_DISCARD; per-fn
 * "discards" bit drives a call-graph BFS from each entry point.
 * `discard_at[i]` records the AST node of the first DISCARD we saw
 * in that fn (used for the diagnostic span). */
static void scan_discards(WGSLNode *n, WGSLNode **out) {
    if (!n || *out) return;
    if (n->kind == WGSL_NODE_STMT_DISCARD) { *out = n; return; }
    for (uint32_t i = 0; i < n->child_count && !*out; i++) {
        scan_discards(n->children[i], out);
    }
}

typedef struct {
    WGSLValidator *v;
    uint32_t       stage_mask;
    WGSLNode      *first_discard;   /* set on first hit, used for diag */
    uint8_t       *visited;
    WGSLNode     **discard_at;
} DiscardCtx;

static void discard_edge_cb(WGSLValidator *v, const WGSLSymbol *to, const WGSLNode *at, void *ctx) {
    DiscardCtx *cx = (void *)ctx;
    int idx = wgsl_val_sym_index(v, to);
    if (idx < 0 || cx->visited[idx]) return;
    cx->visited[idx] = 1;
    if (cx->discard_at[idx] && !cx->first_discard) {
        cx->first_discard = cx->discard_at[idx];
    }
    wgsl_val_enum_call_edges(v, to, discard_edge_cb, cx);
}

void validate_discard_stages(WGSLValidator *v) {
    if (!v->res) return;
    size_t n = v->res->all_decl_count;
    WGSLNode **discard_at = (WGSLNode **)calloc(n, sizeof *discard_at);
    uint8_t   *visited    = (uint8_t   *)calloc(n, sizeof *visited);
    if (!discard_at || !visited) { free(discard_at); free(visited); return; }
    /* Per-fn pre-scan: record the first discard node (if any). */
    for (size_t i = 0; i < n; i++) {
        WGSLSymbol *s = v->res->all_decls[i];
        if (s->kind != WGSL_SYM_FUNCTION || !s->ast || s->ast->child_count == 0) continue;
        WGSLNode *body = s->ast->children[s->ast->child_count - 1];
        scan_discards(body, &discard_at[i]);
    }
    /* Per-entry-point BFS through call-graph; if any reachable fn (or
     * the entry itself) carries a discard AND the entry stage isn't
     * fragment, emit on the first found discard. */
    for (size_t i = 0; i < n; i++) {
        WGSLSymbol *s = v->res->all_decls[i];
        if (s->kind != WGSL_SYM_FUNCTION) continue;
        uint32_t stage = 0;
        if (s->flags & WGSL_SYM_FLAG_VERTEX)   stage = STAGE_VX;
        if (s->flags & WGSL_SYM_FLAG_FRAGMENT) stage = STAGE_FR;
        if (s->flags & WGSL_SYM_FLAG_COMPUTE)  stage = STAGE_CO;
        if (stage == 0 || stage == STAGE_FR) continue;
        memset(visited, 0, n);
        DiscardCtx cx = { v, stage, NULL, visited, discard_at };
        if (discard_at[i]) cx.first_discard = discard_at[i];
        visited[i] = 1;
        wgsl_val_enum_call_edges(v, s, discard_edge_cb, &cx);
        if (cx.first_discard) {
            wgsl_val_error(v, cx.first_discard,
                "'discard' is only valid in code reachable from a "
                "fragment entry point (entry '%.*s' is %s)",
                (int)s->name_len, s->name,
                stage == STAGE_VX ? "@vertex" : "@compute");
        }
    }
    free(visited);
    free(discard_at);
}
