/**
 * @file access_discard.c — discard stage reachability (§9.4.11)
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

