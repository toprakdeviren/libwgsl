/**
 * @file signature.c — signature help
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


/* signature help. */

/* Deepest call expression whose argument-list region contains offset.
 * Heuristic: node span contains offset and kind is EXPR_CALL. */
static const WGSLNode *call_at(const WGSLNode *n, uint32_t offset, int depth) {
    if (!n || depth > WGSL_MAX_AST_DEPTH) return NULL;
    if (offset < n->span_offset ||
        offset >= n->span_offset + n->span_length)
        return NULL;
    const WGSLNode *best = NULL;
    if (n->kind == WGSL_NODE_EXPR_CALL) best = n;
    for (uint32_t i = 0; i < n->child_count; i++) {
        const WGSLNode *hit = call_at(n->children[i], offset, depth + 1);
        if (hit) best = hit;
    }
    return best;
}

/* Active parameter index: count top-level commas in the arg list before offset. */
static uint32_t active_arg_index(const WGSLResult *r, const WGSLNode *call,
                                 uint32_t offset)
{
    if (!r || !call || call->child_count < 1) return 0;
    /* children[0] = callee; children[1..] = args.  Prefer counting
     * completed arg nodes whose span ends before the cursor. */
    uint32_t idx = 0;
    for (uint32_t i = 1; i < call->child_count; i++) {
        const WGSLNode *a = call->children[i];
        if (!a) continue;
        if (a->span_offset + a->span_length <= offset) idx = i; /* 1-based → later -1 */
        else break;
    }
    /* idx is last completed arg's child index; active is that index-1, or 0. */
    if (idx == 0) return 0;
    return idx - 1;
}

static WGSLSignatureHelp empty_sig(void) {
    WGSLSignatureHelp h = {0};
    h.label = "";
    return h;
}

/* Build "name(p0: T0, p1: T1) -> R" into the result arena. */
static const char *format_fn_signature(WGSLResult *r, WGSLSymbol *fn_sym) {
    if (!fn_sym || !fn_sym->ast || fn_sym->ast->kind != WGSL_NODE_DECL_FUNCTION)
        return "";
    const WGSLNode *fn = fn_sym->ast;
    uint32_t FA = wgsl_fn_attr_count(fn);
    uint32_t P  = wgsl_fn_param_count(fn);
    uint32_t RA = wgsl_fn_ret_attr_count(fn);
    uint32_t HR = wgsl_fn_has_return_type(fn);

    char buf[512];
    size_t used = 0;
    #define APP(s) do { \
        const char *_s = (s); size_t _n = strlen(_s); \
        if (used + _n < sizeof buf) { memcpy(buf + used, _s, _n); used += _n; } \
    } while (0)
    #define APPN(p, n) do { \
        size_t _n = (n); if (used + _n < sizeof buf) { memcpy(buf + used, (p), _n); used += _n; } \
    } while (0)

    APPN(fn_sym->name, fn_sym->name_len);
    APP("(");
    for (uint32_t k = 0; k < P; k++) {
        if (k) APP(", ");
        uint32_t ci = FA + k;
        if (ci >= fn->child_count) break;
        const WGSLNode *p = fn->children[ci];
        if (!p || p->kind != WGSL_NODE_DECL_PARAM) continue;
        uint32_t pno = wgsl_node_name_span(p).offset;
        uint32_t pnl = wgsl_node_name_span(p).length;
        if (pnl && r->src_copy && pno + pnl <= r->src_len)
            APPN(r->src_copy + pno, pnl);
        APP(": ");
        uint32_t PA = wgsl_param_attr_count(p);
        WGSLTypeInfo *pt = (PA < p->child_count)
            ? wgsl_typecheck_type_of(&r->tc, p->children[PA]) : NULL;
        char tbuf[96];
        if (pt) {
            size_t tn = wgsl_type_format(pt, tbuf, sizeof tbuf);
            if (tn >= sizeof tbuf) tn = sizeof tbuf - 1;
            APPN(tbuf, tn);
        } else {
            APP("?");
        }
    }
    APP(")");
    if (HR) {
        APP(" -> ");
        uint32_t rti = FA + P + RA;
        WGSLTypeInfo *rt = (rti < fn->child_count)
            ? wgsl_typecheck_type_of(&r->tc, fn->children[rti]) : NULL;
        char tbuf[96];
        if (rt) {
            size_t tn = wgsl_type_format(rt, tbuf, sizeof tbuf);
            if (tn >= sizeof tbuf) tn = sizeof tbuf - 1;
            APPN(tbuf, tn);
        } else {
            APP("?");
        }
    }
    #undef APP
    #undef APPN
    buf[used] = '\0';
    char *owned = (char *)wgsl_arena_alloc(&r->arena, used + 1);
    if (!owned) return "";
    memcpy(owned, buf, used + 1);
    return owned;
}

WGSLSignatureHelp wgsl_signature_help_at(const WGSLResult *r, uint32_t offset) {
    if (!r || !r->ast.root) return empty_sig();
    const WGSLNode *call = call_at(r->ast.root, offset, 0);
    if (!call || call->child_count == 0) return empty_sig();

    const WGSLNode *callee = call->children[0];
    WGSLSymbol *sym = NULL;
    if (callee && callee->kind == WGSL_NODE_EXPR_IDENT)
        sym = wgsl_node_resolved_symbol(callee);
    else if (callee && callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
             callee->child_count > 0)
        sym = wgsl_node_resolved_symbol(callee->children[0]);

    if (!sym) return empty_sig();

    WGSLSignatureHelp h = empty_sig();
    h.present = 1;
    h.active_parameter = active_arg_index(r, call, offset);

    if (sym->ast && sym->ast->kind == WGSL_NODE_DECL_FUNCTION) {
        uint32_t P = wgsl_fn_param_count(sym->ast);
        h.parameter_count = P;
        if (h.active_parameter >= P && P > 0) h.active_parameter = P - 1;
        h.label = format_fn_signature((WGSLResult *)r, sym);
    } else {
        /* Builtin / predeclared: just the name + "(…)" */
        char tmp[160];
        size_t n = sym->name_len < sizeof tmp - 4 ? sym->name_len : sizeof tmp - 4;
        memcpy(tmp, sym->name, n);
        memcpy(tmp + n, "(…)", 4);
        char *owned = (char *)wgsl_arena_alloc((WGSLArena *)&r->arena, n + 4);
        if (owned) { memcpy(owned, tmp, n + 4); h.label = owned; }
        else h.label = "";
        h.parameter_count = 0;
        h.active_parameter = 0;
    }
    return h;
}

void wgsl_signature_help_at_into(
    const WGSLResult *r, uint32_t offset, WGSLSignatureHelp *out)
{
    if (!out) return;
    *out = wgsl_signature_help_at(r, offset);
}

