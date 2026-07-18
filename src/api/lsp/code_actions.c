/**
 * @file code_actions.c — code actions / quick fixes
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


/* code actions / quickfix. */

/* Parse `requires 'enable NAME;'` (and the trailing-semicolon-inside-
 * quotes form the frontend emits).  Writes NAME into `out` (NUL-term). */
static int extract_enable_ext(const char *msg, char *out, size_t cap) {
    if (!msg || !out || cap < 2) return 0;
    const char *p = strstr(msg, "requires 'enable ");
    if (!p) p = strstr(msg, "requires \"enable ");
    if (!p) return 0;
    p = strstr(p, "enable ");
    if (!p) return 0;
    p += 7; /* strlen("enable ") */
    size_t n = 0;
    while (p[n] && p[n] != '\'' && p[n] != '"' && p[n] != ';' &&
           p[n] != ' ' && p[n] != '\n' && n + 1 < cap)
        n++;
    if (n == 0) return 0;
    memcpy(out, p, n);
    out[n] = '\0';
    return 1;
}

/* True if the module already has `enable NAME` (any directive). */
static int module_has_enable(const WGSLResult *r, const char *name) {
    if (!r || !r->ast.root || !name) return 0;
    size_t nl = strlen(name);
    const WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        const WGSLNode *n = tu->children[i];
        if (!n || n->kind != WGSL_NODE_DIR_ENABLE) continue;
        for (uint32_t k = 0; k < n->child_count; k++) {
            const WGSLNode *id = n->children[k];
            if (!id || id->kind != WGSL_NODE_EXPR_IDENT) continue;
            uint32_t o = wgsl_node_name_span(id).offset;
            uint32_t l = wgsl_node_name_span(id).length;
            if (l == nl && r->src_copy && o + l <= r->src_len &&
                memcmp(r->src_copy + o, name, nl) == 0)
                return 1;
        }
    }
    return 0;
}

/* Insert point: just after the last enable/requires/diagnostic directive
 * (end of its span), else 0. */
static uint32_t enable_insert_offset(const WGSLResult *r) {
    uint32_t off = 0;
    if (!r || !r->ast.root) return 0;
    const WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        const WGSLNode *n = tu->children[i];
        if (!n) continue;
        if (n->kind != WGSL_NODE_DIR_ENABLE &&
            n->kind != WGSL_NODE_DIR_REQUIRES &&
            n->kind != WGSL_NODE_DIR_DIAGNOSTIC)
            continue;
        uint32_t end = n->span_offset + n->span_length;
        if (end > off) off = end;
    }
    return off;
}

/* ActBuf: lsp_priv.h */

static int act_push(ActBuf *b, WGSLCodeAction a) {
    if (b->n + 1 > b->cap) {
        int nc = b->cap ? b->cap * 2 : 4;
        WGSLCodeAction *g = (WGSLCodeAction *)realloc(b->v, (size_t)nc * sizeof *g);
        if (!g) return 0;
        b->v = g; b->cap = nc;
    }
    b->v[b->n++] = a;
    return 1;
}

static int act_has_kind(const ActBuf *b, const char *kind) {
    for (int i = 0; i < b->n; i++)
        if (b->v[i].kind && strcmp(b->v[i].kind, kind) == 0) return 1;
    return 0;
}

/* Push a single-edit code action; frees partial state on OOM. */
static int act_push_edit(ActBuf *b, const char *title, const char *kind,
                         uint32_t start, uint32_t end, const char *text)
{
    if (act_has_kind(b, kind)) return 1;
    WGSLTextEdit *ed = (WGSLTextEdit *)calloc(1, sizeof *ed);
    if (!ed) return 0;
    ed->start_offset = start;
    ed->end_offset = end;
    ed->new_text = lsp_strdup(text);
    if (!ed->new_text) { free(ed); return 0; }

    WGSLCodeAction a;
    memset(&a, 0, sizeof a);
    a.title = lsp_strdup(title);
    a.kind = lsp_strdup(kind);
    a.edits = ed;
    a.edit_count = 1;
    if (!a.title || !a.kind) {
        free(a.title); free(a.kind);
        free(ed->new_text); free(ed);
        return 0;
    }
    if (!act_push(b, a)) {
        free(a.title); free(a.kind);
        free(ed->new_text); free(ed);
        return 0;
    }
    return 1;
}

/* Attr name match on parent->children[first..first+count). */
static int ca_has_attr(const WGSLResult *r, const WGSLNode *parent,
                       uint32_t first, uint32_t count, const char *want)
{
    if (!r || !parent || !want || !r->src_copy) return 0;
    size_t wl = strlen(want);
    for (uint32_t i = 0; i < count && first + i < parent->child_count; i++) {
        const WGSLNode *a = parent->children[first + i];
        if (!a || a->kind != WGSL_NODE_ATTRIBUTE || a->child_count == 0)
            continue;
        const WGSLNode *nm = a->children[0];
        if (!nm || nm->kind != WGSL_NODE_EXPR_IDENT) continue;
        uint32_t o = wgsl_node_name_span(nm).offset;
        uint32_t l = wgsl_node_name_span(nm).length;
        if (l == wl && o + l <= r->src_len &&
            memcmp(r->src_copy + o, want, wl) == 0)
            return 1;
    }
    return 0;
}

static int ca_fn_stage(const WGSLResult *r, const WGSLNode *fn) {
    /* 1 = vertex, 2 = fragment, 3 = compute, 0 = none */
    if (!fn) return 0;
    uint32_t FA = wgsl_fn_attr_count(fn);
    if (ca_has_attr(r, fn, 0, FA, "vertex")) return 1;
    if (ca_has_attr(r, fn, 0, FA, "fragment")) return 2;
    if (ca_has_attr(r, fn, 0, FA, "compute")) return 3;
    return 0;
}

/* Collect @location(N) bits already present on direct EP params/return. */
static uint64_t ca_collect_locations(const WGSLResult *r, const WGSLNode *fn) {
    uint64_t bits = 0;
    if (!r || !fn || !r->src_copy) return 0;
    uint32_t FA = wgsl_fn_attr_count(fn);
    uint32_t P  = wgsl_fn_param_count(fn);
    uint32_t RA = wgsl_fn_ret_attr_count(fn);
    uint32_t HR = wgsl_fn_has_return_type(fn);

    for (uint32_t k = 0; k < P && FA + k < fn->child_count; k++) {
        const WGSLNode *p = fn->children[FA + k];
        if (!p || p->kind != WGSL_NODE_DECL_PARAM) continue;
        uint32_t PA = wgsl_param_attr_count(p);
        for (uint32_t ai = 0; ai < PA && ai < p->child_count; ai++) {
            const WGSLNode *a = p->children[ai];
            if (!a || a->kind != WGSL_NODE_ATTRIBUTE || a->child_count < 2)
                continue;
            const WGSLNode *nm = a->children[0];
            if (!nm || nm->kind != WGSL_NODE_EXPR_IDENT) continue;
            uint32_t o = wgsl_node_name_span(nm).offset;
            uint32_t l = wgsl_node_name_span(nm).length;
            if (!(l == 8 && o + l <= r->src_len &&
                  memcmp(r->src_copy + o, "location", 8) == 0))
                continue;
            const WGSLNode *arg = a->children[1];
            if (!arg || arg->kind != WGSL_NODE_EXPR_LITERAL_INT) continue;
            /* Int literals store suffix flags in payload; source is the span. */
            uint32_t ao = arg->span_offset;
            uint32_t al = arg->span_length;
            if (al == 0 || (size_t)ao + (size_t)al > r->src_len) continue;
            long long v = 0;
            for (uint32_t i = 0; i < al; i++) {
                char c = r->src_copy[ao + i];
                if (c < '0' || c > '9') { v = -1; break; }
                v = v * 10 + (c - '0');
            }
            if (v >= 0 && v < 64) bits |= (1ull << (unsigned)v);
        }
    }
    if (HR) {
        for (uint32_t ai = 0; ai < RA && FA + P + ai < fn->child_count; ai++) {
            const WGSLNode *a = fn->children[FA + P + ai];
            if (!a || a->kind != WGSL_NODE_ATTRIBUTE || a->child_count < 2)
                continue;
            const WGSLNode *nm = a->children[0];
            if (!nm || nm->kind != WGSL_NODE_EXPR_IDENT) continue;
            uint32_t o = wgsl_node_name_span(nm).offset;
            uint32_t l = wgsl_node_name_span(nm).length;
            if (!(l == 8 && o + l <= r->src_len &&
                  memcmp(r->src_copy + o, "location", 8) == 0))
                continue;
            const WGSLNode *arg = a->children[1];
            if (!arg || arg->kind != WGSL_NODE_EXPR_LITERAL_INT) continue;
            uint32_t ao = arg->span_offset;
            uint32_t al = arg->span_length;
            if (al == 0 || (size_t)ao + (size_t)al > r->src_len) continue;
            long long v = 0;
            for (uint32_t i = 0; i < al; i++) {
                char c = r->src_copy[ao + i];
                if (c < '0' || c > '9') { v = -1; break; }
                v = v * 10 + (c - '0');
            }
            if (v >= 0 && v < 64) bits |= (1ull << (unsigned)v);
        }
    }
    return bits;
}

static int ca_next_location(uint64_t *bits) {
    for (int i = 0; i < 64; i++) {
        if ((*bits & (1ull << i)) == 0) {
            *bits |= (1ull << i);
            return i;
        }
    }
    return -1;
}

static int ca_push_location(ActBuf *b, uint32_t ins, int loc) {
    char title[64], kind[64], text[32];
    snprintf(title, sizeof title, "Add '@location(%d)'", loc);
    snprintf(kind, sizeof kind, "quickfix.add_location.%u", ins);
    snprintf(text, sizeof text, "@location(%d) ", loc);
    return act_push_edit(b, title, kind, ins, ins, text);
}

/* Missing @location on vertex/fragment direct IO slots. */
static int ca_location_fixes(const WGSLResult *r, ActBuf *b) {
    if (!r || !r->ast.root) return 1;
    const WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        const WGSLNode *fn = tu->children[i];
        if (!fn || fn->kind != WGSL_NODE_DECL_FUNCTION) continue;
        int stage = ca_fn_stage(r, fn);
        if (stage != 1 && stage != 2) continue; /* vertex / fragment only */

        uint32_t FA = wgsl_fn_attr_count(fn);
        uint32_t P  = wgsl_fn_param_count(fn);
        uint32_t RA = wgsl_fn_ret_attr_count(fn);
        uint32_t HR = wgsl_fn_has_return_type(fn);
        uint64_t used = ca_collect_locations(r, fn);

        for (uint32_t k = 0; k < P && FA + k < fn->child_count; k++) {
            const WGSLNode *p = fn->children[FA + k];
            if (!p || p->kind != WGSL_NODE_DECL_PARAM) continue;
            uint32_t PA = wgsl_param_attr_count(p);
            if (ca_has_attr(r, p, 0, PA, "location")) continue;
            if (ca_has_attr(r, p, 0, PA, "builtin")) continue;
            /* Struct-typed params own locations on members — skip. */
            if (PA < p->child_count) {
                const WGSLNode *ts = p->children[PA];
                WGSLTypeInfo *pt = ts ? wgsl_typecheck_type_of(&r->tc, ts) : NULL;
                if (pt && pt->kind == WGSL_TYPE_STRUCT) continue;
            }
            int loc = ca_next_location(&used);
            if (loc < 0) continue;
            if (!ca_push_location(b, p->span_offset, loc)) return 0;
        }

        if (HR && FA + P + RA < fn->child_count) {
            if (!ca_has_attr(r, fn, FA + P, RA, "location") &&
                !ca_has_attr(r, fn, FA + P, RA, "builtin"))
            {
                const WGSLNode *ret_spec = fn->children[FA + P + RA];
                WGSLTypeInfo *rt = ret_spec
                    ? wgsl_typecheck_type_of(&r->tc, ret_spec) : NULL;
                if (!(rt && rt->kind == WGSL_TYPE_STRUCT)) {
                    uint32_t ins = ret_spec ? ret_spec->span_offset
                                            : fn->span_offset;
                    if (RA > 0 && FA + P < fn->child_count &&
                        fn->children[FA + P])
                        ins = fn->children[FA + P]->span_offset;
                    int loc = ca_next_location(&used);
                    if (loc >= 0) {
                        if (!ca_push_location(b, ins, loc)) return 0;
                    }
                }
            }
        }
    }
    return 1;
}

static int is_pow2_u64(uint64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

static uint64_t next_pow2_u64(uint64_t n) {
    if (n <= 1) return 1;
    n--;
    n |= n >> 1; n |= n >> 2; n |= n >> 4;
    n |= n >> 8; n |= n >> 16; n |= n >> 32;
    return n + 1;
}

/* @align(N) where N is not a power of two: rewrite N. */
static int ca_align_fixes(const WGSLResult *r, ActBuf *b) {
    if (!r || !r->ast.root || !r->src_copy) return 1;
    const WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        const WGSLNode *st = tu->children[i];
        if (!st || st->kind != WGSL_NODE_DECL_STRUCT) continue;
        for (uint32_t mi = 0; mi < st->child_count; mi++) {
            const WGSLNode *m = st->children[mi];
            if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            uint32_t MA = wgsl_member_attr_count(m);
            for (uint32_t ai = 0; ai < MA && ai < m->child_count; ai++) {
                const WGSLNode *a = m->children[ai];
                if (!a || a->kind != WGSL_NODE_ATTRIBUTE || a->child_count < 2)
                    continue;
                const WGSLNode *nm = a->children[0];
                if (!nm || nm->kind != WGSL_NODE_EXPR_IDENT) continue;
                uint32_t no = wgsl_node_name_span(nm).offset;
                uint32_t nl = wgsl_node_name_span(nm).length;
                if (!(nl == 5 && no + nl <= r->src_len &&
                      memcmp(r->src_copy + no, "align", 5) == 0))
                    continue;
                const WGSLNode *arg = a->children[1];
                if (!arg || arg->kind != WGSL_NODE_EXPR_LITERAL_INT) continue;
                /* Int literal payload is suffix flags; use the node span. */
                uint32_t ao = arg->span_offset;
                uint32_t al = arg->span_length;
                if (al == 0 || (size_t)ao + (size_t)al > r->src_len) continue;
                uint64_t v = 0;
                int ok = 1;
                for (uint32_t k = 0; k < al; k++) {
                    char c = r->src_copy[ao + k];
                    if (c < '0' || c > '9') { ok = 0; break; }
                    v = v * 10 + (uint64_t)(c - '0');
                }
                if (!ok || v == 0 || is_pow2_u64(v)) continue;
                uint64_t np = next_pow2_u64(v);
                char title[80], kind[80], text[32];
                snprintf(title, sizeof title,
                         "Change @align(%llu) to @align(%llu)",
                         (unsigned long long)v, (unsigned long long)np);
                snprintf(kind, sizeof kind, "quickfix.align_pow2.%u", ao);
                snprintf(text, sizeof text, "%llu", (unsigned long long)np);
                if (!act_push_edit(b, title, kind, ao, ao + al, text))
                    return 0;
            }
        }
    }
    return 1;
}

/* Optimizer findings → one whole-document quickfix.  The optimizer owns
 * conflict resolution across DCE/unused/const-branch/CSE edits; LSP just
 * exposes the resulting safe rewrite when it differs from the checked source. */
static int ca_optimize_apply_fix(const WGSLResult *r, ActBuf *b) {
    if (!r || !r->src_copy) return 1;
    if (!wgsl_ok(r)) return 1;
    char *opt = wgsl_optimize_apply(r->src_copy);
    if (!opt) return 0;
    int changed = strcmp(opt, r->src_copy) != 0;
    int ok = 1;
    if (changed) {
        ok = act_push_edit(b, "Apply safe WGSL optimizations",
                           "quickfix.opt.apply", 0, (uint32_t)r->src_len, opt);
    }
    wgsl_free_string(opt);
    return ok;
}

int wgsl_code_actions(const WGSLResult *r,
                      WGSLCodeAction **out, int *out_count)
{
    if (out) *out = NULL;
    if (out_count) *out_count = 0;
    if (!r || !out || !out_count) return 0;

    ActBuf b = {0};
    int nd = wgsl_diagnostic_count(r);
    for (int i = 0; i < nd; i++) {
        const WGSLDiagnostic *d = wgsl_diagnostic(r, i);
        if (!d || !d->message) continue;
        char ext[64];
        if (!extract_enable_ext(d->message, ext, sizeof ext)) continue;
        if (module_has_enable(r, ext)) continue;

        char kind[96];
        snprintf(kind, sizeof kind, "quickfix.enable.%s", ext);
        if (act_has_kind(&b, kind)) continue;

        char title[96];
        snprintf(title, sizeof title, "Add 'enable %s;'", ext);

        char text[80];
        /* Prefer a leading newline when inserting after existing text
         * that does not already end with one. */
        uint32_t ins = enable_insert_offset(r);
        int need_nl_before = 0;
        if (ins > 0 && r->src_copy && ins <= r->src_len &&
            r->src_copy[ins - 1] != '\n')
            need_nl_before = 1;
        snprintf(text, sizeof text, "%senable %s;\n",
                 need_nl_before ? "\n" : "", ext);

        if (!act_push_edit(&b, title, kind, ins, ins, text)) {
            wgsl_code_actions_free(b.v, b.n);
            return 0;
        }
    }

    /* Location / align auto-fixes (text edits, no AST rewrite). */
    if (!ca_location_fixes(r, &b) || !ca_align_fixes(r, &b) ||
        !ca_optimize_apply_fix(r, &b)) {
        wgsl_code_actions_free(b.v, b.n);
        return 0;
    }

    *out = b.v;
    *out_count = b.n;
    return 1;
}

void wgsl_code_actions_free(WGSLCodeAction *actions, int count) {
    if (!actions) return;
    for (int i = 0; i < count; i++) {
        free(actions[i].title);
        free(actions[i].kind);
        if (actions[i].edits) {
            for (int e = 0; e < actions[i].edit_count; e++)
                free(actions[i].edits[e].new_text);
            free(actions[i].edits);
        }
    }
    free(actions);
}
