/**
 * @file apply.c — source rewrite half of the optimizer.
 */
#include "wgsl.h"
#include "internal/result.h"
#include "internal/ast.h"
#include "internal/resolver.h"
#include "internal/check.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int node_is_terminal(const WGSLNode *n) {
    return n && (n->kind == WGSL_NODE_STMT_RETURN ||
                 n->kind == WGSL_NODE_STMT_DISCARD);
}

static int attr_named(const WGSLResult *r, const WGSLNode *parent,
                      uint32_t first, uint32_t count, const char *want)
{
    if (!r || !r->src_copy || !parent || !want) return 0;
    size_t wl = strlen(want);
    for (uint32_t i = 0; i < count && first + i < parent->child_count; i++) {
        const WGSLNode *a = parent->children[first + i];
        if (!a || a->kind != WGSL_NODE_ATTRIBUTE || a->child_count == 0) continue;
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

static int fn_is_entry(const WGSLResult *r, const WGSLNode *fn) {
    if (!fn || fn->kind != WGSL_NODE_DECL_FUNCTION) return 0;
    uint32_t FA = wgsl_fn_attr_count(fn);
    return attr_named(r, fn, 0, FA, "vertex") ||
           attr_named(r, fn, 0, FA, "fragment") ||
           attr_named(r, fn, 0, FA, "compute");
}

static int module_has_entry(const WGSLResult *r) {
    if (!r || !r->ast.root) return 0;
    const WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        const WGSLNode *n = tu->children[i];
        if (n && n->kind == WGSL_NODE_DECL_FUNCTION && fn_is_entry(r, n))
            return 1;
    }
    return 0;
}

/* unused locals / functions. */

typedef struct {
    WGSLSymbol *sym;
    uint32_t    off, len;
    int         uses; /* number of IDENT references (not counting decl) */
    int         is_fn;
    int         is_entry;
} UseRec;

typedef struct {
    UseRec *v;
    int n, cap;
} UseTab;

static int use_push(UseTab *t, UseRec u) {
    if (t->n + 1 > t->cap) {
        int nc = t->cap ? t->cap * 2 : 16;
        UseRec *g = (UseRec *)realloc(t->v, (size_t)nc * sizeof *g);
        if (!g) return 0;
        t->v = g; t->cap = nc;
    }
    t->v[t->n++] = u;
    return 1;
}

static UseRec *use_find(UseTab *t, WGSLSymbol *s) {
    for (int i = 0; i < t->n; i++)
        if (t->v[i].sym == s) return &t->v[i];
    return NULL;
}

static void count_uses(UseTab *t, const WGSLNode *n) {
    if (!n) return;
    if (n->kind == WGSL_NODE_EXPR_IDENT) {
        WGSLSymbol *s = wgsl_node_resolved_symbol(n);
        UseRec *u = use_find(t, s);
        if (u) u->uses++;
    }
    for (uint32_t i = 0; i < n->child_count; i++)
        count_uses(t, n->children[i]);
}

static int literal_bool(const WGSLNode *n, int *out) {
    if (!n) return 0;
    if (n->kind == WGSL_NODE_EXPR_PAREN && n->child_count > 0)
        return literal_bool(n->children[0], out);
    if (n->kind == WGSL_NODE_EXPR_LITERAL_BOOL) {
        *out = wgsl_lit_bool(n);
        return 1;
    }
    return 0;
}

static uint64_t fnv_span(const char *src, size_t src_len, uint32_t o, uint32_t l) {
    uint64_t h = 14695981039346656037ull;
    if (!src || (size_t)o + (size_t)l > src_len) return h;
    for (uint32_t i = 0; i < l; i++) {
        h ^= (unsigned char)src[o + i];
        h *= 1099511628211ull;
    }
    return h;
}

typedef struct { uint32_t off, len; } Span;

typedef struct {
    uint32_t start, end;
    char    *text;
} Edit;

typedef struct {
    Edit *v;
    int n, cap;
} Edits;

static char *opt_strndup(const char *s, size_t n) {
    char *out = (char *)malloc(n + 1);
    if (!out) return NULL;
    if (n) memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

static char *opt_strdup(const char *s) {
    return opt_strndup(s ? s : "", s ? strlen(s) : 0);
}

static void edits_free(Edits *e) {
    if (!e) return;
    for (int i = 0; i < e->n; i++) free(e->v[i].text);
    free(e->v);
    memset(e, 0, sizeof *e);
}

static int edit_push_owned(Edits *e, uint32_t start, uint32_t end, char *text) {
    if (!text) return 0;
    if (e->n + 1 > e->cap) {
        int nc = e->cap ? e->cap * 2 : 8;
        Edit *g = (Edit *)realloc(e->v, (size_t)nc * sizeof *g);
        if (!g) { free(text); return 0; }
        e->v = g; e->cap = nc;
    }
    e->v[e->n++] = (Edit){ start, end, text };
    return 1;
}

static int edit_push_copy(Edits *e, uint32_t start, uint32_t end,
                          const char *text) {
    return edit_push_owned(e, start, end, opt_strdup(text));
}

static int edit_push_slice(Edits *e, uint32_t start, uint32_t end,
                           const char *src, uint32_t off, uint32_t len) {
    return edit_push_owned(e, start, end, opt_strndup(src + off, len));
}

static int edit_cmp(const void *a, const void *b) {
    const Edit *x = (const Edit *)a, *y = (const Edit *)b;
    if (x->start < y->start) return -1;
    if (x->start > y->start) return 1;
    uint32_t xl = x->end - x->start;
    uint32_t yl = y->end - y->start;
    if (xl > yl) return -1; /* wider replacement wins overlap ties */
    if (xl < yl) return 1;
    return 0;
}

static char *apply_edits(const char *src, Edits *edits) {
    size_t n = strlen(src);
    if (!edits || edits->n == 0) {
        char *c = (char *)malloc(n + 1);
        if (c) { memcpy(c, src, n); c[n] = 0; }
        return c;
    }
    qsort(edits->v, (size_t)edits->n, sizeof edits->v[0], edit_cmp);

    size_t cap = n + 1;
    for (int i = 0; i < edits->n; i++)
        cap += edits->v[i].text ? strlen(edits->v[i].text) : 0;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;

    size_t cursor = 0, o = 0;
    for (int i = 0; i < edits->n; i++) {
        Edit *ed = &edits->v[i];
        if (ed->start > n) continue;
        if (ed->end > n) ed->end = (uint32_t)n;
        if (ed->end < ed->start) continue;
        if ((size_t)ed->start < cursor) continue; /* overlapped by earlier edit */
        if ((size_t)ed->start > cursor) {
            size_t chunk = (size_t)ed->start - cursor;
            memcpy(out + o, src + cursor, chunk);
            o += chunk;
        }
        if (ed->text) {
            size_t tn = strlen(ed->text);
            memcpy(out + o, ed->text, tn);
            o += tn;
        }
        cursor = ed->end;
    }
    if (cursor < n) {
        memcpy(out + o, src + cursor, n - cursor);
        o += n - cursor;
    }
    out[o] = 0;
    return out;
}

static int span_ok(const char *src, size_t n, uint32_t off, uint32_t len) {
    (void)src;
    return (size_t)off <= n && (size_t)len <= n - (size_t)off;
}

static Span line_delete_span(const char *src, size_t n,
                             uint32_t off, uint32_t len) {
    size_t start = off;
    size_t end = (size_t)off + len;
    size_t line = start;
    while (line > 0 && src[line - 1] != '\n') line--;
    int only_indent = 1;
    for (size_t i = line; i < start; i++) {
        if (src[i] != ' ' && src[i] != '\t') { only_indent = 0; break; }
    }
    if (only_indent) start = line;
    while (end < n && (src[end] == ' ' || src[end] == '\t')) end++;
    if (end < n && src[end] == '\n') end++;
    return (Span){ (uint32_t)start, (uint32_t)(end - start) };
}

static int node_expr_safe_pure(const WGSLNode *n) {
    if (!n) return 0;
    switch (n->kind) {
    case WGSL_NODE_EXPR_LITERAL_BOOL:
    case WGSL_NODE_EXPR_LITERAL_INT:
    case WGSL_NODE_EXPR_LITERAL_FLOAT:
    case WGSL_NODE_EXPR_IDENT:
        return 1;
    case WGSL_NODE_EXPR_PAREN:
    case WGSL_NODE_EXPR_UNARY:
    case WGSL_NODE_EXPR_BINARY:
        for (uint32_t i = 0; i < n->child_count; i++)
            if (!node_expr_safe_pure(n->children[i])) return 0;
        return 1;
    default:
        return 0;
    }
}

static void collect_dce_edits(Edits *edits, const char *src, size_t n,
                              const WGSLNode *compound) {
    if (!compound || compound->kind != WGSL_NODE_STMT_COMPOUND) return;
    int dead = 0;
    for (uint32_t i = 0; i < compound->child_count; i++) {
        const WGSLNode *s = compound->children[i];
        if (!s || s->kind == WGSL_NODE_ATTRIBUTE) continue;
        if (dead) {
            if (span_ok(src, n, s->span_offset, s->span_length)) {
                Span d = line_delete_span(src, n, s->span_offset, s->span_length);
                (void)edit_push_copy(edits, d.off, d.off + d.len, "");
            }
            continue;
        }
        if (node_is_terminal(s)) dead = 1;
        if (s->kind == WGSL_NODE_STMT_COMPOUND)
            collect_dce_edits(edits, src, n, s);
        else {
            for (uint32_t k = 0; k < s->child_count; k++)
                if (s->children[k] &&
                    s->children[k]->kind == WGSL_NODE_STMT_COMPOUND)
                    collect_dce_edits(edits, src, n, s->children[k]);
        }
    }
}

static const WGSLNode *first_compound_child(const WGSLNode *n) {
    if (!n) return NULL;
    if (n->kind == WGSL_NODE_STMT_COMPOUND) return n;
    for (uint32_t i = 0; i < n->child_count; i++)
        if (n->children[i] && n->children[i]->kind == WGSL_NODE_STMT_COMPOUND)
            return n->children[i];
    return NULL;
}

static int compound_inner_span(const char *src, size_t n,
                               const WGSLNode *compound,
                               uint32_t *off, uint32_t *len) {
    if (!compound || compound->kind != WGSL_NODE_STMT_COMPOUND ||
        !span_ok(src, n, compound->span_offset, compound->span_length))
        return 0;
    uint32_t start = compound->span_offset;
    uint32_t end = compound->span_offset + compound->span_length;
    while (start < end && src[start] != '{') start++;
    while (end > start && src[end - 1] != '}') end--;
    if (start >= end) return 0;
    start++;
    end--;
    *off = start;
    *len = end >= start ? end - start : 0;
    return 1;
}

static void collect_const_branch_edits(Edits *edits, const char *src, size_t n,
                                       const WGSLNode *node) {
    if (!node) return;
    if (node->kind == WGSL_NODE_STMT_IF && node->child_count >= 2) {
        int v = 0;
        if (literal_bool(node->children[0], &v) &&
            span_ok(src, n, node->span_offset, node->span_length))
        {
            const WGSLNode *chosen = NULL;
            if (v) {
                chosen = first_compound_child(node->children[1]);
            } else {
                for (uint32_t i = 2; i < node->child_count; i++) {
                    const WGSLNode *c = node->children[i];
                    if (c && c->kind == WGSL_NODE_STMT_ELSE) {
                        chosen = first_compound_child(c);
                        break;
                    }
                    if (c && c->kind == WGSL_NODE_STMT_ELSE_IF) {
                        chosen = NULL; /* preserve chained condition */
                        break;
                    }
                }
            }
            if (chosen) {
                uint32_t bo = 0, bl = 0;
                if (compound_inner_span(src, n, chosen, &bo, &bl))
                    (void)edit_push_slice(edits, node->span_offset,
                                          node->span_offset + node->span_length,
                                          src, bo, bl);
            } else if (!v) {
                Span d = line_delete_span(src, n, node->span_offset,
                                          node->span_length);
                (void)edit_push_copy(edits, d.off, d.off + d.len, "");
            }
            return; /* do not also rewrite dead children inside this span */
        }
    }
    for (uint32_t i = 0; i < node->child_count; i++)
        collect_const_branch_edits(edits, src, n, node->children[i]);
}

static void collect_local_decls(const WGSLResult *r, const WGSLNode *n,
                                UseTab *tab, int pure_only) {
    if (!n) return;
    if (n->kind == WGSL_NODE_DECL_LET || n->kind == WGSL_NODE_DECL_VAR) {
        WGSLSymbol *s = wgsl_resolver_symbol_for_decl(&r->res, n);
        int safe = 1;
        if (pure_only) {
            safe = 0;
            if (n->kind == WGSL_NODE_DECL_VAR && n->child_count == 0) {
                safe = 1;
            } else if (n->child_count > 0) {
                const WGSLNode *init = n->children[n->child_count - 1];
                safe = node_expr_safe_pure(init);
            }
        }
        if (s && safe) {
            UseRec u = {
                .sym = s, .off = n->span_offset, .len = n->span_length,
                .uses = 0, .is_fn = 0, .is_entry = 0,
            };
            (void)use_push(tab, u);
        }
    }
    for (uint32_t i = 0; i < n->child_count; i++)
        collect_local_decls(r, n->children[i], tab, pure_only);
}

static void collect_unused_edits(Edits *edits, const char *src, size_t n,
                                 const WGSLResult *r) {
    if (!r || !r->ast.root) return;
    UseTab tab = {0};
    const WGSLNode *tu = r->ast.root;
    int has_entry = module_has_entry(r);
    for (uint32_t i = 0; i < tu->child_count; i++) {
        const WGSLNode *node = tu->children[i];
        if (!node) continue;
        if (node->kind == WGSL_NODE_DECL_FUNCTION) {
            WGSLSymbol *s = wgsl_resolver_symbol_for_decl(&r->res, node);
            if (s) {
                UseRec u = {
                    .sym = s, .off = node->span_offset,
                    .len = node->span_length, .uses = 0,
                    .is_fn = 1, .is_entry = fn_is_entry(r, node),
                };
                (void)use_push(&tab, u);
            }
            for (uint32_t k = 0; k < node->child_count; k++)
                if (node->children[k] &&
                    node->children[k]->kind == WGSL_NODE_STMT_COMPOUND)
                    collect_local_decls(r, node->children[k], &tab, 1);
        }
    }
    count_uses(&tab, tu);
    for (int i = 0; i < tab.n; i++) {
        UseRec *u = &tab.v[i];
        if (u->is_fn && (u->is_entry || !has_entry)) continue;
        if (u->uses > 0) continue;
        if (!span_ok(src, n, u->off, u->len)) continue;
        Span d = line_delete_span(src, n, u->off, u->len);
        (void)edit_push_copy(edits, d.off, d.off + d.len, "");
    }
    free(tab.v);
}

static char *apply_non_cse(const char *src) {
    WGSLResult *r = wgsl_check(src);
    if (!r || !r->ast.root) {
        if (r) wgsl_free(r);
        return opt_strdup(src);
    }
    Edits edits = {0};
    size_t n = strlen(src);
    const WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        const WGSLNode *fn = tu->children[i];
        if (!fn || fn->kind != WGSL_NODE_DECL_FUNCTION) continue;
        for (uint32_t k = 0; k < fn->child_count; k++)
            if (fn->children[k] &&
                fn->children[k]->kind == WGSL_NODE_STMT_COMPOUND)
                collect_dce_edits(&edits, src, n, fn->children[k]);
    }
    collect_unused_edits(&edits, src, n, r);
    collect_const_branch_edits(&edits, src, n, tu);
    wgsl_free(r);
    char *out = apply_edits(src, &edits);
    edits_free(&edits);
    return out;
}

typedef struct {
    uint32_t expr_off, expr_len, stmt_off;
} CseOcc;

typedef struct {
    char   *expr;
    uint32_t len;
    uint64_t hash;
    CseOcc *occ;
    int n, cap;
} CseGroup;

typedef struct {
    CseGroup *v;
    int n, cap;
} CseGroups;

static void cse_groups_free(CseGroups *g) {
    if (!g) return;
    for (int i = 0; i < g->n; i++) {
        free(g->v[i].expr);
        free(g->v[i].occ);
    }
    free(g->v);
    memset(g, 0, sizeof *g);
}

static int cse_occ_push(CseGroup *g, uint32_t expr_off, uint32_t expr_len,
                        uint32_t stmt_off) {
    if (g->n + 1 > g->cap) {
        int nc = g->cap ? g->cap * 2 : 4;
        CseOcc *v = (CseOcc *)realloc(g->occ, (size_t)nc * sizeof *v);
        if (!v) return 0;
        g->occ = v; g->cap = nc;
    }
    g->occ[g->n++] = (CseOcc){ expr_off, expr_len, stmt_off };
    return 1;
}

static int cse_group_push(CseGroups *groups, const char *src, size_t n,
                          const WGSLNode *expr, uint32_t stmt_off) {
    if (!span_ok(src, n, expr->span_offset, expr->span_length)) return 1;
    uint64_t h = fnv_span(src, n, expr->span_offset, expr->span_length);
    for (int i = 0; i < groups->n; i++) {
        CseGroup *g = &groups->v[i];
        if (g->hash == h && g->len == expr->span_length &&
            memcmp(g->expr, src + expr->span_offset, expr->span_length) == 0)
            return cse_occ_push(g, expr->span_offset, expr->span_length, stmt_off);
    }
    if (groups->n + 1 > groups->cap) {
        int nc = groups->cap ? groups->cap * 2 : 8;
        CseGroup *v = (CseGroup *)realloc(groups->v, (size_t)nc * sizeof *v);
        if (!v) return 0;
        groups->v = v; groups->cap = nc;
    }
    CseGroup *g = &groups->v[groups->n++];
    memset(g, 0, sizeof *g);
    g->expr = opt_strndup(src + expr->span_offset, expr->span_length);
    if (!g->expr) return 0;
    g->len = expr->span_length;
    g->hash = h;
    return cse_occ_push(g, expr->span_offset, expr->span_length, stmt_off);
}

static void collect_cse_exprs(CseGroups *groups, const char *src, size_t n,
                              const WGSLNode *expr, uint32_t stmt_off) {
    if (!expr) return;
    if (expr->kind == WGSL_NODE_EXPR_BINARY && expr->span_length >= 3 &&
        node_expr_safe_pure(expr))
        (void)cse_group_push(groups, src, n, expr, stmt_off);
    for (uint32_t i = 0; i < expr->child_count; i++)
        collect_cse_exprs(groups, src, n, expr->children[i], stmt_off);
}

static void collect_cse_compound(CseGroups *groups, const char *src, size_t n,
                                 const WGSLNode *compound) {
    if (!compound || compound->kind != WGSL_NODE_STMT_COMPOUND) return;
    for (uint32_t i = 0; i < compound->child_count; i++) {
        const WGSLNode *stmt = compound->children[i];
        if (!stmt || stmt->kind == WGSL_NODE_ATTRIBUTE) continue;
        collect_cse_exprs(groups, src, n, stmt, stmt->span_offset);
        for (uint32_t k = 0; k < stmt->child_count; k++) {
            if (stmt->children[k] &&
                stmt->children[k]->kind == WGSL_NODE_STMT_COMPOUND)
                collect_cse_compound(groups, src, n, stmt->children[k]);
        }
    }
}

static int span_overlaps(uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    return a0 < b1 && b0 < a1;
}

static int cse_group_has_overlaps(const CseGroup *g) {
    for (int i = 0; i < g->n; i++)
        for (int k = i + 1; k < g->n; k++)
            if (span_overlaps(g->occ[i].expr_off, g->occ[i].expr_off + g->occ[i].expr_len,
                              g->occ[k].expr_off, g->occ[k].expr_off + g->occ[k].expr_len))
                return 1;
    return 0;
}

static char *make_cse_insert_text(const char *src, uint32_t stmt_off,
                                  const char *name, const char *expr) {
    size_t line = stmt_off;
    while (line > 0 && src[line - 1] != '\n') line--;
    size_t indent_len = (size_t)stmt_off - line;
    size_t name_len = strlen(name), expr_len = strlen(expr);
    size_t total = strlen("let  = ;\n") + name_len + expr_len + indent_len + 1;
    char *out = (char *)malloc(total);
    if (!out) return NULL;
    size_t o = 0;
    o += (size_t)snprintf(out + o, total - o, "let %s = %s;\n", name, expr);
    if (indent_len) {
        memcpy(out + o, src + line, indent_len);
        o += indent_len;
    }
    out[o] = 0;
    return out;
}

static char *apply_one_cse(const char *src, int temp_index, int *changed) {
    *changed = 0;
    WGSLResult *r = wgsl_check(src);
    if (!r || !r->ast.root) {
        if (r) wgsl_free(r);
        return opt_strdup(src);
    }
    CseGroups groups = {0};
    size_t n = strlen(src);
    const WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        const WGSLNode *fn = tu->children[i];
        if (!fn || fn->kind != WGSL_NODE_DECL_FUNCTION) continue;
        for (uint32_t k = 0; k < fn->child_count; k++)
            if (fn->children[k] &&
                fn->children[k]->kind == WGSL_NODE_STMT_COMPOUND)
                collect_cse_compound(&groups, src, n, fn->children[k]);
    }
    wgsl_free(r);

    int best = -1;
    for (int i = 0; i < groups.n; i++) {
        if (groups.v[i].n < 2) continue;
        if (cse_group_has_overlaps(&groups.v[i])) continue;
        if (best < 0 || groups.v[i].len > groups.v[best].len ||
            (groups.v[i].len == groups.v[best].len &&
             groups.v[i].n > groups.v[best].n))
            best = i;
    }
    if (best < 0) {
        cse_groups_free(&groups);
        return opt_strdup(src);
    }

    char name[48];
    snprintf(name, sizeof name, "wgsl_opt_cse%d", temp_index);
    while (strstr(src, name) != NULL && temp_index < 1000)
        snprintf(name, sizeof name, "wgsl_opt_cse%d", ++temp_index);

    CseGroup *g = &groups.v[best];
    Edits edits = {0};
    char *ins = make_cse_insert_text(src, g->occ[0].stmt_off, name, g->expr);
    (void)edit_push_owned(&edits, g->occ[0].stmt_off, g->occ[0].stmt_off, ins);
    for (int i = 0; i < g->n; i++)
        (void)edit_push_copy(&edits, g->occ[i].expr_off,
                             g->occ[i].expr_off + g->occ[i].expr_len, name);
    char *out = apply_edits(src, &edits);
    edits_free(&edits);
    cse_groups_free(&groups);
    if (out) *changed = strcmp(out, src) != 0;
    return out;
}

char *wgsl_optimize_apply(const char *src) {
    if (!src) src = "";
    char *cur = apply_non_cse(src);
    if (!cur) return NULL;
    for (int i = 0; i < 8; i++) {
        int changed = 0;
        char *next = apply_one_cse(cur, i, &changed);
        if (!next) { free(cur); return NULL; }
        free(cur);
        cur = next;
        if (!changed) break;
    }
    return cur;
}
