/**
 * @file analyze.c — static optimization analysis passes.
 *
 * Static analyses over a checked `WGSLResult`:
 *   - DCE: statements after unconditional return/discard
 *   - unused: locals never read; non-entry functions never called
 *   - const-branch: if (true|false) with literal condition
 *   - CSE: repeated pure expression spellings inside a function
 *   - loop: const-bounded for-loops → trip-count hints
 *
 * Emits a JSON report.  `wgsl_optimize_apply` runs a small source-preserving
 * rewrite pipeline for the safe subset: unreachable siblings after
 * return/discard, unused pure locals/uncalled non-entry functions,
 * literal-bool branches, and repeated pure expression spellings.
 */
#include "wgsl.h"
#include "internal/result.h"
#include "internal/ast.h"
#include "internal/resolver.h"
#include "internal/check.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* tiny JSON builder. */

typedef struct {
    char  *buf;
    size_t len, cap;
    int    oom;
} JB;

static void jb_grow(JB *j, size_t n) {
    if (j->oom) return;
    if (j->len + n + 1 <= j->cap) return;
    size_t nc = j->cap ? j->cap : 256;
    while (nc < j->len + n + 1) {
        if (nc > SIZE_MAX / 2) { j->oom = 1; return; }
        nc *= 2;
    }
    char *g = (char *)realloc(j->buf, nc);
    if (!g) { j->oom = 1; return; }
    j->buf = g; j->cap = nc;
}

static void jb_putn(JB *j, const char *s, size_t n) {
    if (!s || j->oom) return;
    jb_grow(j, n);
    if (j->oom) return;
    memcpy(j->buf + j->len, s, n);
    j->len += n;
    j->buf[j->len] = 0;
}
static void jb_puts(JB *j, const char *s) { if (s) jb_putn(j, s, strlen(s)); }
static void jb_putc(JB *j, char c) { jb_putn(j, &c, 1); }
static void jb_put_u(JB *j, unsigned long long v) {
    char t[32];
    int n = snprintf(t, sizeof t, "%llu", v);
    if (n > 0) jb_putn(j, t, (size_t)n);
}
static void jb_put_jstr(JB *j, const char *s) {
    jb_putc(j, '"');
    if (!s) { jb_putc(j, '"'); return; }
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '"' || *p == '\\') { jb_putc(j, '\\'); jb_putc(j, (char)*p); }
        else if (*p == '\n') jb_puts(j, "\\n");
        else if (*p == '\t') jb_puts(j, "\\t");
        else if (*p < 0x20) {
            char t[8];
            snprintf(t, sizeof t, "\\u%04x", *p);
            jb_puts(j, t);
        } else jb_putc(j, (char)*p);
    }
    jb_putc(j, '"');
}

/* finding list. */

typedef struct {
    const char *pass;     /* "dce" | "unused" | "const_branch" | "cse" | "loop" */
    const char *kind;
    const char *message;
    uint32_t    offset;
    uint32_t    length;
    int         savings_hint; /* rough estimated ops/bytes saved; 0 = n/a */
    int         message_owned; /* 1 → free(message) in findings_free */
} Finding;

typedef struct {
    Finding *v;
    int n, cap;
    int cnt_dce, cnt_unused, cnt_branch, cnt_cse, cnt_loop;
} Findings;

static int fnd_push(Findings *f, Finding x) {
    if (f->n + 1 > f->cap) {
        int nc = f->cap ? f->cap * 2 : 8;
        Finding *g = (Finding *)realloc(f->v, (size_t)nc * sizeof *g);
        if (!g) return 0;
        f->v = g; f->cap = nc;
    }
    f->v[f->n++] = x;
    if (strcmp(x.pass, "dce") == 0) f->cnt_dce++;
    else if (strcmp(x.pass, "unused") == 0) f->cnt_unused++;
    else if (strcmp(x.pass, "const_branch") == 0) f->cnt_branch++;
    else if (strcmp(x.pass, "cse") == 0) f->cnt_cse++;
    else if (strcmp(x.pass, "loop") == 0) f->cnt_loop++;
    return 1;
}

/* AST helpers. */

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

/* DCE: after return/discard in a compound. */

static void dce_compound(Findings *f, const WGSLNode *compound) {
    if (!compound || compound->kind != WGSL_NODE_STMT_COMPOUND) return;
    int dead = 0;
    for (uint32_t i = 0; i < compound->child_count; i++) {
        const WGSLNode *s = compound->children[i];
        if (!s) continue;
        if (s->kind == WGSL_NODE_ATTRIBUTE) continue;
        if (dead) {
            Finding x = {
                .pass = "dce",
                .kind = "unreachable_after_terminal",
                .message = "unreachable statement after return/discard "
                           "(dead-code elimination candidate)",
                .offset = s->span_offset,
                .length = s->span_length,
                .savings_hint = 1,
            };
            (void)fnd_push(f, x);
            continue;
        }
        if (node_is_terminal(s)) dead = 1;
        /* Recurse into nested compounds / if / loop bodies. */
        if (s->kind == WGSL_NODE_STMT_COMPOUND)
            dce_compound(f, s);
        else {
            for (uint32_t k = 0; k < s->child_count; k++) {
                if (s->children[k] &&
                    s->children[k]->kind == WGSL_NODE_STMT_COMPOUND)
                    dce_compound(f, s->children[k]);
            }
        }
    }
}

static void pass_dce(Findings *f, const WGSLResult *r) {
    if (!r || !r->ast.root) return;
    const WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        const WGSLNode *fn = tu->children[i];
        if (!fn || fn->kind != WGSL_NODE_DECL_FUNCTION) continue;
        for (uint32_t k = 0; k < fn->child_count; k++) {
            if (fn->children[k] &&
                fn->children[k]->kind == WGSL_NODE_STMT_COMPOUND)
                dce_compound(f, fn->children[k]);
        }
    }
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

static void pass_unused(Findings *f, const WGSLResult *r) {
    if (!r || !r->ast.root) return;
    UseTab tab = {0};
    const WGSLNode *tu = r->ast.root;

    for (uint32_t i = 0; i < tu->child_count; i++) {
        const WGSLNode *n = tu->children[i];
        if (!n) continue;
        if (n->kind == WGSL_NODE_DECL_FUNCTION) {
            WGSLSymbol *s = wgsl_resolver_symbol_for_decl(&r->res, n);
            if (!s) continue;
            uint32_t no = wgsl_node_name_span(n).offset;
            uint32_t nl = wgsl_node_name_span(n).length;
            UseRec u = {
                .sym = s, .off = no, .len = nl, .uses = 0,
                .is_fn = 1, .is_entry = fn_is_entry(r, n),
            };
            (void)use_push(&tab, u);
        }
    }

    /* Local decls inside functions */
    for (uint32_t i = 0; i < tu->child_count; i++) {
        const WGSLNode *fn = tu->children[i];
        if (!fn || fn->kind != WGSL_NODE_DECL_FUNCTION) continue;
        for (uint32_t k = 0; k < fn->child_count; k++) {
            const WGSLNode *body = fn->children[k];
            if (!body || body->kind != WGSL_NODE_STMT_COMPOUND) continue;
            /* DFS collect */
            const WGSLNode **stack = NULL;
            int sn = 0, scap = 0;
            #define PUSH(N) do { \
                if (sn + 1 > scap) { \
                    int nc = scap ? scap * 2 : 32; \
                    const WGSLNode **g = (const WGSLNode **)realloc( \
                        stack, (size_t)nc * sizeof *g); \
                    if (!g) break; \
                    stack = g; \
                    scap = nc; \
                } \
                stack[sn++] = (N); \
            } while (0)
            PUSH(body);
            while (sn > 0) {
                const WGSLNode *cur = stack[--sn];
                if (!cur) continue;
                if (cur->kind == WGSL_NODE_DECL_LET ||
                    cur->kind == WGSL_NODE_DECL_VAR)
                {
                    WGSLSymbol *s = wgsl_resolver_symbol_for_decl(&r->res, cur);
                    if (s) {
                        uint32_t no = wgsl_node_name_span(cur).offset;
                        uint32_t nl = wgsl_node_name_span(cur).length;
                        UseRec u = {
                            .sym = s, .off = no, .len = nl, .uses = 0,
                            .is_fn = 0, .is_entry = 0,
                        };
                        (void)use_push(&tab, u);
                    }
                }
                for (uint32_t c = 0; c < cur->child_count; c++)
                    if (cur->children[c]) PUSH(cur->children[c]);
            }
            free(stack);
            #undef PUSH
        }
    }

    /* Count uses over whole TU. */
    count_uses(&tab, tu);

    for (int i = 0; i < tab.n; i++) {
        UseRec *u = &tab.v[i];
        if (u->is_fn) {
            if (u->is_entry) continue;
            if (u->uses > 0) continue; /* called somewhere */
            Finding x = {
                .pass = "unused",
                .kind = "unused_function",
                .message = "function is never called and is not an entry point",
                .offset = u->off,
                .length = u->len,
                .savings_hint = 2,
            };
            (void)fnd_push(f, x);
        } else {
            if (u->uses > 0) continue;
            Finding x = {
                .pass = "unused",
                .kind = "unused_local",
                .message = "local variable is never read "
                           "(dead store / dead-code candidate)",
                .offset = u->off,
                .length = u->len,
                .savings_hint = 1,
            };
            (void)fnd_push(f, x);
        }
    }
    free(tab.v);
}

/* const branch. */

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

static void walk_const_branch(Findings *f, const WGSLNode *n) {
    if (!n) return;
    if (n->kind == WGSL_NODE_STMT_IF && n->child_count > 0) {
        int v = 0;
        if (literal_bool(n->children[0], &v)) {
            Finding x = {
                .pass = "const_branch",
                .kind = v ? "if_true" : "if_false",
                .message = v
                    ? "condition is always true — else branch is dead; "
                      "consider simplifying to the then-body"
                    : "condition is always false — then branch is dead; "
                      "consider simplifying to the else-body (if any)",
                .offset = n->span_offset,
                .length = n->span_length,
                .savings_hint = 2,
            };
            (void)fnd_push(f, x);
        }
    }
    for (uint32_t i = 0; i < n->child_count; i++)
        walk_const_branch(f, n->children[i]);
}

static void pass_const_branch(Findings *f, const WGSLResult *r) {
    if (r && r->ast.root) walk_const_branch(f, r->ast.root);
}

/* CSE: repeated pure expression spellings. */

typedef struct {
    uint32_t off, len;
    uint64_t hash;
    int      count;
    int      reported;
} CseEnt;

static uint64_t fnv_span(const char *src, size_t src_len, uint32_t o, uint32_t l) {
    uint64_t h = 14695981039346656037ull;
    if (!src || (size_t)o + (size_t)l > src_len) return h;
    for (uint32_t i = 0; i < l; i++) {
        h ^= (unsigned char)src[o + i];
        h *= 1099511628211ull;
    }
    return h;
}

static int expr_is_cse_candidate(const WGSLNode *n) {
    if (!n) return 0;
    switch (n->kind) {
    case WGSL_NODE_EXPR_BINARY:
    case WGSL_NODE_EXPR_INDEX:
    case WGSL_NODE_EXPR_MEMBER:
        return n->span_length >= 3; /* skip trivial */
    default:
        return 0;
    }
}

static void cse_collect(const WGSLResult *r, const WGSLNode *n,
                        CseEnt **out, int *n_out, int *cap)
{
    if (!n) return;
    if (expr_is_cse_candidate(n) && r->src_copy) {
        uint64_t h = fnv_span(r->src_copy, r->src_len, n->span_offset, n->span_length);
        int found = -1;
        for (int i = 0; i < *n_out; i++) {
            if ((*out)[i].hash == h &&
                (*out)[i].len == n->span_length &&
                (size_t)(*out)[i].off + (*out)[i].len <= r->src_len &&
                (size_t)n->span_offset + n->span_length <= r->src_len &&
                memcmp(r->src_copy + (*out)[i].off,
                       r->src_copy + n->span_offset, n->span_length) == 0)
            {
                found = i;
                break;
            }
        }
        if (found >= 0) {
            (*out)[found].count++;
        } else {
            if (*n_out + 1 > *cap) {
                int nc = *cap ? *cap * 2 : 16;
                CseEnt *g = (CseEnt *)realloc(*out, (size_t)nc * sizeof *g);
                if (!g) return;
                *out = g; *cap = nc;
            }
            (*out)[*n_out] = (CseEnt){
                .off = n->span_offset, .len = n->span_length,
                .hash = h, .count = 1, .reported = 0,
            };
            (*n_out)++;
        }
    }
    for (uint32_t i = 0; i < n->child_count; i++)
        cse_collect(r, n->children[i], out, n_out, cap);
}

static void pass_cse(Findings *f, const WGSLResult *r) {
    if (!r || !r->ast.root || !r->src_copy) return;
    const WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        const WGSLNode *fn = tu->children[i];
        if (!fn || fn->kind != WGSL_NODE_DECL_FUNCTION) continue;
        CseEnt *ents = NULL;
        int ne = 0, cap = 0;
        for (uint32_t k = 0; k < fn->child_count; k++) {
            if (fn->children[k] &&
                fn->children[k]->kind == WGSL_NODE_STMT_COMPOUND)
                cse_collect(r, fn->children[k], &ents, &ne, &cap);
        }
        for (int e = 0; e < ne; e++) {
            if (ents[e].count < 2) continue;
            Finding x = {
                .pass = "cse",
                .kind = "common_subexpression",
                .message = "expression appears multiple times in this function "
                           "— consider a local let (common-subexpression elimination)",
                .offset = ents[e].off,
                .length = ents[e].len,
                .savings_hint = ents[e].count - 1,
            };
            (void)fnd_push(f, x);
        }
        free(ents);
    }
}

/* loop: const-bounded for. */

static int parse_int_lit(const WGSLResult *r, const WGSLNode *n, long long *out) {
    if (!n || !r || !r->src_copy) return 0;
    if (n->kind == WGSL_NODE_EXPR_PAREN && n->child_count > 0)
        return parse_int_lit(r, n->children[0], out);
    if (n->kind == WGSL_NODE_EXPR_UNARY && n->child_count > 0) {
        /* unary minus */
        long long v = 0;
        if (!parse_int_lit(r, n->children[0], &v)) return 0;
        *out = -v;
        return 1;
    }
    if (n->kind != WGSL_NODE_EXPR_LITERAL_INT) return 0;
    uint32_t o = n->span_offset, l = n->span_length;
    if (l == 0 || (size_t)o + l > r->src_len) return 0;
    long long v = 0;
    int neg = 0;
    uint32_t i = 0;
    if (r->src_copy[o] == '-') { neg = 1; i = 1; }
    for (; i < l; i++) {
        char c = r->src_copy[o + i];
        if (c >= '0' && c <= '9') v = v * 10 + (c - '0');
        else if (c == 'i' || c == 'u') break;
        else if (c == 'x' || c == 'X') return 0; /* hex literal not handled here */
        else return 0;
    }
    *out = neg ? -v : v;
    return 1;
}

static void walk_loops(Findings *f, const WGSLResult *r, const WGSLNode *n) {
    if (!n) return;
    /* Best-effort: STMT_FOR span reprint — look at child structure.
     * Parser for-loop: children vary; we scan source of the for header
     * for `i < N` patterns via AST when available.
     * If node is STMT_FOR, report span as an analyzable loop. */
    if (n->kind == WGSL_NODE_STMT_FOR) {
        /* Try children: often [init?][cond?][update?][body] with flags */
        long long hi = -1;
        for (uint32_t i = 0; i < n->child_count; i++) {
            const WGSLNode *c = n->children[i];
            if (!c) continue;
            if (c->kind == WGSL_NODE_EXPR_BINARY) {
                /* cond: i < N */
                if (c->child_count >= 2) {
                    long long v = 0;
                    if (parse_int_lit(r, c->children[1], &v) && v > 0) {
                        hi = v;
                    }
                }
            }
        }
        char msg[256];
        if (hi > 0) {
            snprintf(msg, sizeof msg,
                     "for-loop upper bound is constant %lld — trip count ≈ %lld; "
                     "consider unrolling or fusing with adjacent loops "
                     "(interp cost model can quantify savings)",
                     hi, hi);
        } else {
            snprintf(msg, sizeof msg,
                     "for-loop — inspect trip bounds; fuse adjacent loops if "
                     "they share the same index domain to cut memory traffic");
        }
        /* Store message on heap... Finding holds const char* — use static
         * buffer per finding by strdup. */
        char *m = (char *)malloc(strlen(msg) + 1);
        if (m) memcpy(m, msg, strlen(msg) + 1);
        Finding x = {
            .pass = "loop",
            .kind = hi > 0 ? "const_trip_count" : "loop_candidate",
            .message = m ? m : "for-loop optimization candidate",
            .offset = n->span_offset,
            .length = n->span_length,
            .savings_hint = hi > 0 ? (int)(hi > 32 ? 4 : 2) : 1,
            .message_owned = m ? 1 : 0,
        };
        (void)fnd_push(f, x);
    }
    if (n->kind == WGSL_NODE_STMT_LOOP) {
        Finding x = {
            .pass = "loop",
            .kind = "loop_candidate",
            .message = "loop — check continuing-block for reduction patterns; "
                       "workgroup/shared memory may replace multi-pass storage traffic",
            .offset = n->span_offset,
            .length = n->span_length,
            .savings_hint = 1,
            .message_owned = 0, /* string literal */
        };
        (void)fnd_push(f, x);
    }
    for (uint32_t i = 0; i < n->child_count; i++)
        walk_loops(f, r, n->children[i]);
}

static void pass_loop(Findings *f, const WGSLResult *r) {
    if (r && r->ast.root) walk_loops(f, r, r->ast.root);
}

/* emit JSON. */

static char *findings_to_json(Findings *f) {
    JB j = {0};
    jb_puts(&j, "{\"passes\":[\"dce\",\"unused\",\"const_branch\",\"cse\",\"loop\"],");
    jb_puts(&j, "\"summary\":{");
    jb_puts(&j, "\"dce\":"); jb_put_u(&j, (unsigned)f->cnt_dce); jb_putc(&j, ',');
    jb_puts(&j, "\"unused\":"); jb_put_u(&j, (unsigned)f->cnt_unused); jb_putc(&j, ',');
    jb_puts(&j, "\"const_branch\":"); jb_put_u(&j, (unsigned)f->cnt_branch); jb_putc(&j, ',');
    jb_puts(&j, "\"cse\":"); jb_put_u(&j, (unsigned)f->cnt_cse); jb_putc(&j, ',');
    jb_puts(&j, "\"loop\":"); jb_put_u(&j, (unsigned)f->cnt_loop);
    jb_puts(&j, "},\"findings\":[");
    for (int i = 0; i < f->n; i++) {
        if (i) jb_putc(&j, ',');
        Finding *x = &f->v[i];
        jb_puts(&j, "{\"pass\":"); jb_put_jstr(&j, x->pass);
        jb_puts(&j, ",\"kind\":"); jb_put_jstr(&j, x->kind);
        jb_puts(&j, ",\"message\":"); jb_put_jstr(&j, x->message);
        jb_puts(&j, ",\"offset\":"); jb_put_u(&j, x->offset);
        jb_puts(&j, ",\"length\":"); jb_put_u(&j, x->length);
        jb_puts(&j, ",\"savings_hint\":"); jb_put_u(&j, (unsigned)x->savings_hint);
        jb_putc(&j, '}');
    }
    jb_puts(&j, "]}");
    if (j.oom) { free(j.buf); return NULL; }
    if (!j.buf) {
        j.buf = (char *)malloc(3);
        if (j.buf) { j.buf[0] = '{'; j.buf[1] = '}'; j.buf[2] = 0; }
    }
    return j.buf;
}

static void findings_free(Findings *f) {
    for (int i = 0; i < f->n; i++) {
        /* Free by owned flag, never by scanning message text. */
        if (f->v[i].message_owned && f->v[i].message)
            free((void *)f->v[i].message);
    }
    free(f->v);
    memset(f, 0, sizeof *f);
}

static char *run_analysis(const WGSLResult *r) {
    Findings f = {0};
    pass_dce(&f, r);
    pass_unused(&f, r);
    pass_const_branch(&f, r);
    pass_cse(&f, r);
    pass_loop(&f, r);
    char *json = findings_to_json(&f);
    findings_free(&f);
    return json;
}

/* public API. */

char *wgsl_optimize_json(const WGSLResult *r) {
    if (!r || !r->ast.root) {
        char *e = (char *)malloc(48);
        if (e) memcpy(e, "{\"passes\":[],\"summary\":{},\"findings\":[]}", 41);
        return e;
    }
    return run_analysis(r);
}

char *wgsl_optimize_json_src(const char *src) {
    WGSLResult *r = wgsl_check(src ? src : "");
    if (!r) return NULL;
    char *j = wgsl_optimize_json(r);
    wgsl_free(r);
    return j;
}
