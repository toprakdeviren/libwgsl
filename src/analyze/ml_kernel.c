/**
 * @file ml_kernel.c — ML-kernel-oriented static analysis.
 *
 * Produces a JSON report for compute (and other) modules:
 *   - binding layout (group/binding → type + address space)
 *   - shape hints (array lengths, vec widths on resources)
 *   - pattern tags (softmax, attention-ish, layernorm, gelu/silu/relu,
 *     matmul-ish, reduction, elementwise)
 *   - workgroup_size autotune suggestions (power-of-two sweep)
 *
 * Heuristic / structural — not a full tensor IR.  Free with
 * `wgsl_free_string`.
 */
#include "wgsl.h"
#include "internal/result.h"
#include "internal/ast.h"
#include "internal/resolver.h"
#include "internal/types.h"
#include "internal/check.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
static void jb_put_i(JB *j, long long v) {
    char t[32];
    int n = snprintf(t, sizeof t, "%lld", v);
    if (n > 0) jb_putn(j, t, (size_t)n);
}
static void jb_put_jstr(JB *j, const char *s) {
    jb_putc(j, '"');
    if (!s) { jb_putc(j, '"'); return; }
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '"' || *p == '\\') { jb_putc(j, '\\'); jb_putc(j, (char)*p); }
        else if (*p == '\n') jb_puts(j, "\\n");
        else if (*p < 0x20) {
            char t[8]; snprintf(t, sizeof t, "\\u%04x", *p); jb_puts(j, t);
        } else jb_putc(j, (char)*p);
    }
    jb_putc(j, '"');
}
static void jb_put_src(JB *j, const char *src, size_t src_len,
                       uint32_t o, uint32_t l) {
    jb_putc(j, '"');
    if (src && (size_t)o + (size_t)l <= src_len) {
        for (uint32_t i = 0; i < l; i++) {
            unsigned char c = (unsigned char)src[o + i];
            if (c == '"' || c == '\\') { jb_putc(j, '\\'); jb_putc(j, (char)c); }
            else if (c == '\n') jb_puts(j, "\\n");
            else if (c < 0x20) {
                char t[8]; snprintf(t, sizeof t, "\\u%04x", c); jb_puts(j, t);
            } else jb_putc(j, (char)c);
        }
    }
    jb_putc(j, '"');
}

/* pattern flags. */

enum {
    PAT_SOFTMAX     = 1u << 0,
    PAT_ATTENTION   = 1u << 1,
    PAT_LAYERNORM   = 1u << 2,
    PAT_GELU        = 1u << 3,
    PAT_SILU        = 1u << 4,
    PAT_RELU        = 1u << 5,
    PAT_MATMUL      = 1u << 6,
    PAT_REDUCTION   = 1u << 7,
    PAT_ELEMENTWISE = 1u << 8,
    PAT_EMBEDDING   = 1u << 9,
    PAT_ROPE        = 1u << 10,
};

typedef struct {
    int      group, binding;
    uint32_t name_off, name_len;
    char     as_name[24];
    char     access_name[16];
    char     type_repr[64];
    int      array_len;   /* 0 = runtime / unknown */
    int      vec_width;   /* 0 = scalar / unknown */
    int      access;
    int      is_atomic;
    int      is_f16;
} Binding;

typedef struct {
    Binding *v;
    int n, cap;
} BindTab;

typedef struct {
    uint32_t patterns;
    int has_exp, has_log, has_rsqrt, has_sqrt;
    int has_dot, has_mul, has_div, has_max, has_min;
    int has_subgroup, has_atomic, has_barrier;
    int has_sin, has_cos, has_tanh;
    int workgroup[3];
    int has_workgroup;
    int is_compute;
    char entry[64];
    int call_exp, call_sum_like, call_normalize;
} Scan;

typedef struct {
    int seq_len;
    int head_dim;
    int softmax_axis;
    int evidence_bindings;
    int matmul_m;
    int matmul_n;
    int matmul_k;
    int matmul_broadcast;
    int matmul_evidence_bindings;
} ShapeFlow;

static int bt_push(BindTab *t, Binding b) {
    if (t->n + 1 > t->cap) {
        int nc = t->cap ? t->cap * 2 : 8;
        Binding *g = (Binding *)realloc(t->v, (size_t)nc * sizeof *g);
        if (!g) return 0;
        t->v = g; t->cap = nc;
    }
    t->v[t->n++] = b;
    return 1;
}

static int name_eq(const char *src, size_t src_len,
                   uint32_t o, uint32_t l, const char *s) {
    size_t n = strlen(s);
    return l == n && src && (size_t)o + n <= src_len &&
           memcmp(src + o, s, n) == 0;
}

static long long lit_int(const char *src, size_t src_len, const WGSLNode *n) {
    if (!n || n->kind != WGSL_NODE_EXPR_LITERAL_INT || !src) return -1;
    uint32_t o = n->span_offset, l = n->span_length;
    if ((size_t)o + l > src_len) return -1;
    long long v = 0;
    for (uint32_t i = 0; i < l; i++) {
        char c = src[o + i];
        if (c >= '0' && c <= '9') v = v * 10 + (c - '0');
        else break;
    }
    return v;
}

static void binding_set_type_repr(Binding *b, const char *s) {
    if (!b) return;
    if (!s) s = "";
    size_t n = strlen(s);
    if (n >= sizeof b->type_repr) n = sizeof b->type_repr - 1;
    memcpy(b->type_repr, s, n);
    b->type_repr[n] = '\0';
}

static void type_shape(const WGSLTypeInfo *t, Binding *b) {
    if (!t) return;
    if (t->kind == WGSL_TYPE_F16) b->is_f16 = 1;
    if (t->kind == WGSL_TYPE_ATOMIC) {
        b->is_atomic = 1;
        snprintf(b->type_repr, sizeof b->type_repr, "atomic");
        return;
    }
    if (t->kind == WGSL_TYPE_ARRAY) {
        b->array_len = (int)t->array_len;
        if (t->ref) type_shape((const WGSLTypeInfo *)t->ref, b);
        char tmp[128];
        snprintf(tmp, sizeof tmp, "array<%s,%d>",
                 b->type_repr[0] ? b->type_repr : "T",
                 b->array_len);
        binding_set_type_repr(b, tmp);
        return;
    }
    if (t->kind == WGSL_TYPE_VEC) {
        b->vec_width = (int)t->width;
        const char *el = "f32";
        if (t->ref) {
            WGSLTypeKind ek = (WGSLTypeKind)((const WGSLTypeInfo *)t->ref)->kind;
            if (ek == WGSL_TYPE_I32) el = "i32";
            else if (ek == WGSL_TYPE_U32) el = "u32";
            else if (ek == WGSL_TYPE_F16) { el = "f16"; b->is_f16 = 1; }
            else if (ek == WGSL_TYPE_BOOL) el = "bool";
        }
        snprintf(b->type_repr, sizeof b->type_repr, "vec%d<%s>",
                 b->vec_width, el);
        return;
    }
    if (t->kind == WGSL_TYPE_F32 || t->kind == WGSL_TYPE_ABSTRACT_FLOAT)
        snprintf(b->type_repr, sizeof b->type_repr, "f32");
    else if (t->kind == WGSL_TYPE_I32 || t->kind == WGSL_TYPE_ABSTRACT_INT)
        snprintf(b->type_repr, sizeof b->type_repr, "i32");
    else if (t->kind == WGSL_TYPE_U32)
        snprintf(b->type_repr, sizeof b->type_repr, "u32");
    else if (t->kind == WGSL_TYPE_BOOL)
        snprintf(b->type_repr, sizeof b->type_repr, "bool");
    else
        snprintf(b->type_repr, sizeof b->type_repr, "%s",
                 wgsl_type_kind_name((WGSLTypeKind)t->kind));
}

static int binding_tensor_rank(const Binding *b) {
    if (!b) return 0;
    if (b->array_len > 0 && b->vec_width > 0) return 2;
    if (b->array_len > 0 || b->vec_width > 0) return 1;
    return 0;
}

static void derive_shape_flow(const BindTab *bt, const Scan *sc, ShapeFlow *sf) {
    memset(sf, 0, sizeof *sf);
    sf->softmax_axis = -1;
    if (!bt || !sc) return;

    if (sc->patterns & PAT_ATTENTION) {
        int seq = 0, head = 0, matching = 0;
        for (int i = 0; i < bt->n; i++) {
            const Binding *b = &bt->v[i];
            if (strcmp(b->as_name, "storage") != 0) continue;
            if (binding_tensor_rank(b) != 2) continue;
            if (!seq) {
                seq = b->array_len;
                head = b->vec_width;
                matching = 1;
            } else if (b->array_len == seq && b->vec_width == head) {
                matching++;
            }
        }

        if (matching >= 2) {
            sf->seq_len = seq;
            sf->head_dim = head;
            sf->softmax_axis = 0; /* shape is [seq_len, head_dim] */
            sf->evidence_bindings = matching;
        }
    }

    if (!(sc->patterns & PAT_MATMUL)) return;

    const Binding *a = NULL, *b = NULL, *out = NULL;
    for (int i = 0; i < bt->n; i++) {
        const Binding *cur = &bt->v[i];
        if (strcmp(cur->as_name, "storage") != 0) continue;
        if (binding_tensor_rank(cur) != 2) continue;
        if (cur->access == WGSL_ACCESS_READ) {
            if (!a) a = cur;
            else if (!b) b = cur;
        } else if (cur->access == WGSL_ACCESS_READ_WRITE && !out) {
            out = cur;
        }
    }

    if (a && b && out) {
        int m = a->array_len;
        int k = a->vec_width;
        int n = b->vec_width;
        int broadcast = 0;
        int compatible_k = (b->array_len == k);
        if (!compatible_k && b->array_len == 1) {
            compatible_k = 1;
            broadcast = 1;
        }
        if (compatible_k && out->array_len == m && out->vec_width == n) {
            sf->matmul_m = m;
            sf->matmul_n = n;
            sf->matmul_k = k;
            sf->matmul_broadcast = broadcast;
            sf->matmul_evidence_bindings = 3;
        }
    }
}

static void as_name(uint8_t as, char *out, size_t n) {
    const char *s = "unknown";
    switch (as) {
    case WGSL_AS_STORAGE: s = "storage"; break;
    case WGSL_AS_UNIFORM: s = "uniform"; break;
    case WGSL_AS_WORKGROUP: s = "workgroup"; break;
    case WGSL_AS_PRIVATE: s = "private"; break;
    case WGSL_AS_FUNCTION: s = "function"; break;
    case WGSL_AS_HANDLE: s = "handle"; break;
    default: break;
    }
    snprintf(out, n, "%s", s);
}

static int attr_int(const char *src, size_t src_len, const WGSLNode *attr) {
    if (!attr || attr->child_count < 2) return -1;
    return (int)lit_int(src, src_len, attr->children[1]);
}

static int attr_is(const char *src, size_t src_len, const WGSLNode *attr,
                   const char *name) {
    if (!attr || attr->child_count == 0) return 0;
    const WGSLNode *nm = attr->children[0];
    if (!nm || nm->kind != WGSL_NODE_EXPR_IDENT) return 0;
    uint32_t o = wgsl_node_name_span(nm).offset;
    uint32_t l = wgsl_node_name_span(nm).length;
    return name_eq(src, src_len, o, l, name);
}

static void collect_bindings(const WGSLResult *r, BindTab *bt) {
    if (!r || !r->ast.root || !r->src_copy) return;
    const WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        const WGSLNode *n = tu->children[i];
        if (!n || n->kind != WGSL_NODE_DECL_VAR) continue;
        WGSLSymbol *sym = wgsl_resolver_symbol_for_decl(&r->res, n);
        if (!sym) continue;
        if (sym->as != WGSL_AS_STORAGE && sym->as != WGSL_AS_UNIFORM &&
            sym->as != WGSL_AS_HANDLE && sym->as != WGSL_AS_WORKGROUP)
            continue;

        Binding b;
        memset(&b, 0, sizeof b);
        b.group = b.binding = -1;
        b.name_off = wgsl_node_name_span(n).offset;
        b.name_len = wgsl_node_name_span(n).length;
        as_name(sym->as, b.as_name, sizeof b.as_name);
        b.access = sym->am;
        snprintf(b.access_name, sizeof b.access_name, "%s",
                 wgsl_access_mode_name((WGSLAccessMode)sym->am));
        if (sym->type) type_shape(sym->type, &b);

        uint32_t attrs = wgsl_varlike_attr_count(n);
        for (uint32_t a = 0; a < attrs && a < n->child_count; a++) {
            const WGSLNode *at = n->children[a];
            if (attr_is(r->src_copy, r->src_len, at, "group"))
                b.group = attr_int(r->src_copy, r->src_len, at);
            if (attr_is(r->src_copy, r->src_len, at, "binding"))
                b.binding = attr_int(r->src_copy, r->src_len, at);
        }
        (void)bt_push(bt, b);
    }
}

static int nm_is(const char *nm, uint32_t nl, const char *s) {
    size_t n = strlen(s);
    return nl == n && nm && memcmp(nm, s, n) == 0;
}

static void scan_calls(const WGSLResult *r, const WGSLNode *n, Scan *sc, int depth) {
    if (!n || !r || !r->src_copy || depth > 64) return;
    if (n->kind == WGSL_NODE_EXPR_CALL && n->child_count > 0) {
        const WGSLNode *cal = n->children[0];
        const char *nm = NULL;
        uint32_t nl = 0;
        if (cal && cal->kind == WGSL_NODE_EXPR_IDENT) {
            uint32_t o = wgsl_node_name_span(cal).offset;
            nl = wgsl_node_name_span(cal).length;
            if ((size_t)o + nl <= r->src_len) nm = r->src_copy + o;
        } else if (cal && cal->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
                   cal->child_count > 0 && cal->children[0] &&
                   cal->children[0]->kind == WGSL_NODE_EXPR_IDENT) {
            uint32_t o = wgsl_node_name_span(cal->children[0]).offset;
            nl = wgsl_node_name_span(cal->children[0]).length;
            if ((size_t)o + nl <= r->src_len) nm = r->src_copy + o;
        }
        if (nm && nl) {
            if (nm_is(nm, nl, "exp") || nm_is(nm, nl, "exp2"))
            { sc->has_exp = 1; sc->call_exp++; }
            if (nm_is(nm, nl, "log") || nm_is(nm, nl, "log2"))
                sc->has_log = 1;
            if (nm_is(nm, nl, "inverseSqrt"))
                sc->has_rsqrt = 1;
            if (nm_is(nm, nl, "sqrt"))
                sc->has_sqrt = 1;
            if (nm_is(nm, nl, "dot"))
            { sc->has_dot = 1; sc->patterns |= PAT_MATMUL; }
            if (nm_is(nm, nl, "max")) sc->has_max = 1;
            if (nm_is(nm, nl, "min")) sc->has_min = 1;
            if (nm_is(nm, nl, "sin")) sc->has_sin = 1;
            if (nm_is(nm, nl, "cos")) sc->has_cos = 1;
            if (nm_is(nm, nl, "tanh"))
            { sc->has_tanh = 1; sc->patterns |= PAT_GELU; }
            if (nl >= 8 && memcmp(nm, "subgroup", 8) == 0)
            { sc->has_subgroup = 1; sc->patterns |= PAT_REDUCTION; }
            if (nl >= 6 && memcmp(nm, "atomic", 6) == 0)
            { sc->has_atomic = 1; sc->patterns |= PAT_REDUCTION; }
            if (nm_is(nm, nl, "workgroupBarrier") || nm_is(nm, nl, "storageBarrier"))
                sc->has_barrier = 1;
            if (nl >= 4 && (memcmp(nm, "relu", 4) == 0 || memcmp(nm, "ReLU", 4) == 0))
                sc->patterns |= PAT_RELU;
            if (nl >= 4 && (memcmp(nm, "gelu", 4) == 0 || memcmp(nm, "GELU", 4) == 0))
                sc->patterns |= PAT_GELU;
            if ((nl >= 4 && memcmp(nm, "silu", 4) == 0) ||
                (nl >= 5 && memcmp(nm, "swish", 5) == 0))
                sc->patterns |= PAT_SILU;
            if (nl >= 7 && memcmp(nm, "softmax", 7) == 0)
                sc->patterns |= PAT_SOFTMAX;
            if (nl >= 9 && memcmp(nm, "attention", 9) == 0)
                sc->patterns |= PAT_ATTENTION;
            if ((nl >= 9 && memcmp(nm, "layernorm", 9) == 0) ||
                (nl >= 10 && memcmp(nm, "layer_norm", 10) == 0) ||
                (nl >= 7 && memcmp(nm, "rmsnorm", 7) == 0))
                sc->patterns |= PAT_LAYERNORM;
            if (nl >= 4 && memcmp(nm, "rope", 4) == 0)
                sc->patterns |= PAT_ROPE;
            if (nl >= 5 && memcmp(nm, "embed", 5) == 0)
                sc->patterns |= PAT_EMBEDDING;
        }
    }
    if (n->kind == WGSL_NODE_EXPR_BINARY) {
        WGSLTokenKind op = (WGSLTokenKind)wgsl_node_op_kind_u32(n);
        const char *ops = wgsl_token_kind_name(op);
        if (ops && ops[0] == '*') sc->has_mul = 1;
        if (ops && ops[0] == '/') { sc->has_div = 1; sc->call_normalize++; }
        if (ops && ops[0] == '+') sc->call_sum_like++;
    }
    for (uint32_t i = 0; i < n->child_count; i++)
        scan_calls(r, n->children[i], sc, depth + 1);
}

static void scan_module(const WGSLResult *r, Scan *sc) {
    memset(sc, 0, sizeof *sc);
    sc->workgroup[0] = sc->workgroup[1] = sc->workgroup[2] = 1;
    if (!r || !r->ast.root || !r->src_copy) return;
    const WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        const WGSLNode *fn = tu->children[i];
        if (!fn || fn->kind != WGSL_NODE_DECL_FUNCTION) continue;
        uint32_t FA = wgsl_fn_attr_count(fn);
        int is_entry = 0;
        for (uint32_t a = 0; a < FA && a < fn->child_count; a++) {
            const WGSLNode *at = fn->children[a];
            if (attr_is(r->src_copy, r->src_len, at, "compute")) {
                sc->is_compute = 1;
                is_entry = 1;
            }
            if (attr_is(r->src_copy, r->src_len, at, "vertex") ||
                attr_is(r->src_copy, r->src_len, at, "fragment"))
                is_entry = 1;
            if (attr_is(r->src_copy, r->src_len, at, "workgroup_size")) {
                sc->has_workgroup = 1;
                for (uint32_t k = 1; k < at->child_count && k <= 3; k++) {
                    long long v = lit_int(r->src_copy, r->src_len, at->children[k]);
                    if (v > 0) sc->workgroup[k - 1] = (int)v;
                }
            }
        }
        if (is_entry && sc->entry[0] == 0) {
            uint32_t o = wgsl_node_name_span(fn).offset;
            uint32_t l = wgsl_node_name_span(fn).length;
            if (l && (size_t)o + l <= r->src_len) {
                size_t cpy = l < sizeof sc->entry - 1 ? l : sizeof sc->entry - 1;
                memcpy(sc->entry, r->src_copy + o, cpy);
                sc->entry[cpy] = 0;
            }
        }
        /* fn name hints */
        {
            uint32_t o = wgsl_node_name_span(fn).offset;
            uint32_t l = wgsl_node_name_span(fn).length;
            if (l >= 4 && (size_t)o + l <= r->src_len) {
                const char *nm = r->src_copy + o;
                if (l >= 7 && memcmp(nm, "softmax", 7) == 0) sc->patterns |= PAT_SOFTMAX;
                if (l >= 9 && memcmp(nm, "attention", 9) == 0) sc->patterns |= PAT_ATTENTION;
                if (l >= 4 && memcmp(nm, "gemm", 4) == 0) sc->patterns |= PAT_MATMUL;
                if (l >= 6 && memcmp(nm, "matmul", 6) == 0) sc->patterns |= PAT_MATMUL;
            }
        }
        scan_calls(r, fn, sc, 0);
    }

    /* Derive composite patterns */
    if (sc->has_exp && (sc->has_div || sc->call_normalize > 0) && sc->call_sum_like > 0)
        sc->patterns |= PAT_SOFTMAX;
    if ((sc->patterns & PAT_SOFTMAX) && (sc->has_dot || sc->has_mul))
        sc->patterns |= PAT_ATTENTION;
    if (sc->has_rsqrt || (sc->has_sqrt && sc->has_div && sc->call_sum_like > 0))
        sc->patterns |= PAT_LAYERNORM;
    if (sc->has_mul && !sc->has_exp &&
        !(sc->patterns & (PAT_SOFTMAX | PAT_MATMUL | PAT_LAYERNORM)))
        sc->patterns |= PAT_ELEMENTWISE;
    if (sc->has_sin && sc->has_cos)
        sc->patterns |= PAT_ROPE;
}

static void emit_pattern_array(JB *j, uint32_t p) {
    jb_putc(j, '[');
    int first = 1;
    #define P(bit, name) do { \
        if (p & (bit)) { if (!first) jb_putc(j, ','); first=0; jb_put_jstr(j, name); } \
    } while (0)
    P(PAT_SOFTMAX, "softmax");
    P(PAT_ATTENTION, "attention");
    P(PAT_LAYERNORM, "layernorm");
    P(PAT_GELU, "gelu");
    P(PAT_SILU, "silu");
    P(PAT_RELU, "relu");
    P(PAT_MATMUL, "matmul");
    P(PAT_REDUCTION, "reduction");
    P(PAT_ELEMENTWISE, "elementwise");
    P(PAT_EMBEDDING, "embedding");
    P(PAT_ROPE, "rope");
    #undef P
    jb_putc(j, ']');
}

static void workgroup_candidates(const Scan *sc, const int **cands, int *nc) {
    static const int red[] = { 32, 64, 128, 256 };
    static const int attn[] = { 16, 32, 64, 128 };
    static const int elem[] = { 64, 128, 256, 512 };
    static const int def[] = { 32, 64, 128, 256 };
    if (sc->patterns & (PAT_REDUCTION | PAT_SOFTMAX)) {
        *cands = red; *nc = 4;
    } else if (sc->patterns & PAT_ATTENTION) {
        *cands = attn; *nc = 4;
    } else if (sc->patterns & PAT_ELEMENTWISE) {
        *cands = elem; *nc = 4;
    } else {
        *cands = def; *nc = 4;
    }
}

static void emit_workgroup_hints(JB *j, const Scan *sc) {
    jb_puts(j, "\"workgroup_hints\":{");
    jb_puts(j, "\"current\":[");
    jb_put_u(j, (unsigned)sc->workgroup[0]); jb_putc(j, ',');
    jb_put_u(j, (unsigned)sc->workgroup[1]); jb_putc(j, ',');
    jb_put_u(j, (unsigned)sc->workgroup[2]);
    jb_puts(j, "],\"suggested_x\":[");
    const int *cands;
    int nc;
    workgroup_candidates(sc, &cands, &nc);
    for (int i = 0; i < nc; i++) {
        if (i) jb_putc(j, ',');
        jb_put_u(j, (unsigned)cands[i]);
    }
    jb_puts(j, "],\"notes\":");
    if (sc->patterns & PAT_ATTENTION)
        jb_put_jstr(j, "attention-like: prefer tile sizes matching head_dim; "
                       "try 32–128 for occupancy vs register pressure");
    else if (sc->patterns & PAT_SOFTMAX)
        jb_put_jstr(j, "softmax/reduction: 64–256 often best for warp-level reduce");
    else if (sc->patterns & PAT_ELEMENTWISE)
        jb_put_jstr(j, "elementwise: larger workgroups (128–512) hide latency");
    else
        jb_put_jstr(j, "sweep power-of-two X with host autotune; measure occupancy");
    jb_putc(j, '}');
}

static void emit_autotune_protocol(JB *j, const Scan *sc) {
    const int *cands;
    int nc;
    workgroup_candidates(sc, &cands, &nc);
    jb_puts(j, "\"autotune_protocol\":{\"version\":1");
    jb_puts(j, ",\"parameter\":\"workgroup_x\"");
    jb_puts(j, ",\"values\":[");
    for (int i = 0; i < nc; i++) {
        if (i) jb_putc(j, ',');
        jb_put_u(j, (unsigned)cands[i]);
    }
    jb_puts(j, "],\"metrics\":[\"ok\",\"elapsed_ns\"]");
    jb_puts(j, ",\"result_fields\":[\"workgroup_x\",\"ok\",\"elapsed_ns\"]}");
}

static void emit_shape_flow(JB *j, const ShapeFlow *sf) {
    jb_puts(j, "\"shape_flow\":{");
    int any = 0;
    if (sf && sf->head_dim > 0 && sf->seq_len > 0) {
        any = 1;
        jb_puts(j, "\"seq_len\":"); jb_put_u(j, (unsigned)sf->seq_len);
        jb_puts(j, ",\"head_dim\":"); jb_put_u(j, (unsigned)sf->head_dim);
        jb_puts(j, ",\"softmax_axis\":"); jb_put_i(j, sf->softmax_axis);
        jb_puts(j, ",\"softmax_axis_name\":\"seq_len\"");
        jb_puts(j, ",\"evidence\":\"storage_array_vec_bindings\"");
        jb_puts(j, ",\"evidence_bindings\":");
        jb_put_u(j, (unsigned)sf->evidence_bindings);
    }
    if (sf && sf->matmul_m > 0 && sf->matmul_n > 0 && sf->matmul_k > 0) {
        if (any) jb_putc(j, ',');
        jb_puts(j, "\"matmul\":{\"m\":"); jb_put_u(j, (unsigned)sf->matmul_m);
        jb_puts(j, ",\"n\":"); jb_put_u(j, (unsigned)sf->matmul_n);
        jb_puts(j, ",\"k\":"); jb_put_u(j, (unsigned)sf->matmul_k);
        jb_puts(j, ",\"broadcast\":");
        jb_puts(j, sf->matmul_broadcast ? "true" : "false");
        jb_puts(j, ",\"evidence\":\"storage_array_vec_bindings\"");
        jb_puts(j, ",\"evidence_bindings\":");
        jb_put_u(j, (unsigned)sf->matmul_evidence_bindings);
        jb_putc(j, '}');
    }
    jb_putc(j, '}');
}

static char *build_report(const WGSLResult *r) {
    BindTab bt = {0};
    Scan sc;
    ShapeFlow sf;
    collect_bindings(r, &bt);
    scan_module(r, &sc);
    derive_shape_flow(&bt, &sc, &sf);

    JB j = {0};
    jb_puts(&j, "{\"version\":1,\"kind\":\"ml_kernel\",");
    jb_puts(&j, "\"entry\":"); jb_put_jstr(&j, sc.entry[0] ? sc.entry : "");
    jb_puts(&j, ",\"is_compute\":"); jb_puts(&j, sc.is_compute ? "true" : "false");
    jb_puts(&j, ",\"patterns\":"); emit_pattern_array(&j, sc.patterns);
    jb_putc(&j, ',');
    emit_workgroup_hints(&j, &sc);
    jb_putc(&j, ',');
    emit_autotune_protocol(&j, &sc);

    jb_puts(&j, ",\"signals\":{");
    jb_puts(&j, "\"exp\":"); jb_puts(&j, sc.has_exp ? "true" : "false");
    jb_puts(&j, ",\"dot\":"); jb_puts(&j, sc.has_dot ? "true" : "false");
    jb_puts(&j, ",\"rsqrt\":"); jb_puts(&j, sc.has_rsqrt ? "true" : "false");
    jb_puts(&j, ",\"subgroup\":"); jb_puts(&j, sc.has_subgroup ? "true" : "false");
    jb_puts(&j, ",\"atomic\":"); jb_puts(&j, sc.has_atomic ? "true" : "false");
    jb_puts(&j, ",\"barrier\":"); jb_puts(&j, sc.has_barrier ? "true" : "false");
    jb_puts(&j, "},\"bindings\":[");
    for (int i = 0; i < bt.n; i++) {
        if (i) jb_putc(&j, ',');
        Binding *b = &bt.v[i];
        jb_puts(&j, "{\"name\":");
        if (r->src_copy)
            jb_put_src(&j, r->src_copy, r->src_len, b->name_off, b->name_len);
        else
            jb_puts(&j, "\"\"");
        jb_puts(&j, ",\"group\":"); jb_put_u(&j, b->group < 0 ? 0 : (unsigned)b->group);
        jb_puts(&j, ",\"binding\":"); jb_put_u(&j, b->binding < 0 ? 0 : (unsigned)b->binding);
        jb_puts(&j, ",\"address_space\":"); jb_put_jstr(&j, b->as_name);
        jb_puts(&j, ",\"access\":"); jb_put_jstr(&j, b->access_name);
        jb_puts(&j, ",\"type\":"); jb_put_jstr(&j, b->type_repr[0] ? b->type_repr : "unknown");
        jb_puts(&j, ",\"array_len\":"); jb_put_u(&j, (unsigned)(b->array_len > 0 ? b->array_len : 0));
        jb_puts(&j, ",\"vec_width\":"); jb_put_u(&j, (unsigned)(b->vec_width > 0 ? b->vec_width : 0));
        jb_puts(&j, ",\"f16\":"); jb_puts(&j, b->is_f16 ? "true" : "false");
        jb_puts(&j, ",\"atomic\":"); jb_puts(&j, b->is_atomic ? "true" : "false");
        jb_putc(&j, '}');
    }
    jb_puts(&j, "],\"shape_hints\":[");
    int sh_first = 1;
    for (int i = 0; i < bt.n; i++) {
        Binding *b = &bt.v[i];
        if (b->array_len <= 0 && b->vec_width <= 0) continue;
        if (!sh_first) jb_putc(&j, ',');
        sh_first = 0;
        jb_puts(&j, "{\"binding\":");
        jb_put_u(&j, b->binding < 0 ? 0 : (unsigned)b->binding);
        if (b->array_len > 0) {
            jb_puts(&j, ",\"extent\":"); jb_put_u(&j, (unsigned)b->array_len);
        }
        if (b->vec_width > 0) {
            jb_puts(&j, ",\"lanes\":"); jb_put_u(&j, (unsigned)b->vec_width);
        }
        int rank = binding_tensor_rank(b);
        if (rank > 0) {
            jb_puts(&j, ",\"rank\":"); jb_put_u(&j, (unsigned)rank);
            jb_puts(&j, ",\"shape\":[");
            int first_dim = 1;
            if (b->array_len > 0) {
                jb_put_u(&j, (unsigned)b->array_len);
                first_dim = 0;
            }
            if (b->vec_width > 0) {
                if (!first_dim) jb_putc(&j, ',');
                jb_put_u(&j, (unsigned)b->vec_width);
            }
            jb_putc(&j, ']');
        }
        jb_putc(&j, '}');
    }
    jb_puts(&j, "],");
    emit_shape_flow(&j, &sf);
    jb_puts(&j, ",\"advice\":[");
    int adv = 0;
    if (sc.patterns & PAT_SOFTMAX) {
        if (adv++) jb_putc(&j, ',');
        jb_put_jstr(&j, "softmax: consider online max+sum (two-pass or Welford) "
                        "to improve numeric stability; fuse exp/sum/div if possible");
    }
    if (sc.patterns & PAT_ATTENTION) {
        if (adv++) jb_putc(&j, ',');
        jb_put_jstr(&j, "attention: tile K/V into workgroup memory; match "
                        "workgroup.x to head_dim or warp size");
    }
    if (sc.patterns & PAT_LAYERNORM) {
        if (adv++) jb_putc(&j, ',');
        jb_put_jstr(&j, "layernorm/rmsnorm: fuse mean/var/affine; use "
                        "subgroupReduce for channel reductions");
    }
    if (sc.patterns & PAT_MATMUL) {
        if (adv++) jb_putc(&j, ',');
        jb_put_jstr(&j, "matmul-ish: check memory coalescing on row-major "
                        "A/B; consider f16 paths if enable f16 is present");
    }
    int any_f16 = 0;
    for (int i = 0; i < bt.n; i++) if (bt.v[i].is_f16) any_f16 = 1;
    if (any_f16) {
        if (adv++) jb_putc(&j, ',');
        jb_put_jstr(&j, "f16 present: verify enable f16; packed matmul may "
                        "use tensor-core friendly layouts");
    }
    if (sc.is_compute && !sc.has_workgroup) {
        if (adv++) jb_putc(&j, ',');
        jb_put_jstr(&j, "compute entry missing @workgroup_size — required by WGSL");
    }
    jb_puts(&j, "]}");

    free(bt.v);
    if (j.oom) { free(j.buf); return NULL; }
    if (!j.buf) {
        j.buf = (char *)malloc(3);
        if (j.buf) { j.buf[0] = '{'; j.buf[1] = '}'; j.buf[2] = 0; }
    }
    return j.buf;
}

char *wgsl_ml_analyze_json(const WGSLResult *r) {
    if (!r || !r->ast.root) {
        /* Empty-report JSON is 73 chars + NUL; keep a little headroom. */
        char *e = (char *)malloc(96);
        if (e) snprintf(e, 96, "{\"version\":1,\"kind\":\"ml_kernel\",\"patterns\":[],"
                               "\"bindings\":[],\"advice\":[]}");
        return e;
    }
    return build_report(r);
}

char *wgsl_ml_analyze_json_src(const char *src) {
    WGSLResult *r = wgsl_check(src ? src : "");
    if (!r) return NULL;
    char *j = wgsl_ml_analyze_json(r);
    wgsl_free(r);
    return j;
}
