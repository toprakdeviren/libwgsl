/**
 * @file layout.c — §14.4 memory layout + §13.3.2 resource bindings.
 *
 *   - Per-struct member: `@align(n)` multiplicity vs RequiredAlignOf,
 *     `@size(n)` ≥ SizeOf(T), generation-fixed-footprint gate on the
 *     member type when `@size` is present.
 *   - Per-`var<uniform>` / `var<storage>` resource binding:
 *     host-shareable store-type gate, uniform-AS array stride / struct
 *     offset rules, runtime-sized-array rejection in uniform, and the
 *     (group, binding) uniqueness check across all resource vars.
 *
 * Co-extensive with WGSL §14.4 (memory layout) + §13.3.2 (resource
 * interface) + §6.4.2 (host-shareable).
 *
 * (Distinct from `src/layout.c` — that file is the AlignOf / SizeOf /
 * StrideOf math helper module.  This file is the validator-side gate
 * that consumes it.)
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
#include "internal/layout.h"
#include "internal/check.h"

void validate_layouts(WGSLValidator *v, WGSLNode *tu) {
    if (!tu) return;
    WGSLLayoutCtx lcx = { v->tc, v->cev, v->src, 0 };
    /* Struct member layout checks, including @size and @align. */
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (n && n->kind == WGSL_NODE_DECL_STRUCT) {
            for (uint32_t k = 0; k < n->child_count; k++) {
                WGSLNode *m = n->children[k];
                if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
                uint32_t MA = wgsl_member_attr_count(m);
                WGSLTypeInfo *mt = NULL;
                if (MA < m->child_count) mt = wgsl_typecheck_type_of(v->tc, m->children[MA]);
                WGSLNode *sz = wgsl_val_find_attr(v, m, 0, MA, "size");
                if (sz) {
                    long long val = wgsl_val_attr_int_arg(v, sz);
                    uint32_t real_size = wgsl_layout_size_of(&lcx, mt, WGSL_AS_NONE);
                    if (val < (long long)real_size) wgsl_val_error(v, sz, "@size must be at least the size of the type (%u)", real_size);
                    /* §12.13 — `@size(n)` may only be applied to members
                     * whose type has a generation-fixed footprint.  Runtime-
                     * sized arrays (and structs containing one as their
                     * tail member) are excluded. */
                    if (mt && !wgsl_type_has_generation_fixed_footprint(mt)) {
                        wgsl_val_error(v, sz,
                            "@size requires a member type with a "
                            "generation-fixed footprint (got runtime-sized)");
                    }
                }
                /* §12.1 — @align(n) must be a positive multiple of
                 * RequiredAlignOf(T, default-AS).  Power-of-2 + positive
                 * is checked elsewhere; here we add the multiplicity gate. */
                WGSLNode *al = wgsl_val_find_attr(v, m, 0, MA, "align");
                if (al && mt) {
                    long long val = wgsl_val_attr_int_arg(v, al);
                    if (val > 0) {
                        uint32_t ra = wgsl_layout_required_align_of(&lcx, mt, WGSL_AS_NONE);
                        if (ra > 0 && (uint32_t)val % ra != 0) {
                            wgsl_val_error(v, al,
                                "@align value %lld must be a multiple of "
                                "RequiredAlignOf(%s) = %u",
                                val, wgsl_type_kind_name((WGSLTypeKind)mt->kind),
                                (unsigned)ra);
                        }
                    }
                }
            }
        }
    }
    /* Uniform-address-space variable layout checks.  Struct-of-struct
     *    member offsets must satisfy roundUp(16, AlignOf(member_T, uniform)).
     *    `wgsl_layout_struct(uniform)` already encodes this in the
     *    offset walk; here we just trigger the walk and have it report
     *    any member that breaks the rule.  Variable-level @align must
     *    be a multiple of RequiredAlignOf(T, var's-AS). */
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n || n->kind != WGSL_NODE_DECL_VAR) continue;
        WGSLSymbol *sym = wgsl_val_find_symbol_for_ast(v, n);
        if (!sym || !sym->type) continue;
        WGSLAddressSpace as = (WGSLAddressSpace)sym->as;
        /* Variable @align (rare; struct-member @align is the common form). */
        uint32_t A = wgsl_varlike_attr_count(n);
        WGSLNode *al = wgsl_val_find_attr(v, n, 0, A, "align");
        if (al) {
            long long val = wgsl_val_attr_int_arg(v, al);
            if (val > 0) {
                uint32_t ra = wgsl_layout_required_align_of(&lcx, sym->type, as);
                if (ra > 0 && (uint32_t)val % ra != 0) {
                    wgsl_val_error(v, al,
                        "@align value %lld must be a multiple of "
                        "RequiredAlignOf(T, %s) = %u",
                        val,
                        as == WGSL_AS_UNIFORM   ? "uniform"   :
                        as == WGSL_AS_STORAGE   ? "storage"   :
                        as == WGSL_AS_WORKGROUP ? "workgroup" :
                        as == WGSL_AS_PRIVATE   ? "private"   : "function",
                        (unsigned)ra);
                }
            }
        }
        /* §14.4.5 — uniform AS struct member offset rule.  When a
         * struct (potentially nested) is bound in `var<uniform>`, every
         * member's offset must satisfy roundUp(16, AlignOf(member_T)).
         * `wgsl_layout_struct(uniform)` computes offsets honouring
         * that rule; we re-check here that struct authors haven't
         * @align'd to a value that breaks it. */
        int uniform_std_layout =
            v->tc && (v->tc->supported_features & WGSL_FEAT_UNIFORM_BUF_STD_LAYOUT);
        if (as == WGSL_AS_UNIFORM && !uniform_std_layout &&
            sym->type->kind == WGSL_TYPE_STRUCT)
        {
            const WGSLNode *sd = (const WGSLNode *)sym->type->ref;
            uint32_t nmem = 0;
            for (uint32_t k = 0; sd && k < sd->child_count; k++)
                if (sd->children[k] &&
                    sd->children[k]->kind == WGSL_NODE_DECL_STRUCT_MEMBER) nmem++;
            uint32_t *offs = (uint32_t *)malloc((nmem ? nmem : 1) * sizeof *offs);
            uint32_t salign = 0, ssize = 0;
            if (offs && wgsl_layout_struct(&lcx, sym->type, WGSL_AS_UNIFORM,
                                           offs, nmem, &salign, &ssize)) {
                uint32_t midx = 0;
                for (uint32_t k = 0; sd && k < sd->child_count; k++) {
                    WGSLNode *m = sd->children[k];
                    if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
                    uint32_t MA = wgsl_member_attr_count(m);
                    WGSLTypeInfo *mt = NULL;
                    if (MA < m->child_count) mt = wgsl_typecheck_type_of(v->tc, m->children[MA]);
                    if (!mt) { midx++; continue; }
                    /* Required offset alignment for uniform AS:
                     *   roundUp(16, AlignOf(T)) for struct/array members. */
                    if (mt->kind == WGSL_TYPE_STRUCT || mt->kind == WGSL_TYPE_ARRAY) {
                        uint32_t base = wgsl_layout_align_of(&lcx, mt, WGSL_AS_UNIFORM);
                        uint32_t need = (16 + base - 1) / base * base;
                        if (need < 16) need = 16;
                        if (offs[midx] % need != 0) {
                            wgsl_val_error(v, m,
                                "var<uniform> struct member offset %u must be "
                                "a multiple of roundUp(16, AlignOf(%s)) = %u",
                                (unsigned)offs[midx],
                                wgsl_type_kind_name((WGSLTypeKind)mt->kind),
                                (unsigned)need);
                        }
                    }
                    midx += 1;
                }
            }
            free(offs);
        }
    }
}

/* Deep host-shareable walk — the predicate in types.c is shallow on
 * struct members; this version recurses into struct ref-trees, using
 * the type-checker's resolved type-of map. */
static int is_host_shareable_deep_impl(WGSLValidator *v, const WGSLTypeInfo *t,
                                       const WGSLNode *at, int depth) {
    if (!t) return 0;
    /* Bound recursion on the resolved type graph: a deep struct/alias chain
     * would overflow the stack.  Report once and reject. */
    if (depth > WGSL_MAX_AST_DEPTH) {
        wgsl_val_error(v, at,
            "type nested too deeply (max %d)", WGSL_MAX_AST_DEPTH);
        return 0;
    }
    if (t->kind == WGSL_TYPE_STRUCT) {
        const WGSLNode *sd = (const WGSLNode *)t->ref;
        if (!sd) return 0;
        for (uint32_t i = 0; i < sd->child_count; i++) {
            WGSLNode *m = sd->children[i];
            if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            uint32_t MA = wgsl_member_attr_count(m);
            const WGSLTypeInfo *mt = NULL;
            if (MA < m->child_count) {
                mt = wgsl_typecheck_type_of(v->tc, m->children[MA]);
            }
            if (!is_host_shareable_deep_impl(v, mt, at, depth + 1)) return 0;
        }
        return 1;
    }
    if (t->kind == WGSL_TYPE_ARRAY) {
        return is_host_shareable_deep_impl(
            v, (const WGSLTypeInfo *)t->ref, at, depth + 1);
    }
    return wgsl_type_is_host_shareable(t);
}

static int is_host_shareable_deep(WGSLValidator *v, const WGSLTypeInfo *t,
                                  const WGSLNode *at) {
    return is_host_shareable_deep_impl(v, t, at, 0);
}

typedef struct {
    int64_t group;
    int64_t binding;
    WGSLNode *at;
    WGSLSymbol *sym;
} ResourceBindingInfo;

/* Multi-word bitset over resource indices (was uint64 → silent >64 cliff). */
static int res_bit_words(int resource_count) {
    return resource_count <= 0 ? 0 : (resource_count + 63) / 64;
}
static void res_bit_set(uint64_t *bits, int i) {
    bits[(unsigned)i >> 6] |= (uint64_t)1u << ((unsigned)i & 63u);
}
static int res_bit_test(const uint64_t *bits, int i) {
    return (bits[(unsigned)i >> 6] >> ((unsigned)i & 63u)) & 1u;
}
static void res_bit_or_into(uint64_t *dst, const uint64_t *src, int nwords) {
    for (int w = 0; w < nwords; w++) dst[w] |= src[w];
}

typedef struct {
    WGSLValidator *v;
    const ResourceBindingInfo *resources;
    int resource_count;
    uint64_t *bits;   /* nwords words */
    int nwords;
} ResourceScanCtx;

static void scan_resource_refs(ResourceScanCtx *cx, WGSLNode *n) {
    if (!n) return;
    if (n->kind == WGSL_NODE_EXPR_IDENT) {
        WGSLSymbol *sym = wgsl_node_resolved_symbol(n);
        for (int i = 0; i < cx->resource_count; i++) {
            if (sym == cx->resources[i].sym) {
                res_bit_set(cx->bits, i);
                break;
            }
        }
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        scan_resource_refs(cx, n->children[i]);
    }
}

typedef struct {
    WGSLValidator *v;
    uint64_t *direct_bits; /* [decl_count][nwords] row-major */
    int nwords;
    uint8_t *visited;
    uint64_t *bits;        /* nwords working set */
} ResourceReachCtx;

static void resource_reach_edge_cb(
    WGSLValidator *v, const WGSLSymbol *to, const WGSLNode *at, void *ctx)
{
    (void)at;
    ResourceReachCtx *cx = (ResourceReachCtx *)ctx;
    int idx = wgsl_val_sym_index(v, to);
    if (idx < 0 || cx->visited[idx]) return;
    cx->visited[idx] = 1;
    res_bit_or_into(cx->bits, cx->direct_bits + (size_t)idx * (size_t)cx->nwords,
                    cx->nwords);
    wgsl_val_enum_call_edges(v, to, resource_reach_edge_cb, cx);
}

static void resource_bits_for_entry(
    WGSLValidator *v, const WGSLSymbol *entry,
    uint64_t *direct_bits, int nwords, uint8_t *visited, uint64_t *out_bits)
{
    size_t n = v->res->all_decl_count;
    memset(visited, 0, n * sizeof *visited);
    memset(out_bits, 0, (size_t)nwords * sizeof *out_bits);
    int idx = wgsl_val_sym_index(v, entry);
    if (idx >= 0) {
        visited[idx] = 1;
        res_bit_or_into(out_bits, direct_bits + (size_t)idx * (size_t)nwords,
                        nwords);
    }
    ResourceReachCtx cx = { v, direct_bits, nwords, visited, out_bits };
    wgsl_val_enum_call_edges(v, entry, resource_reach_edge_cb, &cx);
}

static void emit_duplicate_bindings_for_bits(
    WGSLValidator *v, const ResourceBindingInfo *resources,
    int resource_count, const uint64_t *bits)
{
    for (int i = 0; i < resource_count; i++) {
        if (!res_bit_test(bits, i)) continue;
        for (int j = i + 1; j < resource_count; j++) {
            if (!res_bit_test(bits, j)) continue;
            if (resources[i].group == resources[j].group &&
                resources[i].binding == resources[j].binding)
            {
                wgsl_val_error(v, resources[j].at,
                    "duplicate resource binding @group(%lld) "
                    "@binding(%lld) (previously bound)",
                    (long long)resources[j].group,
                    (long long)resources[j].binding);
            }
        }
    }
}

static void validate_duplicate_resource_bindings(
    WGSLValidator *v, const ResourceBindingInfo *resources, int resource_count)
{
    if (!v || !v->res || resource_count <= 1) return;
    size_t n = v->res->all_decl_count;
    int nwords = res_bit_words(resource_count);
    if (nwords <= 0) return;
    uint64_t *direct_bits = (uint64_t *)calloc(
        n * (size_t)nwords, sizeof *direct_bits);
    uint8_t *visited = (uint8_t *)calloc(n, sizeof *visited);
    uint64_t *work = (uint64_t *)calloc((size_t)nwords, sizeof *work);
    uint64_t *scan = (uint64_t *)calloc((size_t)nwords, sizeof *scan);
    if (!direct_bits || !visited || !work || !scan) {
        free(direct_bits);
        free(visited);
        free(work);
        free(scan);
        return;
    }

    for (size_t i = 0; i < n; i++) {
        WGSLSymbol *s = v->res->all_decls[i];
        if (!s || s->kind != WGSL_SYM_FUNCTION ||
            !s->ast || s->ast->child_count == 0)
        {
            continue;
        }
        WGSLNode *body = s->ast->children[s->ast->child_count - 1];
        memset(scan, 0, (size_t)nwords * sizeof *scan);
        ResourceScanCtx sc = {
            v, resources, resource_count, scan, nwords
        };
        scan_resource_refs(&sc, body);
        memcpy(direct_bits + i * (size_t)nwords, scan,
               (size_t)nwords * sizeof *scan);
    }

    for (size_t i = 0; i < n; i++) {
        WGSLSymbol *s = v->res->all_decls[i];
        if (!s || s->kind != WGSL_SYM_FUNCTION) continue;
        if (!(s->flags & (WGSL_SYM_FLAG_VERTEX |
                          WGSL_SYM_FLAG_FRAGMENT |
                          WGSL_SYM_FLAG_COMPUTE)))
        {
            continue;
        }
        resource_bits_for_entry(v, s, direct_bits, nwords, visited, work);
        emit_duplicate_bindings_for_bits(v, resources, resource_count, work);
    }

    free(direct_bits);
    free(visited);
    free(work);
    free(scan);
}

void validate_resource_bindings(WGSLValidator *v, WGSLNode *tu) {
    if (!tu) return;
    WGSLLayoutCtx lcx = { v->tc, v->cev, v->src, 0 };
    /* Growable: old fixed [64] + uint64 bitset silently stopped
     * dup-check past 64 resources.  Not a WGSL limit — pure analysis cap. */
    ResourceBindingInfo *resources = NULL;
    int resource_n = 0, resource_cap = 0;

    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n || n->kind != WGSL_NODE_DECL_VAR) continue;
        uint32_t A = wgsl_varlike_attr_count(n);
        WGSLSymbol *sym = wgsl_val_find_symbol_for_ast(v, n);
        if (!sym) continue;
        /* Resource address spaces (uniform / storage) and handle vars
         * (textures, samplers in the handle AS — they default to AS
         * none in the typechecker but live conceptually in handle). */
        int is_resource =
            (sym->as == WGSL_AS_UNIFORM) ||
            (sym->as == WGSL_AS_STORAGE) ||
            (sym->type && (sym->type->kind == WGSL_TYPE_TEXTURE ||
                           sym->type->kind == WGSL_TYPE_SAMPLER));
        if (is_resource) {
            WGSLNode *gp = wgsl_val_find_attr(v, n, 0, A, "group");
            WGSLNode *bd = wgsl_val_find_attr(v, n, 0, A, "binding");
            if (!gp) {
                wgsl_val_error(v, n, "resource variable requires @group attribute");
            }
            if (!bd) {
                wgsl_val_error(v, n, "resource variable requires @binding attribute");
            }
            /* §6.4.2 / §14.4.5 — uniform & storage address spaces require
             * a host-shareable store type.  Bool, ptr, ref, texture,
             * sampler, and types containing them are excluded. */
            if ((sym->as == WGSL_AS_UNIFORM || sym->as == WGSL_AS_STORAGE) &&
                sym->type)
            {
                if (!is_host_shareable_deep(v, sym->type, n)) {
                    wgsl_val_error(v, n,
                        "var<%s, ...> store type must be host-shareable "
                        "(got %s)",
                        sym->as == WGSL_AS_UNIFORM ? "uniform" : "storage",
                        wgsl_type_kind_name((WGSLTypeKind)sym->type->kind));
                }
            }
            /* §14.4.5 — uniform address space layout constraints. */
            if (sym->as == WGSL_AS_UNIFORM && sym->type) {
                const WGSLTypeInfo *st = sym->type;
                int uniform_std_layout =
                    v->tc && (v->tc->supported_features & WGSL_FEAT_UNIFORM_BUF_STD_LAYOUT);
                /* Uniform AS forbids runtime-sized arrays (their size
                 * isn't known at pipeline generation). */
                if (st->kind == WGSL_TYPE_ARRAY && st->array_len == 0) {
                    wgsl_val_error(v, n,
                        "var<uniform> must not have a runtime-sized "
                        "array store type");
                }
                /* Array stride in uniform AS must be a multiple of 16
                 * (the RoundUp(16, …) bump per §14.4.5).  Applies to
                 * a top-level array store type; for arrays nested in
                 * structs the rule still holds for each contained
                 * array, but the recursive walk is left for the full
                 * layout calculator. */
                if (!uniform_std_layout &&
                    st->kind == WGSL_TYPE_ARRAY && st->array_len > 0)
                {
                    const WGSLTypeInfo *elem = (const WGSLTypeInfo *)st->ref;
                    uint32_t es = wgsl_layout_size_of(&lcx, elem, WGSL_AS_UNIFORM);
                    uint32_t ea = wgsl_layout_align_of(&lcx, elem, WGSL_AS_UNIFORM);
                    uint32_t stride = ea ? ((es + ea - 1) / ea) * ea : es;
                    if (stride % 16 != 0) {
                        wgsl_val_error(v, n,
                            "var<uniform> array element stride must be "
                            "a multiple of 16 (got %u for element type "
                            "%s)",
                            (unsigned)stride,
                            wgsl_type_kind_name((WGSLTypeKind)elem->kind));
                    }
                }
            }
            if (gp && bd) {
                if (resource_n == resource_cap) {
                    int nc = resource_cap ? resource_cap * 2 : 16;
                    ResourceBindingInfo *g = (ResourceBindingInfo *)realloc(
                        resources, (size_t)nc * sizeof *g);
                    if (!g) {
                        /* OOM: still free what we have; skip further adds. */
                        free(resources);
                        return;
                    }
                    resources = g;
                    resource_cap = nc;
                }
                resources[resource_n].group = wgsl_val_attr_int_arg(v, gp);
                resources[resource_n].binding = wgsl_val_attr_int_arg(v, bd);
                resources[resource_n].at = n;
                resources[resource_n].sym = sym;
                resource_n += 1;
            }
        }
    }
    validate_duplicate_resource_bindings(v, resources, resource_n);
    free(resources);
}
