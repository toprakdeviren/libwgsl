/**
 * @file io.c — §13 entry-point IO shape + builtin-name validation.
 *
 * Per-entry-point checks:
 *   - @builtin / @location placement, stage, direction, type
 *   - struct param + @location forbidden
 *   - nested IO structs forbidden
 *   - dual-source-blending fragment-output struct shape
 *   - compute entry-points: no user-defined IO, no return type
 *
 * Reads `BuiltinEntry` rows from attrs.c via `wgsl_val_find_builtin_entry`
 * (declared in validate_priv.h).
 *
 * Co-extensive with WGSL §13.
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

/* IO walker state — tracks `@location` indices seen (per direction)
 * and each `@builtin(name)` so duplicates / shape mismatches can fire.
 *
 * §13.3.1.3 says "no two inputs share @location" AND "no two outputs
 * share @location" — but inputs and outputs are independent.
 *
 * Previous form used a uint64 bitset (0..63 only) + fixed [16] builtin arrays;
 * values outside those windows silently skipped uniqueness checks.
 * Not a WGSL hard max — pure analysis buffer. Growable lists now. */
typedef struct {
    const char *name;
    uint32_t len;
} SeenBuiltin;

typedef struct {
    int64_t *locs;
    int n, cap;
} LocSet;

typedef struct {
    SeenBuiltin *items;
    int n, cap;
} BuiltinSet;

typedef struct {
    LocSet     in_locations;
    LocSet     out_locations;
    BuiltinSet seen_in_builtins;
    BuiltinSet seen_out_builtins;
} IOState;

static void locset_free(LocSet *s) {
    free(s->locs);
    s->locs = NULL; s->n = 0; s->cap = 0;
}
static void builtinset_free(BuiltinSet *s) {
    free(s->items);
    s->items = NULL; s->n = 0; s->cap = 0;
}
static void iostate_free(IOState *io) {
    locset_free(&io->in_locations);
    locset_free(&io->out_locations);
    builtinset_free(&io->seen_in_builtins);
    builtinset_free(&io->seen_out_builtins);
}

/* Returns 1 if newly added, 0 if already present (duplicate).
 * On OOM returns -1 (caller should treat as analysis failure). */
static int locset_add(LocSet *s, int64_t l) {
    for (int i = 0; i < s->n; i++) {
        if (s->locs[i] == l) return 0;
    }
    if (s->n == s->cap) {
        int nc = s->cap ? s->cap * 2 : 8;
        int64_t *g = (int64_t *)realloc(s->locs, (size_t)nc * sizeof *g);
        if (!g) return -1;
        s->locs = g;
        s->cap = nc;
    }
    s->locs[s->n++] = l;
    return 1;
}

/* Returns 1 if newly added, 0 if duplicate, -1 on OOM. */
static int builtinset_add(BuiltinSet *s, const char *nm, uint32_t len) {
    for (int i = 0; i < s->n; i++) {
        if (s->items[i].len == len && memcmp(s->items[i].name, nm, len) == 0)
            return 0;
    }
    if (s->n == s->cap) {
        int nc = s->cap ? s->cap * 2 : 8;
        SeenBuiltin *g = (SeenBuiltin *)realloc(
            s->items, (size_t)nc * sizeof *g);
        if (!g) return -1;
        s->items = g;
        s->cap = nc;
    }
    s->items[s->n].name = nm;
    s->items[s->n].len = len;
    s->n += 1;
    return 1;
}

static int builtin_type_matches(
    const WGSLValidator *v, const WGSLTypeInfo *t, BuiltinTypeHint hint)
{
    if (hint == BT_NONE || !t) return 1;
    switch (hint) {
    case BT_U32:        return t == v->tc->types->t_u32;
    case BT_I32:        return t == v->tc->types->t_i32;
    case BT_F32:        return t == v->tc->types->t_f32;
    case BT_BOOL:       return t == v->tc->types->t_bool;
    case BT_VEC3_U32:
        return t->kind == WGSL_TYPE_VEC && t->width == 3 &&
               t->ref == v->tc->types->t_u32;
    case BT_VEC4_F32:
        return t->kind == WGSL_TYPE_VEC && t->width == 4 &&
               t->ref == v->tc->types->t_f32;
    case BT_ARRAY_F32_1_8:
        return t->kind == WGSL_TYPE_ARRAY &&
               t->array_len >= 1 && t->array_len <= 8 &&
               t->ref == v->tc->types->t_f32;
    default: return 1;
    }
}

static const char *builtin_type_name(BuiltinTypeHint hint) {
    switch (hint) {
    case BT_U32:        return "u32";
    case BT_I32:        return "i32";
    case BT_F32:        return "f32";
    case BT_BOOL:       return "bool";
    case BT_VEC3_U32:   return "vec3<u32>";
    case BT_VEC4_F32:   return "vec4<f32>";
    case BT_ARRAY_F32_1_8: return "array<f32, N> where 1 <= N <= 8";
    default:            return "(any)";
    }
}

static int type_is_integer_scalar_or_vec(
    const WGSLValidator *v, const WGSLTypeInfo *t)
{
    if (!t) return 0;
    if (t == v->tc->types->t_i32 || t == v->tc->types->t_u32) return 1;
    return t->kind == WGSL_TYPE_VEC &&
           (t->ref == v->tc->types->t_i32 ||
            t->ref == v->tc->types->t_u32);
}

static int attr_arg_name_eq(
    const WGSLValidator *v, const WGSLNode *attr,
    uint32_t arg_index, const char *want)
{
    if (!attr || arg_index >= attr->child_count) return 0;
    const WGSLNode *arg = attr->children[arg_index];
    if (!arg || arg->kind != WGSL_NODE_EXPR_IDENT) return 0;
    uint32_t off = wgsl_node_name_span(arg).offset;
    uint32_t len = wgsl_node_name_span(arg).length;
    size_t wl = strlen(want);
    return len == wl && memcmp(v->src->bytes + off, want, wl) == 0;
}

static void check_interpolation_slot(
    WGSLValidator *v, WGSLNode *at, WGSLNode *interp,
    const WGSLTypeInfo *slot_type, uint32_t stage_mask, int is_input)
{
    if (interp) {
        int is_flat = attr_arg_name_eq(v, interp, 1, "flat");
        int is_perspective = attr_arg_name_eq(v, interp, 1, "perspective");
        int is_linear = attr_arg_name_eq(v, interp, 1, "linear");
        if (interp->child_count >= 3) {
            int center = attr_arg_name_eq(v, interp, 2, "center");
            int centroid = attr_arg_name_eq(v, interp, 2, "centroid");
            int sample = attr_arg_name_eq(v, interp, 2, "sample");
            int first = attr_arg_name_eq(v, interp, 2, "first");
            int either = attr_arg_name_eq(v, interp, 2, "either");
            if (is_flat && (center || centroid || sample)) {
                wgsl_val_error(v, interp,
                    "@interpolate(flat, ...) sampling must be 'first' "
                    "or 'either'");
            }
            if ((is_perspective || is_linear) && (first || either)) {
                wgsl_val_error(v, interp,
                    "@interpolate(%s, ...) sampling must be 'center', "
                    "'centroid', or 'sample'",
                    is_perspective ? "perspective" : "linear");
            }
        }
    }

    int inter_stage =
        (is_input && (stage_mask & STAGE_FR)) ||
        (!is_input && (stage_mask & STAGE_VX));
    if (inter_stage &&
        type_is_integer_scalar_or_vec(v, slot_type) &&
        (!interp || !attr_arg_name_eq(v, interp, 1, "flat")))
    {
        wgsl_val_error(v, interp ? interp : at,
            "integer user-defined IO must use @interpolate(flat)");
    }
}

/* Per-slot check: validate one @builtin / @location attribute set on
 * either a direct param/return or a struct member.  `slot_type` is the
 * resolved declared type. */
static void check_io_slot(
    WGSLValidator *v,
    WGSLNode *at, WGSLNode *attr_parent,
    uint32_t attr_first, uint32_t attr_count,
    const WGSLTypeInfo *slot_type,
    IOState *io, uint32_t stage_mask, int is_input)
{
    WGSLNode *loc      = wgsl_val_find_attr(v, attr_parent, attr_first, attr_count, "location");
    WGSLNode *blt      = wgsl_val_find_attr(v, attr_parent, attr_first, attr_count, "builtin");
    WGSLNode *interp   = wgsl_val_find_attr(v, attr_parent, attr_first, attr_count, "interpolate");

    /* §13.3.1.2 — every entry-point IO slot must carry either
     * @location or @builtin. */
    if (!loc && !blt) {
        wgsl_val_error(v, at,
            "entry-point parameter / return / member must have a "
            "@location or @builtin attribute");
    }

    if (loc) {
        long long l = 0;
        int loc_is_int = wgsl_val_attr_int_arg_typed(v, loc, &l);
        if (stage_mask & STAGE_CO) {
            wgsl_val_error(v, loc,
                "@compute entry point must not have user-defined IO "
                "(@location is forbidden)");
        }
        /* §13.3.1.3 — when `@blend_src` is also present on the slot the
         * location is shared between the (0/1) pair, so don't flag it
         * as a duplicate.  The dual-source struct shape itself is
         * validated separately in `check_io_type`. */
        WGSLNode *bs_attr_for_loc = wgsl_val_find_attr(v, attr_parent, attr_first, attr_count, "blend_src");
        /* Any non-negative @location participates in uniqueness (§13.3.1.3).
         * Negative values are not a legal location index. */
        if (loc_is_int && !bs_attr_for_loc) {
            if (l < 0) {
                wgsl_val_error(v, loc,
                    "@location value must be a non-negative integer "
                    "(got %lld)", l);
            } else {
                LocSet *ls = is_input ? &io->in_locations : &io->out_locations;
                int add = locset_add(ls, l);
                if (add == 0) {
                    wgsl_val_error(v, loc,
                        "duplicate @location(%lld) on entry-point %s",
                        l, is_input ? "input" : "output");
                } else if (add < 0) {
                    wgsl_val_error(v, loc,
                        "@location uniqueness analysis truncated "
                        "(out of memory)");
                }
            }
        }
        /* §13.3.1.2 — @location-tagged IO must be numeric scalar /
         * vector.  bool / matrix / struct / atomic etc. are rejected. */
        if (slot_type) {
            int ok_num =
                wgsl_type_is_numeric(slot_type) ||
                (slot_type->kind == WGSL_TYPE_VEC &&
                 wgsl_type_is_numeric((const WGSLTypeInfo *)slot_type->ref));
            if (!ok_num) {
                wgsl_val_error(v, at,
                    "@location IO must have a numeric scalar or vector "
                    "type (got %s)",
                    wgsl_type_kind_name((WGSLTypeKind)slot_type->kind));
            }
        }
        check_interpolation_slot(
            v, at, interp, slot_type, stage_mask, is_input);
    }

    /* §12.3 — `@blend_src` is fragment output only, and only on a
     * structure member.  Per-IO context is needed to know stage /
     * direction, so part of the check lives here rather than in
     * `check_attr`. */
    {
        WGSLNode *bs = wgsl_val_find_attr(v, attr_parent, attr_first, attr_count, "blend_src");
        if (bs) {
            if (!attr_parent || attr_parent->kind != WGSL_NODE_DECL_STRUCT_MEMBER) {
                wgsl_val_error(v, bs,
                    "@blend_src is only valid on a structure member");
            }
            if (!(stage_mask & STAGE_FR)) {
                wgsl_val_error(v, bs,
                    "@blend_src is only valid in a fragment entry point");
            } else if (is_input) {
                wgsl_val_error(v, bs,
                    "@blend_src is only valid on a fragment output, not "
                    "an input");
            }
            /* Type must be a numeric scalar / vector. */
            if (slot_type) {
                int ok_num =
                    wgsl_type_is_numeric(slot_type) ||
                    (slot_type->kind == WGSL_TYPE_VEC &&
                     wgsl_type_is_numeric((const WGSLTypeInfo *)slot_type->ref));
                if (!ok_num) {
                    wgsl_val_error(v, at,
                        "@blend_src target must have a numeric scalar or "
                        "vector type (got %s)",
                        wgsl_type_kind_name((WGSLTypeKind)slot_type->kind));
                }
            }
        }
    }

    if (blt && blt->child_count >= 2) {
        WGSLNode *arg = blt->children[1];
        if (arg && arg->kind == WGSL_NODE_EXPR_IDENT) {
            uint32_t off = wgsl_node_name_span(arg).offset;
            uint32_t len = wgsl_node_name_span(arg).length;
            const char *nm = v->src->bytes + off;
            /* Duplicate check (per entry point and direction).  Builtins
             * like sample_mask may legally appear once as input and once
             * as output on the same fragment entry point. */
            BuiltinSet *bs = is_input
                ? &io->seen_in_builtins : &io->seen_out_builtins;
            int add = builtinset_add(bs, nm, len);
            if (add == 0) {
                wgsl_val_error(v, blt,
                    "@builtin(%.*s) must not be specified more than "
                    "once on a single entry point", (int)len, nm);
            } else if (add < 0) {
                wgsl_val_error(v, blt,
                    "@builtin uniqueness analysis truncated "
                    "(out of memory)");
            }
            /* Stage / direction / type matrix from kBuiltinTable. */
            const BuiltinEntry *be = wgsl_val_find_builtin_entry(nm, len);
            if (be) {
                if ((be->stage_mask & stage_mask) == 0) {
                    wgsl_val_error(v, blt,
                        "@builtin(%.*s) is not valid on this shader "
                        "stage", (int)len, nm);
                }
                if (is_input && !be->dir_in) {
                    wgsl_val_error(v, blt,
                        "@builtin(%.*s) is an output, not an input",
                        (int)len, nm);
                }
                if (!is_input && !be->dir_out) {
                    wgsl_val_error(v, blt,
                        "@builtin(%.*s) is an input, not an output",
                        (int)len, nm);
                }
                if (len == 8 && memcmp(nm, "position", 8) == 0) {
                    if (is_input && !(stage_mask & STAGE_FR)) {
                        wgsl_val_error(v, blt,
                            "@builtin(position) is only an input for "
                            "fragment entry points");
                    }
                    if (!is_input && !(stage_mask & STAGE_VX)) {
                        wgsl_val_error(v, blt,
                            "@builtin(position) is only an output for "
                            "vertex entry points");
                    }
                }
                if (slot_type &&
                    !builtin_type_matches(v, slot_type, be->type))
                {
                    wgsl_val_error(v, at,
                        "@builtin(%.*s) requires type %s (got %s)",
                        (int)len, nm,
                        builtin_type_name(be->type),
                        wgsl_type_kind_name((WGSLTypeKind)slot_type->kind));
                }
            }
        }
    }
}

static void reject_struct_direct_io_attrs(
    WGSLValidator *v,
    WGSLNode *parent,
    uint32_t first,
    uint32_t count,
    const char *what)
{
    for (uint32_t i = 0; i < count && first + i < parent->child_count; i++) {
        WGSLNode *a = parent->children[first + i];
        if (!a || a->kind != WGSL_NODE_ATTRIBUTE) continue;
        if (!attr_is_entry_point_io(v, a)) continue;
        const char *nm = NULL;
        uint32_t nl = 0;
        attr_name_span(v, a, &nm, &nl);
        wgsl_val_error(v, a,
            "entry-point IO attribute '@%.*s' may not apply to a "
            "struct-typed entry %s; place it on the struct member "
            "instead",
            (int)nl, nm ? nm : "", what);
    }
}

static void validate_blend_src_struct_shape(WGSLValidator *v, const WGSLNode *sd) {
    if (!sd || sd->kind != WGSL_NODE_DECL_STRUCT) return;
    int blend_src_seen[2] = {0, 0};
    const WGSLTypeInfo *blend_src_type[2] = {NULL, NULL};
    long long blend_src_loc[2] = {-1, -1};
    WGSLNode *first_blend_attr = NULL;
    int location_count = 0;

    for (uint32_t i = 0; i < sd->child_count; i++) {
        WGSLNode *m = sd->children[i];
        if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
        uint32_t MA = wgsl_member_attr_count(m);
        WGSLNode *loc = wgsl_val_find_attr(v, m, 0, MA, "location");
        if (loc) location_count += 1;

        WGSLNode *bs_attr = wgsl_val_find_attr(v, m, 0, MA, "blend_src");
        if (!bs_attr) continue;
        if (!first_blend_attr) first_blend_attr = bs_attr;
        long long bn;
        if (!wgsl_val_attr_int_arg_typed(v, bs_attr, &bn) ||
            (bn != 0 && bn != 1))
        {
            continue;
        }
        blend_src_seen[bn] += 1;
        if (MA < m->child_count) {
            blend_src_type[bn] = wgsl_typecheck_type_of(v->tc, m->children[MA]);
        }
        if (loc) {
            long long lv;
            if (wgsl_val_attr_int_arg_typed(v, loc, &lv)) {
                blend_src_loc[bn] = lv;
            }
        }
    }

    if (!first_blend_attr) return;
    if (location_count != 2) {
        wgsl_val_error(v, sd,
            "dual-source blending requires exactly two @location "
            "members in the structure (got %d)", location_count);
    }
    if (blend_src_seen[0] != 1 || blend_src_seen[1] != 1) {
        wgsl_val_error(v, first_blend_attr,
            "dual-source blending requires exactly one @blend_src(0) "
            "and one @blend_src(1) member");
    }
    if (blend_src_seen[0] == 1 && blend_src_seen[1] == 1) {
        if (blend_src_loc[0] != 0 || blend_src_loc[1] != 0) {
            wgsl_val_error(v, first_blend_attr,
                "dual-source blending members must both be @location(0)");
        }
        if (blend_src_type[0] && blend_src_type[1] &&
            blend_src_type[0] != blend_src_type[1])
        {
            wgsl_val_error(v, first_blend_attr,
                "dual-source blending members must have the same type");
        }
    }
}

static void check_io_type(
    WGSLValidator *v, const WGSLNode *at, const WGSLTypeInfo *type,
    IOState *io, int is_input, uint32_t stage_mask, int depth)
{
    if (!type) return;
    if (type->kind == WGSL_TYPE_STRUCT) {
        if (depth > 0) { wgsl_val_error(v, at, "entry-point IO structs cannot nest"); return; }
        const WGSLNode *sd = (const WGSLNode *)type->ref;
        if (!sd) return;
        for (uint32_t i = 0; i < sd->child_count; i++) {
            WGSLNode *m = sd->children[i];
            if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            uint32_t MA = wgsl_member_attr_count(m);
            /* Resolve member type for shape / numeric / nested checks. */
            const WGSLTypeInfo *mt = NULL;
            if (MA < m->child_count) {
                mt = wgsl_typecheck_type_of(v->tc, m->children[MA]);
            }
            /* §13.3.1.3 — IO structs cannot nest.  Recurse one level
             * deep with depth+1 so the inner walk emits the diagnostic
             * on first encounter. */
            if (mt && mt->kind == WGSL_TYPE_STRUCT) {
                check_io_type(v, m, mt, io, is_input, stage_mask, depth + 1);
                continue;
            }
            check_io_slot(v, (WGSLNode *)m, (WGSLNode *)m, 0, MA, mt, io,
                          stage_mask, is_input);
        }
    }
}

static void validate_fn_io(WGSLValidator *v, WGSLNode *fn, uint32_t stage_mask) {
    IOState state = {0};
    uint32_t FA = wgsl_fn_attr_count(fn);
    uint32_t P  = wgsl_fn_param_count(fn);
    uint32_t RA = wgsl_fn_ret_attr_count(fn);
    uint32_t HR = wgsl_fn_has_return_type(fn);
    for (uint32_t k = 0; k < P; k++) {
        if (FA + k >= fn->child_count) break;
        WGSLNode *p = fn->children[FA + k];
        if (!p || p->kind != WGSL_NODE_DECL_PARAM) continue;
        uint32_t PA = wgsl_param_attr_count(p);
        WGSLTypeInfo *pt = NULL;
        if (PA < p->child_count) pt = wgsl_typecheck_type_of(v->tc, p->children[PA]);

        /* §13.3.1.3 — `@location` may not apply to a struct-typed
         * param; the same direct-attribute rule applies to all entry
         * IO attrs on struct-typed params (members own their IO attrs). */
        if (pt && pt->kind == WGSL_TYPE_STRUCT) {
            reject_struct_direct_io_attrs(v, p, 0, PA, "parameter");
            check_io_type(v, p, pt, &state, 1, stage_mask, 0);
            continue;
        }
        /* §13.3.1.2 — direct (non-struct) param needs @location or
         * @builtin; §13.3.1.4 etc. apply per-slot. */
        check_io_slot(v, p, p, 0, PA, pt, &state, stage_mask, 1);

        /* §13.3.1.2 — compute stage user-defined IO is checked per slot. */
    }
    if (HR) {
        WGSLTypeInfo *rt = NULL;
        WGSLNode *ret_spec = NULL;
        if (FA + P + RA < fn->child_count) {
            ret_spec = fn->children[FA + P + RA];
            rt = wgsl_typecheck_type_of(v->tc, ret_spec);
        }
        if (rt && rt->kind == WGSL_TYPE_STRUCT) {
            reject_struct_direct_io_attrs(v, fn, FA + P, RA, "return type");
            check_io_type(v, ret_spec ? ret_spec : fn, rt, &state, 0, stage_mask, 0);
        } else {
            check_io_slot(v, ret_spec ? ret_spec : fn, fn, FA + P, RA, rt,
                          &state, stage_mask, 0);
            /* §13.3.1.2 — compute stage user-defined IO is checked per slot. */
        }
    }
    iostate_free(&state);
}

static int name_has_prefix(const char *name, uint32_t len, const char *prefix) {
    size_t plen = strlen(prefix);
    return len >= plen && memcmp(name, prefix, plen) == 0;
}

static int name_eq(const char *name, uint32_t len, const char *want) {
    size_t wlen = strlen(want);
    return len == wlen && memcmp(name, want, wlen) == 0;
}

static int is_subgroup_or_quad_builtin(const WGSLSymbol *s) {
    return s && s->kind == WGSL_SYM_PREDECLARED_FN &&
           (name_has_prefix(s->name, s->name_len, "subgroup") ||
            name_has_prefix(s->name, s->name_len, "quad"));
}

static int is_atomic_builtin(const WGSLSymbol *s) {
    return s && s->kind == WGSL_SYM_PREDECLARED_FN &&
           name_has_prefix(s->name, s->name_len, "atomic");
}

static int is_compute_only_builtin(const WGSLSymbol *s) {
    if (!s || s->kind != WGSL_SYM_PREDECLARED_FN) return 0;
    const char *name = s->name;
    uint32_t len = s->name_len;
    return name_eq(name, len, "workgroupBarrier") ||
           name_eq(name, len, "storageBarrier") ||
           name_eq(name, len, "textureBarrier") ||
           name_eq(name, len, "workgroupUniformLoad");
}

static int is_fragment_only_builtin(const WGSLSymbol *s) {
    if (!s || s->kind != WGSL_SYM_PREDECLARED_FN) return 0;
    const char *name = s->name;
    uint32_t len = s->name_len;
    return name_eq(name, len, "dpdx") ||
           name_eq(name, len, "dpdxCoarse") ||
           name_eq(name, len, "dpdxFine") ||
           name_eq(name, len, "dpdy") ||
           name_eq(name, len, "dpdyCoarse") ||
           name_eq(name, len, "dpdyFine") ||
           name_eq(name, len, "fwidth") ||
           name_eq(name, len, "fwidthCoarse") ||
           name_eq(name, len, "fwidthFine") ||
           name_eq(name, len, "textureSample") ||
           name_eq(name, len, "textureSampleBias") ||
           name_eq(name, len, "textureSampleCompare");
}

static WGSLSymbol *call_callee_symbol(const WGSLNode *call) {
    if (!call || call->kind != WGSL_NODE_EXPR_CALL || call->child_count < 1) {
        return NULL;
    }
    const WGSLNode *callee = call->children[0];
    if (!callee) return NULL;
    if (callee->kind == WGSL_NODE_EXPR_IDENT) {
        return wgsl_node_resolved_symbol(callee);
    }
    if (callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
        callee->child_count >= 1 &&
        callee->children[0])
    {
        return wgsl_node_resolved_symbol(callee->children[0]);
    }
    return NULL;
}

static void validate_stage_reachable_calls_node(
    WGSLValidator *v, WGSLNode *n, uint32_t stage_mask,
    uint8_t *visited, size_t visited_n)
{
    if (!n) return;

    if (n->kind == WGSL_NODE_EXPR_CALL) {
        WGSLSymbol *callee = call_callee_symbol(n);
        if (is_compute_only_builtin(callee)) {
            if (stage_mask & (STAGE_VX | STAGE_FR)) {
                wgsl_val_error(v, n,
                    "builtin '%.*s' is only valid in code reachable "
                    "from a compute entry point",
                    (int)callee->name_len, callee->name);
            }
        } else if (is_fragment_only_builtin(callee)) {
            if (stage_mask & (STAGE_VX | STAGE_CO)) {
                wgsl_val_error(v, n,
                    "builtin '%.*s' is only valid in code reachable "
                    "from a fragment entry point",
                    (int)callee->name_len, callee->name);
            }
        } else if (is_subgroup_or_quad_builtin(callee) || is_atomic_builtin(callee)) {
            if (stage_mask & STAGE_VX) {
                wgsl_val_error(v, n,
                    "builtin '%.*s' is not valid in vertex stage code",
                    (int)callee->name_len, callee->name);
            }
        } else if (callee && callee->kind == WGSL_SYM_FUNCTION && callee->ast) {
            int idx = wgsl_val_sym_index(v, callee);
            if (idx >= 0 && (size_t)idx < visited_n && !visited[idx]) {
                visited[idx] = 1;
                validate_stage_reachable_calls_node(
                    v, callee->ast, stage_mask, visited, visited_n);
            }
        }
    }

    for (uint32_t i = 0; i < n->child_count; i++) {
        validate_stage_reachable_calls_node(
            v, n->children[i], stage_mask, visited, visited_n);
    }
}

static void validate_stage_reachable_calls(
    WGSLValidator *v, WGSLNode *entry, uint32_t stage_mask)
{
    if (!v || !v->res || !entry) return;
    size_t n = v->res->all_decl_count;
    if (n == 0) {
        validate_stage_reachable_calls_node(v, entry, stage_mask, NULL, 0);
        return;
    }
    if (n > SIZE_MAX / sizeof(uint8_t)) return;
    uint8_t *visited = (uint8_t *)calloc(n, sizeof *visited);
    if (!visited) return;
    WGSLSymbol *entry_sym = wgsl_val_find_symbol_for_ast(v, entry);
    int entry_idx = wgsl_val_sym_index(v, entry_sym);
    if (entry_idx >= 0 && (size_t)entry_idx < n) visited[entry_idx] = 1;
    validate_stage_reachable_calls_node(v, entry, stage_mask, visited, n);
    free(visited);
}

void validate_entry_points(WGSLValidator *v, WGSLNode *tu) {
    if (!tu) return;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (n && n->kind == WGSL_NODE_DECL_STRUCT) {
            validate_blend_src_struct_shape(v, n);
        }
    }
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *fn = tu->children[i];
        if (!fn || fn->kind != WGSL_NODE_DECL_FUNCTION) continue;

        uint32_t FA = wgsl_fn_attr_count(fn);
        WGSLNode *vx = wgsl_val_find_attr(v, fn, 0, FA, "vertex");
        WGSLNode *fr = wgsl_val_find_attr(v, fn, 0, FA, "fragment");
        WGSLNode *co = wgsl_val_find_attr(v, fn, 0, FA, "compute");
        WGSLNode *ws = wgsl_val_find_attr(v, fn, 0, FA, "workgroup_size");

        int stage_count = (vx ? 1 : 0) + (fr ? 1 : 0) + (co ? 1 : 0);
        if (stage_count > 1) {
            wgsl_val_error(v, fn, "more than one shader-stage attribute on a function");
        }

        if (vx || fr || co) {
            uint32_t mask = (vx ? STAGE_VX : 0) | (fr ? STAGE_FR : 0) | (co ? STAGE_CO : 0);
            validate_fn_io(v, fn, mask);
            validate_stage_reachable_calls(v, fn, mask);
        }

        if (co && !ws) {
            wgsl_val_error(v, fn, "entry point using `@compute` requires a `@workgroup_size` attribute");
        }

        if (ws && !co) {
            wgsl_val_error(v, ws, "`@workgroup_size` is only valid on `@compute` entry points");
        }

        /* §13.2 — a compute entry point must not have a return type. */
        if (co) {
            uint32_t HR_co = wgsl_fn_has_return_type(fn);
            if (HR_co) {
                /* Point the diagnostic at the return type spec if we
                 * can locate it. */
                uint32_t P_co  = wgsl_fn_param_count(fn);
                uint32_t RA_co = wgsl_fn_ret_attr_count(fn);
                WGSLNode *ret_spec = NULL;
                if (FA + P_co + RA_co < fn->child_count) {
                    ret_spec = fn->children[FA + P_co + RA_co];
                }
                wgsl_val_error(v, ret_spec ? ret_spec : fn,
                    "@compute entry point must not have a return type");
            }
        }

        /* §12.14 — @workgroup_size(x [, y [, z]]) per-arg checks.
         * Attribute argument types aren't carried on the side-table
         * (the type checker only walks fn bodies / param tspecs /
         * return tspecs, not attribute children), so we drive these
         * checks through const-eval and inspect the resolved value
         * kind. */
        if (ws && v->cev) {
            WGSLTypeInfo *common_t = NULL;
            for (uint32_t k = 1; k < ws->child_count; k++) {
                WGSLNode *arg = ws->children[k];
                if (!arg) continue;
                size_t sc = v->diag->count;
                int    se = v->diag->error_count;
                int    cv = v->cev->store.had_error;
                WGSLValue val = {0};
                int ok_eval = wgsl_consteval_expr(v->cev, arg, &val);
                if (!ok_eval) {
                    /* Non-const arg.  Roll back the "not a constant"
                     * diagnostic and verify the arg is at least an
                     * override-expression (literal / paren / unary /
                     * binary / const-ident / override-ident).  `var` /
                     * `let` / fn-param refs are spec-illegal here. */
                    v->diag->count       = sc;
                    v->diag->error_count = se;
                    v->cev->store.had_error    = cv;
                    if (!wgsl_typecheck_is_override_expr(arg)) {
                        wgsl_val_error(v, arg,
                            "@workgroup_size argument must be a const- "
                            "or override-expression");
                    }
                    continue;
                }
                if (val.kind == WGSL_VAL_FLOAT ||
                    (val.type && (val.type == v->tc->types->t_f32 ||
                                  val.type == v->tc->types->t_f16 ||
                                  val.type == v->tc->types->t_abstract_float)))
                {
                    wgsl_val_error(v, arg,
                        "@workgroup_size arguments must be i32 or u32, "
                        "not a floating-point type (f32 / f16)");
                    continue;
                }
                if (val.kind == WGSL_VAL_INT && val.u.i <= 0) {
                    wgsl_val_error(v, arg,
                        "@workgroup_size argument must be a positive "
                        "const- or override-expression (got %lld)",
                        (long long)val.u.i);
                }
                /* Type-compatibility across args: pin to first concrete
                 * type seen; AbstractInt is compatible with either i32
                 * or u32. */
                if (val.type) {
                    if (!common_t) {
                        common_t = val.type;
                    } else if (val.type != common_t &&
                               val.type != v->tc->types->t_abstract_int &&
                               common_t != v->tc->types->t_abstract_int)
                    {
                        wgsl_val_error(v, arg,
                            "@workgroup_size arguments must all be the "
                            "same type (i32 or u32)");
                    }
                }
            }
        }

        /* §12.12 — `@must_use` requires the function to have a return
         * type; applying it to a void fn is meaningless. */
        WGSLNode *mu = wgsl_val_find_attr(v, fn, 0, FA, "must_use");
        if (mu) {
            uint32_t HR = wgsl_fn_has_return_type(fn);
            if (!HR) {
                wgsl_val_error(v, mu,
                    "@must_use is only valid on a function with a return type");
            }
        }
    }
}
