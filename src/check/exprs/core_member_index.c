/**
 * @file core_member_index.c — member/swizzle and index expression typing
 */
#include "internal/exprs_priv.h"


/* EXPR_MEMBER: swizzle (vec) + field access (struct). */

static int swizzle_index(char c) {
    switch (c) {
    case 'x': case 'r': return 0;
    case 'y': case 'g': return 1;
    case 'z': case 'b': return 2;
    case 'w': case 'a': return 3;
    default:            return -1;
    }
}

/* §8.5.1 — `xyzw` and `rgba` letterings must not be mixed within a
 * swizzle.  Returns 1 if every character in `name[0..len-1]` belongs to
 * the same set, 0 otherwise. */
int swizzle_lettering_consistent(const char *name, uint32_t len) {
    int saw_xyzw = 0, saw_rgba = 0;
    for (uint32_t i = 0; i < len; i++) {
        char c = name[i];
        if (c == 'x' || c == 'y' || c == 'z' || c == 'w') saw_xyzw = 1;
        else if (c == 'r' || c == 'g' || c == 'b' || c == 'a') saw_rgba = 1;
    }
    return !(saw_xyzw && saw_rgba);
}

int type_member(WGSLTypeChecker *tc, WGSLNode *n) {
    if (n->child_count < 2) return 0;
    WGSLNode *obj    = n->children[0];
    WGSLNode *member = n->children[1];
    if (!obj || !member) return 0;
    if (!wgsl_tc_type_expr(tc, obj)) return 0;
    WGSLTypeInfo *ot = wgsl_tc_store_type_of(tc, obj);
    if (!ot) return 0;
    int obj_is_ref = (obj->flags & WGSL_FLAG_IS_REF) != 0;
    if (ot->kind == WGSL_TYPE_PTR) {
        ot = (WGSLTypeInfo *)ot->ref;
        obj_is_ref = 1;
    }

    uint32_t off = wgsl_node_name_span(member).offset;
    uint32_t len = wgsl_node_name_span(member).length;
    const char *name = tc->src->bytes + off;

    /* Vector swizzle. */
    if (ot->kind == WGSL_TYPE_VEC) {
        WGSLTypeInfo *elem = (WGSLTypeInfo *)ot->ref;
        if (len < 1 || len > 4) {
            wgsl_tc_error(tc, member, "swizzle must be 1-4 characters");
            return 0;
        }
        for (uint32_t i = 0; i < len; i++) {
            int idx = swizzle_index(name[i]);
            if (idx < 0) {
                wgsl_tc_error(tc, member,
                    "invalid swizzle character '%c'", name[i]);
                return 0;
            }
            if (idx >= ot->width) {
                wgsl_tc_error(tc, member,
                    "swizzle component '%c' out of range for vec%u",
                    name[i], (unsigned)ot->width);
                return 0;
            }
        }
        /* §8.5.1 — `xyzw` and `rgba` letterings must not be mixed. */
        if (!swizzle_lettering_consistent(name, len)) {
            wgsl_tc_error(tc, member,
                "mixed swizzle letterings: components must all be from "
                "the {x,y,z,w} set or the {r,g,b,a} set, not both");
            return 0;
        }
        if (len == 1) {
            /* A single-component swizzle on a Ref<vec> is itself a Ref;
             * a multi-component swizzle is always a value (not LHS). */
            return wgsl_tc_set_type(tc, n, elem, obj_is_ref);
        }
        WGSLTypeInfo *vt = wgsl_type_vec(tc->types, (uint8_t)len, elem);
        return wgsl_tc_set_type(tc, n, vt, 0);
    }

    /* Struct field access. */
    if (ot->kind == WGSL_TYPE_STRUCT) {
        WGSLNode *sd = (WGSLNode *)ot->ref;
        if (!sd) {
            wgsl_tc_error(tc, n, "struct has no field information");
            return 0;
        }
        for (uint32_t i = 0; i < sd->child_count; i++) {
            WGSLNode *m = sd->children[i];
            if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            uint32_t mn_off = wgsl_node_name_span(m).offset;
            uint32_t mn_len = wgsl_node_name_span(m).length;
            if (mn_len == len &&
                memcmp(tc->src->bytes + mn_off, name, len) == 0)
            {
                uint32_t attrs = wgsl_member_attr_count(m);
                if (attrs >= m->child_count) return 0;
                WGSLTypeInfo *ft = wgsl_tc_resolve_type_spec(tc, m->children[attrs]);
                if (!ft) return 0;
                return wgsl_tc_set_type(tc, n, ft, obj_is_ref);
            }
        }
        wgsl_tc_error(tc, member,
            "struct has no field '%.*s'", (int)len, name);
        return 0;
    }

    /* Synthetic atomic CX result struct. */
    if (ot->kind == WGSL_TYPE_ATOMIC_CX_RESULT) {
        if (len == 9 && memcmp(name, "old_value", 9) == 0) {
            return wgsl_tc_set_type(tc, n, (WGSLTypeInfo *)ot->ref, obj_is_ref);
        }
        if (len == 9 && memcmp(name, "exchanged", 9) == 0) {
            return wgsl_tc_set_type(tc, n, tc->types->t_bool, obj_is_ref);
        }
    }

    /* §17 — `__frexp_result<T>` { fract: T, exp: I } where I is i32 if T
     * is scalar, or vecN<i32> matching T's vec width. */
    if (ot->kind == WGSL_TYPE_FREXP_RESULT) {
        WGSLTypeInfo *T = (WGSLTypeInfo *)ot->ref;
        if (len == 5 && memcmp(name, "fract", 5) == 0) {
            return wgsl_tc_set_type(tc, n, T, obj_is_ref);
        }
        if (len == 3 && memcmp(name, "exp", 3) == 0) {
            WGSLTypeInfo *I = tc->types->t_i32;
            if (T && T->kind == WGSL_TYPE_VEC) {
                I = wgsl_type_vec(tc->types, T->width, tc->types->t_i32);
            }
            return wgsl_tc_set_type(tc, n, I, obj_is_ref);
        }
        wgsl_tc_error(tc, n, "__frexp_result has no field '%.*s' "
                 "(expected 'fract' or 'exp')", (int)len, name);
        return 0;
    }

    /* §17 — `__modf_result<T>` { fract: T, whole: T }. */
    if (ot->kind == WGSL_TYPE_MODF_RESULT) {
        WGSLTypeInfo *T = (WGSLTypeInfo *)ot->ref;
        if (len == 5 && memcmp(name, "fract", 5) == 0) {
            return wgsl_tc_set_type(tc, n, T, obj_is_ref);
        }
        if (len == 5 && memcmp(name, "whole", 5) == 0) {
            return wgsl_tc_set_type(tc, n, T, obj_is_ref);
        }
        wgsl_tc_error(tc, n, "__modf_result has no field '%.*s' "
                 "(expected 'fract' or 'whole')", (int)len, name);
        return 0;
    }

    /* Permissive fallback: keep the object type when member layout is not
     * available at this point.  The validator reports strict struct-field
     * errors with full type context. */
    return wgsl_tc_set_type(tc, n, ot, obj_is_ref);
}

/* EXPR_INDEX: vec / mat / array indexing. */

int type_index(WGSLTypeChecker *tc, WGSLNode *n) {
    if (n->child_count != 2 || !n->children[0] || !n->children[1]) {
        wgsl_tc_error(tc, n, "malformed index expression");
        return 0;
    }
    if (!wgsl_tc_type_expr(tc, n->children[0])) return 0;
    if (!wgsl_tc_type_expr(tc, n->children[1])) return 0;
    WGSLTypeInfo *base = wgsl_tc_store_type_of(tc, n->children[0]);
    if (!base) return 0;
    WGSLTypeInfo *index = wgsl_tc_store_type_of(tc, n->children[1]);
    if (!index) return 0;
    int base_is_ref = (n->children[0]->flags & WGSL_FLAG_IS_REF) != 0;
    if (base->kind == WGSL_TYPE_PTR) {
        base = (WGSLTypeInfo *)base->ref;
        base_is_ref = 1;
    }

    WGSLTypeInfo *result = NULL;
    uint32_t bound = 0;          /* 0 = no static bound (runtime array)  */
    const char *kind_name = "";
    switch ((WGSLTypeKind)base->kind) {
    case WGSL_TYPE_VEC:
        result = (WGSLTypeInfo *)base->ref;
        bound = base->width;
        kind_name = "vec";
        break;
    case WGSL_TYPE_MAT:
        /* mat[i] returns a column: vec<rows, T>. */
        result = wgsl_type_vec(tc->types, base->rows, (WGSLTypeInfo *)base->ref);
        bound = base->width;
        kind_name = "matrix";
        break;
    case WGSL_TYPE_ARRAY:
        result = (WGSLTypeInfo *)base->ref;
        bound = base->array_len;  /* 0 → runtime-sized; skip the check.  */
        kind_name = "array";
        break;
    default:
        wgsl_tc_error(tc, n,
            "type %s does not support indexing",
            wgsl_type_kind_name((WGSLTypeKind)base->kind));
        return 0;
    }

    if (!wgsl_type_is_scalar(index) || !wgsl_type_is_integer(index)) {
        char got[96];
        wgsl_type_format(index, got, sizeof got);
        wgsl_tc_error(tc, n->children[1],
            "%s index has type %s; expected integer scalar",
            kind_name, got);
        return 0;
    }

    /* §8.5.{1,2,3} — const-expression index must be in range.  Drive
     * `wgsl_consteval_expr` over the index so we silently roll back any
     * "not a constant" diagnostics for runtime indices, but keep the
     * "out of range" / "negative" diagnostic the materializer emits. */
    if (tc->cev && tc->diag) {
        size_t saved_count = tc->diag->count;
        int    saved_errs  = tc->diag->error_count;
        int    saved_cev   = tc->cev->store.had_error;
        WGSLValue v = {0};
        if (wgsl_consteval_expr(tc->cev, n->children[1], &v) &&
            v.kind == WGSL_VAL_INT)
        {
            int64_t iv = v.u.i;
            if (iv < 0) {
                wgsl_tc_error(tc, n->children[1],
                    "negative const-expression index '%lld' for %s",
                    (long long)iv, kind_name);
            } else if (bound > 0 && (uint64_t)iv >= bound) {
                wgsl_tc_error(tc, n->children[1],
                    "const-expression index %lld out of range for %s of "
                    "length %u", (long long)iv, kind_name, (unsigned)bound);
            }
        } else {
            /* Index isn't a const-expression — drop any consteval
             * diagnostics emitted along the way; the runtime bound
             * check belongs to the executor, not the type checker. */
            tc->diag->count       = saved_count;
            tc->diag->error_count = saved_errs;
            tc->cev->store.had_error    = saved_cev;
        }
    }

    /* Indexing into a Ref<…> yields a Ref<elem>; into a value yields a value. */
    return wgsl_tc_set_type(tc, n, result, base_is_ref);
}
