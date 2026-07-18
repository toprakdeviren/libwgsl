/**
 * @file ref_dispatch.c — expression typing module (was exprs/ref_dispatch.inc).
 */
#include "internal/exprs_priv.h"

void wgsl_tc_get_ref_space_and_mode(WGSLTypeChecker *tc, const WGSLNode *n, WGSLAddressSpace *as, WGSLAccessMode *am) {
    if (!n) return;
    WGSLTypeInfo *typed = wgsl_typecheck_type_of(tc, n);
    if (typed &&
        (typed->kind == WGSL_TYPE_PTR || typed->kind == WGSL_TYPE_REF))
    {
        *as = (WGSLAddressSpace)(typed->flags & 0xFFu);
        *am = (WGSLAccessMode)((typed->flags >> 8) & 0xFFu);
        return;
    }
    if (n->kind == WGSL_NODE_EXPR_PAREN) {
        if (n->child_count > 0) {
            wgsl_tc_get_ref_space_and_mode(tc, n->children[0], as, am);
        }
        return;
    }
    if (n->kind == WGSL_NODE_EXPR_IDENT) {
        WGSLSymbol *s = wgsl_node_resolved_symbol((WGSLNode*)n);
        if (s) { *as = (WGSLAddressSpace)s->as; *am = (WGSLAccessMode)s->am; }
        return;
    }
    if (n->kind == WGSL_NODE_EXPR_MEMBER || n->kind == WGSL_NODE_EXPR_INDEX) {
        if (n->child_count > 0) wgsl_tc_get_ref_space_and_mode(tc, n->children[0], as, am);
        return;
    }
    if (n->kind == WGSL_NODE_EXPR_INDIRECTION) {
        if (n->child_count > 0) {
            WGSLTypeInfo *pt = wgsl_tc_store_type_of(tc, n->children[0]);
            if (pt && pt->kind == WGSL_TYPE_PTR) {
                *as = (WGSLAddressSpace)(pt->flags & 0xFF);
                *am = (WGSLAccessMode)((pt->flags >> 8) & 0xFF);
            }
        }
        return;
    }
}

int type_addr_of(WGSLTypeChecker *tc, WGSLNode *n) {
    if (n->child_count == 0 || !n->children[0]) return 0;
    WGSLNode *expr = n->children[0];
    if (!wgsl_tc_type_expr(tc, expr)) return 0;

    if (!wgsl_typecheck_is_ref(tc, expr)) {
        wgsl_tc_error(tc, expr, "cannot take address of value; operand must be a reference");
        return 0;
    }

    /* §8.13 — `&` cannot take the address of a single vector component
     * (`&v.x`).  `v.x` is a ref<vec elem> per the single-letter swizzle
     * rule, which would otherwise satisfy the ref check above.  Detect
     * the EXPR_MEMBER-on-a-vec pattern with a 1-character member name. */
    if (expr->kind == WGSL_NODE_EXPR_MEMBER && expr->child_count >= 2) {
        WGSLNode *base = expr->children[0];
        WGSLNode *mem  = expr->children[1];
        WGSLTypeInfo *bt = base ? wgsl_tc_store_type_of(tc, base) : NULL;
        if (bt && bt->kind == WGSL_TYPE_VEC && mem) {
            uint32_t mlen = wgsl_node_name_span(mem).length;
            if (mlen == 1) {
                wgsl_tc_error(tc, n,
                    "cannot take address of a vector component "
                    "(spec §8.13)");
                return 0;
            }
        }
    }

    WGSLTypeInfo *elem = wgsl_tc_store_type_of(tc, expr);
    if (!elem) return 0;

    /* §8.13 — `&t` on a texture / sampler is rejected.  Handle-AS
     * resources cannot have their address taken; only var-allocated
     * memory in function/private/workgroup/uniform/storage can. */
    if (elem->kind == WGSL_TYPE_TEXTURE || elem->kind == WGSL_TYPE_SAMPLER) {
        wgsl_tc_error(tc, n,
            "cannot take address of a handle-address-space variable "
            "(spec §8.13)");
        return 0;
    }

    WGSLAddressSpace as = WGSL_AS_NONE;
    WGSLAccessMode   am = WGSL_ACCESS_NONE;
    wgsl_tc_get_ref_space_and_mode(tc, expr, &as, &am);
    if (as == WGSL_AS_HANDLE) {
        wgsl_tc_error(tc, n,
            "cannot take address of a handle-address-space variable "
            "(spec §8.13)");
        return 0;
    }

    WGSLTypeInfo *ptr = wgsl_type_ptr(tc->types, as, elem, am);
    return wgsl_tc_set_type(tc, n, ptr, 0);
}

int type_indirection(WGSLTypeChecker *tc, WGSLNode *n) {
    if (n->child_count == 0 || !n->children[0]) return 0;
    WGSLNode *expr = n->children[0];
    if (!wgsl_tc_type_expr(tc, expr)) return 0;

    WGSLTypeInfo *ptr = wgsl_tc_store_type_of(tc, expr);
    if (!ptr || ptr->kind != WGSL_TYPE_PTR) {
        wgsl_tc_error(tc, expr, "cannot dereference non-pointer type");
        return 0;
    }

    WGSLTypeInfo *elem = (WGSLTypeInfo*)ptr->ref;
    return wgsl_tc_set_type(tc, n, elem, 1);
}

int wgsl_tc_type_expr_inner(
    WGSLTypeChecker *tc, WGSLNode *n, WGSLTypeInfo *expected);

/* Depth-guarded entry.  All sub-expression type-checking funnels through
 * these (type_unary/binary/member/index/etc. only recurse by calling
 * them again), so one guard bounds every deep-AST expression vector.
 * The cheap early-outs run BEFORE the depth counter so they never leak a
 * balance.
 *
 * §4.3 — `expected` is the contextual type from the enclosing conversion
 * site (decl init, assignment, return, call arg, …).  NULL = free
 * bottom-up inference. */
int wgsl_tc_type_expr(WGSLTypeChecker *tc, WGSLNode *n) {
    return wgsl_tc_type_expr_exp(tc, n, NULL);
}

int wgsl_tc_type_expr_exp(
    WGSLTypeChecker *tc, WGSLNode *n, WGSLTypeInfo *expected)
{
    if (!n) return 0;
    if (n->flags & WGSL_FLAG_TYPED) {
        /* Already typed — do not re-stamp.  Use-site conversion rank +
         * range checks still apply to the stored abstract type. */
        (void)expected;
        return 1;
    }
    if (++tc->expr_depth > WGSL_MAX_AST_DEPTH) {
        wgsl_tc_error(tc, n, "expression nested too deeply (max %d)",
                      WGSL_MAX_AST_DEPTH);
        --tc->expr_depth;
        return 0;
    }
    int r = wgsl_tc_type_expr_inner(tc, n, expected);
    --tc->expr_depth;
    return r;
}

int wgsl_tc_type_expr_inner(
    WGSLTypeChecker *tc, WGSLNode *n, WGSLTypeInfo *expected)
{
    if (!n) return 0;
    if (n->flags & WGSL_FLAG_TYPED) return 1;

    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_LITERAL_INT:
        return type_literal_int(tc, n, expected);
    case WGSL_NODE_EXPR_LITERAL_FLOAT:
        return type_literal_float(tc, n, expected);
    case WGSL_NODE_EXPR_LITERAL_BOOL:
        return wgsl_tc_set_type(tc, n, tc->types->t_bool, 0);

    case WGSL_NODE_EXPR_IDENT: return type_ident(tc, n);

    case WGSL_NODE_EXPR_PAREN: {
        if (n->child_count == 0 || !n->children[0]) return 0;
        if (!wgsl_tc_type_expr_exp(tc, n->children[0], expected)) return 0;
        WGSLTypeInfo *t = wgsl_tc_store_type_of(tc, n->children[0]);
        int is_ref = (n->children[0]->flags & WGSL_FLAG_IS_REF) != 0;
        return wgsl_tc_set_type(tc, n, t, is_ref);
    }

    case WGSL_NODE_EXPR_UNARY:  return type_unary (tc, n, expected);
    case WGSL_NODE_EXPR_BINARY: return type_binary(tc, n, expected);
    case WGSL_NODE_EXPR_CALL:   return type_call  (tc, n, expected);
    case WGSL_NODE_EXPR_MEMBER: return type_member(tc, n);
    case WGSL_NODE_EXPR_INDEX:  return type_index (tc, n);

    case WGSL_NODE_EXPR_ADDR_OF:        return type_addr_of(tc, n);
    case WGSL_NODE_EXPR_INDIRECTION:    return type_indirection(tc, n);

    /* templated-ident-as-value: walk children for descendant typing, but do
     * not assign a value type here. */
    case WGSL_NODE_EXPR_TEMPLATED_IDENT:
        for (uint32_t i = 0; i < n->child_count; i++) {
            wgsl_tc_type_expr(tc, n->children[i]);
        }
        return 1;

    default:
        return 1;
    }
}
