/**
 * @file check.c — WGSL type checker driver.
 *
 * Slim driver that ties together the type checker's four translation
 * units: this file (driver / side-table / walk_stmt / function-decl /
 * directives / entry point), types.c (type-spec resolution),
 * decls.c (decl typing + init-vs-decl compatibility), and
 * exprs.c (expression typing + builtin overload resolution).
 *
 * Cross-file sharing happens through `internal/check_priv.h`.
 */
#include "internal/check.h"
#include "internal/check_priv.h"
#include "internal/check_walk_priv.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal/token.h"

/* Diagnostics. */

void wgsl_tc_error(
    WGSLTypeChecker *tc, const WGSLNode *at, const char *fmt, ...)
{
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    uint32_t off = at ? at->span_offset : 0;
    uint32_t len = at ? at->span_length : 0;
    wgsl_diag_emit_at(tc->diag, tc->src, WGSL_DIAG_ERROR,
                      off, len, NULL, "%s", buf);
    tc->had_error = 1;
}

/* Side-table (open-addressed node* hash, §4.1). */

#define TC_NODE_HTAB_EMPTY ((size_t)-1)

static uint64_t tc_ptr_hash(const void *p) {
    uint64_t x = (uint64_t)(uintptr_t)p;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return x;
}

static int tc_node_htab_rehash(WGSLTypeChecker *tc, size_t new_cap) {
    if (new_cap < 16) new_cap = 16;
    size_t cap = 16;
    while (cap < new_cap) {
        if (cap > SIZE_MAX / 2) return 0;
        cap *= 2;
    }
    size_t *nt = (size_t *)malloc(cap * sizeof *nt);
    if (!nt) return 0;
    for (size_t i = 0; i < cap; i++) nt[i] = TC_NODE_HTAB_EMPTY;
    for (size_t i = 0; i < tc->count; i++) {
        uint64_t h = tc_ptr_hash(tc->table[i].node);
        size_t mask = cap - 1, slot = (size_t)h & mask;
        while (nt[slot] != TC_NODE_HTAB_EMPTY) slot = (slot + 1) & mask;
        nt[slot] = i;
    }
    free(tc->node_htab);
    tc->node_htab = nt;
    tc->node_htab_cap = cap;
    return 1;
}

static size_t tc_node_lookup(const WGSLTypeChecker *tc, const WGSLNode *n) {
    if (!tc->node_htab || !tc->node_htab_cap || !n) return TC_NODE_HTAB_EMPTY;
    uint64_t h = tc_ptr_hash(n);
    size_t mask = tc->node_htab_cap - 1, slot = (size_t)h & mask;
    for (size_t k = 0; k < tc->node_htab_cap; k++) {
        size_t idx = tc->node_htab[slot];
        if (idx == TC_NODE_HTAB_EMPTY) return TC_NODE_HTAB_EMPTY;
        if (idx < tc->count && tc->table[idx].node == n) return idx;
        slot = (slot + 1) & mask;
    }
    return TC_NODE_HTAB_EMPTY;
}

static int tc_node_htab_insert(WGSLTypeChecker *tc, size_t idx) {
    if (!tc->node_htab || tc->node_htab_cap == 0) {
        if (!tc_node_htab_rehash(tc, 64)) return 0;
    }
    if (tc->count * 10 >= tc->node_htab_cap * 7) {
        if (!tc_node_htab_rehash(tc, tc->node_htab_cap * 2)) return 0;
    }
    const WGSLNode *n = tc->table[idx].node;
    uint64_t h = tc_ptr_hash(n);
    size_t mask = tc->node_htab_cap - 1, slot = (size_t)h & mask;
    for (;;) {
        size_t cur = tc->node_htab[slot];
        if (cur == TC_NODE_HTAB_EMPTY) {
            tc->node_htab[slot] = idx;
            return 1;
        }
        if (cur < tc->count && tc->table[cur].node == n) {
            tc->node_htab[slot] = idx; /* update to latest */
            return 1;
        }
        slot = (slot + 1) & mask;
    }
}

int wgsl_tc_set_type(
    WGSLTypeChecker *tc, WGSLNode *n, WGSLTypeInfo *t, int is_ref)
{
    if (!n || !t) return 0;
    WGSLTypeInfo *stored = t;
    if (is_ref && t->kind != WGSL_TYPE_REF) {
        WGSLAddressSpace as = WGSL_AS_NONE;
        WGSLAccessMode   am = WGSL_ACCESS_NONE;
        wgsl_tc_get_ref_space_and_mode(tc, n, &as, &am);
        stored = wgsl_type_ref(tc->types, (uint8_t)as, t, am);
        if (!stored) return 0;
    }
    /* Update in place if already recorded. */
    size_t existing = tc_node_lookup(tc, n);
    if (existing != TC_NODE_HTAB_EMPTY && existing < tc->count) {
        tc->table[existing].type  = stored;
        tc->table[existing].flags = is_ref ? WGSL_FLAG_IS_REF : 0;
        n->flags |= WGSL_FLAG_TYPED;
        if (is_ref) n->flags |= WGSL_FLAG_IS_REF;
        else        n->flags &= (uint16_t)~WGSL_FLAG_IS_REF;
        return 1;
    }
    if (tc->count == tc->capacity) {
        if (tc->capacity > SIZE_MAX / 2) return 0;
        size_t cap = tc->capacity ? tc->capacity * 2 : 64;
        if (cap > SIZE_MAX / sizeof *tc->table) return 0;
        WGSLNodeType *g = (WGSLNodeType *)realloc(
            tc->table, cap * sizeof *g);
        if (!g) return 0;
        tc->table = g;
        tc->capacity = cap;
    }
    size_t idx = tc->count;
    tc->table[tc->count].node  = n;
    tc->table[tc->count].type  = stored;
    tc->table[tc->count].flags = is_ref ? WGSL_FLAG_IS_REF : 0;
    tc->count += 1;
    (void)tc_node_htab_insert(tc, idx);
    n->flags |= WGSL_FLAG_TYPED;
    if (is_ref) n->flags |= WGSL_FLAG_IS_REF;
    else        n->flags &= (uint16_t)~WGSL_FLAG_IS_REF;
    return 1;
}

WGSLTypeInfo *wgsl_typecheck_type_of(
    const WGSLTypeChecker *tc, const WGSLNode *n)
{
    if (!tc || !n) return NULL;
    size_t idx = tc_node_lookup(tc, n);
    if (idx != TC_NODE_HTAB_EMPTY && idx < tc->count)
        return tc->table[idx].type;
    if (n->kind == WGSL_NODE_EXPR_IDENT) {
        WGSLSymbol *s = wgsl_node_resolved_symbol(n);
        if (s && s->type) return s->type;
    }
    return NULL;
}

WGSLTypeInfo *wgsl_tc_unwrap_ref_type(WGSLTypeInfo *t) {
    if (t && t->kind == WGSL_TYPE_REF) return (WGSLTypeInfo *)t->ref;
    return t;
}

WGSLTypeInfo *wgsl_tc_store_type_of(
    const WGSLTypeChecker *tc, const WGSLNode *n)
{
    return wgsl_tc_unwrap_ref_type(wgsl_typecheck_type_of(tc, n));
}

int wgsl_typecheck_is_ref(
    const WGSLTypeChecker *tc, const WGSLNode *n)
{
    if (!tc || !n) return 0;
    if (n->flags & WGSL_FLAG_IS_REF) return 1;
    WGSLTypeInfo *t = wgsl_typecheck_type_of(tc, n);
    return t && t->kind == WGSL_TYPE_REF;
}

static WGSLTypeInfo *tc_elem_type(WGSLTypeInfo *t) {
    if (!t) return NULL;
    if (wgsl_type_is_scalar(t)) return t;
    if (t->kind == WGSL_TYPE_VEC ||
        t->kind == WGSL_TYPE_MAT ||
        t->kind == WGSL_TYPE_ARRAY ||
        t->kind == WGSL_TYPE_ATOMIC)
    {
        return (WGSLTypeInfo *)t->ref;
    }
    return NULL;
}

int tc_same_shape(WGSLTypeInfo *a, WGSLTypeInfo *b) {
    if (!a || !b) return 0;
    if (wgsl_type_is_scalar(a) && wgsl_type_is_scalar(b)) return 1;
    if (a->kind == WGSL_TYPE_VEC && b->kind == WGSL_TYPE_VEC) {
        return a->width == b->width;
    }
    if (a->kind == WGSL_TYPE_MAT && b->kind == WGSL_TYPE_MAT) {
        return a->width == b->width && a->rows == b->rows;
    }
    return 0;
}

int tc_can_convert(WGSLTypeInfo *from, WGSLTypeInfo *to) {
    return from && to && wgsl_type_conversion_rank(from, to) >= 0;
}

WGSLTokenKind compound_binary_op(WGSLTokenKind op) {
    switch (op) {
    case WGSL_TOK_PLUS_EQUAL:              return WGSL_TOK_PLUS;
    case WGSL_TOK_MINUS_EQUAL:             return WGSL_TOK_MINUS;
    case WGSL_TOK_STAR_EQUAL:              return WGSL_TOK_STAR;
    case WGSL_TOK_SLASH_EQUAL:             return WGSL_TOK_SLASH;
    case WGSL_TOK_PERCENT_EQUAL:           return WGSL_TOK_PERCENT;
    case WGSL_TOK_AMP_EQUAL:               return WGSL_TOK_AMP;
    case WGSL_TOK_PIPE_EQUAL:              return WGSL_TOK_PIPE;
    case WGSL_TOK_CARET_EQUAL:             return WGSL_TOK_CARET;
    case WGSL_TOK_LESS_LESS_EQUAL:         return WGSL_TOK_LESS_LESS;
    case WGSL_TOK_GREATER_GREATER_EQUAL:   return WGSL_TOK_GREATER_GREATER;
    default:                               return WGSL_TOK_INVALID;
    }
}

int validate_compound_assignment(
    WGSLTypeChecker *tc, WGSLNode *stmt, WGSLTypeInfo *lt, WGSLTypeInfo *rt)
{
    if (!tc || !stmt || !lt || !rt) return 0;
    WGSLTokenKind bin = compound_binary_op((WGSLTokenKind)wgsl_node_op_kind_u32(stmt));
    WGSLTypeInfo *le = tc_elem_type(lt);
    WGSLTypeInfo *re = tc_elem_type(rt);
    if (!le || !re) {
        wgsl_tc_error(tc, stmt, "compound assignment requires typed operands");
        return 0;
    }

    switch (bin) {
    case WGSL_TOK_PLUS:
    case WGSL_TOK_MINUS:
    case WGSL_TOK_SLASH:
    case WGSL_TOK_PERCENT:
        if (!(tc_same_shape(lt, rt) ||
              ((lt->kind == WGSL_TYPE_VEC || lt->kind == WGSL_TYPE_MAT) &&
               wgsl_type_is_scalar(rt))))
        {
            wgsl_tc_error(tc, stmt,
                "compound assignment operands must have compatible shapes");
            return 0;
        }
        if (!wgsl_type_is_numeric(le) || !tc_can_convert(re, le)) {
            wgsl_tc_error(tc, stmt,
                "compound assignment operator requires numeric operands");
            return 0;
        }
        if ((bin == WGSL_TOK_SLASH || bin == WGSL_TOK_PERCENT) &&
            stmt->child_count >= 2)
        {
            wgsl_tc_check_constexpr_divisor_nonzero(
                tc, stmt->children[1], lt,
                bin == WGSL_TOK_SLASH ? "division by zero" : "modulo by zero");
        }
        return 1;
    case WGSL_TOK_STAR: {
        int shape_ok = tc_same_shape(lt, rt) ||
            ((lt->kind == WGSL_TYPE_VEC || lt->kind == WGSL_TYPE_MAT) &&
             wgsl_type_is_scalar(rt));
        if (!shape_ok) {
            wgsl_tc_error(tc, stmt,
                "compound assignment operands must have compatible shapes");
            return 0;
        }
        if (!wgsl_type_is_numeric(le) || !tc_can_convert(re, le)) {
            wgsl_tc_error(tc, stmt,
                "compound assignment operator requires numeric operands");
            return 0;
        }
        return 1;
    }
    case WGSL_TOK_AMP:
    case WGSL_TOK_PIPE:
    case WGSL_TOK_CARET:
        if (!tc_same_shape(lt, rt)) {
            wgsl_tc_error(tc, stmt,
                "compound assignment operands must have matching shapes");
            return 0;
        }
        if (le == tc->types->t_bool || re == tc->types->t_bool) {
            if (bin == WGSL_TOK_CARET ||
                le != tc->types->t_bool || re != tc->types->t_bool)
            {
                wgsl_tc_error(tc, stmt,
                    "compound bitwise assignment requires integer operands");
                return 0;
            }
            return 1;
        }
        if (!wgsl_type_is_integer(le) || !tc_can_convert(re, le)) {
            wgsl_tc_error(tc, stmt,
                "compound bitwise assignment requires integer operands");
            return 0;
        }
        return 1;
    case WGSL_TOK_LESS_LESS:
    case WGSL_TOK_GREATER_GREATER: {
        int lhs_ok = wgsl_type_is_integer(le);
        int rhs_ok = re == tc->types->t_u32 || re == tc->types->t_abstract_int;
        if (!lhs_ok || !rhs_ok || !tc_same_shape(lt, rt)) {
            wgsl_tc_error(tc, stmt,
                "compound shift assignment requires integer lhs and u32 rhs");
            return 0;
        }
        return 1;
    }
    default:
        wgsl_tc_error(tc, stmt, "unsupported compound assignment operator");
        return 0;
    }
}

void wgsl_typecheck_destroy(WGSLTypeChecker *tc) {
    if (!tc) return;
    free(tc->table);
    free(tc->node_htab);
    free(tc->override_ids);
    memset(tc, 0, sizeof *tc);
}

/* Symbol lookup helper. */

WGSLSymbol *wgsl_tc_find_decl_symbol(
    const WGSLTypeChecker *tc, const WGSLNode *decl)
{
    if (!tc || !tc->res) return NULL;
    return wgsl_resolver_symbol_for_decl(tc->res, decl);
}


