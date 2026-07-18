/**
 * @file overloads.c — overload tables and §6.1.3 resolver.
 */
#include "internal/exprs_priv.h"
#include "check/builtins.gen.h"

/* overload resolution machinery. */

int builtin_call_is_const(const WGSLNode *n);
int builtin_call_is_override(const WGSLNode *n);

/* A const-expression predicate (§7.2.1 / §15).  Walks `n` and returns 1
 * iff the value is fixed at shader-generation time.  Conservative: any
 * shape we do not explicitly recognize is treated as non-const, which
 * matches the step-2 elimination intent (prefer concrete when in doubt). */
int is_const_expr(const WGSLNode *n) {
    if (!n) return 0;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_LITERAL_BOOL:
    case WGSL_NODE_EXPR_LITERAL_INT:
    case WGSL_NODE_EXPR_LITERAL_FLOAT:
        return 1;
    case WGSL_NODE_EXPR_PAREN:
        return n->child_count == 1 && is_const_expr(n->children[0]);
    case WGSL_NODE_EXPR_UNARY:
        return n->child_count == 1 && is_const_expr(n->children[0]);
    case WGSL_NODE_EXPR_BINARY:
        return n->child_count == 2 &&
               is_const_expr(n->children[0]) &&
               is_const_expr(n->children[1]);
    case WGSL_NODE_EXPR_IDENT: {
        WGSLSymbol *s = wgsl_node_resolved_symbol((WGSLNode *)n);
        return s && s->kind == WGSL_SYM_CONST;
    }
    case WGSL_NODE_EXPR_CALL:
        if (builtin_call_is_const(n)) return 1;
        /* §11.3 pure user functions (const/let + return of const-exprs). */
        return user_fn_call_is_const_expr(n);
    default:
        return 0;
    }
}

/* §7.2.2 / §15: an override-expression is a superset of const-expression
 * that also accepts references to module-scope `override` symbols.
 * Used to validate `override X = init;` where init may reference other
 * overrides (`override Y = X * 2;` is legal) but not non-const vars or
 * fn params. */
int wgsl_tc_is_override_expr(const WGSLNode *n) {
    if (!n) return 0;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_LITERAL_BOOL:
    case WGSL_NODE_EXPR_LITERAL_INT:
    case WGSL_NODE_EXPR_LITERAL_FLOAT:
        return 1;
    case WGSL_NODE_EXPR_PAREN:
        return n->child_count == 1 && wgsl_tc_is_override_expr(n->children[0]);
    case WGSL_NODE_EXPR_UNARY:
        return n->child_count == 1 && wgsl_tc_is_override_expr(n->children[0]);
    case WGSL_NODE_EXPR_BINARY:
        return n->child_count == 2 &&
               wgsl_tc_is_override_expr(n->children[0]) &&
               wgsl_tc_is_override_expr(n->children[1]);
    case WGSL_NODE_EXPR_INDEX:
        return n->child_count == 2 &&
               wgsl_tc_is_override_expr(n->children[0]) &&
               wgsl_tc_is_override_expr(n->children[1]);
    case WGSL_NODE_EXPR_MEMBER:
        return n->child_count >= 1 &&
               wgsl_tc_is_override_expr(n->children[0]);
    case WGSL_NODE_EXPR_IDENT: {
        WGSLSymbol *s = wgsl_node_resolved_symbol((WGSLNode *)n);
        return s && (s->kind == WGSL_SYM_CONST || s->kind == WGSL_SYM_OVERRIDE);
    }
    case WGSL_NODE_EXPR_CALL:
        return builtin_call_is_override(n);
    default:
        return 0;
    }
}

int wgsl_typecheck_is_override_expr(const WGSLNode *n) {
    return wgsl_tc_is_override_expr(n);
}

/* Types: exprs_priv.h */

/* §3.4 — format helpers for detailed overload diagnostics. */

void ol_append(char *buf, size_t cap, size_t *used, const char *s) {
    if (!buf || !used || !s || cap == 0) return;
    size_t n = strlen(s);
    if (*used + n + 1 >= cap) n = (*used < cap - 1) ? (cap - 1 - *used) : 0;
    if (n) memcpy(buf + *used, s, n);
    *used += n;
    buf[*used] = '\0';
}

void ol_append_type(char *buf, size_t cap, size_t *used, WGSLTypeInfo *t) {
    char tmp[96];
    if (!t) { ol_append(buf, cap, used, "?"); return; }
    size_t n = wgsl_type_format(t, tmp, sizeof tmp);
    if (n >= sizeof tmp) n = sizeof tmp - 1;
    tmp[n] = '\0';
    ol_append(buf, cap, used, tmp);
}

void ol_append_args(char *buf, size_t cap, size_t *used,
                           WGSLTypeInfo *const *arg_types, int arg_count)
{
    ol_append(buf, cap, used, "(");
    for (int i = 0; i < arg_count; i++) {
        if (i) ol_append(buf, cap, used, ", ");
        ol_append_type(buf, cap, used, arg_types[i]);
    }
    ol_append(buf, cap, used, ")");
}

void ol_append_cand(char *buf, size_t cap, size_t *used,
                           const WGSLOverloadCandidate *c, int arg_count)
{
    ol_append(buf, cap, used, "(");
    for (int i = 0; i < arg_count; i++) {
        if (i) ol_append(buf, cap, used, ", ");
        ol_append_type(buf, cap, used, c->params[i]);
    }
    ol_append(buf, cap, used, ") -> ");
    ol_append_type(buf, cap, used, c->result);
}

/* §6.1.3 — overload resolution.
 *
 *  Step 1: drop candidates with an infeasible conversion on any arg.
 *  Step 2: drop candidates where one parameter is abstract while
 *          another argument is not a const-expression.
 *  Step 3: rank candidates pairwise — C1 is preferred over C2 iff
 *          R_C1[i] ≤ R_C2[i] for every i, with strict inequality on
 *          at least one i.
 *  Step 4: if exactly one candidate is preferred over every other
 *          survivor, succeed.  Otherwise fail.
 *
 * Returns the chosen index or -1 on failure.  Emits a multi-line
 * diagnostic (§3.4) explaining which candidates were
 * eliminated at which step. */
int resolve_overload(
    WGSLTypeChecker *tc,
    const WGSLOverloadCandidate *cands, int cand_count,
    WGSLTypeInfo *const *arg_types, const int *arg_is_const, int arg_count,
    const char *what, const WGSLNode *at)
{
    if (cand_count <= 0 || arg_count <= 0 || arg_count > WGSL_OL_MAX_PARAMS) {
        return -1;
    }

    int ranks[WGSL_OL_MAX_CANDS][WGSL_OL_MAX_PARAMS];
    int alive[WGSL_OL_MAX_CANDS] = {0};
    int elim_step[WGSL_OL_MAX_CANDS]; /* 0=alive, 1/2=eliminated at step */
    int elim_arg[WGSL_OL_MAX_CANDS];  /* step-1 bad arg index */
    int alive_count = 0;
    for (int c = 0; c < cand_count; c++) {
        elim_step[c] = 0;
        elim_arg[c] = -1;
    }

    /* Step 1: conversion-rank feasibility. */
    for (int c = 0; c < cand_count; c++) {
        alive[c] = 1;
        for (int i = 0; i < arg_count; i++) {
            int r = wgsl_type_conversion_rank(arg_types[i], cands[c].params[i]);
            if (r < 0) {
                alive[c] = 0;
                elim_step[c] = 1;
                elim_arg[c] = i;
                break;
            }
            ranks[c][i] = r;
        }
        if (alive[c]) alive_count += 1;
    }
    if (alive_count == 0) {
        char msg[768];
        size_t u = 0;
        msg[0] = '\0';
        ol_append(msg, sizeof msg, &u, "no matching overload for '");
        ol_append(msg, sizeof msg, &u, what ? what : "?");
        ol_append(msg, sizeof msg, &u, "'\n  arguments: ");
        ol_append_args(msg, sizeof msg, &u, arg_types, arg_count);
        ol_append(msg, sizeof msg, &u,
                  "\n  step 1 (conversion): every candidate rejected:");
        int shown = 0;
        for (int c = 0; c < cand_count && shown < 4; c++) {
            if (elim_step[c] != 1) continue;
            ol_append(msg, sizeof msg, &u, "\n    ");
            ol_append_cand(msg, sizeof msg, &u, &cands[c], arg_count);
            if (elim_arg[c] >= 0) {
                char argno[16];
                char ta[64], tp[64];
                wgsl_type_format(arg_types[elim_arg[c]], ta, sizeof ta);
                wgsl_type_format(cands[c].params[elim_arg[c]], tp, sizeof tp);
                snprintf(argno, sizeof argno, "%d", elim_arg[c]);
                ol_append(msg, sizeof msg, &u, " — arg ");
                ol_append(msg, sizeof msg, &u, argno);
                ol_append(msg, sizeof msg, &u, ": no conversion from ");
                ol_append(msg, sizeof msg, &u, ta);
                ol_append(msg, sizeof msg, &u, " to ");
                ol_append(msg, sizeof msg, &u, tp);
            }
            shown++;
        }
        if (cand_count > shown)
            ol_append(msg, sizeof msg, &u, "\n    …");
        wgsl_tc_error(tc, at, "%s", msg);
        return -1;
    }

    /* Step 2: eliminate abstract-after-conversion + non-const-elsewhere. */
    if (arg_count > 1) {
        int has_nonconst = 0;
        for (int j = 0; j < arg_count; j++) {
            if (!arg_is_const[j]) { has_nonconst = 1; break; }
        }
        if (has_nonconst) {
            for (int c = 0; c < cand_count; c++) {
                if (!alive[c]) continue;
                int any_abstract_param = 0;
                for (int i = 0; i < arg_count; i++) {
                    if (wgsl_type_is_abstract(cands[c].params[i])) {
                        any_abstract_param = 1;
                        break;
                    }
                }
                if (any_abstract_param) {
                    alive[c] = 0;
                    elim_step[c] = 2;
                    alive_count -= 1;
                }
            }
        }
        if (alive_count == 0) {
            char msg[768];
            size_t u = 0;
            msg[0] = '\0';
            ol_append(msg, sizeof msg, &u, "no matching overload for '");
            ol_append(msg, sizeof msg, &u, what ? what : "?");
            ol_append(msg, sizeof msg, &u, "'\n  arguments: ");
            ol_append_args(msg, sizeof msg, &u, arg_types, arg_count);
            ol_append(msg, sizeof msg, &u,
                      "\n  step 2 (abstract vs non-const): at least one argument "
                      "is not a const-expression, so candidates with abstract "
                      "parameter types are rejected:");
            int shown = 0;
            for (int c = 0; c < cand_count && shown < 4; c++) {
                if (elim_step[c] != 2) continue;
                ol_append(msg, sizeof msg, &u, "\n    ");
                ol_append_cand(msg, sizeof msg, &u, &cands[c], arg_count);
                shown++;
            }
            wgsl_tc_error(tc, at, "%s", msg);
            return -1;
        }
    }

    /* Step 3+4: pairwise preference, pick unique winner. */
    int best = -1;
    int survivors[WGSL_OL_MAX_CANDS];
    int nsurv = 0;
    for (int c = 0; c < cand_count; c++) {
        if (alive[c]) survivors[nsurv++] = c;
    }

    for (int c = 0; c < cand_count; c++) {
        if (!alive[c]) continue;
        int preferred_over_all = 1;
        for (int o = 0; o < cand_count; o++) {
            if (o == c || !alive[o]) continue;
            int all_le = 1, some_lt = 0;
            for (int i = 0; i < arg_count; i++) {
                if (ranks[c][i] > ranks[o][i]) { all_le = 0; break; }
                if (ranks[c][i] < ranks[o][i]) some_lt = 1;
            }
            if (!(all_le && some_lt)) { preferred_over_all = 0; break; }
        }
        if (preferred_over_all) {
            if (best == -1) {
                best = c;
            } else {
                best = -1;
                break;
            }
        }
    }
    if (best == -1) {
        char msg[768];
        size_t u = 0;
        msg[0] = '\0';
        ol_append(msg, sizeof msg, &u, "ambiguous overload for '");
        ol_append(msg, sizeof msg, &u, what ? what : "?");
        ol_append(msg, sizeof msg, &u, "'\n  arguments: ");
        ol_append_args(msg, sizeof msg, &u, arg_types, arg_count);
        ol_append(msg, sizeof msg, &u,
                  "\n  step 3 (ranking): no unique preferred candidate among:");
        for (int i = 0; i < nsurv && i < 6; i++) {
            ol_append(msg, sizeof msg, &u, "\n    ");
            ol_append_cand(msg, sizeof msg, &u, &cands[survivors[i]], arg_count);
        }
        wgsl_tc_error(tc, at, "%s", msg);
        return -1;
    }
    return best;
}

/* Lift a scalar T into the shape carried by `shape` (scalar / vec / mat). */
WGSLTypeInfo *lift_to_shape(
    WGSLTypeStore *ts, WGSLTypeInfo *T, const WGSLTypeInfo *shape)
{
    if (!shape || !T) return T;
    if (shape->kind == WGSL_TYPE_VEC) {
        return wgsl_type_vec(ts, shape->width, T);
    }
    if (shape->kind == WGSL_TYPE_MAT) {
        return wgsl_type_mat(ts, shape->width, shape->rows, T);
    }
    return T;
}

/* Pick a "shape source" arg — prefer vec/mat over scalar so a mixed
 * scalar/composite call still finds the composite shape. */
const WGSLTypeInfo *pick_shape_arg(
    WGSLTypeInfo *const *args, int n)
{
    for (int i = 0; i < n; i++) {
        if (!args[i]) continue;
        if (args[i]->kind == WGSL_TYPE_VEC || args[i]->kind == WGSL_TYPE_MAT) {
            return args[i];
        }
    }
    return n > 0 ? args[0] : NULL;
}

/* Build the candidate set for a "T-pattern" builtin: all `arity` params
 * share the same shape (derived from the args), with the element ranging
 * over the family's scalar set.  Returns the number of candidates. */
int build_t_candidates(
    WGSLTypeChecker *tc, WGSLParamFamily fam,
    WGSLTypeInfo *const *args, int arity,
    WGSLOverloadCandidate *out_cands)
{
    const WGSLTypeInfo *shape = pick_shape_arg(args, arity);
    if (!shape) return 0;

    WGSLTypeInfo *scalars[8];
    int sc = 0;
    switch (fam) {
    case PF_FLOAT_T:
        scalars[sc++] = tc->types->t_abstract_float;
        scalars[sc++] = tc->types->t_f32;
        if (tc->enabled_extensions & WGSL_EXT_F16) {
            scalars[sc++] = tc->types->t_f16;
        }
        break;
    case PF_NUMERIC_T:
        scalars[sc++] = tc->types->t_abstract_int;
        scalars[sc++] = tc->types->t_abstract_float;
        scalars[sc++] = tc->types->t_i32;
        scalars[sc++] = tc->types->t_u32;
        scalars[sc++] = tc->types->t_f32;
        if (tc->enabled_extensions & WGSL_EXT_F16) {
            scalars[sc++] = tc->types->t_f16;
        }
        break;
    case PF_INTEGER_T:
        scalars[sc++] = tc->types->t_abstract_int;
        scalars[sc++] = tc->types->t_i32;
        scalars[sc++] = tc->types->t_u32;
        break;
    case PF_SIGN_T:
        scalars[sc++] = tc->types->t_abstract_int;
        scalars[sc++] = tc->types->t_abstract_float;
        scalars[sc++] = tc->types->t_i32;
        scalars[sc++] = tc->types->t_f32;
        if (tc->enabled_extensions & WGSL_EXT_F16) {
            scalars[sc++] = tc->types->t_f16;
        }
        break;
    case PF_F32_T:
        scalars[sc++] = tc->types->t_f32;
        break;
    default:
        return 0;
    }

    int n = 0;
    for (int i = 0; i < sc && n < WGSL_OL_MAX_CANDS; i++) {
        WGSLTypeInfo *lifted = lift_to_shape(tc->types, scalars[i], shape);
        WGSLOverloadCandidate *cnd = &out_cands[n];
        for (int a = 0; a < arity; a++) cnd->params[a] = lifted;
        cnd->result = lifted;
        n += 1;
    }
    return n;
}

/* Builtin overload table. */

/* Builtin/texture row types: exprs_priv.h */

/* Generated from def/wgsl.def.  The fam / arity fields drive §6.1.3
 * overload resolution; the flags field carries explicit @const / @must_use
 * metadata, and special hooks cover mixed-shape overloads that do not fit
 * the compact T-pattern.
 *
 * kBuiltinTable and kBuiltinOverloadIndex are sorted by name (§4.2) so
 * lookups are binary-search O(log n); overload rows for a name form a
 * contiguous range [begin, begin+count) in kBuiltinOverloads. */

/* Lexicographic compare of (name, name_len) vs a C string key. */
int name_cmp_cstr(const char *name, uint32_t name_len, const char *key)
{
    size_t kl = strlen(key);
    size_t n = name_len < kl ? name_len : kl;
    int c = memcmp(name, key, n);
    if (c != 0) return c;
    if (name_len < kl) return -1;
    if (name_len > kl) return 1;
    return 0;
}

/* kBuiltinTable is sorted by name — binary search. */
const WGSLBuiltinEntry *find_builtin_entry(
    const char *name, uint32_t name_len)
{
    size_t lo = 0, hi = (size_t)WGSL_BUILTIN_TABLE_COUNT;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        const WGSLBuiltinEntry *e = &kBuiltinTable[mid];
        int c = name_cmp_cstr(name, name_len, e->name);
        if (c == 0) return e;
        if (c < 0) hi = mid;
        else lo = mid + 1;
    }
    return NULL;
}

/* kBuiltinOverloadIndex is sorted by name — binary search → row range. */
const WGSLBuiltinOverloadIndex *find_overload_index(
    const char *name, uint32_t name_len)
{
    size_t lo = 0, hi = (size_t)WGSL_BUILTIN_OVERLOAD_INDEX_COUNT;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        const WGSLBuiltinOverloadIndex *e = &kBuiltinOverloadIndex[mid];
        int c = name_cmp_cstr(name, name_len, e->name);
        if (c == 0) return e;
        if (c < 0) hi = mid;
        else lo = mid + 1;
    }
    return NULL;
}

int wgsl_tc_builtin_must_use(const char *name, uint32_t name_len) {
    const WGSLBuiltinEntry *e = find_builtin_entry(name, name_len);
    return e && (e->flags & WGSL_BIF_MUST_USE);
}

int sym_is_constructor_callable(const WGSLSymbol *sym) {
    if (!sym) return 0;
    switch ((WGSLSymKind)sym->kind) {
    case WGSL_SYM_PREDECLARED_TYPE:
    case WGSL_SYM_PREDECLARED_TYPEGEN:
    case WGSL_SYM_PREDECLARED_TYPE_ALIAS:
    case WGSL_SYM_TYPE_ALIAS:
    case WGSL_SYM_STRUCT:
        return 1;
    default:
        return 0;
    }
}

int builtin_call_is_const(const WGSLNode *n) {
    if (!n || n->kind != WGSL_NODE_EXPR_CALL || n->child_count < 1) return 0;
    const WGSLNode *callee = n->children[0];
    if (!callee) return 0;

    WGSLSymbol *sym = NULL;
    if (callee->kind == WGSL_NODE_EXPR_IDENT) {
        sym = wgsl_node_resolved_symbol((WGSLNode *)callee);
    } else if (callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
               callee->child_count >= 1 &&
               callee->children[0])
    {
        sym = wgsl_node_resolved_symbol(callee->children[0]);
    }
    if (!sym) return 0;
    if (sym->kind == WGSL_SYM_PREDECLARED_FN) {
        const WGSLBuiltinEntry *e = find_builtin_entry(sym->name, sym->name_len);
        if (!e || !(e->flags & WGSL_BIF_CONST)) return 0;
    } else if (!sym_is_constructor_callable(sym)) {
        return 0;
    }
    for (uint32_t i = 1; i < n->child_count; i++) {
        if (!is_const_expr(n->children[i])) return 0;
    }
    return 1;
}

int builtin_call_is_override(const WGSLNode *n) {
    if (!n || n->kind != WGSL_NODE_EXPR_CALL || n->child_count < 1) return 0;
    const WGSLNode *callee = n->children[0];
    if (!callee) return 0;

    WGSLSymbol *sym = NULL;
    if (callee->kind == WGSL_NODE_EXPR_IDENT) {
        sym = wgsl_node_resolved_symbol((WGSLNode *)callee);
    } else if (callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
               callee->child_count >= 1 &&
               callee->children[0])
    {
        sym = wgsl_node_resolved_symbol(callee->children[0]);
    }
    if (!sym) return 0;
    if (sym->kind == WGSL_SYM_PREDECLARED_FN) {
        const WGSLBuiltinEntry *e = find_builtin_entry(sym->name, sym->name_len);
        if (!e || !(e->flags & WGSL_BIF_CONST)) return 0;
    } else if (!sym_is_constructor_callable(sym)) {
        return 0;
    }
    for (uint32_t i = 1; i < n->child_count; i++) {
        if (!wgsl_tc_is_override_expr(n->children[i])) return 0;
    }
    return 1;
}


void check_const_builtin_call(
    WGSLTypeChecker *tc, WGSLNode *call, const WGSLBuiltinEntry *entry)
{
    if (!entry || !(entry->flags & WGSL_BIF_CONST)) return;
    if (!is_const_expr(call)) return;
    WGSLValue ignored = {0};
    (void)try_consteval_expr_soft(tc, call, &ignored);
}
