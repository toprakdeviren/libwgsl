/**
 * @file decls_init.c — init-type compatibility and constructible-deep helpers
 */
#include "internal/check_priv.h"

#include <stdlib.h>
#include <string.h>

/* Init-type compatibility. */

/* §6.2.12 deep-constructible check.  The shallow predicate in types.c
 * conservatively returns 1 for struct kinds because struct member types
 * live on the side-table (keyed by member-typespec AST node), not on
 * the struct WGSLTypeInfo.  This walker pulls each member's resolved
 * type via `wgsl_typecheck_type_of` and recurses, mirroring the
 * "all members must be constructible" spec rule. */
static int wgsl_tc_is_constructible_deep_impl(
    const WGSLTypeChecker *tc, const WGSLTypeInfo *t, int depth)
{
    if (!t) return 0;
    /* Deep struct/alias type chain — assume constructible; the over-deep
     * type is rejected elsewhere.  See decl_type_contains_runtime_array. */
    if (depth > WGSL_MAX_AST_DEPTH) return 1;
    if (!wgsl_type_is_constructible(t)) return 0;
    if (t->kind == WGSL_TYPE_ARRAY) {
        return wgsl_tc_is_constructible_deep_impl(
            tc, (const WGSLTypeInfo *)t->ref, depth + 1);
    }
    if (t->kind == WGSL_TYPE_STRUCT) {
        const WGSLNode *sd = (const WGSLNode *)t->ref;
        if (!sd) return 1;  /* unresolved; trust the shallow answer */
        for (uint32_t i = 0; i < sd->child_count; i++) {
            const WGSLNode *m = sd->children[i];
            if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            uint32_t attrs = wgsl_member_attr_count(m);
            if (attrs >= m->child_count) continue;
            const WGSLNode *tspec = m->children[attrs];
            WGSLTypeInfo *mt = wgsl_tc_store_type_of(tc, tspec);
            if (!mt) continue;
            if (!wgsl_tc_is_constructible_deep_impl(tc, mt, depth + 1)) return 0;
        }
        return 1;
    }
    return 1;
}

int wgsl_tc_is_constructible_deep(
    const WGSLTypeChecker *tc, const WGSLTypeInfo *t)
{
    return wgsl_tc_is_constructible_deep_impl(tc, t, 0);
}

static int decl_type_contains_runtime_array_impl(
    const WGSLTypeChecker *tc, const WGSLTypeInfo *t,
    const WGSLTypeInfo **seen, int seen_n, int depth)
{
    if (!t) return 0;
    /* Bound recursion on the resolved type graph: a deep struct/alias
     * chain (`struct S1{a:S0}…`) is neither AST-deep nor template-bounded
     * and would otherwise overflow the stack.  Deeper = treated as "no
     * runtime array here"; the over-deep type is rejected elsewhere. */
    if (depth > WGSL_MAX_AST_DEPTH) return 0;
    if (t->kind == WGSL_TYPE_ARRAY) {
        if (t->array_len == 0 && t->flags == 0 && t->pad_ == 0) return 1;
        return decl_type_contains_runtime_array_impl(
            tc, (const WGSLTypeInfo *)t->ref, seen, seen_n, depth + 1);
    }
    if (t->kind == WGSL_TYPE_STRUCT) {
        for (int i = 0; i < seen_n; i++) {
            if (seen[i] == t) return 0;
        }
        const WGSLTypeInfo *local_seen[64];
        int local_n = seen_n;
        if (local_n > 64) local_n = 64;
        for (int i = 0; i < local_n; i++) local_seen[i] = seen[i];
        if (local_n < 64) local_seen[local_n++] = t;

        const WGSLNode *sd = (const WGSLNode *)t->ref;
        if (!sd) return 0;
        for (uint32_t i = 0; i < sd->child_count; i++) {
            const WGSLNode *m = sd->children[i];
            if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            uint32_t attrs = wgsl_member_attr_count(m);
            if (attrs >= m->child_count) continue;
            const WGSLTypeInfo *mt =
                wgsl_tc_store_type_of(tc, m->children[attrs]);
            if (decl_type_contains_runtime_array_impl(
                    tc, mt, local_seen, local_n, depth + 1))
            {
                return 1;
            }
        }
    }
    return 0;
}

int decl_type_contains_runtime_array(
    const WGSLTypeChecker *tc, const WGSLTypeInfo *t)
{
    const WGSLTypeInfo *seen[1] = {0};
    return decl_type_contains_runtime_array_impl(tc, t, seen, 0, 0);
}

static int decl_type_contains_override_array_impl(
    const WGSLTypeChecker *tc, const WGSLTypeInfo *t,
    const WGSLTypeInfo **seen, int seen_n, int depth)
{
    if (!t) return 0;
    if (depth > WGSL_MAX_AST_DEPTH) return 0;   /* deep type graph — see above */
    if (t->kind == WGSL_TYPE_ARRAY) {
        if (t->array_len == 0 && (t->flags != 0 || t->pad_ != 0)) return 1;
        return decl_type_contains_override_array_impl(
            tc, (const WGSLTypeInfo *)t->ref, seen, seen_n, depth + 1);
    }
    if (t->kind == WGSL_TYPE_STRUCT) {
        for (int i = 0; i < seen_n; i++) {
            if (seen[i] == t) return 0;
        }
        const WGSLTypeInfo *local_seen[64];
        int local_n = seen_n;
        if (local_n > 64) local_n = 64;
        for (int i = 0; i < local_n; i++) local_seen[i] = seen[i];
        if (local_n < 64) local_seen[local_n++] = t;

        const WGSLNode *sd = (const WGSLNode *)t->ref;
        if (!sd) return 0;
        for (uint32_t i = 0; i < sd->child_count; i++) {
            const WGSLNode *m = sd->children[i];
            if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            uint32_t attrs = wgsl_member_attr_count(m);
            if (attrs >= m->child_count) continue;
            const WGSLTypeInfo *mt =
                wgsl_tc_store_type_of(tc, m->children[attrs]);
            if (decl_type_contains_override_array_impl(
                    tc, mt, local_seen, local_n, depth + 1))
            {
                return 1;
            }
        }
    }
    return 0;
}

int decl_type_contains_override_array(
    const WGSLTypeChecker *tc, const WGSLTypeInfo *t)
{
    const WGSLTypeInfo *seen[1] = {0};
    return decl_type_contains_override_array_impl(tc, t, seen, 0, 0);
}

static int decl_type_contains_atomic_impl(
    const WGSLTypeChecker *tc, const WGSLTypeInfo *t, int depth)
{
    if (!t) return 0;
    if (depth > WGSL_MAX_AST_DEPTH) return 0;   /* deep type graph — see above */
    if (t->kind == WGSL_TYPE_ATOMIC) return 1;
    if (t->kind == WGSL_TYPE_ARRAY) {
        return decl_type_contains_atomic_impl(
            tc, (const WGSLTypeInfo *)t->ref, depth + 1);
    }
    if (t->kind == WGSL_TYPE_STRUCT) {
        const WGSLNode *sd = (const WGSLNode *)t->ref;
        if (!sd) return 0;
        for (uint32_t i = 0; i < sd->child_count; i++) {
            const WGSLNode *m = sd->children[i];
            if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            uint32_t attrs = wgsl_member_attr_count(m);
            if (attrs >= m->child_count) continue;
            const WGSLTypeInfo *mt =
                wgsl_tc_store_type_of(tc, m->children[attrs]);
            if (decl_type_contains_atomic_impl(tc, mt, depth + 1)) return 1;
        }
    }
    return 0;
}

int decl_type_contains_atomic(
    const WGSLTypeChecker *tc, const WGSLTypeInfo *t)
{
    return decl_type_contains_atomic_impl(tc, t, 0);
}

static int wgsl_tc_type_equal(
    const WGSLTypeInfo *a, const WGSLTypeInfo *b)
{
    if (a == b) return 1;
    if (!a || !b) return 0;
    if (a->kind != b->kind) return 0;

    switch ((WGSLTypeKind)a->kind) {
    case WGSL_TYPE_VEC:
        return a->width == b->width &&
               wgsl_tc_type_equal((const WGSLTypeInfo *)a->ref,
                                  (const WGSLTypeInfo *)b->ref);
    case WGSL_TYPE_MAT:
        return a->width == b->width &&
               a->rows == b->rows &&
               wgsl_tc_type_equal((const WGSLTypeInfo *)a->ref,
                                  (const WGSLTypeInfo *)b->ref);
    case WGSL_TYPE_ARRAY:
        return a->array_len == b->array_len &&
               a->flags == b->flags &&
               a->pad_ == b->pad_ &&
               wgsl_tc_type_equal((const WGSLTypeInfo *)a->ref,
                                  (const WGSLTypeInfo *)b->ref);
    case WGSL_TYPE_ATOMIC:
    case WGSL_TYPE_ATOMIC_CX_RESULT:
    case WGSL_TYPE_FREXP_RESULT:
    case WGSL_TYPE_MODF_RESULT:
        return wgsl_tc_type_equal((const WGSLTypeInfo *)a->ref,
                                  (const WGSLTypeInfo *)b->ref);
    case WGSL_TYPE_TEXTURE:
        return a->width == b->width &&
               a->rows == b->rows &&
               a->array_len == b->array_len &&
               wgsl_tc_type_equal((const WGSLTypeInfo *)a->ref,
                                  (const WGSLTypeInfo *)b->ref);
    case WGSL_TYPE_SAMPLER:
        return a->width == b->width;
    case WGSL_TYPE_PTR:
    case WGSL_TYPE_REF:
        return a->flags == b->flags &&
               wgsl_tc_type_equal((const WGSLTypeInfo *)a->ref,
                                  (const WGSLTypeInfo *)b->ref);
    case WGSL_TYPE_STRUCT:
        return a->ref == b->ref;
    default:
        return 1;
    }
}

/* Returns 1 if `init_type` is NOT compatible with `decl_type`.
 * Uses conversion-rank for abstract→concrete paths, then structural
 * equality for concrete / opaque / pointer shapes.  This matters for
 * §11.2 pointer parameters: `ptr<AS, T, AM>` must match address space,
 * store type, and access mode at call sites. */
int wgsl_tc_init_type_mismatch(
    WGSLTypeInfo *init_type, WGSLTypeInfo *decl_type)
{
    if (!init_type || !decl_type) return 0;  /* can't verify → allow */
    if (init_type == decl_type) return 0;     /* interned identity */
    if (wgsl_type_conversion_rank(init_type, decl_type) >= 0) return 0;
    return !wgsl_tc_type_equal(init_type, decl_type);
}

/* §6.1.2 / §6.2.1 — When an abstract initializer is concretized to a
 * declared type, the conversion table requires the value to fit the
 * target ("Identity if the value is in i32. Produces a shader-generation
 * error otherwise.").  This helper does a best-effort const-eval of
 * `init`; on success, `wgsl_consteval_materialize` enforces the range
 * check and emits an "out of range" error if needed.
 *
 * On failures that are noise (WGSL_CEV_NOT_CONST / NOT_IMPLEMENTED for
 * runtime initializers), the diagnostics consteval emitted along the
 * way are rolled back — runtime exprs aren't required to be const, and
 * we don't want to surface speculative const-eval diagnostics as user errors.
 * Real VALUE_ERROR (div0/overflow) is kept through typed status.
 *
 * Skip when:
 *   - there's no init or no target,
 *   - the target is abstract (no narrowing happens),
 *   - the init type isn't abstract (concrete→concrete is handled by
 *     `wgsl_tc_init_type_mismatch` upstream, with no value-side ambiguity).
 *
 * Caller's responsibility to skip for `DECL_CONST` — the consteval
 * module pass already materializes those and would otherwise emit the
 * range diagnostic twice. */
void wgsl_tc_check_init_concretization_range(
    WGSLTypeChecker *tc, WGSLNode *init, WGSLTypeInfo *target)
{
    if (!tc->cev || !tc->diag || !init || !target) return;
    if (wgsl_type_is_abstract(target)) return;

    /* The "narrowing" range check only matters when an abstract value
     * is being materialized into a concrete target.  But const-eval of
     * the init may *also* discover value-side errors (div by zero,
     * shift out of range, signed overflow) that are spec violations
     * regardless of type narrowing — so the consteval below always
     * runs, and the noise filter below salvages real value errors. */
    WGSLTypeInfo *init_type = wgsl_tc_store_type_of(tc, init);
    int has_narrowing = init_type && wgsl_type_is_abstract(init_type);

    size_t saved_count = tc->diag->count;
    int    saved_errs  = tc->diag->error_count;
    int    saved_cev_err = tc->cev->store.had_error;

    WGSLValue v = {0};
    int ok = wgsl_consteval_expr(tc->cev, init, &v);
    if (!ok) {
        /* Typed status: surface VALUE_ERROR, roll back noise. */
        if (tc->cev->store.last_status == WGSL_CEV_VALUE_ERROR) {
            tc->had_error = 1;
        } else {
            tc->diag->count       = saved_count;
            tc->diag->error_count = saved_errs;
            tc->cev->store.had_error    = saved_cev_err;
            tc->cev->store.last_status  = WGSL_CEV_OK;
        }
        return;
    }
    /* Only materialize when there's a real narrowing to perform.  For
     * non-narrowing inits (e.g. `let x = 1u << 33u;`) the eval itself
     * already caught any value-side errors above. */
    if (has_narrowing &&
        !wgsl_consteval_materialize(tc->cev, &v, target, init))
    {
        /* Range violation — materialize already emitted the diagnostic
         * through cev_error.  Propagate to the type checker's error
         * flag so the pass return-value reflects the failure. */
        tc->had_error = 1;
    }
}
