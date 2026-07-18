/**
 * @file check_priv.h — internal-only sharing between check*.c files.
 *
 * Functions declared here are private to the type checker but split
 * across check.c (driver / walk) / types.c (type-spec) /
 * decls.c (decl typing + init compat) / exprs.c (expr
 * typing + overload + calls).  None of these are visible to consumers
 * of the checker — the public surface is `internal/check.h`.
 */
#ifndef WGSL_INTERNAL_CHECK_PRIV_H
#define WGSL_INTERNAL_CHECK_PRIV_H

#include "internal/check.h"

/* Diagnostics & side-table (check.c). */

void wgsl_tc_error(
    WGSLTypeChecker *tc, const WGSLNode *at, const char *fmt, ...);
int  wgsl_tc_set_type(
    WGSLTypeChecker *tc, WGSLNode *n, WGSLTypeInfo *t, int is_ref);
WGSLTypeInfo *wgsl_tc_store_type_of(
    const WGSLTypeChecker *tc, const WGSLNode *n);
WGSLTypeInfo *wgsl_tc_unwrap_ref_type(WGSLTypeInfo *t);

/* Symbol lookup helper (check.c). */
WGSLSymbol *wgsl_tc_find_decl_symbol(
    const WGSLTypeChecker *tc, const WGSLNode *decl);

/* Type-spec resolution (types.c). */

/* Resolve a type-spec AST subtree to a WGSLTypeInfo and record it on
 * the side-table.  NULL on error (with a diagnostic emitted). */
WGSLTypeInfo *wgsl_tc_resolve_type_spec(WGSLTypeChecker *tc, WGSLNode *spec);

/* Enumerant template-arg resolvers (types.c).  Used by both the
 * type-spec resolver (`ptr<AS, T>`, `texture_storage_*<fmt, am>`) and
 * by decls.c when reading `var<AS>` / `var<AS, AM>` template
 * lists at decl level. */
WGSLAddressSpace wgsl_tc_resolve_addr_space_arg(
    WGSLTypeChecker *tc, WGSLNode *arg);
WGSLAccessMode   wgsl_tc_resolve_access_mode_arg(
    WGSLTypeChecker *tc, WGSLNode *arg);

/* Init compatibility (decls.c). */

/* "Constructible" predicate with struct-recursion (§6.2.12).  The
 * shallow predicate `wgsl_type_is_constructible` short-circuits structs
 * as constructible; this version walks members. */
int wgsl_tc_is_constructible_deep(
    const WGSLTypeChecker *tc, const WGSLTypeInfo *t);

/* §7 init-vs-decl compatibility check.  Returns 0 if `init` can be
 * converted to `decl` (per §6.1.2 rank table), 1 otherwise. */
int wgsl_tc_init_type_mismatch(
    WGSLTypeInfo *init_type, WGSLTypeInfo *decl_type);

/* §6.2.1 — runs const-eval on `init` and reports if the materialised
 * value overflows the target's concrete representation. */
void wgsl_tc_check_init_concretization_range(
    WGSLTypeChecker *tc, WGSLNode *init, WGSLTypeInfo *decl_type);

/* §8 arithmetic value-side checks for partially constant divisors. */
void wgsl_tc_check_constexpr_divisor_nonzero(
    WGSLTypeChecker *tc,
    WGSLNode *rhs,
    WGSLTypeInfo *rhs_type,
    const char *message);

/* Decl typing (decls_init.c / decls_typing.c). */
int decl_type_contains_runtime_array(const WGSLTypeChecker *tc, const WGSLTypeInfo *t);
int decl_type_contains_override_array(const WGSLTypeChecker *tc, const WGSLTypeInfo *t);
int decl_type_contains_atomic(const WGSLTypeChecker *tc, const WGSLTypeInfo *t);
int wgsl_tc_type_decl_const_or_let(WGSLTypeChecker *tc, WGSLNode *n);
int wgsl_tc_type_decl_var         (WGSLTypeChecker *tc, WGSLNode *n);
int wgsl_tc_type_decl_override    (WGSLTypeChecker *tc, WGSLNode *n);
int wgsl_tc_type_decl_alias       (WGSLTypeChecker *tc, WGSLNode *n);
int wgsl_tc_type_decl_struct      (WGSLTypeChecker *tc, WGSLNode *n);

/* Expression typing (exprs.c). */

/* Top-level dispatcher; handles every EXPR_* node kind.
 * `wgsl_tc_type_expr` is bottom-up (no expected type).
 * `wgsl_tc_type_expr_exp` is bidirectional (§4.3 / §6.2.1): when
 * `expected` is a concrete type that the expression's abstract type
 * converts to, abstract literals and intermediate results materialize
 * in that context (e.g. `let x: u32 = ~0` / `countOneBits(7)`). */
int wgsl_tc_type_expr(WGSLTypeChecker *tc, WGSLNode *n);
int wgsl_tc_type_expr_exp(
    WGSLTypeChecker *tc, WGSLNode *n, WGSLTypeInfo *expected);

/* Override-expression predicate (§7.2.2 / §15) — needed by walk_stmt
 * for `var<private>` init / @workgroup_size cross-check. */
int wgsl_tc_is_override_expr(const WGSLNode *n);

/* Generated builtin metadata query.  Used by STMT_FN_CALL to enforce
 * `@must_use` on predeclared functions as well as user functions. */
int wgsl_tc_builtin_must_use(const char *name, uint32_t name_len);

/* Helper used by walk_stmt's STMT_ASSIGN arm to pull the AS / AM of
 * the LHS reference (§9.2.1 access-mode check).  Lives in exprs.c
 * because it shares structure with type_ident / type_addr_of. */
void wgsl_tc_get_ref_space_and_mode(
    WGSLTypeChecker *tc, const WGSLNode *n,
    WGSLAddressSpace *as, WGSLAccessMode *am);

/* §5 — forward module-scope type resolution.  When `type_ident`
 * encounters a use of a not-yet-typed module-scope decl, it walks the
 * decl through `type_module_decl` to populate its type.  Defined in
 * check.c. */
int wgsl_tc_type_module_decl(WGSLTypeChecker *tc, WGSLNode *n);

#endif /* WGSL_INTERNAL_CHECK_PRIV_H */
