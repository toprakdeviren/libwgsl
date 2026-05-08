/**
 * @file validate.h — WGSL validator (Phase 8).
 *
 * Runs after the resolver / const-evaluator / type checker.  Does NOT
 * mutate the AST; emits diagnostics and writes a small module summary
 * for downstream consumers (LSP, the public C API).
 *
 * v1 covers eight rule families per `PLAN.md`:
 *
 *   1. recursion       — no user-fn call-graph cycles (§11.4)
 *   2. struct cycle    — no value-recursive struct types (§6.2.10)
 *   3. alias cycle     — no alias type cycles (§6.7)
 *   4. const cycle     — no `const` decl init cycles (§7.2)
 *   5. override cycle  — no `override` decl init cycles (§7.2)
 *   6. attribute conf. — `@attr(...)` placement + arity + const-ness (§12)
 *   7. entry point     — stage attr / IO shape rules (§13)
 *   8. behavior set    — `Next ⊆ {Next, Return}` for fn body (§9.7)
 *
 * Address-space / layout / uniformity / alias-analysis families are
 * deferred to Iter C; the spec rule families above land in Iters A–B
 * (see `PLAN.md` Phase 8 status row).
 */
#ifndef WGSL_INTERNAL_VALIDATE_H
#define WGSL_INTERNAL_VALIDATE_H

#include <stddef.h>
#include <stdint.h>

#include "internal/arena.h"
#include "internal/check.h"
#include "internal/consteval.h"
#include "internal/diag.h"
#include "internal/resolver.h"
#include "internal/source.h"
#include "internal/types.h"

/** Per-fn computed behavior set (subset of §9.7 — Next bit is the only
 *  load-bearing one in v1; the others are tracked for completeness). */
typedef enum {
    WGSL_BEHAVIOR_NEXT     = 1u << 0,
    WGSL_BEHAVIOR_BREAK    = 1u << 1,
    WGSL_BEHAVIOR_CONTINUE = 1u << 2,
    WGSL_BEHAVIOR_RETURN   = 1u << 3,
} WGSLBehaviorBits;

typedef struct WGSLValidator {
    WGSLArena          *arena;
    WGSLDiagBag        *diag;
    WGSLTypeStore      *types;
    const WGSLSource   *src;
    WGSLResolver       *res;
    WGSLConstEvaluator *cev;
    WGSLTypeChecker    *tc;

    /* Diagnostic-only flag for whether at least one entry point was
     * seen (validator does not error if zero entry points exist; that
     * is a "library module" — common for the corpus). */
    int                 entry_point_count;

    int                 had_error;
} WGSLValidator;

/**
 * Validate the full translation unit.
 *
 * @return 1 if no error-severity diagnostics emitted; 0 otherwise.
 */
int wgsl_validate(
    WGSLAst             *ast,
    const WGSLSource    *src,
    WGSLArena           *arena,
    WGSLDiagBag         *diag,
    WGSLTypeStore       *types,
    WGSLResolver        *res,
    WGSLConstEvaluator  *cev,
    WGSLTypeChecker     *tc,
    WGSLValidator       *out);

/** Free anything the validator owns that isn't in the arena. */
void wgsl_validator_destroy(WGSLValidator *v);

#endif /* WGSL_INTERNAL_VALIDATE_H */
