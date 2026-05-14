/**
 * @file consteval.h — WGSL const-expression evaluator.
 *
 * Phase 6 v1 deliberately caps what the spec demands at *shader-generation
 * time* (per `PLAN.md` §6 risk register / `DESIGN.md` rev 1):
 *
 *   - `const NAME = expr;`       (§7.2)  must const-eval
 *   - `override NAME = expr?;`   (§7.2)  default value is override-expr
 *   - `const_assert <expr>;`     (§10.1) must eval to literal `true`
 *   - template-arg const-expr   (e.g. `array<f32, N>`)
 *   - abstract→concrete materialization (§6.2.1)
 *   - scalar + vector FP builtin-call domains for §15.7.7, including
 *     fold-through for common @const numeric builtins (`abs`, `min`,
 *     `max`, `mix`, `ldexp`, `length`, `normalize`, etc.)
 *
 * **Out of scope for v1** (postponed to v1.x):
 *   - user-defined `const fn` graph evaluation (§11.3)
 *   - memoising const-call evaluator
 *   - full aggregate / matrix constructors and SPIR-V/runtime-required
 *     bitfield / pack-unpack builtins
 *
 * The evaluator runs after the resolver (Phase 5).  It uses every
 * `EXPR_IDENT`'s resolved `WGSLSymbol *` (stashed in `payload[2]`) to
 * look up the bound value of a const symbol.
 *
 * Successful evaluation of a `DECL_CONST` / `DECL_OVERRIDE` populates
 * a side-table keyed by `WGSLSymbol *` so later uses (in template args,
 * other consts, `const_assert`) can fetch the value back.  Untyped
 * `const` values keep their abstract numeric type; materialization is
 * performed only for typed consts here, or later by the consuming context.
 */
#ifndef WGSL_INTERNAL_CONSTEVAL_H
#define WGSL_INTERNAL_CONSTEVAL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "internal/arena.h"
#include "internal/ast.h"
#include "internal/diag.h"
#include "internal/resolver.h"
#include "internal/source.h"
#include "internal/types.h"

/** Tagged value union.  All instances are arena-allocated by the
 *  evaluator; callers do not free.  Aggregates point at flat element
 *  arrays in the same arena. */
typedef enum {
    WGSL_VAL_INVALID = 0,
    WGSL_VAL_BOOL,
    WGSL_VAL_INT,        /* AbstractInt / i32 / u32 — interpret with `type` */
    WGSL_VAL_FLOAT,      /* AbstractFloat / f32 / f16 — interpret with `type` */
    WGSL_VAL_VEC,        /* `agg.count` elems = type width                  */
    WGSL_VAL_MAT,        /* `agg.count` elems = cols * rows (column-major)  */
    WGSL_VAL_ARRAY,      /* `agg.count` elems = array length                */
    WGSL_VAL_STRUCT,     /* `agg.count` elems = struct member count         */
} WGSLValKind;

typedef struct WGSLValue {
    uint16_t       kind;          /* WGSLValKind                          */
    uint16_t       flags_;        /* reserved                             */
    uint32_t       pad_;
    WGSLTypeInfo  *type;          /* tagged type (Abstract* or concrete)  */
    union {
        bool     b;
        int64_t  i;
        double   f;
        struct {
            uint32_t          count;
            struct WGSLValue *elems;     /* arena-owned                   */
        } agg;
    } u;
} WGSLValue;

/* Side-table entry: WGSLSymbol → cached evaluated value. */
typedef struct {
    const WGSLSymbol *sym;
    WGSLValue        *value;       /* arena-owned                         */
} WGSLConstBinding;

typedef struct WGSLConstEvaluator {
    WGSLArena         *arena;        /* used for values + bindings table  */
    WGSLDiagBag       *diag;
    WGSLTypeStore     *types;
    const WGSLSource  *src;
    WGSLResolver      *res;          /* for EXPR_IDENT → symbol lookups   */

    WGSLConstBinding  *bindings;     /* heap-grown, freed in destroy      */
    size_t             binding_count;
    size_t             binding_capacity;

    int                had_error;
} WGSLConstEvaluator;

/* ── Initialization / destruction ──────────────────────────────────── */

/**
 * Initialise the evaluator state (no walking).  After this, individual
 * expressions can be evaluated with `wgsl_consteval_expr`, or the full
 * module can be processed with `wgsl_consteval` (Iter B).
 *
 * `res` may be NULL — in that case, identifier lookups fail and only
 * literal-arithmetic expressions can be evaluated.  Iter A tests use
 * this mode.
 */
void wgsl_consteval_init(
    WGSLConstEvaluator *cev,
    WGSLArena          *arena,
    WGSLDiagBag        *diag,
    WGSLTypeStore      *types,
    const WGSLSource   *src,
    WGSLResolver       *res);

/** Release the bindings table.  Values themselves live in the arena. */
void wgsl_consteval_destroy(WGSLConstEvaluator *cev);

/* ── Single-expression evaluation (Iter A entry) ───────────────────── */

/**
 * Evaluate one expression node as a const-expression.
 *
 * On success returns 1, populates `*out` with `kind != INVALID`.
 * On failure returns 0, emits one or more diagnostics on the bag, and
 * leaves `*out` zero-initialized.
 *
 * Side effects: NONE on the AST itself.  (Iter B will additionally
 * cache bindings for `DECL_CONST` / `DECL_OVERRIDE` symbols.)
 */
int wgsl_consteval_expr(
    WGSLConstEvaluator *cev,
    WGSLNode           *expr,
    WGSLValue          *out);

/* ── Materialisation (§6.2.1) ──────────────────────────────────────── */

/**
 * Materialise abstract numeric values to concrete types per §6.2.1:
 *   AbstractInt    → i32 (default) or u32 / f32 / f16 if `target` requires it
 *   AbstractFloat  → f32 (default) or f16 if `target` requires it
 *
 * `target` may be NULL — that means "default concretisation": AbstractInt → i32,
 * AbstractFloat → f32.  Already-concrete values are returned unchanged
 * if the conversion-rank table allows; otherwise an error diagnostic is
 * emitted at `at_node`.
 *
 * Returns 1 on success (possibly with `*v` rewritten in place); 0 on
 * failure.
 */
int wgsl_consteval_materialize(
    WGSLConstEvaluator *cev,
    WGSLValue          *v,
    WGSLTypeInfo       *target,        /* NULL = default concrete tower */
    const WGSLNode     *at_node);

/* ── Module-level pass (Iter B entry; not yet used in Iter A) ──────── */

/**
 * Walk the translation unit, evaluating every `DECL_CONST`,
 * `DECL_OVERRIDE`, `DECL_CONST_ASSERT` (module + fn-scope).  Caches
 * resolved values on the side-table.  Emits diagnostics for non-const
 * expressions, divide-by-zero, overflow at materialise, …
 *
 * Defined in Iter B.  Stubbed to return 1 in Iter A.
 *
 * @return 1 if no error-severity diagnostics emitted; 0 otherwise.
 */
int wgsl_consteval(
    WGSLAst            *ast,
    const WGSLSource   *src,
    WGSLArena          *arena,
    WGSLDiagBag        *diag,
    WGSLTypeStore      *types,
    WGSLResolver       *res,
    WGSLConstEvaluator *out);

/** Symbol→value binding lookup.  Returns NULL if `sym` has no value. */
const WGSLValue *wgsl_consteval_value_of(
    const WGSLConstEvaluator *cev, const WGSLSymbol *sym);

#endif /* WGSL_INTERNAL_CONSTEVAL_H */
