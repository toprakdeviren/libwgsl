/**
 * @file check.h — WGSL type checker / overload resolver (Phase 7).
 *
 * Tags every expression node with an effective `WGSLTypeInfo *` and an
 * `is_ref` bit (per §6.4.3 Reference vs Pointer).  Tags every user-
 * declared symbol with its resolved type.
 *
 * Storage layout:
 *
 *   - **Symbol type** — written into `WGSLSymbol.type` (resolver fills
 *     this for predeclared types; the type checker fills it for every
 *     user const / let / var / param / function / struct / alias).
 *
 *   - **Expression type** — kept on a side-table keyed by `WGSLNode *`
 *     (linear scan in v1; swap for a hash table when a profile demands
 *     it).  `EXPR_IDENT` does not need an entry — its type comes from
 *     `wgsl_node_resolved_symbol(n)->type`.
 *
 *   - **`is_ref` bit** — `WGSL_FLAG_IS_REF` set on the expression node
 *     itself; `WGSL_FLAG_TYPED` marks "the type checker visited me".
 *
 * Iter A scope (this iteration):
 *
 *   - Type-specifier resolution: scalar, `vec*<T>`, `mat*x*<T>`,
 *     `atomic<T>`, `array<T>`, `array<T, N>` (where N is a const-eval'd
 *     symbol), and pre-declared aliases (`vec3f`, `mat4x4f`, …).
 *   - Module-scope decl typing: every `const` / `override` / `var` /
 *     `let` / `param` / `struct member` / `alias` gets `sym->type`.
 *   - Function-body walk that types: literals, identifiers, parens,
 *     unary `- ! ~`, binary numeric / comparison / logical / bitwise.
 *
 * Iter B will add: operator overloads on vec/mat, constructors,
 * member access (swizzle + struct field), index, address-of /
 * indirection / ref-vs-ptr semantics, assignment-statement validation.
 *
 * Iter C will adopt the Tint-style `def/wgsl.def` DSL + codegen for the
 * full builtin function overload table.
 */
#ifndef WGSL_INTERNAL_CHECK_H
#define WGSL_INTERNAL_CHECK_H

#include <stddef.h>
#include <stdint.h>

#include "internal/arena.h"
#include "internal/ast.h"
#include "internal/consteval.h"
#include "internal/diag.h"
#include "internal/resolver.h"
#include "internal/source.h"
#include "internal/types.h"

/** One side-table row: a node and the type the checker assigned to it. */
typedef struct {
    const WGSLNode *node;
    WGSLTypeInfo   *type;
    uint16_t        flags;     /* mirrors WGSL_FLAG_IS_REF + reserved   */
} WGSLNodeType;

typedef struct WGSLTypeChecker {
    WGSLArena          *arena;
    WGSLDiagBag        *diag;
    WGSLTypeStore      *types;
    const WGSLSource   *src;
    WGSLResolver       *res;
    WGSLConstEvaluator *cev;

    WGSLNodeType *table;
    size_t        count;
    size_t        capacity;

    int           had_error;
} WGSLTypeChecker;

/**
 * Type-check the entire translation unit.  Walks decls in source
 * order, types every expression in fn bodies, and writes:
 *   - `sym->type` for every user-declared symbol;
 *   - `payload[2]` *not* used (we maintain a side-table instead);
 *   - `WGSL_FLAG_TYPED` on every typed node;
 *   - `WGSL_FLAG_IS_REF` per §6.4.3.
 *
 * @return 1 if zero error-severity diagnostics were emitted; 0 otherwise.
 */
int wgsl_typecheck(
    WGSLAst             *ast,
    const WGSLSource    *src,
    WGSLArena           *arena,
    WGSLDiagBag         *diag,
    WGSLTypeStore       *types,
    WGSLResolver        *res,
    WGSLConstEvaluator  *cev,
    WGSLTypeChecker     *out);

/** Look up the type the checker assigned to `n`.  NULL if not typed.  */
WGSLTypeInfo *wgsl_typecheck_type_of(
    const WGSLTypeChecker *tc, const WGSLNode *n);

/** Returns 1 iff the typed expression is a reference (§6.4.3). */
int wgsl_typecheck_is_ref(
    const WGSLTypeChecker *tc, const WGSLNode *n);

/** Free the side-table.  TypeInfos themselves live in the type store. */
void wgsl_typecheck_destroy(WGSLTypeChecker *tc);

#endif /* WGSL_INTERNAL_CHECK_H */
