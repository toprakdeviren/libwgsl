/**
 * @file check.h — WGSL type checker / overload resolver (Phase 7).
 *
 * Tags every expression node with an effective `WGSLTypeInfo *`.
 * Reference expressions carry a structural `WGSL_TYPE_REF`
 * (`ref<AS,T,AM>`) plus the legacy `is_ref` bit for cheap predicates.
 * Tags every user-declared symbol with its resolved store/value type.
 *
 * Storage layout:
 *
 *   - **Symbol type** — written into `WGSLSymbol.type` (resolver fills
 *     this for predeclared types; the type checker fills it for every
 *     user const / let / var / param / function / struct / alias).
 *
 *   - **Expression type** — kept on a side-table keyed by `WGSLNode *`
 *     (linear scan in v1; swap for a hash table when a profile demands
 *     it).  `EXPR_IDENT` entries are recorded too, so `var` uses can
 *     expose their structural `ref<AS,T,AM>` type without changing the
 *     declaration symbol's store type.
 *
 *   - **`is_ref` bit** — `WGSL_FLAG_IS_REF` set on the expression node
 *     itself for compatibility and fast checks; `WGSL_FLAG_TYPED` marks
 *     "the type checker visited me".
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

typedef enum {
    WGSL_EXT_F16                  = 1u << 0,
    WGSL_EXT_CLIP_DISTANCES       = 1u << 1,
    WGSL_EXT_DUAL_SOURCE_BLENDING = 1u << 2,
    WGSL_EXT_SUBGROUPS            = 1u << 3,
    WGSL_EXT_PRIMITIVE_INDEX      = 1u << 4,
} WGSLExtension;

typedef enum {
    WGSL_FEAT_RO_RW_STORAGE_TEXTURES   = 1u << 0,
    WGSL_FEAT_PACKED_4X8_INT_DOT       = 1u << 1,
    WGSL_FEAT_UNRESTRICTED_PTR_PARAMS  = 1u << 2,
    WGSL_FEAT_POINTER_COMPOSITE_ACCESS = 1u << 3,
    WGSL_FEAT_UNIFORM_BUF_STD_LAYOUT   = 1u << 4,
    WGSL_FEAT_SUBGROUP_ID              = 1u << 5,
    WGSL_FEAT_SUBGROUP_UNIFORMITY      = 1u << 6,
    WGSL_FEAT_TEXTURE_AND_SAMPLER_LET  = 1u << 7,
    WGSL_FEAT_TEXTURE_FORMATS_TIER1    = 1u << 8,
    WGSL_FEAT_LINEAR_INDEXING          = 1u << 9,
    WGSL_FEAT_IMMEDIATE_DATA           = 1u << 10,
} WGSLLanguageFeature;

#define WGSL_FEAT_ALL_SUPPORTED \
    (WGSL_FEAT_RO_RW_STORAGE_TEXTURES | \
     WGSL_FEAT_PACKED_4X8_INT_DOT | \
     WGSL_FEAT_UNRESTRICTED_PTR_PARAMS | \
     WGSL_FEAT_POINTER_COMPOSITE_ACCESS | \
     WGSL_FEAT_UNIFORM_BUF_STD_LAYOUT | \
     WGSL_FEAT_SUBGROUP_ID | \
     WGSL_FEAT_SUBGROUP_UNIFORMITY | \
     WGSL_FEAT_TEXTURE_AND_SAMPLER_LET | \
     WGSL_FEAT_TEXTURE_FORMATS_TIER1 | \
     WGSL_FEAT_LINEAR_INDEXING | \
     WGSL_FEAT_IMMEDIATE_DATA)

typedef struct WGSLTypeChecker {
    WGSLArena          *arena;
    WGSLDiagBag        *diag;
    WGSLTypeStore      *types;
    const WGSLSource   *src;
    WGSLResolver       *res;
    WGSLConstEvaluator *cev;

    uint32_t      enabled_extensions;
    uint32_t      supported_features;
    uint32_t      required_features;

    WGSLNodeType *table;
    size_t        count;
    size_t        capacity;

    /* §7.2.2 @id uniqueness — heap-grown; type_decl_override pushes
     * each `@id(N)` after a duplicate scan.  Heap rather than inline
     * to keep WGSLTypeChecker's size from rippling through
     * WGSLResult and tripping the heap-allocator's layout heuristics. */
    int32_t      *override_ids;
    int           override_id_count;
    int           override_id_capacity;

    int           had_error;
    int           in_function;   /* 1 while inside type_function body */
    int           suppress_consteval_value_errors;

    /* §9.4.10 return type checking — set on entry to a function body,
     * cleared on exit.  NULL means "no return type written" (void). */
    WGSLTypeInfo *current_fn_return;
    /* §9.4.9 — depth of nested STMT_CONTINUING bodies we're currently
     * walking.  `return` / `continue` inside a continuing block are
     * spec errors. */
    int           continuing_depth;
    /* §9.4.6 — depth of nested loop / switch bodies entered *inside*
     * the current continuing block.  When `continuing_depth > 0` and
     * `inner_break_target_depth == 0`, a `break` would target the
     * enclosing loop and exit the continuing block — forbidden. */
    int           inner_break_target_depth;
    /* §9.4.8 / §9.7 — depth of nested LOOP / FOR / WHILE bodies entered
     * *inside* the current continuing block.  Switch is a `break`
     * target but not a `continue` target, so this counter is bumped
     * only on real loops.  When `continuing_depth > 0` and
     * `inner_continue_target_depth == 0`, a `continue` would target
     * the enclosing loop's continuing block — forbidden. */
    int           inner_continue_target_depth;
} WGSLTypeChecker;

/**
 * Type-check the entire translation unit.  Walks decls in source
 * order, types every expression in fn bodies, and writes:
 *   - `sym->type` for every user-declared symbol;
 *   - `payload[2]` *not* used (we maintain a side-table instead);
 *   - `WGSL_FLAG_TYPED` on every typed node;
 *   - structural `WGSL_TYPE_REF` for reference expressions, with
 *     `WGSL_FLAG_IS_REF` mirrored per §6.4.3.
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

/** Look up the type the checker assigned to `n`.  Reference expressions
 *  return `WGSL_TYPE_REF` (`ref<AS,T,AM>`), not just the store type.
 *  NULL if not typed. */
WGSLTypeInfo *wgsl_typecheck_type_of(
    const WGSLTypeChecker *tc, const WGSLNode *n);

/** Returns 1 iff the typed expression is a reference (§6.4.3). */
int wgsl_typecheck_is_ref(
    const WGSLTypeChecker *tc, const WGSLNode *n);

/** §7.2.2 / §15 — override-expression predicate.  Recognises literals,
 *  parens, unary, binary, and IDENT references that resolve to a
 *  `const` or `override` symbol.  Returns 1 iff `n` is a valid
 *  override-expression. */
int wgsl_typecheck_is_override_expr(const WGSLNode *n);

/** Free the side-table.  TypeInfos themselves live in the type store. */
void wgsl_typecheck_destroy(WGSLTypeChecker *tc);

#endif /* WGSL_INTERNAL_CHECK_H */
