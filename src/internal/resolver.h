/**
 * @file resolver.h — Name resolver + scope tree (WGSL §5).
 *
 * Owns three kinds of scope (per WGSL §5.1):
 *   - **predeclared** — populated once per resolver: every predeclared
 *     scalar / vec & mat alias / type generator / address-space and
 *     access-mode name / a starter set of builtin function names.
 *   - **module**      — top-level decls of the current TU (order-
 *                       independent forward-references).
 *   - **function / block** — pushed and popped during body walking;
 *                            order-sensitive shadowing is allowed in
 *                            inner scopes, never of the predeclared
 *                            scope.
 *
 * Resolution result for an identifier expression is stored in the AST
 * node's `payload[2]` slot — a `(uintptr_t)WGSLSymbol *` cast back to
 * the symbol on read.  Same for `EXPR_TEMPLATED_IDENT` (its
 * children[0] carries the resolution).
 *
 * Phase 5 deliberately does **not** resolve the right-hand side of
 * `obj.field` (member access) — that needs the type of `obj`, which
 * is the type-checker's job in Phase 7.
 *
 * Diagnostic vocabulary:
 *   - "use of undeclared identifier 'X'" — name does not resolve.
 *   - "'X' is used before its declaration" — function-body forward
 *     reference (illegal per §5.2).
 *   - "redeclaration of 'X' in the same scope".
 */
#ifndef WGSL_INTERNAL_RESOLVER_H
#define WGSL_INTERNAL_RESOLVER_H

#include <stddef.h>
#include <stdint.h>

#include "internal/arena.h"
#include "internal/ast.h"
#include "internal/diag.h"
#include "internal/parser.h"
#include "internal/source.h"
#include "internal/types.h"

typedef enum {
    WGSL_SYM_INVALID = 0,

    /* Predeclared (loaded once per resolver). */
    WGSL_SYM_PREDECLARED_TYPE,        /* bool, i32, u32, f32, f16 — `type` is set */
    WGSL_SYM_PREDECLARED_TYPEGEN,     /* vec2/3/4, mat*x*, atomic, array, ptr — generator */
    WGSL_SYM_PREDECLARED_TYPE_ALIAS,  /* vec3f, mat4x4f, … — `type` is set */
    WGSL_SYM_PREDECLARED_ADDR_SPACE,  /* storage / uniform / workgroup / private / function */
    WGSL_SYM_PREDECLARED_ACCESS_MODE, /* read / write / read_write */
    WGSL_SYM_PREDECLARED_FN,          /* sin, cos, dot, dpdx, textureSample, … */
    WGSL_SYM_PREDECLARED_VALUE,       /* (no current entries — `true`/`false` are keywords) */

    /* User-declared. */
    WGSL_SYM_TYPE_ALIAS,              /* `alias UV = vec2<f32>;` */
    WGSL_SYM_STRUCT,
    WGSL_SYM_FUNCTION,
    WGSL_SYM_VAR,
    WGSL_SYM_LET,
    WGSL_SYM_CONST,
    WGSL_SYM_OVERRIDE,
    WGSL_SYM_PARAM,

    WGSL_SYM_KIND_COUNT,
} WGSLSymKind;

typedef struct WGSLSymbol {
    const char  *name;      /* borrowed (source bytes or static literal) */
    uint32_t     name_len;
    uint16_t     kind;      /* WGSLSymKind */
    uint16_t     flags;     /* reserved */
    WGSLNode    *ast;       /* declaration node; NULL for predeclared    */
    WGSLTypeInfo *type;     /* resolved type (when applicable)           */
} WGSLSymbol;

typedef struct WGSLScope {
    struct WGSLScope *parent;
    WGSLSymbol      **items;
    size_t            count;
    size_t            capacity;
} WGSLScope;

typedef struct WGSLResolver {
    WGSLArena        *arena;
    WGSLDiagBag      *diag;
    WGSLTypeStore    *types;
    const WGSLSource *src;

    WGSLScope         predeclared;
    WGSLScope         module;
    WGSLScope        *current;     /* function / block scope chain top  */

    /* Flat list of every user-declared symbol the resolver bound (any
     * scope).  Lets later passes (Phase 6 const-eval, Phase 7 type
     * checker) reverse-look-up `WGSLNode *decl → WGSLSymbol *` without
     * keeping live scope chains around. */
    WGSLSymbol      **all_decls;
    size_t            all_decl_count;
    size_t            all_decl_capacity;

    int               had_error;
    int               in_attribute;  /* 1 while walking inside @attr(…) */
} WGSLResolver;

/**
 * Resolve every identifier reference in `ast`.  Mutates the AST nodes
 * by storing a `WGSLSymbol *` into each `EXPR_IDENT` (and the leading
 * ident of `EXPR_TEMPLATED_IDENT`) via `payload[2]`.
 *
 * @return 1 if zero error-severity diagnostics were emitted; 0 otherwise.
 */
int wgsl_resolve(
    WGSLAst         *ast,
    const WGSLSource *src,
    WGSLArena       *arena,
    WGSLDiagBag     *diag,
    WGSLTypeStore   *types,
    WGSLResolver    *out);

/** Helper: extract the resolved symbol for an `EXPR_IDENT` node. */
WGSLSymbol *wgsl_node_resolved_symbol(const WGSLNode *ident);

/** Free anything the resolver owns that isn't in the arena. */
void wgsl_resolver_destroy(WGSLResolver *r);

#endif /* WGSL_INTERNAL_RESOLVER_H */
