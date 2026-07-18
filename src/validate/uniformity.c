/**
 * @file uniformity.c — §15.2 uniformity analysis.
 *
 * Conservative uniformity oracle:
 *   - var<uniform>                     → uniform
 *   - read-only resource roots         → uniform
 *   - fn-scope `var` / `let`           → tracked through init,
 *                                          assignments, branches, and
 *                                          loop / switch joins
 *   - non-read-only module vars        → non-uniform
 *   - fn parameters                    → non-uniform unless they are
 *                                          spec-listed uniform builtins
 *   - literals / const / override      → uniform
 *
 * Combined with per-fn `UR_DERIVATIVE` / `UR_SUBGROUP` requirement
 * bitsets propagated through the call graph, this is enough to drive
 * the §15.6.1 barrier / §15.6.2 derivative / §15.6.3-4 subgroup-and-quad
 * "must be called in uniform control flow" diagnostics.
 *
 * Co-extensive with the validator-facing WGSL §15.2 surface: pointer
 * desugaring, assignment lattice, expression lattice, and call-site
 * requirements.  §15.2.9 full-AST annotation remains a tooling breadth
 * improvement rather than an accept/reject gap.
 *
 * AST-WRITE CONTRACT: this pass is the ONLY validator pass that writes to
 * the AST, and it writes EXACTLY three §15.2.9 analyzer-annotation bits into
 * `WGSLNode.flags` (never structure, children, spans, types, or any other
 * field):
 *   - WGSL_FLAG_DIVERGENT_CF     — per stmt/expr node (check_uniformity_block)
 *   - WGSL_FLAG_NONUNIFORM       — per expr node    (analyze_uniform_expr)
 *   - WGSL_FLAG_SUBGROUP_UNIFORM — per expr node    (analyze_uniform_expr)
 * These are tooling annotations (IDE/LSP surface the §15.2.9 divergence /
 * (sub)uniformity state); they never change accept/reject.  `WGSLNode.flags`
 * is defined as "bit-packed analyzer state" (ast.h) — so this is legitimate
 * annotation, NOT a violated "validate never mutates the AST" invariant.
 * Call-site consumers: `uniform_expr_is_subgroup_uniform` (state.c) reads
 * NONUNIFORM/SUBGROUP_UNIFORM back; keep this contract in sync if bits move.
 */
#include "internal/validate.h"
#include "internal/validate_priv.h"
#include "internal/uniformity_priv.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal/resolver.h"
#include "internal/types.h"
#include "internal/ast.h"
#include "internal/token.h"

/* Conservative uniformity oracle.  It tracks the leaves and simple
 * per-symbol value states the spec commonly cares about:
 *   - var<uniform>  reads      → uniform
 *   - non-read-only module var reads → non-uniform
 *   - ordinary function parameters   → non-uniform (potentially
 *                                      per-invocation at entry-point
 *                                      boundary)
 *   - const / override / let bound to const-exprs → uniform
 *   - literals                 → uniform
 *
 * This is enough to flag `if (frag_input > 0) { dpdx(...); }` etc.
 * It remains intentionally conservative where the spec allows "may
 * trigger" diagnostics. */
/* §15.2.5 helper — pulls a fn-scope `var` / `let` initializer expression
 * and runs the uniformity oracle on it.  Without this, every function-
 * scope `var` would be reported as non-uniform — false-flagging the
 * canonical reduction pattern
 *     `for (var s = 128u; s > 0u; s >>= 1u) { … workgroupBarrier(); }`
 * where `s` is a uniform loop counter.
 *
 * Post-init assignments are tracked through sequential statements,
 * partial/full assignment distinction, branches, and loop/switch joins. */
/* UniformCtx: uniformity_priv.h */
int is_uniform_expr(UniformCtx *cx, WGSLNode *n);
int analyze_uniform_expr(
    UniformCtx *cx, WGSLNode *n, int is_divergent,
    const uint32_t *fn_reqs);
WGSLSymbol *call_user_fn_sym(WGSLNode *call);
void check_uniformity_block(
    UniformCtx *cx, WGSLNode *n, int is_divergent,
    const uint32_t *fn_reqs);
WGSLSymbol *uniform_memory_root(WGSLNode *n, int depth);
int uniform_param_builtin_is_uniform(UniformCtx *cx, WGSLSymbol *sym);
int uniform_param_builtin_is_subgroup_uniform(UniformCtx *cx, WGSLSymbol *sym);
int uniform_module_var_is_readonly(const WGSLSymbol *sym);
int parameter_is_uniform(UniformCtx *cx, WGSLSymbol *sym);

/* UD_* : uniformity_priv.h */
