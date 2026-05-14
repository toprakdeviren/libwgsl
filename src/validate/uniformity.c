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
 */
#include "internal/validate.h"
#include "internal/validate_priv.h"

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
typedef struct {
    WGSLValidator *v;
    const uint8_t *param_nonuniform;
    size_t         nsyms;
    int            init_depth;
    uint8_t       *sym_uniform;
    uint8_t       *sym_known;
} UniformCtx;

static int is_uniform_expr(UniformCtx *cx, WGSLNode *n);
static int analyze_uniform_expr(
    UniformCtx *cx, WGSLNode *n, int is_divergent,
    const uint32_t *fn_reqs);
static WGSLSymbol *call_user_fn_sym(WGSLNode *call);
static void check_uniformity_block(
    UniformCtx *cx, WGSLNode *n, int is_divergent,
    const uint32_t *fn_reqs);
static WGSLSymbol *uniform_memory_root(WGSLNode *n, int depth);
static int uniform_param_builtin_is_uniform(UniformCtx *cx, WGSLSymbol *sym);
static int uniform_param_builtin_is_subgroup_uniform(UniformCtx *cx, WGSLSymbol *sym);
static int uniform_module_var_is_readonly(const WGSLSymbol *sym);
static int parameter_is_uniform(UniformCtx *cx, WGSLSymbol *sym);

enum {
    UD_NONE      = 0,
    UD_WORKGROUP = 1u << 0,
    UD_SUBGROUP  = 1u << 1,
    UD_ALL       = UD_WORKGROUP | UD_SUBGROUP,
};


#include "uniformity/state_behavior.inc"
#include "uniformity/calls_exprs.inc"
#include "uniformity/pointers_block.inc"
