/**
 * @file exprs.c — Phase 7 expression typing + builtin-call
 * overload resolution.
 *
 * Covers every EXPR_* node kind: literals, identifiers, parens,
 * unary / binary operators, member access (swizzle on vec, field
 * access on struct), index (vec / mat / array), address-of, indirection,
 * function calls (type constructors / user-defined fns / builtin
 * functions through the generated `kBuiltinTable` metadata).
 *
 * The §6.1.3 overload resolution machinery (build_t_candidates +
 * resolve_overload) lives here too — it drives `type_call_builtin`'s
 * per-builtin shape inference using each entry's `WGSLParamFamily` +
 * `arity` configuration.  Builtin metadata is generated from
 * `def/wgsl.def`; mixed-shape overloads still use small local validators
 * where a compact T-pattern is not expressive enough.
 *
 * Co-extensive with WGSL §8 (Expressions) + §17 (Built-in Functions).
 */
#include "internal/check.h"
#include "internal/check_priv.h"
#include "internal/token.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Expression typing ─────────────────────────────────────────────── */

int wgsl_tc_type_expr(WGSLTypeChecker *tc, WGSLNode *n);


#include "exprs/core.inc"
#include "exprs/overloads.inc"
#include "exprs/atomic_pack_texture_helpers.inc"
#include "exprs/texture_overloads.inc"
#include "exprs/generated_overloads.inc"
#include "exprs/builtin_calls.inc"
#include "exprs/constructors.inc"
#include "exprs/ref_dispatch.inc"
