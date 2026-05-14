/* ── §6.1.2 / §6.1.3 — overload resolution machinery ──────────────── */

static int builtin_call_is_const(const WGSLNode *n);
static int builtin_call_is_override(const WGSLNode *n);

/* A const-expression predicate (§7.2.1 / §15).  Walks `n` and returns 1
 * iff the value is fixed at shader-generation time.  Conservative: any
 * shape we do not explicitly recognize is treated as non-const, which
 * matches the step-2 elimination intent (prefer concrete when in doubt). */
static int is_const_expr(const WGSLNode *n) {
    if (!n) return 0;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_LITERAL_BOOL:
    case WGSL_NODE_EXPR_LITERAL_INT:
    case WGSL_NODE_EXPR_LITERAL_FLOAT:
        return 1;
    case WGSL_NODE_EXPR_PAREN:
        return n->child_count == 1 && is_const_expr(n->children[0]);
    case WGSL_NODE_EXPR_UNARY:
        return n->child_count == 1 && is_const_expr(n->children[0]);
    case WGSL_NODE_EXPR_BINARY:
        return n->child_count == 2 &&
               is_const_expr(n->children[0]) &&
               is_const_expr(n->children[1]);
    case WGSL_NODE_EXPR_IDENT: {
        WGSLSymbol *s = wgsl_node_resolved_symbol((WGSLNode *)n);
        return s && s->kind == WGSL_SYM_CONST;
    }
    case WGSL_NODE_EXPR_CALL:
        return builtin_call_is_const(n);
    default:
        return 0;
    }
}

/* §7.2.2 / §15: an override-expression is a superset of const-expression
 * that also accepts references to module-scope `override` symbols.
 * Used to validate `override X = init;` where init may reference other
 * overrides (`override Y = X * 2;` is legal) but not non-const vars or
 * fn params. */
int wgsl_tc_is_override_expr(const WGSLNode *n) {
    if (!n) return 0;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_LITERAL_BOOL:
    case WGSL_NODE_EXPR_LITERAL_INT:
    case WGSL_NODE_EXPR_LITERAL_FLOAT:
        return 1;
    case WGSL_NODE_EXPR_PAREN:
        return n->child_count == 1 && wgsl_tc_is_override_expr(n->children[0]);
    case WGSL_NODE_EXPR_UNARY:
        return n->child_count == 1 && wgsl_tc_is_override_expr(n->children[0]);
    case WGSL_NODE_EXPR_BINARY:
        return n->child_count == 2 &&
               wgsl_tc_is_override_expr(n->children[0]) &&
               wgsl_tc_is_override_expr(n->children[1]);
    case WGSL_NODE_EXPR_INDEX:
        return n->child_count == 2 &&
               wgsl_tc_is_override_expr(n->children[0]) &&
               wgsl_tc_is_override_expr(n->children[1]);
    case WGSL_NODE_EXPR_MEMBER:
        return n->child_count >= 1 &&
               wgsl_tc_is_override_expr(n->children[0]);
    case WGSL_NODE_EXPR_IDENT: {
        WGSLSymbol *s = wgsl_node_resolved_symbol((WGSLNode *)n);
        return s && (s->kind == WGSL_SYM_CONST || s->kind == WGSL_SYM_OVERRIDE);
    }
    case WGSL_NODE_EXPR_CALL:
        return builtin_call_is_override(n);
    default:
        return 0;
    }
}

int wgsl_typecheck_is_override_expr(const WGSLNode *n) {
    return wgsl_tc_is_override_expr(n);
}

/* Per-arg parameter family.  Encodes the set of scalar element types T
 * that a builtin's parametric overload set ranges over.  When the field
 * is `PF_NONE`, the entry skips full overload resolution and uses the
 * legacy BR_* return rule directly. */
typedef enum {
    PF_NONE = 0,
    PF_FLOAT_T,         /* T ∈ {AbstractFloat, f32, f16}                */
    PF_NUMERIC_T,       /* T ∈ {AbstractInt, AbstractFloat, i32, u32,   *
                         *      f32, f16}                               */
    PF_INTEGER_T,       /* T ∈ {AbstractInt, i32, u32}                  */
    PF_SIGN_T,          /* T ∈ {AbstractInt, AbstractFloat, i32, f32, f16} */
    PF_F32_T,           /* T ∈ {f32}; abstract args may materialize to f32 */
} WGSLParamFamily;

/* Maximum arity / candidate count tracked by the resolver.  Wide enough
 * for every numeric builtin in the WGSL spec; bumping the candidate cap
 * is cheap (stack-only) if a future family needs it. */
#define WGSL_OL_MAX_PARAMS 4
#define WGSL_OL_MAX_CANDS  8

typedef struct {
    WGSLTypeInfo *params[WGSL_OL_MAX_PARAMS];
    WGSLTypeInfo *result;   /* shape used to derive the return type     */
} WGSLOLCandidate;

/* §6.1.3 — overload resolution.
 *
 *  Step 1: drop candidates with an infeasible conversion on any arg.
 *  Step 2: drop candidates where one parameter is abstract while
 *          another argument is not a const-expression.
 *  Step 3: rank candidates pairwise — C1 is preferred over C2 iff
 *          R_C1[i] ≤ R_C2[i] for every i, with strict inequality on
 *          at least one i.
 *  Step 4: if exactly one candidate is preferred over every other
 *          survivor, succeed.  Otherwise fail.
 *
 * Returns the chosen index or -1 on failure.  Emits a diagnostic on `at`
 * for "no matching overload" and "ambiguous overload". */
static int resolve_overload(
    WGSLTypeChecker *tc,
    const WGSLOLCandidate *cands, int cand_count,
    WGSLTypeInfo *const *arg_types, const int *arg_is_const, int arg_count,
    const char *what, const WGSLNode *at)
{
    if (cand_count <= 0 || arg_count <= 0 || arg_count > WGSL_OL_MAX_PARAMS) {
        return -1;
    }

    int ranks[WGSL_OL_MAX_CANDS][WGSL_OL_MAX_PARAMS];
    int alive[WGSL_OL_MAX_CANDS] = {0};
    int alive_count = 0;

    /* Step 1: conversion-rank feasibility. */
    for (int c = 0; c < cand_count; c++) {
        alive[c] = 1;
        for (int i = 0; i < arg_count; i++) {
            int r = wgsl_type_conversion_rank(arg_types[i], cands[c].params[i]);
            if (r < 0) { alive[c] = 0; break; }
            ranks[c][i] = r;
        }
        if (alive[c]) alive_count += 1;
    }
    if (alive_count == 0) {
        wgsl_tc_error(tc, at, "no matching overload for '%s'", what);
        return -1;
    }

    /* Step 2: eliminate abstract-after-conversion + non-const-elsewhere. */
    if (arg_count > 1) {
        int has_nonconst = 0;
        for (int j = 0; j < arg_count; j++) {
            if (!arg_is_const[j]) { has_nonconst = 1; break; }
        }
        if (has_nonconst) {
            for (int c = 0; c < cand_count; c++) {
                if (!alive[c]) continue;
                int any_abstract_param = 0;
                for (int i = 0; i < arg_count; i++) {
                    if (wgsl_type_is_abstract(cands[c].params[i])) {
                        any_abstract_param = 1;
                        break;
                    }
                }
                if (any_abstract_param) {
                    alive[c] = 0;
                    alive_count -= 1;
                }
            }
        }
        if (alive_count == 0) {
            wgsl_tc_error(tc, at, "no matching overload for '%s'", what);
            return -1;
        }
    }

    /* Step 3+4: pairwise preference, pick unique winner. */
    int best = -1;
    for (int c = 0; c < cand_count; c++) {
        if (!alive[c]) continue;
        int preferred_over_all = 1;
        for (int o = 0; o < cand_count; o++) {
            if (o == c || !alive[o]) continue;
            int all_le = 1, some_lt = 0;
            for (int i = 0; i < arg_count; i++) {
                if (ranks[c][i] > ranks[o][i]) { all_le = 0; break; }
                if (ranks[c][i] < ranks[o][i]) some_lt = 1;
            }
            if (!(all_le && some_lt)) { preferred_over_all = 0; break; }
        }
        if (preferred_over_all) {
            if (best == -1) {
                best = c;
            } else {
                /* Multiple candidates each preferred over all others —
                 * impossible by the partial-order definition, but treat
                 * as ambiguous to be safe. */
                best = -1;
                break;
            }
        }
    }
    if (best == -1) {
        wgsl_tc_error(tc, at, "ambiguous overload for '%s'", what);
        return -1;
    }
    return best;
}

/* Lift a scalar T into the shape carried by `shape` (scalar / vec / mat). */
static WGSLTypeInfo *lift_to_shape(
    WGSLTypeStore *ts, WGSLTypeInfo *T, const WGSLTypeInfo *shape)
{
    if (!shape || !T) return T;
    if (shape->kind == WGSL_TYPE_VEC) {
        return wgsl_type_vec(ts, shape->width, T);
    }
    if (shape->kind == WGSL_TYPE_MAT) {
        return wgsl_type_mat(ts, shape->width, shape->rows, T);
    }
    return T;
}

/* Pick a "shape source" arg — prefer vec/mat over scalar so a mixed
 * scalar/composite call still finds the composite shape. */
static const WGSLTypeInfo *pick_shape_arg(
    WGSLTypeInfo *const *args, int n)
{
    for (int i = 0; i < n; i++) {
        if (!args[i]) continue;
        if (args[i]->kind == WGSL_TYPE_VEC || args[i]->kind == WGSL_TYPE_MAT) {
            return args[i];
        }
    }
    return n > 0 ? args[0] : NULL;
}

/* Build the candidate set for a "T-pattern" builtin: all `arity` params
 * share the same shape (derived from the args), with the element ranging
 * over the family's scalar set.  Returns the number of candidates. */
static int build_t_candidates(
    WGSLTypeChecker *tc, WGSLParamFamily fam,
    WGSLTypeInfo *const *args, int arity,
    WGSLOLCandidate *out_cands)
{
    const WGSLTypeInfo *shape = pick_shape_arg(args, arity);
    if (!shape) return 0;

    WGSLTypeInfo *scalars[8];
    int sc = 0;
    switch (fam) {
    case PF_FLOAT_T:
        scalars[sc++] = tc->types->t_abstract_float;
        scalars[sc++] = tc->types->t_f32;
        if (tc->enabled_extensions & WGSL_EXT_F16) {
            scalars[sc++] = tc->types->t_f16;
        }
        break;
    case PF_NUMERIC_T:
        scalars[sc++] = tc->types->t_abstract_int;
        scalars[sc++] = tc->types->t_abstract_float;
        scalars[sc++] = tc->types->t_i32;
        scalars[sc++] = tc->types->t_u32;
        scalars[sc++] = tc->types->t_f32;
        if (tc->enabled_extensions & WGSL_EXT_F16) {
            scalars[sc++] = tc->types->t_f16;
        }
        break;
    case PF_INTEGER_T:
        scalars[sc++] = tc->types->t_abstract_int;
        scalars[sc++] = tc->types->t_i32;
        scalars[sc++] = tc->types->t_u32;
        break;
    case PF_SIGN_T:
        scalars[sc++] = tc->types->t_abstract_int;
        scalars[sc++] = tc->types->t_abstract_float;
        scalars[sc++] = tc->types->t_i32;
        scalars[sc++] = tc->types->t_f32;
        if (tc->enabled_extensions & WGSL_EXT_F16) {
            scalars[sc++] = tc->types->t_f16;
        }
        break;
    case PF_F32_T:
        scalars[sc++] = tc->types->t_f32;
        break;
    default:
        return 0;
    }

    int n = 0;
    for (int i = 0; i < sc && n < WGSL_OL_MAX_CANDS; i++) {
        WGSLTypeInfo *lifted = lift_to_shape(tc->types, scalars[i], shape);
        WGSLOLCandidate *cnd = &out_cands[n];
        for (int a = 0; a < arity; a++) cnd->params[a] = lifted;
        cnd->result = lifted;
        n += 1;
    }
    return n;
}

/* ── Builtin overload table ────────────────────────────────────────── */

/* Return-type derivation rule for each builtin entry. */
typedef enum {
    BR_VOID,                /* () return                                */
    BR_BOOL,                /* always bool                              */
    BR_I32,                 /* always i32                               */
    BR_U32,                 /* always u32 (`arrayLength`, `textureNum*`)*/
    BR_ARG0,                /* same type as first argument              */
    BR_ARG0_ELEM,           /* element type of first argument           */
    BR_TEMPLATE,            /* template arg of a templated callee       */
    BR_VEC4F,               /* vec4<f32> (texture sampling default)     */
    BR_VEC4U,               /* vec4<u32> (subgroupBallot, unpack4xU8)   */
    BR_VEC4I,               /* vec4<i32> (unpack4xI8)                   */
    BR_VEC2F,               /* vec2<f32> (unpack2x16{s,u}norm/float)    */
    BR_ATOMIC_CX_RESULT,    /* __atomic_compare_exchange_result<T>      */
    BR_FREXP_RESULT,        /* __frexp_result<T> for T = arg0's type    */
    BR_MODF_RESULT,         /* __modf_result<T>  for T = arg0's type    */
} WGSLBuiltinReturn;

typedef enum {
    WGSL_BIF_CONST    = 1u << 0,
    WGSL_BIF_MUST_USE = 1u << 1,
} WGSLBuiltinFlags;

typedef enum {
    BS_NONE = 0,
    BS_LDEXP,
    BS_MIX,
    BS_TEXTURE,
} WGSLBuiltinSpecial;

typedef struct {
    const char       *name;
    WGSLBuiltinReturn ret;
    WGSLParamFamily   fam;     /* PF_NONE → skip full overload resolution */
    uint8_t           arity;   /* number of leading args that share the T */
    uint16_t          flags;   /* WGSL_BIF_* metadata from def/wgsl.def */
    uint8_t           special; /* WGSLBuiltinSpecial validator hook */
} WGSLBuiltinEntry;

static void check_const_builtin_call(
    WGSLTypeChecker *tc, WGSLNode *call, const WGSLBuiltinEntry *entry)
{
    if (!entry || !(entry->flags & WGSL_BIF_CONST)) return;
    if (!is_const_expr(call)) return;
    WGSLValue ignored = {0};
    (void)try_consteval_expr_soft(tc, call, &ignored);
}

typedef enum {
    BOP_VOID_0 = 0,
    BOP_ALL_ANY,
    BOP_SELECT,
    BOP_ARRAY_LENGTH,
    BOP_WORKGROUP_UNIFORM_LOAD,
    BOP_T_FLOAT_1,
    BOP_T_FLOAT_2,
    BOP_T_FLOAT_3,
    BOP_T_NUMERIC_1,
    BOP_T_NUMERIC_2,
    BOP_T_NUMERIC_3,
    BOP_T_SIGN_1,
    BOP_T_INTEGER_1,
    BOP_EXTRACT_BITS,
    BOP_INSERT_BITS,
    BOP_LDEXP,
    BOP_MIX,
    BOP_DOT,
    BOP_DOT4_U32,
    BOP_LENGTH,
    BOP_DETERMINANT,
    BOP_CROSS,
    BOP_TRANSPOSE,
    BOP_VEC_FLOAT_1,
    BOP_VEC_FLOAT_2,
    BOP_VEC_FLOAT_3,
    BOP_REFRACT,
    BOP_T_F32_1,
    BOP_ARG_U32_1,
    BOP_ARG_VEC4F_1,
    BOP_ARG_VEC4I_1,
    BOP_ARG_VEC4U_1,
    BOP_ARG_VEC2F_1,
    BOP_ATOMIC_LOAD,
    BOP_ATOMIC_STORE,
    BOP_ATOMIC_RMW,
    BOP_ATOMIC_CX,
    BOP_BOOL_1,
    BOP_T_NUMERIC_U32_2,
} WGSLBuiltinOverloadPattern;

typedef struct {
    const char                 *name;
    WGSLBuiltinOverloadPattern  pattern;
    WGSLBuiltinReturn           ret;
    uint16_t                    flags;
    uint8_t                     special;
} WGSLBuiltinOverload;

typedef enum {
    BTOV_SAMPLE = 0,
    BTOV_SAMPLE_OFFSET,
    BTOV_SAMPLE_LEVEL,
    BTOV_SAMPLE_LEVEL_OFFSET,
    BTOV_SAMPLE_BIAS,
    BTOV_SAMPLE_BIAS_OFFSET,
    BTOV_SAMPLE_GRAD,
    BTOV_SAMPLE_GRAD_OFFSET,
    BTOV_SAMPLE_COMPARE,
    BTOV_SAMPLE_COMPARE_OFFSET,
    BTOV_SAMPLE_COMPARE_LEVEL,
    BTOV_SAMPLE_COMPARE_LEVEL_OFFSET,
    BTOV_SAMPLE_BASE_CLAMP,
    BTOV_LOAD,
    BTOV_STORE,
    BTOV_GATHER_COMPONENT,
    BTOV_GATHER_COMPONENT_OFFSET,
    BTOV_GATHER_DEPTH,
    BTOV_GATHER_DEPTH_OFFSET,
    BTOV_GATHER_COMPARE,
    BTOV_GATHER_COMPARE_OFFSET,
    BTOV_DIMENSIONS,
    BTOV_DIMENSIONS_LEVEL,
    BTOV_NUM_LAYERS,
    BTOV_NUM_LEVELS,
    BTOV_NUM_SAMPLES,
} WGSLTextureOverloadKind;

typedef enum {
    BTEXEL_ANY = 0,
    BTEXEL_F32,
    BTEXEL_I32,
    BTEXEL_U32,
    BTEXEL_DEPTH,
    BTEXEL_EXTERNAL,
    BTEXEL_STORAGE_ANY,
    BTEXEL_STORAGE_F32,
    BTEXEL_STORAGE_I32,
    BTEXEL_STORAGE_U32,
} WGSLTextureOverloadElem;

typedef struct {
    const char              *name;
    WGSLTextureOverloadKind  kind;
    uint8_t                  dim;       /* WGSLTextureDim */
    uint8_t                  elem;      /* WGSLTextureOverloadElem */
    uint8_t                  access;    /* WGSLAccessMode; NONE = don't care */
} WGSLTextureOverload;

/* Generated from def/wgsl.def.  The fam / arity fields drive §6.1.3
 * overload resolution; the flags field carries explicit @const / @must_use
 * metadata, and special hooks cover mixed-shape overloads that do not fit
 * the compact T-pattern. */
#include "check/builtins.gen.h"

static const WGSLBuiltinEntry *find_builtin_entry(
    const char *name, uint32_t name_len)
{
    for (size_t i = 0; i < sizeof kBuiltinTable / sizeof kBuiltinTable[0]; i++) {
        const WGSLBuiltinEntry *e = &kBuiltinTable[i];
        size_t l = strlen(e->name);
        if (l == name_len && memcmp(e->name, name, l) == 0) return e;
    }
    return NULL;
}

int wgsl_tc_builtin_must_use(const char *name, uint32_t name_len) {
    const WGSLBuiltinEntry *e = find_builtin_entry(name, name_len);
    return e && (e->flags & WGSL_BIF_MUST_USE);
}

static int sym_is_constructor_callable(const WGSLSymbol *sym) {
    if (!sym) return 0;
    switch ((WGSLSymKind)sym->kind) {
    case WGSL_SYM_PREDECLARED_TYPE:
    case WGSL_SYM_PREDECLARED_TYPEGEN:
    case WGSL_SYM_PREDECLARED_TYPE_ALIAS:
    case WGSL_SYM_TYPE_ALIAS:
    case WGSL_SYM_STRUCT:
        return 1;
    default:
        return 0;
    }
}

static int builtin_call_is_const(const WGSLNode *n) {
    if (!n || n->kind != WGSL_NODE_EXPR_CALL || n->child_count < 1) return 0;
    const WGSLNode *callee = n->children[0];
    if (!callee) return 0;

    WGSLSymbol *sym = NULL;
    if (callee->kind == WGSL_NODE_EXPR_IDENT) {
        sym = wgsl_node_resolved_symbol((WGSLNode *)callee);
    } else if (callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
               callee->child_count >= 1 &&
               callee->children[0])
    {
        sym = wgsl_node_resolved_symbol(callee->children[0]);
    }
    if (!sym) return 0;
    if (sym->kind == WGSL_SYM_PREDECLARED_FN) {
        const WGSLBuiltinEntry *e = find_builtin_entry(sym->name, sym->name_len);
        if (!e || !(e->flags & WGSL_BIF_CONST)) return 0;
    } else if (!sym_is_constructor_callable(sym)) {
        return 0;
    }
    for (uint32_t i = 1; i < n->child_count; i++) {
        if (!is_const_expr(n->children[i])) return 0;
    }
    return 1;
}

static int builtin_call_is_override(const WGSLNode *n) {
    if (!n || n->kind != WGSL_NODE_EXPR_CALL || n->child_count < 1) return 0;
    const WGSLNode *callee = n->children[0];
    if (!callee) return 0;

    WGSLSymbol *sym = NULL;
    if (callee->kind == WGSL_NODE_EXPR_IDENT) {
        sym = wgsl_node_resolved_symbol((WGSLNode *)callee);
    } else if (callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
               callee->child_count >= 1 &&
               callee->children[0])
    {
        sym = wgsl_node_resolved_symbol(callee->children[0]);
    }
    if (!sym) return 0;
    if (sym->kind == WGSL_SYM_PREDECLARED_FN) {
        const WGSLBuiltinEntry *e = find_builtin_entry(sym->name, sym->name_len);
        if (!e || !(e->flags & WGSL_BIF_CONST)) return 0;
    } else if (!sym_is_constructor_callable(sym)) {
        return 0;
    }
    for (uint32_t i = 1; i < n->child_count; i++) {
        if (!wgsl_tc_is_override_expr(n->children[i])) return 0;
    }
    return 1;
}

