/**
 * @file exprs_priv.h — shared types and helpers for check/exprs modules.
 */
#ifndef WGSL_INTERNAL_EXPRS_PRIV_H
#define WGSL_INTERNAL_EXPRS_PRIV_H

#include "internal/check.h"
#include "internal/check_priv.h"
#include "internal/token.h"
#include "internal/consteval.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    AT_KIND_LOAD,             /* (ptr) → T */
    AT_KIND_STORE,            /* (ptr, T) → () */
    AT_KIND_RMW,              /* (ptr, T) → T  (Add/Sub/Max/Min/.../Exchange) */
    AT_KIND_CX,               /* (ptr, T, T) → __atomic_cx_result<T> */
} WGSLAtomicKind;

typedef enum {
    BSHAPE_SCALAR_OR_VEC,
    BSHAPE_VEC,
    BSHAPE_VEC3,
    BSHAPE_MAT,
    BSHAPE_MAT_SQUARE,
} WGSLBuiltinShapeReq;

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
} WGSLOverloadCandidate;

/* §3.4 — format helpers for detailed overload diagnostics. */
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

/**
 * Const-eval fold kind (def-driven).  `eval_call_const_builtin`
 * dispatches on this enum instead of a strcmp cascade.  NONE = not
 * folded at shader-creation time (or not yet implemented).
 */
typedef enum {
    WGSL_CFOLD_NONE = 0,
    WGSL_CFOLD_BITCAST,
    WGSL_CFOLD_PACK,
    WGSL_CFOLD_UNPACK,
    WGSL_CFOLD_DOT4,
    WGSL_CFOLD_ANY_ALL,
    WGSL_CFOLD_UNARY_FLOAT,
    WGSL_CFOLD_ABS_SIGN,
    WGSL_CFOLD_BINARY_NUMERIC,
    WGSL_CFOLD_TERNARY_NUMERIC,
    WGSL_CFOLD_LDEXP,
    WGSL_CFOLD_LENGTH,
    WGSL_CFOLD_DISTANCE_DOT,
    WGSL_CFOLD_CROSS,
    WGSL_CFOLD_NORMALIZE,
    WGSL_CFOLD_REFLECT,
    WGSL_CFOLD_FACE_FORWARD,
    WGSL_CFOLD_REFRACT,
    WGSL_CFOLD_TRANSPOSE,
    WGSL_CFOLD_DETERMINANT,
    WGSL_CFOLD_BITFIELD,
    WGSL_CFOLD_INTEGER_BITS,
    WGSL_CFOLD_SELECT,
    WGSL_CFOLD_DECOMPOSE_FLOAT, /* frexp / modf */
} WGSLConstFoldKind;

typedef struct {
    const char       *name;
    WGSLBuiltinReturn ret;
    WGSLParamFamily   fam;     /* diagnostic metadata for parametric families */
    uint8_t           arity;   /* leading T-arity for overload diagnostics */
    uint16_t          flags;   /* WGSL_BIF_* metadata from def/wgsl.def */
    uint8_t           special; /* WGSLBuiltinSpecial validator hook */
    uint8_t           cfold;   /* WGSLConstFoldKind — def-driven consteval */
} WGSLBuiltinEntry;
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


typedef struct {
    const char *name;
    uint16_t    begin;
    uint16_t    count;
} WGSLBuiltinOverloadIndex;

extern const WGSLBuiltinEntry kBuiltinTable[];
extern const WGSLBuiltinOverload kBuiltinOverloads[];
extern const WGSLBuiltinOverloadIndex kBuiltinOverloadIndex[];
#define WGSL_BUILTIN_TABLE_COUNT 146
#define WGSL_BUILTIN_OVERLOAD_COUNT 130
#define WGSL_TEXTURE_OVERLOAD_COUNT 245
#define WGSL_BUILTIN_OVERLOAD_INDEX_COUNT 130

extern const WGSLTextureOverload kTextureOverloads[];


int classify_atomic(const char *name, uint32_t len, WGSLAtomicKind *out);
int atomic_ptr_info(WGSLTypeInfo *arg0, WGSLTypeInfo **out_T, WGSLAddressSpace *out_as, WGSLAccessMode *out_am);
int atomic_ptr_access_ok(WGSLAddressSpace as, WGSLAccessMode am);
WGSLTypeInfo * atomic_pointee_T(WGSLTypeInfo *arg0);
const char * type_label(WGSLTypeInfo *t, char *buf, size_t cap);
int bitcast_scalar_bits(const WGSLTypeInfo *t);
int bitcast_type_bits(const WGSLTypeInfo *t);
void check_atomic_call(WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len, WGSLTypeInfo *arg0_type);
int check_pack_unpack_arg(WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len, WGSLTypeInfo *arg0);
int builtin_name_eq(const char *name, uint32_t len, const char *lit);
int builtin_arg_count(const WGSLNode *call);
int builtin_arg_count_is(WGSLNode *call, int n);
WGSLTypeInfo * builtin_arg_type(WGSLTypeChecker *tc, WGSLNode *call, int arg_index);
int shape_width(const WGSLTypeInfo *t);
WGSLTypeInfo * shape_elem(const WGSLTypeInfo *t);
int type_converts_to(WGSLTypeInfo *from, WGSLTypeInfo *to);
int expect_arg_count(WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len, int min_args, int max_args);
int expect_sampler_arg(WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len, WGSLTypeInfo *t, int comparison);
int expect_float_shape(WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len, WGSLTypeInfo *t, int width, const char *role);
int expect_integer_shape(WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len, WGSLTypeInfo *t, int width, const char *role);
int expect_i32_shape(WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len, WGSLTypeInfo *t, int width, const char *role);
int expect_value_convertible(WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len, WGSLTypeInfo *got_t, WGSLTypeInfo *want_t, const char *role);
int eval_const_int_silent(WGSLTypeChecker *tc, WGSLNode *n, int64_t *out);
int collect_const_int_components(WGSLTypeChecker *tc, WGSLNode *n, int width, int64_t *out);
int check_const_int_range(WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len, int64_t lo, int64_t hi, const char *role);
int texture_is_depth(WGSLTextureDim d);
int texture_is_storage(WGSLTextureDim d);
int texture_is_sampled(WGSLTextureDim d);
int texture_is_multisampled(WGSLTextureDim d);
int texture_is_arrayed(WGSLTextureDim d);
int texture_is_gather_dim(WGSLTextureDim d);
int texture_allows_offset(WGSLTextureDim d);
int texture_offset_width(WGSLTextureDim d);
int texture_coord_width(WGSLTextureDim d);
int texture_dimensions_width(WGSLTextureDim d);
WGSLTypeInfo * storage_texture_elem(WGSLTypeChecker *tc, WGSLTexelFormat f);
WGSLTypeInfo * texture_value_result(WGSLTypeChecker *tc, const char *name, uint32_t name_len, WGSLTypeInfo *tex);
WGSLTypeInfo * texture_builtin_result(WGSLTypeChecker *tc, const char *name, uint32_t name_len, WGSLTypeInfo *tex);
WGSLTypeInfo * texture_arg_type_for_call(WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len, WGSLTypeInfo *arg0_type);
void check_ldexp_call(WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len, WGSLTypeInfo *arg0_type);
void check_mix_call(WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len, WGSLTypeInfo *resolved_shape);
uint32_t const_value_width_tc(const WGSLValue *v);
const WGSLValue * const_value_component_tc(const WGSLValue *v, uint32_t i);
int const_value_gt_tc(WGSLTypeChecker *tc, const WGSLValue *a, const WGSLValue *b, int *out);
int const_value_eq_tc(const WGSLValue *a, const WGSLValue *b, int *out);
void check_clamp_low_high_call(WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len);
void check_smoothstep_edges_call(WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len);
int type_call_builtin(WGSLTypeChecker *tc, WGSLNode *call, WGSLNode *callee, WGSLSymbol *sym);
int ctor_vector_width_from_name(const char *nm, uint32_t len);
int ctor_matrix_dims_from_name(const char *nm, uint32_t len, uint8_t *cols, uint8_t *rows);
int append_ctor_component_type(WGSLTypeChecker *tc, WGSLNode *at, WGSLTypeInfo **items, uint32_t cap, uint32_t *count, WGSLTypeInfo *t);
WGSLTypeInfo * constructor_matrix_elem(WGSLTypeChecker *tc, WGSLNode *at, WGSLTypeInfo *elem);
void check_constructor_arg_range(WGSLTypeChecker *tc, WGSLNode *arg, WGSLTypeInfo *target);
void check_vector_constructor_arg_ranges(WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *elem);
WGSLTypeInfo * infer_typegen_constructor_type(WGSLTypeChecker *tc, WGSLNode *call, WGSLSymbol *sym, int *recognized);
WGSLTypeInfo * resolve_callee_target(WGSLTypeChecker *tc, WGSLNode *callee, WGSLSymbol **out_sym);
int constructor_elem_compatible(WGSLTypeChecker *tc, WGSLTypeInfo *target, WGSLTypeInfo *got);
int check_constructor_elem(WGSLTypeChecker *tc, WGSLNode *arg, WGSLTypeInfo *target_elem, WGSLTypeInfo *got_elem);
int check_explicit_vector_constructor(WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *target);
int check_explicit_matrix_constructor(WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *target);
int check_explicit_struct_constructor(WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *target);
int check_explicit_constructor_args(WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *target);
int type_call(WGSLTypeChecker *tc, WGSLNode *n, WGSLTypeInfo *expected);
WGSLTypeInfo * contextual_type(WGSLTypeInfo *got, WGSLTypeInfo *expected);
int type_literal_int(WGSLTypeChecker *tc, WGSLNode *n, WGSLTypeInfo *expected);
int type_literal_float(WGSLTypeChecker *tc, WGSLNode *n, WGSLTypeInfo *expected);
int type_ident(WGSLTypeChecker *tc, WGSLNode *n);
WGSLTypeInfo * common_scalar_type(WGSLTypeChecker *tc, WGSLTypeInfo *a, WGSLTypeInfo *b, const WGSLNode *at);
WGSLTypeInfo * common_componentwise(WGSLTypeChecker *tc, WGSLTypeInfo *a, WGSLTypeInfo *b, const WGSLNode *at);
WGSLTypeInfo * try_matrix_multiply(WGSLTypeChecker *tc, WGSLTypeInfo *a, WGSLTypeInfo *b, const WGSLNode *at);
int type_unary(WGSLTypeChecker *tc, WGSLNode *n, WGSLTypeInfo *expected);
WGSLTypeInfo * element_type_of(const WGSLTypeInfo *t);
int try_consteval_expr_soft(WGSLTypeChecker *tc, WGSLNode *expr, WGSLValue *out);
int const_value_contains_zero(const WGSLValue *v);
void wgsl_tc_check_constexpr_divisor_nonzero(WGSLTypeChecker *tc, WGSLNode *rhs, WGSLTypeInfo *checked_type, const char *message);
int const_value_shift_amount_out_of_range(const WGSLValue *v, int64_t bit_width, int check_upper, int64_t *bad);
int type_binary(WGSLTypeChecker *tc, WGSLNode *n, WGSLTypeInfo *expected);
int swizzle_lettering_consistent(const char *name, uint32_t len);
int type_member(WGSLTypeChecker *tc, WGSLNode *n);
int type_index(WGSLTypeChecker *tc, WGSLNode *n);
int resolve_overload_silent(WGSLTypeChecker *tc, const WGSLOverloadCandidate *cands, int cand_count, WGSLTypeInfo *const *arg_types, const int *arg_is_const, int arg_count, const char *what, const WGSLNode *at);
int shape_satisfies_req(const WGSLTypeInfo *shape, WGSLBuiltinShapeReq req);
WGSLTypeInfo * resolve_t_family_prefix(WGSLTypeChecker *tc, WGSLNode *call, const char *name, WGSLParamFamily fam, int arity, int total_args, WGSLBuiltinShapeReq req);
WGSLTypeInfo * resolve_t_family_exact(WGSLTypeChecker *tc, WGSLNode *call, const char *name, WGSLParamFamily fam, int arity, WGSLBuiltinShapeReq req);
int match_u32_arg(WGSLTypeChecker *tc, WGSLNode *call, int index);
int match_i32_or_u32_arg(WGSLTypeChecker *tc, WGSLNode *call, int index);
int subgroup_shuffle_second_arg_ok(WGSLTypeChecker *tc, WGSLNode *call, const char *name);
int broadcast_second_arg_ok(WGSLTypeChecker *tc, WGSLNode *call, const char *name);
int match_bool_arg(WGSLTypeChecker *tc, WGSLNode *call, int index);
int match_vec_arg(WGSLTypeChecker *tc, WGSLNode *call, int index, uint8_t width, WGSLTypeInfo *elem);
int match_all_any(WGSLTypeChecker *tc, WGSLNode *call);
WGSLTypeInfo * resolve_select_result(WGSLTypeChecker *tc, WGSLNode *call);
int bitfield_const_range_ok(WGSLTypeChecker *tc, WGSLNode *call, const char *name, int offset_arg, int count_arg);
WGSLTypeInfo * resolve_extract_bits(WGSLTypeChecker *tc, WGSLNode *call, const char *name);
WGSLTypeInfo * resolve_insert_bits(WGSLTypeChecker *tc, WGSLNode *call, const char *name);
WGSLTypeInfo * resolve_ldexp_exact(WGSLTypeChecker *tc, WGSLNode *call, const char *name);
WGSLTypeInfo * resolve_mix_exact(WGSLTypeChecker *tc, WGSLNode *call, const char *name);
WGSLTypeInfo * resolve_refract_exact(WGSLTypeChecker *tc, WGSLNode *call, const char *name);
WGSLTypeInfo * resolve_transpose_exact(WGSLTypeChecker *tc, WGSLNode *call, const char *name);
WGSLTypeInfo * resolve_atomic_exact(WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *arg0_type, WGSLBuiltinOverloadPattern pattern);
int array_length_arg_ok(WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *arg0_type);
void check_array_length_call(WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *arg0_type);
WGSLTypeInfo * resolve_workgroup_uniform_load(WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *arg0_type);
WGSLTypeInfo * result_from_builtin_return(WGSLTypeChecker *tc, WGSLBuiltinReturn ret, WGSLTypeInfo *arg0_type, WGSLTypeInfo *fallback, WGSLTypeInfo *resolved);
int builtin_requires_concrete_result(const char *name, uint32_t name_len);
int match_builtin_overload_row(WGSLTypeChecker *tc, WGSLNode *call, const WGSLBuiltinOverload *row, WGSLTypeInfo *arg0_type, WGSLTypeInfo **out_resolved);
int resolve_generated_builtin_overload(WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len, WGSLTypeInfo *arg0_type, WGSLTypeInfo *fallback, WGSLTypeInfo **out_type);
int builtin_call_is_const(const WGSLNode *n); int builtin_call_is_override(const WGSLNode *n); int is_const_expr(const WGSLNode *n);
/* §11.3 — defined in consteval/user_fn.c */
int user_fn_call_is_const_expr(const WGSLNode *call);
int wgsl_tc_is_override_expr(const WGSLNode *n);
int wgsl_typecheck_is_override_expr(const WGSLNode *n);
void ol_append(char *buf, size_t cap, size_t *used, const char *s);
void ol_append_type(char *buf, size_t cap, size_t *used, WGSLTypeInfo *t);
void ol_append_args(char *buf, size_t cap, size_t *used, WGSLTypeInfo *const *arg_types, int arg_count);
void ol_append_cand(char *buf, size_t cap, size_t *used, const WGSLOverloadCandidate *c, int arg_count);
int resolve_overload(WGSLTypeChecker *tc, const WGSLOverloadCandidate *cands, int cand_count, WGSLTypeInfo *const *arg_types, const int *arg_is_const, int arg_count, const char *what, const WGSLNode *at);
WGSLTypeInfo * lift_to_shape(WGSLTypeStore *ts, WGSLTypeInfo *T, const WGSLTypeInfo *shape);
const WGSLTypeInfo * pick_shape_arg(WGSLTypeInfo *const *args, int n);
int build_t_candidates(WGSLTypeChecker *tc, WGSLParamFamily fam, WGSLTypeInfo *const *args, int arity, WGSLOverloadCandidate *out_cands);
int name_cmp_cstr(const char *name, uint32_t name_len, const char *key);
const WGSLBuiltinEntry * find_builtin_entry(const char *name, uint32_t name_len);
const WGSLBuiltinOverloadIndex * find_overload_index(const char *name, uint32_t name_len);
int wgsl_tc_builtin_must_use(const char *name, uint32_t name_len);
int sym_is_constructor_callable(const WGSLSymbol *sym);
int builtin_call_is_override(const WGSLNode *n);
void wgsl_tc_get_ref_space_and_mode(WGSLTypeChecker *tc, const WGSLNode *n, WGSLAddressSpace *as, WGSLAccessMode *am);
int type_addr_of(WGSLTypeChecker *tc, WGSLNode *n);
int type_indirection(WGSLTypeChecker *tc, WGSLNode *n);
int wgsl_tc_type_expr_inner(WGSLTypeChecker *tc, WGSLNode *n, WGSLTypeInfo *expected); int wgsl_tc_type_expr(WGSLTypeChecker *tc, WGSLNode *n);
int wgsl_tc_type_expr_exp(WGSLTypeChecker *tc, WGSLNode *n, WGSLTypeInfo *expected);
void check_texture_offset_arg(WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len, WGSLTextureDim dim, int arg_index);
void check_texture_sample_call(WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len, WGSLTypeInfo *tex, WGSLTextureDim dim);
void check_texture_load_call(WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len, WGSLTextureDim dim);
void check_texture_store_call(WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len, WGSLTypeInfo *tex, WGSLTextureDim dim);
void check_texture_query_call(WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len, WGSLTextureDim dim);
void check_texture_gather_call(WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len, WGSLTypeInfo *tex, WGSLTextureDim dim, int component_form);
void check_texture_call(WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len, WGSLTypeInfo *tex, int gather_component_form);
int texture_silent_float_shape(WGSLTypeChecker *tc, WGSLNode *call, int arg_index, int width);
int texture_silent_integer_shape(WGSLTypeChecker *tc, WGSLNode *call, int arg_index, int width);
int texture_silent_i32_shape(WGSLTypeChecker *tc, WGSLNode *call, int arg_index, int width);
int texture_silent_sampler(WGSLTypeChecker *tc, WGSLNode *call, int arg_index, int comparison);
int texture_silent_storage_value(WGSLTypeChecker *tc, WGSLNode *call, int arg_index, WGSLTypeInfo *tex);
int texture_row_elem_matches(WGSLTypeChecker *tc, const WGSLTextureOverload *row, WGSLTypeInfo *tex);
int texture_row_texture_matches(WGSLTypeChecker *tc, const WGSLTextureOverload *row, WGSLTypeInfo *tex);
int texture_match_array_index_if_needed(WGSLTypeChecker *tc, WGSLNode *call, WGSLTextureDim dim, int arg_index);
int texture_match_offset(WGSLTypeChecker *tc, WGSLNode *call, WGSLTextureDim dim, int arg_index);
int texture_match_sample_like(WGSLTypeChecker *tc, WGSLNode *call, const WGSLTextureOverload *row, WGSLTypeInfo *tex);
int texture_match_load(WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *tex);
int texture_match_store(WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *tex);
int texture_match_gather(WGSLTypeChecker *tc, WGSLNode *call, const WGSLTextureOverload *row);
int match_texture_overload_row(WGSLTypeChecker *tc, WGSLNode *call, const WGSLTextureOverload *row, WGSLTypeInfo **out_type);
int resolve_generated_texture_overload(WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len, WGSLTypeInfo **out_type);

void check_const_builtin_call(WGSLTypeChecker *tc, WGSLNode *call, const WGSLBuiltinEntry *entry);

#endif /* WGSL_INTERNAL_EXPRS_PRIV_H */
