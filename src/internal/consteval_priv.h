/**
 * @file consteval_priv.h — multi-TU shared surface for consteval modules.
 *
 * consteval used to be a single TU with .inc fragments. Shared helpers are
 * declared here; definitions live in consteval.c or consteval modules.
 */
#ifndef WGSL_INTERNAL_CONSTEVAL_PRIV_H
#define WGSL_INTERNAL_CONSTEVAL_PRIV_H

#include "internal/consteval.h"
#include "internal/ast.h"
#include "internal/types.h"
#include "internal/resolver.h"
#include "internal/token.h"
#include "internal/diag.h"
#include "internal/source.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

double f16_bits_to_double(uint16_t h);
uint16_t double_to_f16_bits(double x);
int value_to_u32_bits(WGSLConstEvaluator *cev, WGSLNode *at, const WGSLValue *v, uint32_t *bits);
int make_bitcast_component(WGSLConstEvaluator *cev, WGSLNode *call, WGSLTypeInfo *target, uint32_t bits, WGSLValue *out);
int eval_call_bitcast_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
int eval_call_unpack_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
int eval_call_pack_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
int eval_call_dot4_packed_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
int eval_unary_float_builtin_scalar(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *arg, WGSLValue *out);
int eval_call_unary_float_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
int fold_abs_sign_scalar(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *arg, WGSLValue *out);
int eval_call_abs_sign_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
int value_numeric_to_double(const WGSLValue *v, double *out);
double det_small(double m[4][4], int n);
int eval_call_determinant_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
int read_u32_const_arg(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, const WGSLValue *v, uint32_t *out);
int bitfield_shape(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *value, WGSLValue *newbits, uint32_t *out_width, WGSLTypeInfo **out_vec_type);
int fold_extract_bits_scalar(WGSLConstEvaluator *cev, WGSLNode *call, WGSLValue *value, uint32_t offset, uint32_t count, WGSLValue *out);
int fold_insert_bits_scalar(WGSLConstEvaluator *cev, WGSLNode *call, WGSLValue *value, WGSLValue *newbits, uint32_t offset, uint32_t count, WGSLValue *out);
int eval_call_bitfield_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
uint32_t reverse32(uint32_t x);
uint32_t popcount32(uint32_t x);
uint32_t clz32(uint32_t x);
uint32_t ctz32(uint32_t x);
int fold_integer_bits_scalar(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *v, WGSLValue *out);
int eval_call_integer_bits_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
int eval_call_select_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
int eval_call_decompose_float_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
int eval_call_any_all_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
int eval_call_const_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
int node_ident_text(const WGSLConstEvaluator *cev, const WGSLNode *n, const char **out, uint32_t *len);
int name_eq(const char *nm, uint32_t len, const char *want);
uint32_t builtin_flop_weight(const char *nm, uint32_t len);
double round_even(double x);
int round_float_value_to_type(WGSLConstEvaluator *cev, WGSLTypeInfo *type, double x, double *out);
int finish_float_builtin_result(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLTypeInfo *result_type, double r, WGSLValue *out);
int finish_float_builtin_result_relaxed_range(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLTypeInfo *result_type, double r, WGSLValue *out);
int require_float_scalar(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *v);
int promote_three(WGSLConstEvaluator *cev, WGSLValue *a, WGSLValue *b, WGSLValue *c, const WGSLNode *at);
uint32_t value_width(const WGSLValue *v);
WGSLValue value_component(const WGSLValue *v, uint32_t i);
int componentwise_shape(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *vals, int argc, int scalar_broadcast_arg, uint32_t *out_width, WGSLTypeInfo **out_vec_type);
int make_componentwise_result(WGSLConstEvaluator *cev, WGSLNode *call, uint32_t width, WGSLTypeInfo *vec_type, WGSLValue *elems, WGSLValue *out);
int make_vector_result(WGSLConstEvaluator *cev, WGSLNode *call, uint32_t width, WGSLTypeInfo *elem_type, WGSLValue *elems, WGSLValue *out);
int require_same_width_vectors(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *vals, uint32_t argc, uint32_t *out_width);
int vector_width_from_name(const char *nm, uint32_t len);
int matrix_dims_from_name(const char *nm, uint32_t len, uint8_t *cols, uint8_t *rows);
WGSLSymbol * callee_head_symbol(const WGSLNode *callee);
uint32_t struct_member_count(const WGSLNode *sd);
int make_zero_value(WGSLConstEvaluator *cev, WGSLTypeInfo *target, const WGSLNode *at, WGSLValue *out);
WGSLTypeInfo * constructor_target_type(const WGSLConstEvaluator *cev, const WGSLNode *callee);
int convert_constructor_component(WGSLConstEvaluator *cev, WGSLValue *v, WGSLTypeInfo *elem, const WGSLNode *at);
int append_constructor_value(WGSLConstEvaluator *cev, WGSLValue *dst, uint32_t width, uint32_t *count, WGSLTypeInfo *elem, WGSLValue *src, const WGSLNode *at);
int common_constructor_type(WGSLConstEvaluator *cev, WGSLValue *vals, uint32_t count, const WGSLNode *at, WGSLTypeInfo **out);
int materialize_constructor_values(WGSLConstEvaluator *cev, WGSLValue *vals, uint32_t count, WGSLTypeInfo *target, const WGSLNode *at);
int convert_scalar_constructor_value(WGSLConstEvaluator *cev, WGSLValue *v, WGSLTypeInfo *target, const WGSLNode *at);
int eval_vector_constructor(WGSLConstEvaluator *cev, WGSLNode *call, WGSLTypeInfo *target, WGSLValue *out);
int eval_matrix_constructor(WGSLConstEvaluator *cev, WGSLNode *call, WGSLTypeInfo *target, WGSLValue *out);
int eval_array_constructor(WGSLConstEvaluator *cev, WGSLNode *call, WGSLTypeInfo *target, WGSLValue *out);
int eval_struct_constructor(WGSLConstEvaluator *cev, WGSLNode *call, WGSLTypeInfo *target, WGSLValue *out);
int eval_constructor_call(WGSLConstEvaluator *cev, WGSLNode *call, WGSLTypeInfo *target, WGSLValue *out);
int append_inferred_component(WGSLConstEvaluator *cev, WGSLValue *dst, uint32_t cap, uint32_t *count, const WGSLValue *src, const WGSLNode *at);
int eval_inferred_vector_constructor(WGSLConstEvaluator *cev, WGSLNode *call, uint8_t width, WGSLValue *out);
int eval_inferred_matrix_constructor(WGSLConstEvaluator *cev, WGSLNode *call, uint8_t cols, uint8_t rows, WGSLValue *out);
int eval_inferred_array_constructor(WGSLConstEvaluator *cev, WGSLNode *call, WGSLValue *out);
int eval_inferred_typegen_constructor(WGSLConstEvaluator *cev, WGSLNode *call, WGSLValue *out);
uint32_t cev_leaf_bytes(const WGSLTypeInfo *t);
void cev_count_load(WGSLConstEvaluator *cev, int base_storage, WGSLValue *out);
int eval_call(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out);
int eval_index(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out);
int swizzle_index(char c, int *set);
int eval_vector_member(WGSLConstEvaluator *cev, WGSLNode *n, const WGSLValue *base, const char *nm, uint32_t len, WGSLValue *out);
int eval_struct_member(WGSLConstEvaluator *cev, WGSLNode *n, const WGSLValue *base, const char *nm, uint32_t len, WGSLValue *out);
int eval_member(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out);
int eval_expr(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out);
int eval_expr_inner(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out);
int wgsl_consteval_expr(WGSLConstEvaluator *cev, WGSLNode *expr, WGSLValue *out);
WGSLTypeInfo * type_from_typespec(const WGSLConstEvaluator *cev, const WGSLNode *tnode);
int eval_decl_const(WGSLConstEvaluator *cev, WGSLNode *n);
int eval_decl_override(WGSLConstEvaluator *cev, WGSLNode *n);
void walk_block(WGSLConstEvaluator *cev, WGSLNode *block);
void walk_function(WGSLConstEvaluator *cev, WGSLNode *fn);
int wgsl_consteval(WGSLAst *ast, const WGSLSource *src, WGSLArena *arena, WGSLDiagBag *diag, WGSLTypeStore *types, WGSLResolver *res, WGSLConstEvaluator *out);
int int_fits_target(int64_t v, WGSLTypeKind k);
int float_fits_target(double v, WGSLTypeKind k);
int wgsl_consteval_materialize(WGSLConstEvaluator *cev, WGSLValue *v, WGSLTypeInfo *target, const WGSLNode *at_node);
int promote_pair(WGSLConstEvaluator *cev, WGSLValue *a, WGSLValue *b, const WGSLNode *at_node);
int int_div_mod(WGSLConstEvaluator *cev, WGSLNode *at, int64_t a, int64_t b, int is_unsigned, int is_mod, int64_t *out);
int checked_abstract_int_result(WGSLConstEvaluator *cev, WGSLNode *at, __int128 value, const char *op_name, int64_t *out);
void cev_bill_op(WGSLConstEvaluator *cev, WGSLTokenKind op, int is_float);
int eval_binary_int(WGSLConstEvaluator *cev, WGSLNode *at, WGSLTokenKind op, WGSLValue *a, WGSLValue *b, WGSLValue *out);
int eval_binary_float(WGSLConstEvaluator *cev, WGSLNode *at, WGSLTokenKind op, WGSLValue *a, WGSLValue *b, WGSLValue *out);
int eval_binary_bool(WGSLConstEvaluator *cev, WGSLNode *at, WGSLTokenKind op, WGSLValue *a, WGSLValue *b, WGSLValue *out);
int eval_binary_componentwise(WGSLConstEvaluator *cev, WGSLNode *n, WGSLTokenKind op, WGSLValue *a, WGSLValue *b, WGSLValue *out);
int eval_numeric_scalar_binary(WGSLConstEvaluator *cev, WGSLNode *n, WGSLTokenKind op, WGSLValue *a, WGSLValue *b, WGSLValue *out);
int make_matrix_binary_result(WGSLConstEvaluator *cev, WGSLNode *n, uint32_t cols, uint32_t rows, WGSLValue *elems, WGSLValue *out);
int make_vec_binary_result(WGSLConstEvaluator *cev, WGSLNode *n, uint32_t width, WGSLValue *elems, WGSLValue *out);
int eval_matrix_binary(WGSLConstEvaluator *cev, WGSLNode *n, WGSLTokenKind op, WGSLValue *a, WGSLValue *b, WGSLValue *out);
int eval_unary_scalar(WGSLConstEvaluator *cev, WGSLNode *n, WGSLTokenKind op, WGSLValue inner, WGSLValue *out);
int eval_unary(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out);
int combine_binary(WGSLConstEvaluator *cev, WGSLNode *n, WGSLTokenKind op, WGSLValue *a, WGSLValue *b, WGSLValue *out);
int eval_binary(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out);
int wgsl_consteval_binop(WGSLConstEvaluator *cev, WGSLNode *ctx, WGSLTokenKind op, const WGSLValue *a, const WGSLValue *b, WGSLValue *out);
int fold_minmax_scalar(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *a, WGSLValue *b, WGSLValue *out);
int fold_float_binary_scalar(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *a, WGSLValue *b, WGSLValue *out);
int eval_call_binary_numeric_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
int fold_clamp_scalar(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *x, WGSLValue *lo, WGSLValue *hi, WGSLValue *out);
int fold_float_ternary_scalar(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *a, WGSLValue *b, WGSLValue *c, WGSLValue *out);
int eval_call_ternary_numeric_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
int eval_call_ldexp_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
int eval_call_length_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
int eval_call_distance_dot_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
int eval_call_cross_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
int eval_call_normalize_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
int fold_float_dot(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, const WGSLValue *a, const WGSLValue *b, uint32_t width, WGSLTypeInfo **out_type, double *out_dot);
int fold_float_inherited_binary(WGSLConstEvaluator *cev, WGSLNode *call, WGSLTypeInfo *type, WGSLTokenKind op, double a, double b, double *out);
int eval_call_reflect_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
int eval_call_face_forward_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
int eval_call_refract_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
int eval_call_transpose_builtin(WGSLConstEvaluator *cev, WGSLNode *call, const char *nm, uint32_t len, WGSLValue *out);
/**
 * Emit a const-eval diagnostic.  Default status = WGSL_CEV_VALUE_ERROR.
 * Use `cev_error_kind` when the failure is NOT_CONST or NOT_IMPLEMENTED
 * so soft-eval can roll it back without scanning message text.
 */
void cev_error(WGSLConstEvaluator *cev, const WGSLNode *at, const char *fmt, ...);
void cev_error_kind(WGSLConstEvaluator *cev, WGSLCevStatus kind,
                    const WGSLNode *at, const char *fmt, ...);
void cev_note_status(WGSLConstEvaluator *cev, WGSLCevStatus kind);
void wgsl_consteval_init(WGSLConstEvaluator *cev, WGSLArena *arena, WGSLDiagBag *diag, WGSLTypeStore *types, const WGSLSource *src, WGSLResolver *res);
void wgsl_consteval_destroy(WGSLConstEvaluator *cev);
int wgsl_consteval_sym_is_per_lane(const WGSLSymbol *sym);
int binding_per_lane(const WGSLConstEvaluator *cev, const WGSLSymbol *sym);
const WGSLValue * wgsl_consteval_value_of(const WGSLConstEvaluator *cev, const WGSLSymbol *sym);
WGSLValue deep_copy_value(WGSLConstEvaluator *cev, const WGSLValue *v);
int binding_write(WGSLConstEvaluator *cev, WGSLConstBinding *b, const WGSLValue *v);
WGSLConstBinding * append_binding(WGSLConstEvaluator *cev, const WGSLSymbol *sym);
int store_binding(WGSLConstEvaluator *cev, const WGSLSymbol *sym, const WGSLValue *v);
int wgsl_consteval_set_value(WGSLConstEvaluator *cev, const WGSLSymbol *sym, const WGSLValue *v);
WGSLSymbol * find_decl_symbol(const WGSLConstEvaluator *cev, const WGSLNode *decl);
int slice_text(const WGSLSource *src, const WGSLNode *n, char *buf, size_t cap);
WGSLTypeInfo * type_for_int_suffix(WGSLConstEvaluator *cev, uint32_t suffix_bits);
WGSLTypeInfo * type_for_float_suffix(WGSLConstEvaluator *cev, uint32_t suffix_bits);
int parse_int_literal(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out);
void truncate_significand(char *buf, int len, int max_sig, int is_hex);
int parse_float_literal(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out);
int eval_literal_bool(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out);
int val_is_int(const WGSLValue *v);
int val_is_float(const WGSLValue *v);
int val_is_numeric(const WGSLValue *v);
int val_is_bool(const WGSLValue *v);

/* §11.3 pure user-function const-eval (user_fn.c) */
int user_fn_is_const_form(const WGSLNode *fn);
int user_fn_call_is_const_expr(const WGSLNode *call);
int eval_user_fn_const_call(WGSLConstEvaluator *cev, WGSLNode *call,
                            WGSLSymbol *sym, WGSLValue *out);

#endif /* WGSL_INTERNAL_CONSTEVAL_PRIV_H */
