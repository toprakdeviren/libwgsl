/**
 * @file uniformity_priv.h
 */
#ifndef WGSL_INTERNAL_UNIFORMITY_PRIV_H
#define WGSL_INTERNAL_UNIFORMITY_PRIV_H
#include "internal/validate.h"
#include "internal/validate_priv.h"

#define UR_DERIVATIVE   0x1u   /* fn needs uniform CF for §15.6.2 */
#define UR_SUBGROUP     0x2u   /* same for §15.6.3 / §15.6.4 */

/* §15.6 — per-builtin-FUNCTION uniformity classification.  Single embedded
 * table + lookup (`builtin_call_uniformity`, call_reqs.c); NOT def-generated. */
enum {
    UNIF_REQ_CF       = 1u << 0, /* must be called in workgroup-uniform CF (§15.6.1/.2) */
    UNIF_REQ_SG_CF    = 1u << 1, /* must be called in subgroup-uniform CF (§15.6.3/.4) */
    UNIF_RESULT_KNOWN = 1u << 2, /* result (workgroup-)uniform regardless of arg uniformity */
    UNIF_RESULT_SG    = 1u << 3, /* result subgroup-uniform when args are */
};


#include "internal/resolver.h"
#include "internal/types.h"
#include "internal/ast.h"
#include "internal/token.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    WGSLValidator *v;
    const uint8_t *param_nonuniform;
    size_t         nsyms;
    int            init_depth;
    uint8_t       *sym_uniform;
    uint8_t       *sym_known;
    const uint8_t *fn_returns_uniform; /* per-decl-index: 1=fn always returns uniform; NULL ok */
} UniformCtx;


enum {
    UD_NONE      = 0,
    UD_WORKGROUP = 1u << 0,
    UD_SUBGROUP  = 1u << 1,
    UD_ALL       = UD_WORKGROUP | UD_SUBGROUP,
};



typedef uint8_t UniformBehavior;


typedef struct {
    WGSLValidator *v;
    uint32_t       bits;       /* output: collected for the fn currently being scanned */
} ReqScanCtx;


typedef struct {
    UniformCtx *ucx;
    uint8_t    *nonuniform;
    int        *changed;
} PtrParamCtx;


void scan_pointer_param_calls(PtrParamCtx *pcx, WGSLNode *n);
uint8_t * compute_param_nonuniform(WGSLValidator *v);
uint8_t * compute_fn_returns_uniform(WGSLValidator *v, const uint32_t *fn_reqs, const uint8_t *param_nonuniform);
void check_uniformity_block(UniformCtx *cx, WGSLNode *n, int is_divergent, const uint32_t *fn_reqs);
void validate_uniformity(WGSLValidator *v, WGSLNode *tu);
int uniform_normalize_divergence(int divergence);
int uniform_expr_is_subgroup_uniform(const WGSLNode *n);
int uniform_diverge_for_condition(int parent_divergence, const WGSLNode *cond, int workgroup_uniform);
int ident_text_eq(WGSLValidator *v, const WGSLNode *n, const char *want, uint32_t want_len);
int diagnostic_attr_filter(WGSLValidator *v, const WGSLNode *attr, const char **out_rule, uint32_t *out_rule_len, int *out_severity);
int set_diagnostic_attr_filters(WGSLValidator *v, WGSLNode *const *attrs, uint32_t attr_count);
int push_diagnostic_attr_scope(WGSLValidator *v, WGSLNode *const *attrs, uint32_t attr_count);
int push_node_diagnostic_attr_scope(WGSLValidator *v, const WGSLNode *n);
int function_filter_severity(WGSLValidator *v, const WGSLNode *fn, const char *rule, uint32_t rule_len);
int uniform_state_get(UniformCtx *cx, WGSLSymbol *sym, int *out);
void uniform_state_set(UniformCtx *cx, WGSLSymbol *sym, int uniform);
int uniform_state_copy(UniformCtx *cx, uint8_t **known, uint8_t **uniform);
void uniform_state_restore(UniformCtx *cx, const uint8_t *known, const uint8_t *uniform);
void uniform_state_join_into(size_t n, uint8_t *acc_known, uint8_t *acc_uniform, const uint8_t *known, const uint8_t *uniform);
void uniform_state_join_current_into(UniformCtx *cx, uint8_t *acc_known, uint8_t *acc_uniform);
void uniform_state_clear_arrays(UniformCtx *cx, uint8_t *known, uint8_t *uniform);
void uniform_state_mark_known_nonuniform(UniformCtx *cx);
int uniform_state_alloc_zero(UniformCtx *cx, uint8_t **known, uint8_t **uniform);
WGSLNode * uniform_decl_init_expr(const WGSLNode *d);
int decl_init_is_uniform(UniformCtx *cx, WGSLSymbol *sym);
int uniform_expr_quiet(UniformCtx *cx, WGSLNode *expr, int is_divergent, const uint32_t *fn_reqs);
WGSLSymbol * uniform_assignment_root(WGSLNode *n);
WGSLSymbol * uniform_memory_root(WGSLNode *n, int depth);
int uniform_root_contents_are_uniform(UniformCtx *cx, WGSLSymbol *sym);
int uniform_pointer_address_is_uniform(UniformCtx *cx, WGSLNode *n, int is_divergent, const uint32_t *fn_reqs);
int uniform_reference_address_is_uniform(UniformCtx *cx, WGSLNode *n, int is_divergent, const uint32_t *fn_reqs);
int uniform_lhs_is_full_reference(WGSLNode *n);
int uniform_expr_node(const WGSLNode *n);
WGSLNode * uniform_for_init(WGSLNode *n);
WGSLNode * uniform_for_cond(WGSLNode *n);
WGSLNode * uniform_for_update(WGSLNode *n);
WGSLNode * uniform_for_body(WGSLNode *n);
UniformBehavior uniform_beh_compound(WGSLNode *n);
UniformBehavior uniform_beh_stmt(WGSLNode *n);
int uniform_stmt_has_partial_interrupt(UniformCtx *cx, WGSLNode *n, int is_divergent, UniformBehavior mask, const uint32_t *fn_reqs);
int uniform_loop_has_partial_interrupt(UniformCtx *cx, WGSLNode *loop, int is_divergent, const uint32_t *fn_reqs);
int uniform_loop_makes_following_divergent(UniformCtx *cx, WGSLNode *n, int is_divergent, const uint32_t *fn_reqs);
UniformBehavior uniform_collect_break_exits(UniformCtx *cx, WGSLNode *n, int is_divergent, const uint32_t *fn_reqs, uint8_t *exit_known, uint8_t *exit_uniform, int *saw_exit);
int uniform_is_param(const WGSLSymbol *sym);
int parameter_is_uniform(UniformCtx *cx, WGSLSymbol *sym);
WGSLSymbol * uniform_fn_param_symbol(WGSLValidator *v, const WGSLSymbol *fn, uint32_t param_idx);
int is_uniform_expr(UniformCtx *cx, WGSLNode *n);
uint32_t builtin_call_uniformity(const char *nm, uint32_t len);
void scan_fn_direct_reqs(ReqScanCtx *cx, WGSLNode *n);
void prop_edge_cb(WGSLValidator *v, const WGSLSymbol *to, const WGSLNode *at, void *ctx);
uint32_t * compute_fn_uniformity_reqs(WGSLValidator *v);
WGSLSymbol * call_user_fn_sym(WGSLNode *call);
int uniform_param_builtin_name(UniformCtx *cx, WGSLSymbol *sym, const char **out_name, uint32_t *out_len);
WGSLSymbol * uniform_param_owner_function(UniformCtx *cx, WGSLSymbol *param);
int uniform_param_builtin_is_uniform(UniformCtx *cx, WGSLSymbol *sym);
int uniform_param_builtin_is_subgroup_uniform(UniformCtx *cx, WGSLSymbol *sym);
int uniform_module_var_is_readonly(const WGSLSymbol *sym);
int call_name(UniformCtx *cx, WGSLNode *call, const char **name, uint32_t *len);
int uniform_param_index(WGSLValidator *v, const WGSLSymbol *fn, const WGSLSymbol *param, uint32_t *out);
WGSLSymbol * uniform_lhs_pointer_param(WGSLNode *lhs);
int uniform_pointer_param_lhs_address_is_uniform(UniformCtx *cx, WGSLNode *lhs, WGSLSymbol *param, WGSLNode *actual, int is_divergent, const uint32_t *fn_reqs);
void apply_user_call_effects_from_stmt(UniformCtx *cx, WGSLSymbol *callee, WGSLNode *call, WGSLNode *stmt, int is_divergent, const uint32_t *fn_reqs);
void apply_user_call_pointer_effects(UniformCtx *cx, WGSLNode *call, int is_divergent, const uint32_t *fn_reqs);
int call_reads_readwrite_storage_texture(UniformCtx *cx, WGSLNode *call, const char *nm, uint32_t len);
int call_result_may_be_nonuniform(UniformCtx *cx, WGSLNode *call, const char *nm, uint32_t len);
void uniform_collect_returns(UniformCtx *cx, WGSLNode *n, const uint32_t *fn_reqs, int *saw_return, int *all_uniform);
int user_function_result_is_uniform(UniformCtx *cx, WGSLSymbol *fn, const uint32_t *fn_reqs);
void check_call_requirements(UniformCtx *cx, WGSLNode *call, int is_divergent, const uint32_t *fn_reqs);
void check_call_param_requirements(UniformCtx *cx, WGSLNode *call, const char *nm, uint32_t len, const int *arg_uniform, uint32_t argc, int is_divergent, const uint32_t *fn_reqs);
int analyze_uniform_expr(UniformCtx *cx, WGSLNode *n, int is_divergent, const uint32_t *fn_reqs);

#endif
