/**
 * @file check_walk_priv.h — shared surface for check walk modules.
 */
#ifndef WGSL_INTERNAL_CHECK_WALK_PRIV_H
#define WGSL_INTERNAL_CHECK_WALK_PRIV_H

#include "internal/check.h"
#include "internal/check_priv.h"
#include "internal/token.h"
#include "internal/consteval.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    WGSLSymbol *sym;
    uint32_t    decl_offset;
    const WGSLNode *decl_node;
} ContUse;


int loop_decls_contain(const ContUse *uses, int n_uses, WGSLSymbol *sym);
int sym_in_loop_body(const WGSLSymbol *sym, uint32_t body_start, uint32_t body_end);
void collect_continuing_uses(const WGSLNode *n, uint32_t body_start, uint32_t body_end, ContUse **uses, int *n_uses, int *cap);
void scan_continues_before(WGSLTypeChecker *tc, WGSLNode *n, const ContUse *uses, int n_uses);
void check_continue_past_decl(WGSLTypeChecker *tc, WGSLNode *loop);
int type_function(WGSLTypeChecker *tc, WGSLNode *fn);
int wgsl_tc_type_module_decl(WGSLTypeChecker *tc, WGSLNode *n);
void walk_stmt(WGSLTypeChecker *tc, WGSLNode *n);
void collect_directives(WGSLTypeChecker *tc, WGSLNode *tu);
int wgsl_typecheck(WGSLAst *ast, const WGSLSource *src, WGSLArena *arena, WGSLDiagBag *diag, WGSLTypeStore *types, WGSLResolver *res, WGSLConstEvaluator *cev, WGSLTypeChecker *out);
int wgsl_typecheck_with_opts(WGSLAst *ast, const WGSLSource *src, WGSLArena *arena, WGSLDiagBag *diag, WGSLTypeStore *types, WGSLResolver *res, WGSLConstEvaluator *cev, WGSLTypeChecker *out, const WGSLTypecheckOpts *opts);
int wgsl_typecheck_ensure_function_body(WGSLTypeChecker *tc, WGSLNode *fn);
void wgsl_tc_error(WGSLTypeChecker *tc, const WGSLNode *at, const char *fmt, ...);
int wgsl_tc_set_type(WGSLTypeChecker *tc, WGSLNode *n, WGSLTypeInfo *t, int is_ref);
WGSLTypeInfo * wgsl_typecheck_type_of(const WGSLTypeChecker *tc, const WGSLNode *n);
WGSLTypeInfo * wgsl_tc_unwrap_ref_type(WGSLTypeInfo *t);
WGSLTypeInfo * wgsl_tc_store_type_of(const WGSLTypeChecker *tc, const WGSLNode *n);
int wgsl_typecheck_is_ref(const WGSLTypeChecker *tc, const WGSLNode *n);
int tc_same_shape(WGSLTypeInfo *a, WGSLTypeInfo *b);
int tc_can_convert(WGSLTypeInfo *from, WGSLTypeInfo *to);
WGSLTokenKind compound_binary_op(WGSLTokenKind op);
int validate_compound_assignment(WGSLTypeChecker *tc, WGSLNode *stmt, WGSLTypeInfo *lt, WGSLTypeInfo *rt);
void wgsl_typecheck_destroy(WGSLTypeChecker *tc);
WGSLSymbol * wgsl_tc_find_decl_symbol(const WGSLTypeChecker *tc, const WGSLNode *decl);

#endif /* WGSL_INTERNAL_CHECK_WALK_PRIV_H */
