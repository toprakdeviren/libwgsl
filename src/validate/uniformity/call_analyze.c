/**
 * @file call_analyze.c — check_call_* + analyze_uniform_expr
 */
#include "internal/uniformity_priv.h"

void check_call_requirements(
    UniformCtx *cx, WGSLNode *call, int is_divergent,
    const uint32_t *fn_reqs)
{
    if (!cx || !call || call->kind != WGSL_NODE_EXPR_CALL) return;
    WGSLValidator *v = cx->v;
    if (call->child_count == 0) return;
    WGSLNode *fn = call->children[0];
    if (!fn) return;
    if (fn->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT && fn->child_count > 0) {
        fn = fn->children[0];
    }
    if (!fn || fn->kind != WGSL_NODE_EXPR_IDENT) return;

    const char *nm = v->src->bytes + wgsl_node_name_span(fn).offset;
    uint32_t len = wgsl_node_name_span(fn).length;
    WGSLSymbol *fsym = wgsl_node_resolved_symbol(fn);
    if (fsym && fsym->name) {
        nm = fsym->name;
        len = fsym->name_len;
    }

    is_divergent = uniform_normalize_divergence(is_divergent);
    uint32_t uf = builtin_call_uniformity(nm, len);
    if ((is_divergent & UD_WORKGROUP) && (uf & UNIF_REQ_CF)) {
        wgsl_val_filterable(v, call, WGSL_DIAG_ERROR, "derivative_uniformity",
            "collective operation '%.*s' must be called in "
            "uniform control flow", (int)len, nm);
    }
    if ((is_divergent & UD_SUBGROUP) && (uf & UNIF_REQ_SG_CF)) {
        wgsl_val_filterable(v, call, WGSL_DIAG_ERROR, "subgroup_uniformity",
            "subgroup / quad operation '%.*s' must be called in "
            "uniform control flow", (int)len, nm);
    }
    if (is_divergent && fn_reqs) {
        WGSLSymbol *callee = call_user_fn_sym(call);
        if (callee) {
            int idx = wgsl_val_sym_index(v, callee);
            if (idx >= 0) {
                uint32_t bits = fn_reqs[idx];
                if ((is_divergent & UD_WORKGROUP) && (bits & UR_DERIVATIVE)) {
                    int sev = function_filter_severity(
                        v, callee->ast, "derivative_uniformity", 21);
                    if (sev < 0 || sev == WGSL_DIAG_ERROR) {
                        wgsl_val_error(v, call,
                            "call to '%.*s' must be in uniform "
                            "control flow (callee transitively "
                            "uses derivative-required ops)",
                            (int)callee->name_len, callee->name);
                    } else if (sev > 0) {
                        wgsl_diag_emit_at(v->diag, v->src,
                            (WGSLDiagSeverity)sev,
                            call->span_offset, call->span_length, NULL,
                            "call to '%.*s' must be in uniform "
                            "control flow (callee transitively "
                            "uses derivative-required ops)",
                            (int)callee->name_len, callee->name);
                    }
                }
                if ((is_divergent & UD_SUBGROUP) && (bits & UR_SUBGROUP)) {
                    int sev = function_filter_severity(
                        v, callee->ast, "subgroup_uniformity", 19);
                    if (sev < 0 || sev == WGSL_DIAG_ERROR) {
                        wgsl_val_error(v, call,
                            "call to '%.*s' must be in uniform "
                            "control flow (callee transitively "
                            "uses subgroup / quad ops)",
                            (int)callee->name_len, callee->name);
                    } else if (sev > 0) {
                        wgsl_diag_emit_at(v->diag, v->src,
                            (WGSLDiagSeverity)sev,
                            call->span_offset, call->span_length, NULL,
                            "call to '%.*s' must be in uniform "
                            "control flow (callee transitively "
                            "uses subgroup / quad ops)",
                            (int)callee->name_len, callee->name);
                    }
                }
            }
        }
    }
}

void check_call_param_requirements(
    UniformCtx *cx, WGSLNode *call, const char *nm, uint32_t len,
    const int *arg_uniform, uint32_t argc, int is_divergent,
    const uint32_t *fn_reqs)
{
    if (!cx || !call || !nm) return;
    if (len == 20 && memcmp(nm, "workgroupUniformLoad", 20) == 0 &&
        call->child_count > 1 &&
        !uniform_pointer_address_is_uniform(
            cx, call->children[1], is_divergent, fn_reqs))
    {
        wgsl_val_filterable(cx->v, call, WGSL_DIAG_ERROR,
            "derivative_uniformity",
            "argument to 'workgroupUniformLoad' must have a uniform address");
    }

    int arg_idx = -1;
    if ((len == 17 && memcmp(nm, "subgroupShuffleUp", 17) == 0) ||
        (len == 19 && memcmp(nm, "subgroupShuffleDown", 19) == 0) ||
        (len == 18 && memcmp(nm, "subgroupShuffleXor", 18) == 0))
    {
        arg_idx = 1; /* delta / mask */
    }
    if (arg_idx >= 0 && (uint32_t)arg_idx < argc &&
        arg_uniform && !arg_uniform[arg_idx])
    {
        const char *what =
            (len == 18 && memcmp(nm, "subgroupShuffleXor", 18) == 0)
                ? "mask" : "delta";
        wgsl_val_filterable(cx->v, call, WGSL_DIAG_ERROR,
            "subgroup_uniformity",
            "argument '%s' to '%.*s' must be uniform",
            what, (int)len, nm);
    }
}

int analyze_uniform_expr(
    UniformCtx *cx, WGSLNode *n, int is_divergent,
    const uint32_t *fn_reqs)
{
    if (!n) return 1;
    is_divergent = uniform_normalize_divergence(is_divergent);
    int u = (is_divergent & UD_WORKGROUP) ? 0 : 1;
    int sg = (is_divergent & UD_SUBGROUP) ? 0 : 1;

    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_LITERAL_BOOL:
    case WGSL_NODE_EXPR_LITERAL_INT:
    case WGSL_NODE_EXPR_LITERAL_FLOAT:
        break;

    case WGSL_NODE_EXPR_IDENT: {
        WGSLSymbol *sym = wgsl_node_resolved_symbol(n);
        int src_u = 1;
        int src_sg = 1;
        if (sym) {
            if (sym->kind == WGSL_SYM_VAR) {
                if (uniform_module_var_is_readonly(sym)) {
                    src_u = 1;
                    src_sg = 1;
                } else if (sym->as == WGSL_AS_FUNCTION || sym->as == WGSL_AS_NONE) {
                    src_u = decl_init_is_uniform(cx, sym);
                    src_sg = src_u;
                } else {
                    src_u = 0;
                    src_sg = 0;
                }
            } else if (sym->kind == WGSL_SYM_LET) {
                WGSLNode *init = uniform_decl_init_expr(sym->ast);
                src_u = decl_init_is_uniform(cx, sym);
                src_sg = src_u || (init && uniform_expr_is_subgroup_uniform(init));
            } else if (sym->kind == WGSL_SYM_PARAM) {
                src_u = uniform_param_builtin_is_uniform(cx, sym) ||
                        parameter_is_uniform(cx, sym);
                src_sg = src_u ||
                         uniform_param_builtin_is_subgroup_uniform(cx, sym);
            } else if (sym->kind == WGSL_SYM_CONST ||
                       sym->kind == WGSL_SYM_OVERRIDE ||
                       sym->kind == WGSL_SYM_PREDECLARED_TYPE ||
                       sym->kind == WGSL_SYM_PREDECLARED_TYPEGEN ||
                       sym->kind == WGSL_SYM_PREDECLARED_TYPE_ALIAS ||
                       sym->kind == WGSL_SYM_PREDECLARED_FN)
            {
                src_u = 1;
                src_sg = 1;
            }
        }
        u = !(is_divergent & UD_WORKGROUP) && src_u;
        sg = !(is_divergent & UD_SUBGROUP) && src_sg;
        break;
    }

    case WGSL_NODE_EXPR_BINARY: {
        if (n->child_count < 2) break;
        WGSLTokenKind op = (WGSLTokenKind)wgsl_node_op_kind_u32(n);
        int lhs_u = analyze_uniform_expr(cx, n->children[0],
                                         is_divergent, fn_reqs);
        int lhs_sg = uniform_expr_is_subgroup_uniform(n->children[0]);
        int rhs_div = is_divergent;
        if (op == WGSL_TOK_AMP_AMP || op == WGSL_TOK_PIPE_PIPE) {
            rhs_div = uniform_diverge_for_condition(rhs_div,
                                                    n->children[0], lhs_u);
        }
        int rhs_u = analyze_uniform_expr(cx, n->children[1],
                                         rhs_div, fn_reqs);
        int rhs_sg = uniform_expr_is_subgroup_uniform(n->children[1]);
        u = lhs_u && rhs_u;
        sg = lhs_sg && rhs_sg && !(is_divergent & UD_SUBGROUP);
        break;
    }

    case WGSL_NODE_EXPR_CALL: {
        check_call_requirements(cx, n, is_divergent, fn_reqs);
        const char *nm = NULL;
        uint32_t len = 0;
        if (!call_name(cx, n, &nm, &len)) {
            nm = "";
            len = 0;
        }
        int args_u[16] = {0};
        uint32_t argc = n->child_count > 0 ? n->child_count - 1 : 0;
        if (argc > (uint32_t)(sizeof args_u / sizeof args_u[0])) {
            argc = (uint32_t)(sizeof args_u / sizeof args_u[0]);
        }
        int all_args_u = !(is_divergent & UD_WORKGROUP);
        int all_args_sg = !(is_divergent & UD_SUBGROUP);
        for (uint32_t i = 0; i < argc; i++) {
            args_u[i] = analyze_uniform_expr(cx, n->children[1 + i],
                                             is_divergent, fn_reqs);
            if (!args_u[i]) all_args_u = 0;
            if (!uniform_expr_is_subgroup_uniform(n->children[1 + i])) {
                all_args_sg = 0;
            }
        }
        check_call_param_requirements(
            cx, n, nm, len, args_u, argc, is_divergent, fn_reqs);
        int known_u = (builtin_call_uniformity(nm, len) & UNIF_RESULT_KNOWN) != 0;
        if (known_u && len == 20 &&
            memcmp(nm, "workgroupUniformLoad", 20) == 0)
        {
            known_u = n->child_count > 1 &&
                uniform_pointer_address_is_uniform(
                    cx, n->children[1], is_divergent, fn_reqs);
        }
        u = known_u
            ? !(is_divergent & UD_WORKGROUP)
            : (all_args_u &&
               !call_result_may_be_nonuniform(cx, n, nm, len));
        WGSLSymbol *callee = call_user_fn_sym(n);
        if (callee && !user_function_result_is_uniform(cx, callee, fn_reqs)) {
            u = 0;
        }
        sg = u || (all_args_sg &&
                   (builtin_call_uniformity(nm, len) & UNIF_RESULT_SG));
        break;
    }

    case WGSL_NODE_EXPR_INDIRECTION: {
        if (n->child_count > 0) {
            analyze_uniform_expr(cx, n->children[0], is_divergent, fn_reqs);
            WGSLSymbol *root = uniform_memory_root(n->children[0], 0);
            int addr_u = uniform_pointer_address_is_uniform(
                cx, n->children[0], is_divergent, fn_reqs);
            u = !(is_divergent & UD_WORKGROUP) &&
                addr_u &&
                uniform_root_contents_are_uniform(cx, root);
            sg = u;
        }
        break;
    }

    case WGSL_NODE_EXPR_INDEX: {
        int obj_u = n->child_count > 0
            ? analyze_uniform_expr(cx, n->children[0], is_divergent, fn_reqs)
            : !(is_divergent & UD_WORKGROUP);
        int obj_sg = n->child_count > 0
            ? uniform_expr_is_subgroup_uniform(n->children[0])
            : !(is_divergent & UD_SUBGROUP);
        int idx_u = n->child_count > 1
            ? analyze_uniform_expr(cx, n->children[1], is_divergent, fn_reqs)
            : !(is_divergent & UD_WORKGROUP);
        int idx_sg = n->child_count > 1
            ? uniform_expr_is_subgroup_uniform(n->children[1])
            : !(is_divergent & UD_SUBGROUP);
        u = obj_u && idx_u;
        sg = obj_sg && idx_sg && !(is_divergent & UD_SUBGROUP);
        break;
    }

    default:
        u = !(is_divergent & UD_WORKGROUP);
        sg = !(is_divergent & UD_SUBGROUP);
        for (uint32_t i = 0; i < n->child_count; i++) {
            if (!analyze_uniform_expr(cx, n->children[i], is_divergent, fn_reqs)) {
                u = 0;
            }
            if (!uniform_expr_is_subgroup_uniform(n->children[i])) {
                sg = 0;
            }
        }
        break;
    }

    if (u) {
        n->flags &= (uint16_t)~WGSL_FLAG_NONUNIFORM;
        n->flags |= (uint16_t)WGSL_FLAG_SUBGROUP_UNIFORM;
    } else {
        n->flags |= (uint16_t)WGSL_FLAG_NONUNIFORM;
        if (sg) n->flags |= (uint16_t)WGSL_FLAG_SUBGROUP_UNIFORM;
        else    n->flags &= (uint16_t)~WGSL_FLAG_SUBGROUP_UNIFORM;
    }
    return u;
}

