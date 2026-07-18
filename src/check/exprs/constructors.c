/**
 * @file constructors.c — expression typing module (was exprs/constructors.inc).
 */
#include "internal/exprs_priv.h"

/* EXPR_CALL: constructor / user-fn / builtin. */

int ctor_vector_width_from_name(const char *nm, uint32_t len) {
    if (len == 4 && memcmp(nm, "vec", 3) == 0 &&
        nm[3] >= '2' && nm[3] <= '4')
    {
        return nm[3] - '0';
    }
    return 0;
}

int ctor_matrix_dims_from_name(
    const char *nm, uint32_t len, uint8_t *cols, uint8_t *rows)
{
    if (len == 6 && memcmp(nm, "mat", 3) == 0 && nm[4] == 'x' &&
        nm[3] >= '2' && nm[3] <= '4' &&
        nm[5] >= '2' && nm[5] <= '4')
    {
        *cols = (uint8_t)(nm[3] - '0');
        *rows = (uint8_t)(nm[5] - '0');
        return 1;
    }
    return 0;
}

int append_ctor_component_type(
    WGSLTypeChecker *tc, WGSLNode *at,
    WGSLTypeInfo **items, uint32_t cap, uint32_t *count, WGSLTypeInfo *t)
{
    if (!t) return 0;
    if (t->kind == WGSL_TYPE_VEC) {
        for (uint32_t i = 0; i < t->width; i++) {
            if (*count >= cap) {
                wgsl_tc_error(tc, at, "too many constructor components");
                return 0;
            }
            items[(*count)++] = (WGSLTypeInfo *)t->ref;
        }
        return 1;
    }
    if (wgsl_type_is_scalar(t)) {
        if (*count >= cap) {
            wgsl_tc_error(tc, at, "too many constructor components");
            return 0;
        }
        items[(*count)++] = t;
        return 1;
    }
    wgsl_tc_error(tc, at, "constructor requires scalar or vector components");
    return 0;
}

static WGSLTypeInfo *common_constructor_type(
    WGSLTypeChecker *tc, WGSLNode *at, WGSLTypeInfo **items, uint32_t count)
{
    if (count == 0) return tc->types->t_abstract_int;
    WGSLTypeInfo *common = items[0];
    for (uint32_t i = 1; i < count; i++) {
        WGSLTypeInfo *t = items[i];
        if (!t) return NULL;
        if (common == t) continue;
        int ab = wgsl_type_conversion_rank(common, t);
        int ba = wgsl_type_conversion_rank(t, common);
        if (ab >= 0 && (ba < 0 || ab <= ba)) {
            common = t;
        } else if (ba >= 0) {
            /* Keep existing common type. */
        } else {
            wgsl_tc_error(tc, at, "constructor arguments have incompatible types");
            return NULL;
        }
    }
    return common;
}

WGSLTypeInfo *constructor_matrix_elem(
    WGSLTypeChecker *tc, WGSLNode *at, WGSLTypeInfo *elem)
{
    if (elem == tc->types->t_abstract_int) elem = tc->types->t_abstract_float;
    if (!wgsl_type_is_float(elem)) {
        wgsl_tc_error(tc, at,
            "matrix constructor requires floating-point components");
        return NULL;
    }
    return elem;
}

void check_constructor_arg_range(
    WGSLTypeChecker *tc, WGSLNode *arg, WGSLTypeInfo *target)
{
    if (!tc || !arg || !target) return;
    wgsl_tc_check_init_concretization_range(tc, arg, target);
}

void check_vector_constructor_arg_ranges(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *elem)
{
    if (!elem) return;
    for (uint32_t i = 1; i < call->child_count; i++) {
        WGSLNode *arg = call->children[i];
        WGSLTypeInfo *t = wgsl_tc_store_type_of(tc, arg);
        if (t && t->kind == WGSL_TYPE_VEC) {
            check_constructor_arg_range(
                tc, arg, wgsl_type_vec(tc->types, t->width, elem));
        } else {
            check_constructor_arg_range(tc, arg, elem);
        }
    }
}

WGSLTypeInfo *infer_typegen_constructor_type(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLSymbol *sym, int *recognized)
{
    *recognized = 0;
    if (!sym || sym->kind != WGSL_SYM_PREDECLARED_TYPEGEN) return NULL;
    const char *nm = sym->name;
    uint32_t len = sym->name_len;
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;

    int width = ctor_vector_width_from_name(nm, len);
    if (width) {
        *recognized = 1;
        if (argc == 0) {
            return wgsl_type_vec(tc->types, (uint8_t)width,
                                 tc->types->t_abstract_int);
        }
        if (argc == 1) {
            WGSLTypeInfo *arg = wgsl_tc_store_type_of(tc, call->children[1]);
            if (!arg) return NULL;
            if (wgsl_type_is_scalar(arg)) {
                return wgsl_type_vec(tc->types, (uint8_t)width, arg);
            }
        }
        WGSLTypeInfo *items[4] = {0};
        uint32_t count = 0;
        for (uint32_t i = 1; i < call->child_count; i++) {
            WGSLTypeInfo *t = wgsl_tc_store_type_of(tc, call->children[i]);
            if (!append_ctor_component_type(
                    tc, call->children[i], items, (uint32_t)width, &count, t))
                return NULL;
        }
        if (count != (uint32_t)width) {
            wgsl_tc_error(tc, call,
                "vector constructor expected %u component(s), got %u",
                (unsigned)width, (unsigned)count);
            return NULL;
        }
        WGSLTypeInfo *elem = common_constructor_type(tc, call, items, count);
        if (elem) check_vector_constructor_arg_ranges(tc, call, elem);
        return elem ? wgsl_type_vec(tc->types, (uint8_t)width, elem) : NULL;
    }

    uint8_t cols = 0, rows = 0;
    if (ctor_matrix_dims_from_name(nm, len, &cols, &rows)) {
        *recognized = 1;
        uint32_t total = (uint32_t)cols * (uint32_t)rows;
        if (argc == 0) {
            return wgsl_type_mat(tc->types, cols, rows,
                                 tc->types->t_abstract_int);
        }
        if (argc == 1) {
            WGSLTypeInfo *arg = wgsl_tc_store_type_of(tc, call->children[1]);
            if (!arg) return NULL;
            if (arg->kind == WGSL_TYPE_MAT &&
                arg->width == cols && arg->rows == rows)
            {
                return arg;
            }
            if (wgsl_type_is_scalar(arg)) {
                if (!wgsl_type_is_numeric(arg)) {
                    wgsl_tc_error(tc, call->children[1],
                        "matrix constructor requires numeric components");
                    return NULL;
                }
                return wgsl_type_mat(tc->types, cols, rows, arg);
            }
        }
        int has_vector_arg = 0;
        for (uint32_t i = 1; i < call->child_count; i++) {
            WGSLTypeInfo *t = wgsl_tc_store_type_of(tc, call->children[i]);
            if (t && t->kind == WGSL_TYPE_VEC) {
                has_vector_arg = 1;
                break;
            }
        }
        if (has_vector_arg) {
            if (argc != cols) {
                wgsl_tc_error(tc, call,
                    "matrix constructor expected %u column vector(s), got %u",
                    (unsigned)cols, (unsigned)argc);
                return NULL;
            }
            WGSLTypeInfo *items[4] = {0};
            for (uint32_t i = 0; i < cols; i++) {
                WGSLNode *argn = call->children[1 + i];
                WGSLTypeInfo *t = wgsl_tc_store_type_of(tc, argn);
                if (!t || t->kind != WGSL_TYPE_VEC || t->width != rows) {
                    wgsl_tc_error(tc, argn,
                        "matrix constructor column vector must have width %u",
                        (unsigned)rows);
                    return NULL;
                }
                items[i] = (WGSLTypeInfo *)t->ref;
            }
            WGSLTypeInfo *elem = common_constructor_type(tc, call, items, cols);
            if (!elem) return NULL;
            elem = constructor_matrix_elem(tc, call, elem);
            if (!elem) return NULL;
            return wgsl_type_mat(tc->types, cols, rows, elem);
        }
        WGSLTypeInfo *items[16] = {0};
        uint32_t count = 0;
        for (uint32_t i = 1; i < call->child_count; i++) {
            WGSLTypeInfo *t = wgsl_tc_store_type_of(tc, call->children[i]);
            if (!append_ctor_component_type(
                    tc, call->children[i], items, total, &count, t))
                return NULL;
        }
        if (count != total) {
            wgsl_tc_error(tc, call,
                "matrix constructor expected %u component(s), got %u",
                (unsigned)total, (unsigned)count);
            return NULL;
        }
        WGSLTypeInfo *elem = common_constructor_type(tc, call, items, count);
        if (!elem) return NULL;
        elem = constructor_matrix_elem(tc, call, elem);
        if (!elem) return NULL;
        return wgsl_type_mat(tc->types, cols, rows, elem);
    }

    if (len == 5 && memcmp(nm, "array", 5) == 0) {
        *recognized = 1;
        if (argc == 0) {
            wgsl_tc_error(tc, call, "array constructor requires at least one element");
            return NULL;
        }
        WGSLTypeInfo **items = (WGSLTypeInfo **)wgsl_arena_alloc(
            tc->arena, sizeof *items * argc);
        if (!items) return NULL;
        for (uint32_t i = 0; i < argc; i++) {
            items[i] = wgsl_tc_store_type_of(tc, call->children[1 + i]);
            if (!items[i]) return NULL;
        }
        WGSLTypeInfo *elem = common_constructor_type(tc, call, items, argc);
        if (elem) {
            for (uint32_t i = 0; i < argc; i++) {
                check_constructor_arg_range(
                    tc, call->children[1 + i], elem);
            }
        }
        return elem ? wgsl_type_array(tc->types, elem, argc) : NULL;
    }

    return NULL;
}

WGSLTypeInfo *resolve_callee_target(
    WGSLTypeChecker *tc, WGSLNode *callee, WGSLSymbol **out_sym)
{
    *out_sym = NULL;
    if (!callee) return NULL;
    if (callee->kind == WGSL_NODE_EXPR_IDENT) {
        WGSLSymbol *sym = wgsl_node_resolved_symbol(callee);
        if (!sym) return NULL;
        *out_sym = sym;
        switch ((WGSLSymKind)sym->kind) {
        case WGSL_SYM_PREDECLARED_TYPE:
        case WGSL_SYM_PREDECLARED_TYPE_ALIAS:
        case WGSL_SYM_TYPE_ALIAS:
        case WGSL_SYM_STRUCT:
            return sym->type;
        default:
            return NULL;
        }
    }
    if (callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT) {
        if (callee->child_count == 0) return NULL;
        WGSLNode *head = callee->children[0];
        if (!head) return NULL;
        WGSLSymbol *sym = wgsl_node_resolved_symbol(head);
        if (sym) *out_sym = sym;
        /* `bitcast<T>(x)` and other templated builtin / user-fn calls
         * are NOT type constructors — let the FN branch handle them. */
        if (sym && (sym->kind == WGSL_SYM_PREDECLARED_FN ||
                    sym->kind == WGSL_SYM_FUNCTION))
        {
            return NULL;
        }
        return wgsl_tc_resolve_type_spec(tc, callee);
    }
    return NULL;
}

int constructor_elem_compatible(
    WGSLTypeChecker *tc, WGSLTypeInfo *target, WGSLTypeInfo *got)
{
    if (!target || !got) return 1;
    if (target == got) return 1;
    if (target == tc->types->t_bool) return got == tc->types->t_bool;
    if (target == tc->types->t_i32) {
        return got == tc->types->t_i32 || got == tc->types->t_abstract_int;
    }
    if (target == tc->types->t_u32) {
        return got == tc->types->t_u32 || got == tc->types->t_abstract_int;
    }
    if (target == tc->types->t_f32) {
        return got == tc->types->t_f32 ||
               got == tc->types->t_abstract_float ||
               got == tc->types->t_abstract_int;
    }
    if (target == tc->types->t_f16) {
        return got == tc->types->t_f16 ||
               got == tc->types->t_abstract_float ||
               got == tc->types->t_abstract_int;
    }
    return 0;
}

int check_constructor_elem(
    WGSLTypeChecker *tc, WGSLNode *arg,
    WGSLTypeInfo *target_elem, WGSLTypeInfo *got_elem)
{
    if (constructor_elem_compatible(tc, target_elem, got_elem)) return 1;
    char got[96], want[96];
    wgsl_tc_error(tc, arg,
        "constructor component has type %s; expected %s",
        type_label(got_elem, got, sizeof got),
        type_label(target_elem, want, sizeof want));
    return 0;
}

int check_explicit_vector_constructor(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *target)
{
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;
    WGSLTypeInfo *elem = (WGSLTypeInfo *)target->ref;
    if (argc == 0) return 1;
    if (argc == 1) {
        WGSLNode *arg = call->children[1];
        WGSLTypeInfo *at = wgsl_tc_store_type_of(tc, arg);
        if (!at) return 1;
        if (at->kind == WGSL_TYPE_VEC &&
            at->width == target->width)
        {
            return 1;
        }
        if (wgsl_type_is_scalar(at)) {
            return check_constructor_elem(tc, arg, elem, at);
        }
    }

    uint32_t count = 0;
    int ok = 1;
    for (uint32_t i = 1; i < call->child_count; i++) {
        WGSLNode *arg = call->children[i];
        WGSLTypeInfo *at = wgsl_tc_store_type_of(tc, arg);
        if (!at) continue;
        if (at->kind == WGSL_TYPE_VEC) {
            count += at->width;
            ok = check_constructor_elem(
                    tc, arg, elem, (WGSLTypeInfo *)at->ref) && ok;
        } else if (wgsl_type_is_scalar(at)) {
            count += 1;
            ok = check_constructor_elem(tc, arg, elem, at) && ok;
        } else {
            wgsl_tc_error(tc, arg,
                "vector constructor requires scalar or vector components");
            ok = 0;
        }
    }
    if (count != target->width) {
        wgsl_tc_error(tc, call,
            "vector constructor expected %u component(s), got %u",
            (unsigned)target->width, (unsigned)count);
        ok = 0;
    }
    return ok;
}

int check_explicit_matrix_constructor(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *target)
{
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;
    WGSLTypeInfo *elem = (WGSLTypeInfo *)target->ref;
    uint32_t total = (uint32_t)target->width * (uint32_t)target->rows;
    if (argc == 0) return 1;
    if (!wgsl_type_is_float(elem)) {
        wgsl_tc_error(tc, call, "matrix constructor requires floating-point components");
        return 0;
    }
    if (argc == 1) {
        WGSLNode *arg = call->children[1];
        WGSLTypeInfo *at = wgsl_tc_store_type_of(tc, arg);
        if (!at) return 1;
        if (at->kind == WGSL_TYPE_MAT &&
            at->width == target->width && at->rows == target->rows)
        {
            WGSLTypeInfo *got_elem = (WGSLTypeInfo *)at->ref;
            if (wgsl_type_is_numeric(elem) && wgsl_type_is_numeric(got_elem)) {
                return 1;
            }
            return check_constructor_elem(
                tc, arg, elem, got_elem);
        }
        if (wgsl_type_is_scalar(at) && wgsl_type_is_numeric(at)) {
            return 1;
        }
    }

    int all_columns = argc == target->width;
    int ok = 1;
    if (all_columns) {
        for (uint32_t i = 1; i < call->child_count; i++) {
            WGSLNode *arg = call->children[i];
            WGSLTypeInfo *at = wgsl_tc_store_type_of(tc, arg);
            if (!at || at->kind != WGSL_TYPE_VEC ||
                at->width != target->rows)
            {
                all_columns = 0;
                break;
            }
        }
    }
    if (all_columns) {
        for (uint32_t i = 1; i < call->child_count; i++) {
            WGSLNode *arg = call->children[i];
            WGSLTypeInfo *at = wgsl_tc_store_type_of(tc, arg);
            ok = check_constructor_elem(
                    tc, arg, elem, (WGSLTypeInfo *)at->ref) && ok;
        }
        return ok;
    }

    uint32_t count = 0;
    for (uint32_t i = 1; i < call->child_count; i++) {
        WGSLNode *arg = call->children[i];
        WGSLTypeInfo *at = wgsl_tc_store_type_of(tc, arg);
        if (!at) continue;
        if (!wgsl_type_is_scalar(at)) {
            wgsl_tc_error(tc, arg,
                "matrix constructor requires scalar components or column vectors");
            ok = 0;
            continue;
        }
        count += 1;
        ok = check_constructor_elem(tc, arg, elem, at) && ok;
    }
    if (count != total) {
        wgsl_tc_error(tc, call,
            "matrix constructor expected %u component(s), got %u",
            (unsigned)total, (unsigned)count);
        ok = 0;
    }
    return ok;
}

int check_explicit_struct_constructor(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *target)
{
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;
    WGSLNode *sd = target ? (WGSLNode *)target->ref : NULL;
    if (!sd) return 1;

    uint32_t member_count = 0;
    for (uint32_t i = 0; i < sd->child_count; i++) {
        WGSLNode *m = sd->children[i];
        if (m && m->kind == WGSL_NODE_DECL_STRUCT_MEMBER) member_count++;
    }
    if (argc == 0) return 1;

    int ok = 1;
    if (argc != member_count) {
        wgsl_tc_error(tc, call,
            "struct constructor expected %u member value(s), got %u",
            (unsigned)member_count, (unsigned)argc);
        ok = 0;
    }

    uint32_t arg_index = 0;
    for (uint32_t i = 0; i < sd->child_count && arg_index < argc; i++) {
        WGSLNode *m = sd->children[i];
        if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
        uint32_t attrs = wgsl_member_attr_count(m);
        if (attrs >= m->child_count) {
            arg_index++;
            continue;
        }
        WGSLTypeInfo *mt = wgsl_tc_store_type_of(tc, m->children[attrs]);
        WGSLNode *arg = call->children[1 + arg_index++];
        WGSLTypeInfo *at = wgsl_tc_store_type_of(tc, arg);
        if (mt && at && wgsl_tc_init_type_mismatch(at, mt)) {
            char got[96], want[96];
            wgsl_tc_error(tc, arg,
                "struct constructor member has type %s; expected %s",
                type_label(at, got, sizeof got),
                type_label(mt, want, sizeof want));
            ok = 0;
        }
        check_constructor_arg_range(tc, arg, mt);
    }
    return ok;
}

int check_explicit_constructor_args(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *target)
{
    if (!target) return 1;
    if (target->kind == WGSL_TYPE_ARRAY &&
        target->array_len == 0 &&
        (target->flags != 0 || target->pad_ != 0))
    {
        wgsl_tc_error(tc, call,
            "override-sized array type cannot be constructed");
        return 0;
    }
    if (target->kind == WGSL_TYPE_VEC) {
        return check_explicit_vector_constructor(tc, call, target);
    }
    if (target->kind == WGSL_TYPE_MAT) {
        return check_explicit_matrix_constructor(tc, call, target);
    }
    if (target->kind == WGSL_TYPE_STRUCT) {
        return check_explicit_struct_constructor(tc, call, target);
    }
    return 1;
}

int type_call(
    WGSLTypeChecker *tc, WGSLNode *n, WGSLTypeInfo *expected)
{
    if (n->child_count < 1 || !n->children[0]) {
        wgsl_tc_error(tc, n, "malformed call expression");
        return 0;
    }

    WGSLNode *callee = n->children[0];
    WGSLSymbol *sym = NULL;
    WGSLTypeInfo *target = resolve_callee_target(tc, callee, &sym);

    /* §4.3 — when the constructor target is known from the callee
     * (e.g. `vec2f(...)`, `S(...)`) or from the contextual expected
     * type for a bare typegen (`vec2(...)` under `vec2u`), type
     * component arguments against the element/member expected type. */
    WGSLTypeInfo *ctor_target = target;
    if (!ctor_target && expected && sym &&
        sym->kind == WGSL_SYM_PREDECLARED_TYPEGEN &&
        callee->kind == WGSL_NODE_EXPR_IDENT)
    {
        /* Accept expected when its shape matches the typegen name. */
        int width = ctor_vector_width_from_name(sym->name, sym->name_len);
        uint8_t cols = 0, rows = 0;
        int is_mat = ctor_matrix_dims_from_name(
            sym->name, sym->name_len, &cols, &rows);
        int is_arr = sym->name_len == 5 && memcmp(sym->name, "array", 5) == 0;
        if (width && expected->kind == WGSL_TYPE_VEC &&
            expected->width == (uint8_t)width)
        {
            ctor_target = expected;
        } else if (is_mat && expected->kind == WGSL_TYPE_MAT &&
                   expected->width == cols && expected->rows == rows)
        {
            ctor_target = expected;
        } else if (is_arr && expected->kind == WGSL_TYPE_ARRAY)
        {
            ctor_target = expected;
        }
    }

    /* Type arguments — with per-arg expected when we know the target. */
    if (ctor_target && ctor_target->kind == WGSL_TYPE_VEC) {
        WGSLTypeInfo *elem = (WGSLTypeInfo *)ctor_target->ref;
        for (uint32_t i = 1; i < n->child_count; i++) {
            wgsl_tc_type_expr_exp(tc, n->children[i], elem);
        }
    } else if (ctor_target && ctor_target->kind == WGSL_TYPE_MAT) {
        WGSLTypeInfo *elem = (WGSLTypeInfo *)ctor_target->ref;
        WGSLTypeInfo *col = wgsl_type_vec(
            tc->types, ctor_target->rows, elem);
        for (uint32_t i = 1; i < n->child_count; i++) {
            /* Scalar form uses elem; column form uses col.  Pass elem
             * first; column vectors re-type via their own shape. */
            WGSLTypeInfo *arg_exp = elem;
            /* Prefer column expected when argc == cols (heuristic;
             * check_explicit_* validates). */
            uint32_t argc = n->child_count - 1;
            if (argc == ctor_target->width) arg_exp = col;
            wgsl_tc_type_expr_exp(tc, n->children[i], arg_exp);
        }
    } else if (ctor_target && ctor_target->kind == WGSL_TYPE_ARRAY) {
        WGSLTypeInfo *elem = (WGSLTypeInfo *)ctor_target->ref;
        for (uint32_t i = 1; i < n->child_count; i++) {
            wgsl_tc_type_expr_exp(tc, n->children[i], elem);
        }
    } else if (ctor_target && ctor_target->kind == WGSL_TYPE_STRUCT) {
        /* Member-typed args filled in check_explicit_struct_constructor
         * after free typing; still type bottom-up here with no expected. */
        for (uint32_t i = 1; i < n->child_count; i++) {
            wgsl_tc_type_expr(tc, n->children[i]);
        }
    } else if (sym && sym->kind == WGSL_SYM_FUNCTION) {
        /* Type args against parameter types when available. */
        WGSLNode *fn = sym->ast;
        uint32_t FA = 0, P = 0;
        if (fn && fn->kind == WGSL_NODE_DECL_FUNCTION) {
            FA = wgsl_fn_attr_count(fn);
            P = wgsl_fn_param_count(fn);
        }
        for (uint32_t i = 1; i < n->child_count; i++) {
            WGSLTypeInfo *pexp = NULL;
            uint32_t k = i - 1;
            if (fn && k < P && (FA + k) < fn->child_count) {
                WGSLNode *param = fn->children[FA + k];
                if (param && param->kind == WGSL_NODE_DECL_PARAM) {
                    uint32_t pattrs =
                        wgsl_param_attr_count(param);
                    if (pattrs < param->child_count) {
                        pexp = wgsl_tc_store_type_of(
                            tc, param->children[pattrs]);
                    }
                }
            }
            wgsl_tc_type_expr_exp(tc, n->children[i], pexp);
        }
    } else {
        /* Builtins / unresolved: bottom-up.  Result is contextually
         * materialized after overload resolution (§4.3). */
        for (uint32_t i = 1; i < n->child_count; i++) {
            wgsl_tc_type_expr(tc, n->children[i]);
        }
    }

    /* Constructor path — target type is known from the callee. */
    if (target) {
        check_explicit_constructor_args(tc, n, target);
        return wgsl_tc_set_type(tc, n, target, 0);
    }

    /* Inferred value constructors: `vec3(1)`, `mat2x2(1,2,3,4)`,
     * `array(1,2)`.  Templated forms are handled by the target path
     * above; this branch covers the predeclared type-generator head
     * without explicit template arguments. */
    if (sym && sym->kind == WGSL_SYM_PREDECLARED_TYPEGEN &&
        callee->kind == WGSL_NODE_EXPR_IDENT)
    {
        int recognized = 0;
        if (ctor_target && ctor_target != target) {
            /* Contextual typegen: treat as explicit constructor of
             * the expected shape. */
            check_explicit_constructor_args(tc, n, ctor_target);
            return wgsl_tc_set_type(tc, n, ctor_target, 0);
        }
        WGSLTypeInfo *inferred =
            infer_typegen_constructor_type(tc, n, sym, &recognized);
        if (inferred) {
            /* Abstract-element constructors stay abstract; the use site
             * converts (so range checks still see AbstractInt).  When
             * `ctor_target` matched `expected` we already returned above. */
            (void)expected;
            return wgsl_tc_set_type(tc, n, inferred, 0);
        }
        return recognized ? 0 : 1;
    }

    /* User-defined function call → return type from `sym->type` (the
     * type checker writes that on the fn symbol when it visits the
     * function decl). */
    if (sym && sym->kind == WGSL_SYM_FUNCTION) {
        /* §11.4 — entry points (@vertex / @fragment / @compute) cannot
         * be call targets.  Reject the call site directly. */
        uint16_t stage_flags = (uint16_t)(
            WGSL_SYM_FLAG_VERTEX | WGSL_SYM_FLAG_FRAGMENT |
            WGSL_SYM_FLAG_COMPUTE);
        if (sym->flags & stage_flags) {
            const char *stage =
                (sym->flags & WGSL_SYM_FLAG_VERTEX)   ? "vertex" :
                (sym->flags & WGSL_SYM_FLAG_FRAGMENT) ? "fragment" :
                                                        "compute";
            wgsl_tc_error(tc, callee,
                "cannot call entry point '%.*s' (@%s); entry points "
                "must never be call targets",
                (int)sym->name_len, sym->name, stage);
        }
        /* §8.10 — argument count and per-arg type checking.  Walk the
         * function decl's param children to recover param types; each
         * param's tspec was resolved during type_function, so the type
         * lives on the side-table. */
        WGSLNode *fn = sym->ast;
        if (fn && fn->kind == WGSL_NODE_DECL_FUNCTION) {
            uint32_t FA = wgsl_fn_attr_count(fn);
            uint32_t P  = wgsl_fn_param_count(fn);
            uint32_t call_argc = (n->child_count >= 1) ? n->child_count - 1 : 0;
            if (call_argc != P) {
                wgsl_tc_error(tc, n,
                    "argument count mismatch for '%.*s': expected %u, got %u",
                    (int)sym->name_len, sym->name,
                    (unsigned)P, (unsigned)call_argc);
            } else {
                /* Per-arg type compatibility — same predicate the
                 * decl-init mismatch check uses. */
                for (uint32_t k = 0; k < P && (FA + k) < fn->child_count; k++) {
                    WGSLNode *param = fn->children[FA + k];
                    if (!param || param->kind != WGSL_NODE_DECL_PARAM) continue;
                    uint32_t pattrs = wgsl_param_attr_count(param);
                    if (pattrs >= param->child_count) continue;
                    WGSLTypeInfo *pt = wgsl_tc_store_type_of(
                        tc, param->children[pattrs]);
                    WGSLNode *arg = n->children[1 + k];
                    WGSLTypeInfo *at = wgsl_tc_store_type_of(tc, arg);
                    if (!pt || !at) continue;
                    if (wgsl_tc_init_type_mismatch(at, pt)) {
                        char got[96], want[96];
                        wgsl_tc_error(tc, arg,
                            "argument %u of '%.*s' has type %s; "
                            "expected %s",
                            (unsigned)(k + 1),
                            (int)sym->name_len, sym->name,
                            type_label(at, got, sizeof got),
                            type_label(pt, want, sizeof want));
                    }
                }
            }
        }
        WGSLTypeInfo *ret = sym->type;
        if (!ret && fn && fn->kind == WGSL_NODE_DECL_FUNCTION) {
            uint32_t FA = wgsl_fn_attr_count(fn);
            uint32_t P  = wgsl_fn_param_count(fn);
            uint32_t RA = wgsl_fn_ret_attr_count(fn);
            uint32_t has_ret = wgsl_fn_has_return_type(fn);
            uint32_t ri = FA + P + RA;
            if (has_ret && ri < fn->child_count) {
                ret = wgsl_tc_resolve_type_spec(tc, fn->children[ri]);
            }
        }
        if (!ret) ret = tc->types->t_void;
        return wgsl_tc_set_type(tc, n, ret, 0);
    }

    /* Builtin function call — generated overload table.  Abstract
     * results stay abstract; use-site materialization (decl/assign/
     * return) converts and range-checks (§4.3 / §6.2.1). */
    if (sym && sym->kind == WGSL_SYM_PREDECLARED_FN) {
        (void)expected;
        return type_call_builtin(tc, n, callee, sym);
    }

    if (sym) {
        wgsl_tc_error(tc, callee,
            "'%.*s' is not callable",
            (int)sym->name_len, sym->name);
        return 0;
    }

    /* Unknown / unresolved callee — quietly punt without erroring; the
     * resolver will already have flagged it if it's truly broken. */
    return 1;
}

