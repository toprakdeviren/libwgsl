static int convert_scalar_constructor_value(
    WGSLConstEvaluator *cev, WGSLValue *v, WGSLTypeInfo *target,
    const WGSLNode *at)
{
    if (!v || !target) return 0;
    WGSLTypeKind tk = (WGSLTypeKind)target->kind;
    if (tk == WGSL_TYPE_BOOL) {
        bool b = false;
        if (v->kind == WGSL_VAL_BOOL) {
            b = v->u.b;
        } else if (v->kind == WGSL_VAL_INT) {
            b = v->u.i != 0;
        } else if (v->kind == WGSL_VAL_FLOAT) {
            b = v->u.f != 0.0;
        } else {
            cev_error(cev, at, "cannot convert value to bool");
            return 0;
        }
        v->kind = WGSL_VAL_BOOL;
        v->type = target;
        v->u.b = b;
        return 1;
    }

    if (tk == WGSL_TYPE_I32 || tk == WGSL_TYPE_U32 ||
        tk == WGSL_TYPE_ABSTRACT_INT)
    {
        int64_t i = 0;
        if (v->kind == WGSL_VAL_BOOL) {
            i = v->u.b ? 1 : 0;
        } else if (v->kind == WGSL_VAL_INT) {
            i = v->u.i;
        } else if (v->kind == WGSL_VAL_FLOAT) {
            if (!isfinite(v->u.f)) {
                cev_error(cev, at, "cannot convert non-finite value to integer");
                return 0;
            }
            double t = trunc(v->u.f);
            if (t < (double)INT64_MIN || t > (double)INT64_MAX) {
                cev_error(cev, at, "value out of range for integer constructor");
                return 0;
            }
            i = (int64_t)t;
        } else {
            cev_error(cev, at, "constructor requires scalar components");
            return 0;
        }
        if (!int_fits_target(i, tk)) {
            cev_error(cev, at, "value %lld out of range for %s",
                      (long long)i, wgsl_type_kind_name(tk));
            return 0;
        }
        v->kind = WGSL_VAL_INT;
        v->type = target;
        v->u.i = i;
        return 1;
    }

    if (tk == WGSL_TYPE_F32 || tk == WGSL_TYPE_F16 ||
        tk == WGSL_TYPE_ABSTRACT_FLOAT)
    {
        double d = 0.0;
        if (v->kind == WGSL_VAL_BOOL) {
            d = v->u.b ? 1.0 : 0.0;
        } else if (v->kind == WGSL_VAL_INT) {
            d = (double)v->u.i;
        } else if (v->kind == WGSL_VAL_FLOAT) {
            d = v->u.f;
        } else {
            cev_error(cev, at, "constructor requires scalar components");
            return 0;
        }
        if (!float_fits_target(d, tk)) {
            cev_error(cev, at, "value out of range for %s",
                      wgsl_type_kind_name(tk));
            return 0;
        }
        v->kind = WGSL_VAL_FLOAT;
        v->type = target;
        v->u.f = d;
        return 1;
    }

    return wgsl_consteval_materialize(cev, v, target, at);
}

static int eval_vector_constructor(
    WGSLConstEvaluator *cev, WGSLNode *call, WGSLTypeInfo *target,
    WGSLValue *out)
{
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;
    if (argc == 0) return make_zero_value(cev, target, call, out);

    uint32_t width = target->width;
    WGSLTypeInfo *elem = (WGSLTypeInfo *)target->ref;
    WGSLValue *elems = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *elems * width);
    if (!elems) {
        cev_error(cev, call, "out of memory while folding vector constructor");
        return 0;
    }

    if (argc == 1) {
        WGSLValue v = {0};
        if (!eval_expr(cev, call->children[1], &v)) return 0;
        if (v.kind != WGSL_VAL_VEC) {
            if (!convert_constructor_component(cev, &v, elem, call->children[1])) {
                return 0;
            }
            for (uint32_t i = 0; i < width; i++) elems[i] = v;
            out->kind = WGSL_VAL_VEC;
            out->type = target;
            out->u.agg.count = width;
            out->u.agg.elems = elems;
            return 1;
        }
    }

    uint32_t count = 0;
    for (uint32_t i = 1; i < call->child_count; i++) {
        WGSLValue v = {0};
        if (!eval_expr(cev, call->children[i], &v)) return 0;
        if (!append_constructor_value(
                cev, elems, width, &count, elem, &v, call->children[i]))
            return 0;
    }
    if (count != width) {
        cev_error(cev, call,
            "vector constructor expected %u component(s), got %u",
            (unsigned)width, (unsigned)count);
        return 0;
    }

    out->kind = WGSL_VAL_VEC;
    out->type = target;
    out->u.agg.count = width;
    out->u.agg.elems = elems;
    return 1;
}

static int eval_matrix_constructor(
    WGSLConstEvaluator *cev, WGSLNode *call, WGSLTypeInfo *target,
    WGSLValue *out)
{
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;
    if (argc == 0) return make_zero_value(cev, target, call, out);

    uint32_t total = (uint32_t)target->width * (uint32_t)target->rows;
    WGSLTypeInfo *elem = (WGSLTypeInfo *)target->ref;

    if (argc == 1) {
        WGSLValue v = {0};
        if (!eval_expr(cev, call->children[1], &v)) return 0;
        if (v.kind == WGSL_VAL_MAT) {
            if (!v.type || v.type->kind != WGSL_TYPE_MAT ||
                v.type->width != target->width || v.type->rows != target->rows)
            {
                cev_error(cev, call->children[1], "matrix constructor shape mismatch");
                return 0;
            }
            WGSLValue *elems = (WGSLValue *)wgsl_arena_alloc(
                cev->arena, sizeof *elems * total);
            if (!elems) {
                cev_error(cev, call, "out of memory while folding matrix constructor");
                return 0;
            }
            for (uint32_t i = 0; i < total; i++) {
                elems[i] = v.u.agg.elems[i];
                if (!convert_constructor_component(
                        cev, &elems[i], elem, call->children[1]))
                    return 0;
            }
            out->kind = WGSL_VAL_MAT;
            out->type = target;
            out->u.agg.count = total;
            out->u.agg.elems = elems;
            return 1;
        }
        if (v.kind == WGSL_VAL_INT || v.kind == WGSL_VAL_FLOAT) {
            if (!convert_constructor_component(cev, &v, elem, call->children[1])) {
                return 0;
            }
            WGSLValue *elems = (WGSLValue *)wgsl_arena_alloc(
                cev->arena, sizeof *elems * total);
            if (!elems) {
                cev_error(cev, call, "out of memory while folding matrix constructor");
                return 0;
            }
            for (uint32_t i = 0; i < total; i++) elems[i] = v;
            out->kind = WGSL_VAL_MAT;
            out->type = target;
            out->u.agg.count = total;
            out->u.agg.elems = elems;
            return 1;
        }
    }

    if (argc > 0) {
        WGSLValue first = {0};
        if (!eval_expr(cev, call->children[1], &first)) return 0;
        if (first.kind == WGSL_VAL_VEC) {
            if (argc != target->width) {
                cev_error(cev, call,
                    "matrix constructor expected %u column vector(s), got %u",
                    (unsigned)target->width, (unsigned)argc);
                return 0;
            }
            WGSLValue *elems = (WGSLValue *)wgsl_arena_alloc(
                cev->arena, sizeof *elems * total);
            if (!elems) {
                cev_error(cev, call, "out of memory while folding matrix constructor");
                return 0;
            }
            for (uint32_t c = 0; c < target->width; c++) {
                WGSLValue col = c == 0 ? first : (WGSLValue){0};
                if (c > 0 && !eval_expr(cev, call->children[1 + c], &col))
                    return 0;
                if (col.kind != WGSL_VAL_VEC ||
                    !col.type || col.type->kind != WGSL_TYPE_VEC ||
                    col.u.agg.count != target->rows)
                {
                    cev_error(cev, call->children[1 + c],
                        "matrix constructor column vector width mismatch");
                    return 0;
                }
                for (uint32_t r = 0; r < target->rows; r++) {
                    WGSLValue cell = col.u.agg.elems[r];
                    if (!convert_constructor_component(
                            cev, &cell, elem, call->children[1 + c]))
                        return 0;
                    elems[c * target->rows + r] = cell;
                }
            }
            out->kind = WGSL_VAL_MAT;
            out->type = target;
            out->u.agg.count = total;
            out->u.agg.elems = elems;
            return 1;
        }
    }

    WGSLValue *elems = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *elems * total);
    if (!elems) {
        cev_error(cev, call, "out of memory while folding matrix constructor");
        return 0;
    }

    uint32_t count = 0;
    for (uint32_t i = 1; i < call->child_count; i++) {
        WGSLValue v = {0};
        if (!eval_expr(cev, call->children[i], &v)) return 0;
        if (!append_constructor_value(
                cev, elems, total, &count, elem, &v, call->children[i]))
            return 0;
    }
    if (count != total) {
        cev_error(cev, call,
            "matrix constructor expected %u component(s), got %u",
            (unsigned)total, (unsigned)count);
        return 0;
    }

    out->kind = WGSL_VAL_MAT;
    out->type = target;
    out->u.agg.count = total;
    out->u.agg.elems = elems;
    return 1;
}

static int eval_array_constructor(
    WGSLConstEvaluator *cev, WGSLNode *call, WGSLTypeInfo *target,
    WGSLValue *out)
{
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;
    uint32_t length = target->array_len ? target->array_len : argc;
    if (argc == 0 && target->array_len > 0) {
        return make_zero_value(cev, target, call, out);
    }
    if (length == 0) {
        cev_error(cev, call, "array constructor requires at least one element");
        return 0;
    }
    if (argc != length) {
        cev_error(cev, call,
            "array constructor expected %u element(s), got %u",
            (unsigned)length, (unsigned)argc);
        return 0;
    }
    WGSLValue *elems = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *elems * length);
    if (!elems) {
        cev_error(cev, call, "out of memory while folding array constructor");
        return 0;
    }
    WGSLTypeInfo *elem_type = (WGSLTypeInfo *)target->ref;
    for (uint32_t i = 0; i < length; i++) {
        if (!eval_expr(cev, call->children[1 + i], &elems[i])) return 0;
        if (!wgsl_consteval_materialize(
                cev, &elems[i], elem_type, call->children[1 + i]))
            return 0;
    }
    out->kind = WGSL_VAL_ARRAY;
    out->type = target->array_len ? target
                                  : wgsl_type_array(cev->types, elem_type, length);
    out->u.agg.count = length;
    out->u.agg.elems = elems;
    return 1;
}

static int eval_struct_constructor(
    WGSLConstEvaluator *cev, WGSLNode *call, WGSLTypeInfo *target,
    WGSLValue *out)
{
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;
    WGSLNode *sd = (WGSLNode *)target->ref;
    uint32_t count = struct_member_count(sd);
    if (argc == 0) return make_zero_value(cev, target, call, out);
    if (argc != count) {
        cev_error(cev, call,
            "struct constructor expected %u member value(s), got %u",
            (unsigned)count, (unsigned)argc);
        return 0;
    }
    WGSLValue *elems = NULL;
    if (count > 0) {
        elems = (WGSLValue *)wgsl_arena_alloc(
            cev->arena, sizeof *elems * count);
        if (!elems) {
            cev_error(cev, call, "out of memory while folding struct constructor");
            return 0;
        }
    }
    uint32_t idx = 0;
    for (uint32_t i = 0; sd && i < sd->child_count; i++) {
        WGSLNode *m = sd->children[i];
        if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
        uint32_t attrs = (uint32_t)(m->payload[1] & 0xFFFFFFFFu);
        if (attrs >= m->child_count) return 0;
        WGSLTypeInfo *mt = type_from_typespec(cev, m->children[attrs]);
        if (!mt) return 0;
        if (!eval_expr(cev, call->children[1 + idx], &elems[idx])) return 0;
        if (!wgsl_consteval_materialize(
                cev, &elems[idx], mt, call->children[1 + idx]))
            return 0;
        idx++;
    }
    out->kind = WGSL_VAL_STRUCT;
    out->type = target;
    out->u.agg.count = count;
    out->u.agg.elems = elems;
    return 1;
}

static int eval_constructor_call(
    WGSLConstEvaluator *cev, WGSLNode *call, WGSLTypeInfo *target,
    WGSLValue *out)
{
    if (!target) return 0;
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;
    if (target->kind == WGSL_TYPE_BOOL ||
        target->kind == WGSL_TYPE_I32 ||
        target->kind == WGSL_TYPE_U32 ||
        target->kind == WGSL_TYPE_F32 ||
        target->kind == WGSL_TYPE_F16 ||
        target->kind == WGSL_TYPE_ABSTRACT_INT ||
        target->kind == WGSL_TYPE_ABSTRACT_FLOAT)
    {
        if (argc == 0) return make_zero_value(cev, target, call, out);
        if (argc != 1) {
            cev_error(cev, call, "scalar constructor expects 0 or 1 argument");
            return 0;
        }
        WGSLValue v = {0};
        if (!eval_expr(cev, call->children[1], &v)) return 0;
        if (!convert_scalar_constructor_value(
                cev, &v, target, call->children[1]))
        {
            return 0;
        }
        *out = v;
        return 1;
    }

    switch ((WGSLTypeKind)target->kind) {
    case WGSL_TYPE_VEC:    return eval_vector_constructor(cev, call, target, out);
    case WGSL_TYPE_MAT:    return eval_matrix_constructor(cev, call, target, out);
    case WGSL_TYPE_ARRAY:  return eval_array_constructor (cev, call, target, out);
    case WGSL_TYPE_STRUCT: return eval_struct_constructor(cev, call, target, out);
    default:
        cev_error(cev, call, "constructor const-eval not implemented yet (Iter C)");
        return 0;
    }
}

static int append_inferred_component(
    WGSLConstEvaluator *cev, WGSLValue *dst, uint32_t cap,
    uint32_t *count, const WGSLValue *src, const WGSLNode *at)
{
    if (src->kind == WGSL_VAL_VEC) {
        for (uint32_t i = 0; i < src->u.agg.count; i++) {
            if (*count >= cap) {
                cev_error(cev, at, "too many constructor components");
                return 0;
            }
            dst[(*count)++] = src->u.agg.elems[i];
        }
        return 1;
    }
    if (src->kind != WGSL_VAL_BOOL &&
        src->kind != WGSL_VAL_INT &&
        src->kind != WGSL_VAL_FLOAT)
    {
        cev_error(cev, at, "constructor requires scalar or vector components");
        return 0;
    }
    if (*count >= cap) {
        cev_error(cev, at, "too many constructor components");
        return 0;
    }
    dst[(*count)++] = *src;
    return 1;
}

static int eval_inferred_vector_constructor(
    WGSLConstEvaluator *cev, WGSLNode *call, uint8_t width, WGSLValue *out)
{
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;
    if (argc == 0) {
        WGSLTypeInfo *target = wgsl_type_vec(cev->types, width, cev->types->t_abstract_int);
        return make_zero_value(cev, target, call, out);
    }

    WGSLValue comps[4] = {{0}};
    uint32_t count = 0;
    if (argc == 1) {
        WGSLValue v = {0};
        if (!eval_expr(cev, call->children[1], &v)) return 0;
        if (v.kind != WGSL_VAL_VEC) {
            WGSLTypeInfo *elem = v.type;
            if (!elem || !materialize_constructor_values(
                    cev, &v, 1, elem, call->children[1]))
                return 0;
            WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(
                cev->arena, sizeof *stored * width);
            if (!stored) {
                cev_error(cev, call, "out of memory while folding vector constructor");
                return 0;
            }
            for (uint32_t i = 0; i < width; i++) stored[i] = v;
            out->kind = WGSL_VAL_VEC;
            out->type = wgsl_type_vec(cev->types, width, elem);
            out->u.agg.count = width;
            out->u.agg.elems = stored;
            return 1;
        }
        if (!append_inferred_component(
                cev, comps, width, &count, &v, call->children[1]))
            return 0;
    } else {
        for (uint32_t i = 1; i < call->child_count; i++) {
            WGSLValue v = {0};
            if (!eval_expr(cev, call->children[i], &v)) return 0;
            if (!append_inferred_component(
                    cev, comps, width, &count, &v, call->children[i]))
                return 0;
        }
    }
    if (count != width) {
        cev_error(cev, call,
            "vector constructor expected %u component(s), got %u",
            (unsigned)width, (unsigned)count);
        return 0;
    }
    WGSLTypeInfo *elem = NULL;
    if (!common_constructor_type(cev, comps, count, call, &elem)) return 0;
    if (!materialize_constructor_values(cev, comps, count, elem, call)) return 0;

    WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *stored * width);
    if (!stored) {
        cev_error(cev, call, "out of memory while folding vector constructor");
        return 0;
    }
    for (uint32_t i = 0; i < width; i++) stored[i] = comps[i];
    out->kind = WGSL_VAL_VEC;
    out->type = wgsl_type_vec(cev->types, width, elem);
    out->u.agg.count = width;
    out->u.agg.elems = stored;
    return 1;
}

static int eval_inferred_matrix_constructor(
    WGSLConstEvaluator *cev, WGSLNode *call,
    uint8_t cols, uint8_t rows, WGSLValue *out)
{
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;
    uint32_t total = (uint32_t)cols * (uint32_t)rows;
    if (argc == 0) {
        WGSLTypeInfo *target = wgsl_type_mat(
            cev->types, cols, rows, cev->types->t_abstract_int);
        return make_zero_value(cev, target, call, out);
    }
    if (argc == 1) {
        WGSLValue v = {0};
        if (!eval_expr(cev, call->children[1], &v)) return 0;
        if (v.kind == WGSL_VAL_MAT && v.type &&
            v.type->kind == WGSL_TYPE_MAT &&
            v.type->width == cols && v.type->rows == rows)
        {
            *out = v;
            return 1;
        }
        if (v.kind == WGSL_VAL_INT || v.kind == WGSL_VAL_FLOAT) {
            WGSLTypeInfo *elem = v.type;
            if (!elem || !materialize_constructor_values(
                    cev, &v, 1, elem, call->children[1]))
                return 0;
            WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(
                cev->arena, sizeof *stored * total);
            if (!stored) {
                cev_error(cev, call, "out of memory while folding matrix constructor");
                return 0;
            }
            for (uint32_t i = 0; i < total; i++) stored[i] = v;
            out->kind = WGSL_VAL_MAT;
            out->type = wgsl_type_mat(cev->types, cols, rows, elem);
            out->u.agg.count = total;
            out->u.agg.elems = stored;
            return 1;
        }
    }

    WGSLValue comps[16] = {{0}};
    uint32_t count = 0;
    for (uint32_t i = 1; i < call->child_count; i++) {
        WGSLValue v = {0};
        if (!eval_expr(cev, call->children[i], &v)) return 0;
        if (!append_inferred_component(
                cev, comps, total, &count, &v, call->children[i]))
            return 0;
    }
    if (count != total) {
        cev_error(cev, call,
            "matrix constructor expected %u component(s), got %u",
            (unsigned)total, (unsigned)count);
        return 0;
    }
    WGSLTypeInfo *elem = NULL;
    if (!common_constructor_type(cev, comps, count, call, &elem)) return 0;
    if (elem == cev->types->t_bool) {
        cev_error(cev, call, "matrix constructor requires numeric components");
        return 0;
    }
    if (!materialize_constructor_values(cev, comps, count, elem, call)) return 0;

    WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *stored * total);
    if (!stored) {
        cev_error(cev, call, "out of memory while folding matrix constructor");
        return 0;
    }
    for (uint32_t i = 0; i < total; i++) stored[i] = comps[i];
    out->kind = WGSL_VAL_MAT;
    out->type = wgsl_type_mat(cev->types, cols, rows, elem);
    out->u.agg.count = total;
    out->u.agg.elems = stored;
    return 1;
}

static int eval_inferred_array_constructor(
    WGSLConstEvaluator *cev, WGSLNode *call, WGSLValue *out)
{
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;
    if (argc == 0) {
        cev_error(cev, call, "array constructor requires at least one element");
        return 0;
    }
    WGSLValue *elems = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *elems * argc);
    if (!elems) {
        cev_error(cev, call, "out of memory while folding array constructor");
        return 0;
    }
    for (uint32_t i = 0; i < argc; i++) {
        if (!eval_expr(cev, call->children[1 + i], &elems[i])) return 0;
    }
    WGSLTypeInfo *elem = NULL;
    if (!common_constructor_type(cev, elems, argc, call, &elem)) return 0;
    if (!materialize_constructor_values(cev, elems, argc, elem, call)) return 0;

    out->kind = WGSL_VAL_ARRAY;
    out->type = wgsl_type_array(cev->types, elem, argc);
    out->u.agg.count = argc;
    out->u.agg.elems = elems;
    return 1;
}

static int eval_inferred_typegen_constructor(
    WGSLConstEvaluator *cev, WGSLNode *call, WGSLValue *out)
{
    WGSLNode *callee = call->child_count > 0 ? call->children[0] : NULL;
    if (!callee || callee->kind != WGSL_NODE_EXPR_IDENT) return 0;
    const char *nm = NULL;
    uint32_t len = 0;
    if (!node_ident_text(cev, callee, &nm, &len)) return 0;
    int width = vector_width_from_name(nm, len);
    if (width) return eval_inferred_vector_constructor(
        cev, call, (uint8_t)width, out);
    uint8_t cols = 0, rows = 0;
    if (matrix_dims_from_name(nm, len, &cols, &rows)) {
        return eval_inferred_matrix_constructor(cev, call, cols, rows, out);
    }
    if (len == 5 && memcmp(nm, "array", 5) == 0) {
        return eval_inferred_array_constructor(cev, call, out);
    }
    return 0;
}

static double f16_bits_to_double(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15);
