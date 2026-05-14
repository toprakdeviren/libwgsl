static int fold_minmax_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *a, WGSLValue *b, WGSLValue *out)
{
    if (!promote_pair(cev, a, b, call)) return 0;
    if (a->kind == WGSL_VAL_INT && b->kind == WGSL_VAL_INT) {
        *out = *a;
        if (a->type == cev->types->t_u32) {
            uint32_t au = (uint32_t)a->u.i, bu = (uint32_t)b->u.i;
            uint32_t r = name_eq(nm, len, "min")
                ? (au < bu ? au : bu)
                : (au > bu ? au : bu);
            out->u.i = (int64_t)r;
        } else {
            out->u.i = name_eq(nm, len, "min")
                ? (a->u.i < b->u.i ? a->u.i : b->u.i)
                : (a->u.i > b->u.i ? a->u.i : b->u.i);
        }
        return 1;
    }
    if (!require_float_scalar(cev, call, nm, len, a) ||
        !require_float_scalar(cev, call, nm, len, b))
    {
        return 0;
    }
    double r = name_eq(nm, len, "min")
        ? (a->u.f < b->u.f ? a->u.f : b->u.f)
        : (a->u.f > b->u.f ? a->u.f : b->u.f);
    return finish_float_builtin_result(cev, call, nm, len, a->type, r, out);
}

static int fold_float_binary_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *a, WGSLValue *b, WGSLValue *out)
{
    if (!promote_pair(cev, a, b, call) ||
        !require_float_scalar(cev, call, nm, len, a) ||
        !require_float_scalar(cev, call, nm, len, b))
    {
        return 0;
    }
    double x = a->u.f, y = b->u.f, r = 0.0;
    const char *domain = NULL;
    if (name_eq(nm, len, "pow")) {
        if (x < 0.0) domain = "base must be >= 0";
        else if (x == 0.0 && y <= 0.0) domain = "zero base requires a positive exponent";
        r = pow(x, y);
    } else if (name_eq(nm, len, "atan2")) {
        r = atan2(x, y);
    } else if (name_eq(nm, len, "step")) {
        r = y < x ? 0.0 : 1.0;
    }
    if (domain) {
        cev_error(cev, call,
            "builtin '%.*s' domain error: %s", (int)len, nm, domain);
        return 0;
    }
    return finish_float_builtin_result(cev, call, nm, len, a->type, r, out);
}

static int eval_call_binary_numeric_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int supported =
        name_eq(nm, len, "min") || name_eq(nm, len, "max") ||
        name_eq(nm, len, "pow") || name_eq(nm, len, "atan2") ||
        name_eq(nm, len, "step");
    if (!supported) return 0;
    if (call->child_count != 3) {
        cev_error(cev, call,
            "builtin '%.*s' expects 2 arguments; got %u",
            (int)len, nm, (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[2] = {{0}};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]))
    {
        return -1;
    }
    uint32_t width = 1;
    WGSLTypeInfo *vec_type = NULL;
    if (!componentwise_shape(cev, call, nm, len, vals, 2, -1,
                             &width, &vec_type))
    {
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue a = value_component(&vals[0], i);
        WGSLValue b = value_component(&vals[1], i);
        int ok = (name_eq(nm, len, "min") || name_eq(nm, len, "max"))
            ? fold_minmax_scalar(cev, call, nm, len, &a, &b, &tmp[i])
            : fold_float_binary_scalar(cev, call, nm, len, &a, &b, &tmp[i]);
        if (!ok) return -1;
    }
    return make_componentwise_result(cev, call, width, vec_type, tmp, out) ? 1 : -1;
}

static int fold_clamp_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len,
    WGSLValue *x, WGSLValue *lo, WGSLValue *hi, WGSLValue *out)
{
    if (!promote_three(cev, x, lo, hi, call)) return 0;
    if (x->kind == WGSL_VAL_INT && lo->kind == WGSL_VAL_INT &&
        hi->kind == WGSL_VAL_INT)
    {
        *out = *x;
        if (x->type == cev->types->t_u32) {
            uint32_t xv = (uint32_t)x->u.i;
            uint32_t lv = (uint32_t)lo->u.i;
            uint32_t hv = (uint32_t)hi->u.i;
            if (lv > hv) {
                cev_error(cev, call,
                    "builtin 'clamp' domain error: low must be <= high");
                return 0;
            }
            if (xv < lv) xv = lv;
            if (xv > hv) xv = hv;
            out->u.i = (int64_t)xv;
        } else {
            if (lo->u.i > hi->u.i) {
                cev_error(cev, call,
                    "builtin 'clamp' domain error: low must be <= high");
                return 0;
            }
            int64_t r = x->u.i < lo->u.i ? lo->u.i : x->u.i;
            if (r > hi->u.i) r = hi->u.i;
            out->u.i = r;
        }
        (void)nm; (void)len;
        return 1;
    }
    if (!require_float_scalar(cev, call, nm, len, x) ||
        !require_float_scalar(cev, call, nm, len, lo) ||
        !require_float_scalar(cev, call, nm, len, hi))
    {
        return 0;
    }
    if (lo->u.f > hi->u.f) {
        cev_error(cev, call,
            "builtin 'clamp' domain error: low must be <= high");
        return 0;
    }
    double r = x->u.f < lo->u.f ? lo->u.f : x->u.f;
    if (r > hi->u.f) r = hi->u.f;
    return finish_float_builtin_result(cev, call, nm, len, x->type, r, out);
}

static int fold_float_ternary_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len,
    WGSLValue *a, WGSLValue *b, WGSLValue *c, WGSLValue *out)
{
    if (!promote_three(cev, a, b, c, call) ||
        !require_float_scalar(cev, call, nm, len, a) ||
        !require_float_scalar(cev, call, nm, len, b) ||
        !require_float_scalar(cev, call, nm, len, c))
    {
        return 0;
    }
    double r = 0.0;
    const char *domain = NULL;
    if (name_eq(nm, len, "mix")) {
        double one_minus_c = 0.0;
        double lhs = 0.0;
        double rhs = 0.0;
        if (!round_float_value_to_type(cev, a->type, 1.0 - c->u.f,
                                       &one_minus_c) ||
            !round_float_value_to_type(cev, a->type, a->u.f * one_minus_c,
                                       &lhs) ||
            !round_float_value_to_type(cev, a->type, b->u.f * c->u.f,
                                       &rhs) ||
            !round_float_value_to_type(cev, a->type, lhs + rhs, &r))
        {
            cev_error(cev, call,
                "builtin '%.*s' produced a non-finite result "
                "(spec §15.7.3 finite-math assumption)",
                (int)len, nm);
            return 0;
        }
    } else if (name_eq(nm, len, "fma")) {
        r = fma(a->u.f, b->u.f, c->u.f);
    } else if (name_eq(nm, len, "smoothstep")) {
        if (a->u.f == b->u.f) {
            domain = "edge0 and edge1 must differ";
        } else {
            double t = (c->u.f - a->u.f) / (b->u.f - a->u.f);
            if (t < 0.0) t = 0.0;
            if (t > 1.0) t = 1.0;
            r = t * t * (3.0 - 2.0 * t);
        }
    }
    if (domain) {
        cev_error(cev, call,
            "builtin '%.*s' domain error: %s", (int)len, nm, domain);
        return 0;
    }
    return finish_float_builtin_result(cev, call, nm, len, a->type, r, out);
}

static int eval_call_ternary_numeric_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int supported =
        name_eq(nm, len, "clamp") || name_eq(nm, len, "mix") ||
        name_eq(nm, len, "fma") || name_eq(nm, len, "smoothstep");
    if (!supported) return 0;
    if (call->child_count != 4) {
        cev_error(cev, call,
            "builtin '%.*s' expects 3 arguments; got %u",
            (int)len, nm, (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[3] = {{0}};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]) ||
        !eval_expr(cev, call->children[3], &vals[2]))
    {
        return -1;
    }
    uint32_t width = 1;
    WGSLTypeInfo *vec_type = NULL;
    int broadcast = name_eq(nm, len, "mix") ? 2 : -1;
    if (!componentwise_shape(cev, call, nm, len, vals, 3, broadcast,
                             &width, &vec_type))
    {
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue a = value_component(&vals[0], i);
        WGSLValue b = value_component(&vals[1], i);
        WGSLValue c = value_component(&vals[2], i);
        int ok = name_eq(nm, len, "clamp")
            ? fold_clamp_scalar(cev, call, nm, len, &a, &b, &c, &tmp[i])
            : fold_float_ternary_scalar(cev, call, nm, len, &a, &b, &c, &tmp[i]);
        if (!ok) return -1;
    }
    return make_componentwise_result(cev, call, width, vec_type, tmp, out) ? 1 : -1;
}

static int eval_call_ldexp_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "ldexp")) return 0;
    if (call->child_count != 3) {
        cev_error(cev, call,
            "builtin 'ldexp' expects 2 arguments; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[2] = {{0}};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]))
    {
        return -1;
    }
    uint32_t width = 1;
    WGSLTypeInfo *vec_type = NULL;
    if (!componentwise_shape(cev, call, nm, len, vals, 2, -1,
                             &width, &vec_type))
    {
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue x = value_component(&vals[0], i);
        WGSLValue e = value_component(&vals[1], i);
        if (!require_float_scalar(cev, call, nm, len, &x)) return -1;
        if (e.kind != WGSL_VAL_INT) {
            cev_error(cev, call, "'ldexp' requires an integer exponent");
            return -1;
        }
        if (e.type == cev->types->t_u32) {
            cev_error(cev, call, "'ldexp' requires a signed integer exponent");
            return -1;
        }
        if (x.type != cev->types->t_abstract_float &&
            e.type == cev->types->t_abstract_int &&
            !wgsl_consteval_materialize(cev, &e, cev->types->t_i32, call))
        {
            return -1;
        }
        int bias = x.type == cev->types->t_f16 ? 15 :
                   x.type == cev->types->t_f32 ? 127 : 1023;
        if (e.u.i > bias + 1) {
            cev_error(cev, call,
                "builtin 'ldexp' domain error: exponent too large");
            return -1;
        }
        double r = e.u.i < -1075 ? 0.0 : ldexp(x.u.f, (int)e.u.i);
        if (!finish_float_builtin_result(cev, call, nm, len, x.type, r, &tmp[i])) {
            return -1;
        }
    }
    return make_componentwise_result(cev, call, width, vec_type, tmp, out) ? 1 : -1;
}

static int eval_call_length_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "length")) return 0;
    if (call->child_count != 2) {
        cev_error(cev, call,
            "builtin 'length' expects 1 argument; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;
    WGSLValue first = value_component(&arg, 0);
    if (!require_float_scalar(cev, call, nm, len, &first)) return -1;
    uint32_t width = value_width(&arg);
    double sum = 0.0;
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue c = value_component(&arg, i);
        if (!promote_pair(cev, &first, &c, call) ||
            !require_float_scalar(cev, call, nm, len, &c))
        {
            return -1;
        }
        sum += c.u.f * c.u.f;
    }
    double r = width == 1 ? fabs(first.u.f) : sqrt(sum);
    return finish_float_builtin_result(cev, call, nm, len, first.type, r, out) ? 1 : -1;
}

static int eval_call_distance_dot_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int is_distance = name_eq(nm, len, "distance");
    int is_dot = name_eq(nm, len, "dot");
    if (!is_distance && !is_dot) return 0;
    if (call->child_count != 3) {
        cev_error(cev, call,
            "builtin '%.*s' expects 2 arguments; got %u",
            (int)len, nm, (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[2] = {{0}};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]))
    {
        return -1;
    }
    if (is_distance && vals[0].kind != WGSL_VAL_VEC &&
        vals[1].kind != WGSL_VAL_VEC)
    {
        WGSLValue a = vals[0];
        WGSLValue b = vals[1];
        if (!promote_pair(cev, &a, &b, call) ||
            !require_float_scalar(cev, call, nm, len, &a) ||
            !require_float_scalar(cev, call, nm, len, &b))
        {
            return -1;
        }
        return finish_float_builtin_result(
            cev, call, nm, len, a.type, fabs(a.u.f - b.u.f), out) ? 1 : -1;
    }
    if (vals[0].kind != WGSL_VAL_VEC || vals[1].kind != WGSL_VAL_VEC ||
        vals[0].u.agg.count != vals[1].u.agg.count)
    {
        cev_error(cev, call,
            "builtin '%.*s' requires same-width vector arguments",
            (int)len, nm);
        return -1;
    }
    WGSLValue a0 = vals[0].u.agg.elems[0];
    WGSLValue b0 = vals[1].u.agg.elems[0];
    if (!promote_pair(cev, &a0, &b0, call)) return -1;
    if (is_distance) {
        if (!require_float_scalar(cev, call, nm, len, &a0) ||
            !require_float_scalar(cev, call, nm, len, &b0))
        {
            return -1;
        }
        double sum = 0.0;
        for (uint32_t i = 0; i < vals[0].u.agg.count; i++) {
            WGSLValue a = vals[0].u.agg.elems[i];
            WGSLValue b = vals[1].u.agg.elems[i];
            if (!promote_pair(cev, &a, &b, call) ||
                !promote_pair(cev, &a0, &a, call) ||
                !promote_pair(cev, &a0, &b, call) ||
                !require_float_scalar(cev, call, nm, len, &a) ||
                !require_float_scalar(cev, call, nm, len, &b))
            {
                return -1;
            }
            double d = a.u.f - b.u.f;
            sum += d * d;
        }
        return finish_float_builtin_result(
            cev, call, nm, len, a0.type, sqrt(sum), out) ? 1 : -1;
    }

    if (a0.kind == WGSL_VAL_INT && b0.kind == WGSL_VAL_INT) {
        int64_t sum = 0;
        int is_u = a0.type == cev->types->t_u32;
        int is_abstract = a0.type == cev->types->t_abstract_int;
        for (uint32_t i = 0; i < vals[0].u.agg.count; i++) {
            WGSLValue a = vals[0].u.agg.elems[i];
            WGSLValue b = vals[1].u.agg.elems[i];
            if (!promote_pair(cev, &a, &b, call) ||
                !promote_pair(cev, &a0, &a, call))
            {
                return -1;
            }
            if (is_u) {
                uint64_t prod = (uint64_t)(uint32_t)a.u.i * (uint64_t)(uint32_t)b.u.i;
                sum = (int64_t)(uint32_t)((uint32_t)sum + (uint32_t)prod);
            } else if (is_abstract) {
                int64_t prod = 0;
                if (!checked_abstract_int_result(
                        cev, call, (__int128)a.u.i * (__int128)b.u.i,
                        "dot", &prod) ||
                    !checked_abstract_int_result(
                        cev, call, (__int128)sum + (__int128)prod,
                        "dot", &sum))
                {
                    return -1;
                }
            } else {
                sum += a.u.i * b.u.i;
                if (a0.type == cev->types->t_i32) sum = (int32_t)sum;
            }
        }
        out->kind = WGSL_VAL_INT;
        out->type = a0.type;
        out->u.i = sum;
        return 1;
    }

    if (!require_float_scalar(cev, call, nm, len, &a0) ||
        !require_float_scalar(cev, call, nm, len, &b0))
    {
        return -1;
    }
    double sum = 0.0;
    for (uint32_t i = 0; i < vals[0].u.agg.count; i++) {
        WGSLValue a = vals[0].u.agg.elems[i];
        WGSLValue b = vals[1].u.agg.elems[i];
        if (!promote_pair(cev, &a, &b, call) ||
            !promote_pair(cev, &a0, &a, call) ||
            !promote_pair(cev, &a0, &b, call) ||
            !require_float_scalar(cev, call, nm, len, &a) ||
            !require_float_scalar(cev, call, nm, len, &b))
        {
            return -1;
        }
        sum += a.u.f * b.u.f;
    }
    return finish_float_builtin_result(cev, call, nm, len, a0.type, sum, out) ? 1 : -1;
}

static int eval_call_cross_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "cross")) return 0;
    if (call->child_count != 3) {
        cev_error(cev, call,
            "builtin 'cross' expects 2 arguments; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[2] = {{0}};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]))
    {
        return -1;
    }
    if (vals[0].kind != WGSL_VAL_VEC || vals[1].kind != WGSL_VAL_VEC ||
        vals[0].u.agg.count != 3 || vals[1].u.agg.count != 3)
    {
        cev_error(cev, call, "builtin 'cross' requires vec3 arguments");
        return -1;
    }
    WGSLValue a0 = vals[0].u.agg.elems[0];
    WGSLValue b0 = vals[1].u.agg.elems[0];
    if (!promote_pair(cev, &a0, &b0, call) ||
        !require_float_scalar(cev, call, nm, len, &a0) ||
        !require_float_scalar(cev, call, nm, len, &b0))
    {
        return -1;
    }
    double av[3], bv[3];
    for (uint32_t i = 0; i < 3; i++) {
        WGSLValue a = vals[0].u.agg.elems[i];
        WGSLValue b = vals[1].u.agg.elems[i];
        if (!promote_pair(cev, &a, &b, call) ||
            !promote_pair(cev, &a0, &a, call) ||
            !promote_pair(cev, &a0, &b, call) ||
            !require_float_scalar(cev, call, nm, len, &a) ||
            !require_float_scalar(cev, call, nm, len, &b))
        {
            return -1;
        }
        av[i] = a.u.f;
        bv[i] = b.u.f;
    }
    WGSLValue tmp[3] = {{0}};
    static const uint8_t idx[3][4] = {
        { 1, 2, 2, 1 },
        { 2, 0, 0, 2 },
        { 0, 1, 1, 0 },
    };
    for (uint32_t i = 0; i < 3; i++) {
        WGSLValue lhs_a = {
            .kind = WGSL_VAL_FLOAT, .type = a0.type, .u.f = av[idx[i][0]],
        };
        WGSLValue lhs_b = {
            .kind = WGSL_VAL_FLOAT, .type = a0.type, .u.f = bv[idx[i][1]],
        };
        WGSLValue rhs_a = {
            .kind = WGSL_VAL_FLOAT, .type = a0.type, .u.f = av[idx[i][2]],
        };
        WGSLValue rhs_b = {
            .kind = WGSL_VAL_FLOAT, .type = a0.type, .u.f = bv[idx[i][3]],
        };
        WGSLValue lhs = {0}, rhs = {0};
        if (!eval_binary_float(cev, call, WGSL_TOK_STAR, &lhs_a, &lhs_b, &lhs) ||
            !eval_binary_float(cev, call, WGSL_TOK_STAR, &rhs_a, &rhs_b, &rhs) ||
            !eval_binary_float(cev, call, WGSL_TOK_MINUS, &lhs, &rhs, &tmp[i]))
        {
            return -1;
        }
    }
    return make_vector_result(cev, call, 3, a0.type, tmp, out) ? 1 : -1;
}

static int eval_call_normalize_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "normalize")) return 0;
    if (call->child_count != 2) {
        cev_error(cev, call,
            "builtin 'normalize' expects 1 argument; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;
    if (arg.kind != WGSL_VAL_VEC || arg.u.agg.count == 0) {
        cev_error(cev, call, "builtin 'normalize' requires a vector argument");
        return -1;
    }
    WGSLValue first = arg.u.agg.elems[0];
    if (!require_float_scalar(cev, call, nm, len, &first)) return -1;
    WGSLValue sumv = {
        .kind = WGSL_VAL_FLOAT,
        .type = first.type,
        .u.f = 0.0,
    };
    for (uint32_t i = 0; i < arg.u.agg.count; i++) {
        WGSLValue c = arg.u.agg.elems[i];
        if (!promote_pair(cev, &first, &c, call) ||
            !require_float_scalar(cev, call, nm, len, &c))
        {
            return -1;
        }
        WGSLValue cv = {
            .kind = WGSL_VAL_FLOAT,
            .type = first.type,
            .u.f = c.u.f,
        };
        WGSLValue prod = {0};
        WGSLValue next = {0};
        if (!eval_binary_float(cev, call, WGSL_TOK_STAR, &cv, &cv, &prod) ||
            !eval_binary_float(cev, call, WGSL_TOK_PLUS, &sumv, &prod, &next))
        {
            return -1;
        }
        sumv = next;
    }
    double lenv = 0.0;
    if (!round_float_value_to_type(cev, first.type, sqrt(sumv.u.f), &lenv)) {
        cev_error(cev, call,
            "builtin 'normalize' produced a non-finite result "
            "(spec §15.7.3 finite-math assumption)");
        return -1;
    }
    if (lenv == 0.0) {
        cev_error(cev, call,
            "builtin 'normalize' domain error: argument must not be the zero vector");
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    WGSLValue denom = {
        .kind = WGSL_VAL_FLOAT,
        .type = first.type,
        .u.f = lenv,
    };
    for (uint32_t i = 0; i < arg.u.agg.count; i++) {
        WGSLValue c = arg.u.agg.elems[i];
        if (!promote_pair(cev, &first, &c, call) ||
            !require_float_scalar(cev, call, nm, len, &c))
        {
            return -1;
        }
        WGSLValue cv = {
            .kind = WGSL_VAL_FLOAT,
            .type = first.type,
            .u.f = c.u.f,
        };
        if (!eval_binary_float(cev, call, WGSL_TOK_SLASH, &cv, &denom, &tmp[i])) {
            return -1;
        }
    }
    return make_componentwise_result(
        cev, call, arg.u.agg.count, arg.type, tmp, out) ? 1 : -1;
}

static int fold_float_dot(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len,
    const WGSLValue *a, const WGSLValue *b,
    uint32_t width, WGSLTypeInfo **out_type, double *out_dot)
{
    WGSLValue at = a->u.agg.elems[0];
    WGSLValue bt = b->u.agg.elems[0];
    if (!promote_pair(cev, &at, &bt, call) ||
        !require_float_scalar(cev, call, nm, len, &at) ||
        !require_float_scalar(cev, call, nm, len, &bt))
    {
        return 0;
    }
    double sum = 0.0;
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue ai = a->u.agg.elems[i];
        WGSLValue bi = b->u.agg.elems[i];
        if (!promote_pair(cev, &ai, &bi, call) ||
            !promote_pair(cev, &at, &ai, call) ||
            !promote_pair(cev, &at, &bi, call) ||
            !require_float_scalar(cev, call, nm, len, &ai) ||
            !require_float_scalar(cev, call, nm, len, &bi))
        {
            return 0;
        }
        WGSLValue av = { .kind = WGSL_VAL_FLOAT, .type = at.type, .u.f = ai.u.f };
        WGSLValue bv = { .kind = WGSL_VAL_FLOAT, .type = at.type, .u.f = bi.u.f };
        WGSLValue prod = {0};
        WGSLValue sv = { .kind = WGSL_VAL_FLOAT, .type = at.type, .u.f = sum };
        WGSLValue next = {0};
        if (!eval_binary_float(cev, call, WGSL_TOK_STAR, &av, &bv, &prod) ||
            !eval_binary_float(cev, call, WGSL_TOK_PLUS, &sv, &prod, &next)) {
            return 0;
        }
        sum = next.u.f;
    }
    if (!isfinite(sum)) {
        cev_error(cev, call,
            "builtin '%.*s' produced a non-finite result "
            "(spec §15.7.3 finite-math assumption)",
            (int)len, nm);
        return 0;
    }
    *out_type = at.type;
    *out_dot = sum;
    return 1;
}

static int fold_float_inherited_binary(
    WGSLConstEvaluator *cev, WGSLNode *call, WGSLTypeInfo *type,
    WGSLTokenKind op, double a, double b, double *out)
{
    WGSLValue av = { .kind = WGSL_VAL_FLOAT, .type = type, .u.f = a };
    WGSLValue bv = { .kind = WGSL_VAL_FLOAT, .type = type, .u.f = b };
    WGSLValue rv = {0};
    if (!eval_binary_float(cev, call, op, &av, &bv, &rv)) return 0;
    *out = rv.u.f;
    return 1;
}

static int eval_call_reflect_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "reflect")) return 0;
    if (call->child_count != 3) {
        cev_error(cev, call,
            "builtin 'reflect' expects 2 arguments; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[2] = {{0}};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]))
    {
        return -1;
    }
    uint32_t width = 0;
    if (!require_same_width_vectors(cev, call, nm, len, vals, 2, &width)) {
        return -1;
    }
    WGSLTypeInfo *rt = NULL;
    double d = 0.0;
    if (!fold_float_dot(cev, call, nm, len, &vals[1], &vals[0],
                        width, &rt, &d))
    {
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue I = vals[0].u.agg.elems[i];
        WGSLValue N = vals[1].u.agg.elems[i];
        if (!promote_pair(cev, &I, &N, call) ||
            !wgsl_consteval_materialize(cev, &I, rt, call) ||
            !wgsl_consteval_materialize(cev, &N, rt, call))
        {
            return -1;
        }
        double r = I.u.f - 2.0 * d * N.u.f;
        if (!finish_float_builtin_result(cev, call, nm, len, rt, r, &tmp[i])) {
            return -1;
        }
    }
    return make_vector_result(cev, call, width, rt, tmp, out) ? 1 : -1;
}

static int eval_call_face_forward_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "faceForward")) return 0;
    if (call->child_count != 4) {
        cev_error(cev, call,
            "builtin 'faceForward' expects 3 arguments; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[3] = {{0}};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]) ||
        !eval_expr(cev, call->children[3], &vals[2]))
    {
        return -1;
    }
    uint32_t width = 0;
    if (!require_same_width_vectors(cev, call, nm, len, vals, 3, &width)) {
        return -1;
    }
    WGSLTypeInfo *rt = NULL;
    double d = 0.0;
    if (!fold_float_dot(cev, call, nm, len, &vals[2], &vals[1],
                        width, &rt, &d))
    {
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    int keep = d < 0.0;
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue N = vals[0].u.agg.elems[i];
        if (!wgsl_consteval_materialize(cev, &N, rt, call)) return -1;
        double r = keep ? N.u.f : -N.u.f;
        if (!finish_float_builtin_result(cev, call, nm, len, rt, r, &tmp[i])) {
            return -1;
        }
    }
    return make_vector_result(cev, call, width, rt, tmp, out) ? 1 : -1;
}

static int eval_call_refract_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "refract")) return 0;
    if (call->child_count != 4) {
        cev_error(cev, call,
            "builtin 'refract' expects 3 arguments; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[2] = {{0}};
    WGSLValue eta = {0};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]) ||
        !eval_expr(cev, call->children[3], &eta))
    {
        return -1;
    }
    uint32_t width = 0;
    if (!require_same_width_vectors(cev, call, nm, len, vals, 2, &width)) {
        return -1;
    }
    WGSLTypeInfo *rt = NULL;
    double d = 0.0;
    if (!fold_float_dot(cev, call, nm, len, &vals[1], &vals[0],
                        width, &rt, &d) ||
        !wgsl_consteval_materialize(cev, &eta, rt, call) ||
        !require_float_scalar(cev, call, nm, len, &eta))
    {
        return -1;
    }

    double eta2 = 0.0;
    double d2 = 0.0;
    double one_minus_d2 = 0.0;
    double term = 0.0;
    double k = 0.0;
    if (!fold_float_inherited_binary(
            cev, call, rt, WGSL_TOK_STAR, eta.u.f, eta.u.f, &eta2) ||
        !fold_float_inherited_binary(
            cev, call, rt, WGSL_TOK_STAR, d, d, &d2) ||
        !fold_float_inherited_binary(
            cev, call, rt, WGSL_TOK_MINUS, 1.0, d2, &one_minus_d2) ||
        !fold_float_inherited_binary(
            cev, call, rt, WGSL_TOK_STAR, eta2, one_minus_d2, &term) ||
        !fold_float_inherited_binary(
            cev, call, rt, WGSL_TOK_MINUS, 1.0, term, &k))
    {
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        double r = 0.0;
        if (k >= 0.0) {
            WGSLValue I = vals[0].u.agg.elems[i];
            WGSLValue N = vals[1].u.agg.elems[i];
            if (!promote_pair(cev, &I, &N, call) ||
                !wgsl_consteval_materialize(cev, &I, rt, call) ||
                !wgsl_consteval_materialize(cev, &N, rt, call))
            {
                return -1;
            }
            double eta_i = 0.0;
            double eta_d = 0.0;
            double sum = 0.0;
            double scaled_n = 0.0;
            WGSLValue sqrt_v = {0};
            if (!fold_float_inherited_binary(
                    cev, call, rt, WGSL_TOK_STAR, eta.u.f, I.u.f, &eta_i) ||
                !fold_float_inherited_binary(
                    cev, call, rt, WGSL_TOK_STAR, eta.u.f, d, &eta_d) ||
                !finish_float_builtin_result(
                    cev, call, nm, len, rt, sqrt(k), &sqrt_v) ||
                !fold_float_inherited_binary(
                    cev, call, rt, WGSL_TOK_PLUS, eta_d, sqrt_v.u.f, &sum) ||
                !fold_float_inherited_binary(
                    cev, call, rt, WGSL_TOK_STAR, sum, N.u.f, &scaled_n) ||
                !fold_float_inherited_binary(
                    cev, call, rt, WGSL_TOK_MINUS, eta_i, scaled_n, &r))
            {
                return -1;
            }
        }
        if (!finish_float_builtin_result_relaxed_range(
                cev, call, nm, len, rt, r, &tmp[i]))
        {
            return -1;
        }
    }
    return make_vector_result(cev, call, width, rt, tmp, out) ? 1 : -1;
}

static int eval_call_transpose_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "transpose")) return 0;
    if (call->child_count != 2) {
        cev_error(cev, call,
            "builtin 'transpose' expects 1 argument; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;
    if (arg.kind != WGSL_VAL_MAT || !arg.type ||
        arg.type->kind != WGSL_TYPE_MAT)
    {
        cev_error(cev, call, "builtin 'transpose' requires a matrix argument");
        return -1;
    }
    uint32_t cols = arg.type->width;
    uint32_t rows = arg.type->rows;
    uint32_t count = cols * rows;
    WGSLValue *elems = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *elems * count);
    if (!elems) {
        cev_error(cev, call, "out of memory while folding transpose");
        return -1;
    }
    for (uint32_t out_col = 0; out_col < rows; out_col++) {
        for (uint32_t out_row = 0; out_row < cols; out_row++) {
            uint32_t src_col = out_row;
            uint32_t src_row = out_col;
            elems[out_col * cols + out_row] =
                arg.u.agg.elems[src_col * rows + src_row];
        }
    }
    out->kind = WGSL_VAL_MAT;
    out->type = wgsl_type_mat(
        cev->types, (uint8_t)rows, (uint8_t)cols,
        (WGSLTypeInfo *)arg.type->ref);
    out->u.agg.count = count;
    out->u.agg.elems = elems;
    return 1;
}

