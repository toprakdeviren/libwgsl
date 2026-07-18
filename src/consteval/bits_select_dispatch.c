/**
 * @file bits_select_dispatch.c — const-expression evaluator module.
 */
#include "internal/consteval_priv.h"

int value_numeric_to_double(const WGSLValue *v, double *out) {
    if (!v || !out) return 0;
    if (v->kind == WGSL_VAL_INT) {
        *out = (double)v->u.i;
        return 1;
    }
    if (v->kind == WGSL_VAL_FLOAT) {
        *out = v->u.f;
        return 1;
    }
    return 0;
}

double det_small(double m[4][4], int n) {
    if (n == 1) return m[0][0];
    if (n == 2) return m[0][0] * m[1][1] - m[0][1] * m[1][0];
    double sum = 0.0;
    for (int col = 0; col < n; col++) {
        double sub[4][4] = {{0.0}};
        for (int r = 1; r < n; r++) {
            int dc = 0;
            for (int c = 0; c < n; c++) {
                if (c == col) continue;
                sub[r - 1][dc++] = m[r][c];
            }
        }
        double term = m[0][col] * det_small(sub, n - 1);
        sum += (col & 1) ? -term : term;
    }
    return sum;
}

int eval_call_determinant_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "determinant")) return 0;
    if (call->child_count != 2) {
        cev_error(cev, call,
            "builtin 'determinant' expects 1 argument; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;
    if (arg.kind != WGSL_VAL_MAT || !arg.type ||
        arg.type->kind != WGSL_TYPE_MAT ||
        arg.type->width != arg.type->rows)
    {
        cev_error(cev, call,
            "builtin 'determinant' requires a square matrix argument");
        return -1;
    }
    int n = (int)arg.type->width;
    if (n < 2 || n > 4) {
        cev_error(cev, call,
            "builtin 'determinant' supports mat2x2, mat3x3, or mat4x4");
        return -1;
    }

    double m[4][4] = {{0.0}};
    for (int c = 0; c < n; c++) {
        for (int r = 0; r < n; r++) {
            WGSLValue cell = arg.u.agg.elems[(uint32_t)c * arg.type->rows + (uint32_t)r];
            if (!value_numeric_to_double(&cell, &m[r][c])) {
                cev_error(cev, call,
                    "builtin 'determinant' requires numeric matrix components");
                return -1;
            }
        }
    }

    WGSLTypeInfo *elem = (WGSLTypeInfo *)arg.type->ref;
    WGSLTypeInfo *rt = elem;
    if (rt == cev->types->t_abstract_int ||
        rt == cev->types->t_i32 ||
        rt == cev->types->t_u32)
    {
        rt = cev->types->t_abstract_float;
    }
    double d = det_small(m, n);
    return finish_float_builtin_result(cev, call, nm, len, rt, d, out) ? 1 : -1;
}

int read_u32_const_arg(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, const WGSLValue *v, uint32_t *out)
{
    WGSLValue x = *v;
    if (x.kind != WGSL_VAL_INT) {
        cev_error(cev, call,
            "builtin '%.*s' requires u32 offset/count arguments",
            (int)len, nm);
        return 0;
    }
    if (!wgsl_consteval_materialize(cev, &x, cev->types->t_u32, call)) {
        return 0;
    }
    *out = (uint32_t)x.u.i;
    return 1;
}

int bitfield_shape(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len,
    WGSLValue *value, WGSLValue *newbits,
    uint32_t *out_width, WGSLTypeInfo **out_vec_type)
{
    WGSLValue vals[2] = {*value, {0}};
    int argc = 1;
    if (newbits) {
        vals[1] = *newbits;
        argc = 2;
    }
    if (!componentwise_shape(cev, call, nm, len, vals, argc, -1,
                             out_width, out_vec_type))
    {
        return 0;
    }
    if (newbits) {
        *value = vals[0];
        *newbits = vals[1];
    } else {
        *value = vals[0];
    }
    return 1;
}

int fold_extract_bits_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    WGSLValue *value, uint32_t offset, uint32_t count, WGSLValue *out)
{
    if (value->kind != WGSL_VAL_INT) {
        cev_error(cev, call, "extractBits value must be integer");
        return 0;
    }
    if (offset + count > 32) {
        cev_error(cev, call,
            "builtin 'extractBits' domain error: offset + count must be <= 32");
        return 0;
    }
    uint32_t bits = (uint32_t)value->u.i;
    uint32_t r = 0;
    if (count != 0) {
        uint32_t mask = count == 32 ? UINT32_MAX : ((1u << count) - 1u);
        r = (bits >> offset) & mask;
        if (value->type == cev->types->t_i32 && count < 32 &&
            (r & (1u << (count - 1u))))
        {
            r |= ~mask;
        }
    }
    out->kind = WGSL_VAL_INT;
    out->type = value->type;
    out->u.i = value->type == cev->types->t_u32
        ? (int64_t)r : (int64_t)(int32_t)r;
    return 1;
}

int fold_insert_bits_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    WGSLValue *value, WGSLValue *newbits,
    uint32_t offset, uint32_t count, WGSLValue *out)
{
    if (value->kind != WGSL_VAL_INT || newbits->kind != WGSL_VAL_INT) {
        cev_error(cev, call, "insertBits values must be integers");
        return 0;
    }
    if (!promote_pair(cev, value, newbits, call)) return 0;
    if (offset + count > 32) {
        cev_error(cev, call,
            "builtin 'insertBits' domain error: offset + count must be <= 32");
        return 0;
    }
    uint32_t v = (uint32_t)value->u.i;
    uint32_t nb = (uint32_t)newbits->u.i;
    uint32_t r = v;
    if (count != 0) {
        uint32_t low = count == 32 ? UINT32_MAX : ((1u << count) - 1u);
        uint32_t mask = low << offset;
        r = (v & ~mask) | ((nb << offset) & mask);
    }
    out->kind = WGSL_VAL_INT;
    out->type = value->type;
    out->u.i = value->type == cev->types->t_u32
        ? (int64_t)r : (int64_t)(int32_t)r;
    return 1;
}

int eval_call_bitfield_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int is_insert = name_eq(nm, len, "insertBits");
    int is_extract = name_eq(nm, len, "extractBits");
    if (!is_insert && !is_extract) return 0;
    uint32_t expected = is_insert ? 5 : 4;
    if (call->child_count != expected) {
        cev_error(cev, call,
            "builtin '%.*s' expects %u arguments; got %u",
            (int)len, nm, (unsigned)(expected - 1),
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue value = {0}, newbits = {0}, offset_v = {0}, count_v = {0};
    if (!eval_expr(cev, call->children[1], &value)) return -1;
    if (is_insert && !eval_expr(cev, call->children[2], &newbits)) return -1;
    uint32_t offset_idx = is_insert ? 3u : 2u;
    if (!eval_expr(cev, call->children[offset_idx], &offset_v) ||
        !eval_expr(cev, call->children[offset_idx + 1u], &count_v))
    {
        return -1;
    }
    uint32_t offset = 0, count = 0;
    if (!read_u32_const_arg(cev, call, nm, len, &offset_v, &offset) ||
        !read_u32_const_arg(cev, call, nm, len, &count_v, &count))
    {
        return -1;
    }

    uint32_t width = 1;
    WGSLTypeInfo *vec_type = NULL;
    if (!bitfield_shape(cev, call, nm, len, &value,
                        is_insert ? &newbits : NULL,
                        &width, &vec_type))
    {
        return -1;
    }

    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue v = value_component(&value, i);
        if (is_insert) {
            WGSLValue nb = value_component(&newbits, i);
            if (!fold_insert_bits_scalar(
                    cev, call, &v, &nb, offset, count, &tmp[i]))
            {
                return -1;
            }
        } else if (!fold_extract_bits_scalar(
                       cev, call, &v, offset, count, &tmp[i]))
        {
            return -1;
        }
    }
    return make_componentwise_result(cev, call, width, vec_type, tmp, out) ? 1 : -1;
}

uint32_t reverse32(uint32_t x) {
    x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);
    x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);
    x = ((x & 0x0f0f0f0fu) << 4) | ((x >> 4) & 0x0f0f0f0fu);
    x = ((x & 0x00ff00ffu) << 8) | ((x >> 8) & 0x00ff00ffu);
    return (x << 16) | (x >> 16);
}

uint32_t popcount32(uint32_t x) {
    uint32_t c = 0;
    while (x) {
        c += x & 1u;
        x >>= 1;
    }
    return c;
}

uint32_t clz32(uint32_t x) {
    if (x == 0) return 32;
    uint32_t n = 0;
    for (int i = 31; i >= 0 && ((x & (1u << i)) == 0); i--) n++;
    return n;
}

uint32_t ctz32(uint32_t x) {
    if (x == 0) return 32;
    uint32_t n = 0;
    while ((x & 1u) == 0) {
        n++;
        x >>= 1;
    }
    return n;
}

int fold_integer_bits_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *v, WGSLValue *out)
{
    if (v->kind != WGSL_VAL_INT) {
        cev_error(cev, call,
            "builtin '%.*s' requires an integer argument", (int)len, nm);
        return 0;
    }
    /* §4.3 / §6.2.1 — keep AbstractInt abstract.  Early materialize to
     * i32 was a false-positive source for `let x: u32 = countOneBits(7)`
     * (i32 result could not convert to u32).  The use-site materializer
     * picks i32 or u32 from context.  Bit ops still use the low 32 bits
     * of the mathematical integer (WGSL §17.5.x domain). */
    int is_u32 = (v->type == cev->types->t_u32);
    int is_i32 = (v->type == cev->types->t_i32);
    int is_abs = (v->type == cev->types->t_abstract_int);
    if (!is_u32 && !is_i32 && !is_abs) {
        cev_error(cev, call,
            "builtin '%.*s' requires an integer argument", (int)len, nm);
        return 0;
    }
    uint32_t x = (uint32_t)v->u.i;
    uint32_t r = 0;
    if (name_eq(nm, len, "countOneBits")) {
        r = popcount32(x);
    } else if (name_eq(nm, len, "countLeadingZeros")) {
        r = clz32(x);
    } else if (name_eq(nm, len, "countTrailingZeros")) {
        r = ctz32(x);
    } else if (name_eq(nm, len, "reverseBits")) {
        r = reverse32(x);
    } else if (name_eq(nm, len, "firstTrailingBit")) {
        r = x == 0 ? UINT32_MAX : ctz32(x);
    } else if (name_eq(nm, len, "firstLeadingBit")) {
        /* Signed firstLeadingBit uses ~x for negative i32; AbstractInt
         * negative values follow the same two's-complement rule on the
         * low 32 bits once the outer context materializes. */
        int signed_neg = (is_i32 || is_abs) && (int32_t)x < 0;
        uint32_t y = signed_neg ? ~x : x;
        r = y == 0 ? UINT32_MAX : (31u - clz32(y));
    } else {
        return 0;
    }

    out->kind = WGSL_VAL_INT;
    out->type = v->type; /* preserve AbstractInt / i32 / u32 */
    if (is_u32) {
        out->u.i = (int64_t)r;
    } else if (is_abs) {
        /* Abstract result: 0..32 or -1-as-all-bits sentinel for the
         * *Bit family zero cases (UINT32_MAX → keep as unsigned bit
         * pattern in the abstract integer). */
        out->u.i = (r == UINT32_MAX) ? (int64_t)(int32_t)r : (int64_t)r;
    } else {
        out->u.i = (int64_t)(int32_t)r;
    }
    return 1;
}

int eval_call_integer_bits_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int supported =
        name_eq(nm, len, "countOneBits") ||
        name_eq(nm, len, "countLeadingZeros") ||
        name_eq(nm, len, "countTrailingZeros") ||
        name_eq(nm, len, "reverseBits") ||
        name_eq(nm, len, "firstLeadingBit") ||
        name_eq(nm, len, "firstTrailingBit");
    if (!supported) return 0;
    if (call->child_count != 2) {
        cev_error(cev, call,
            "builtin '%.*s' expects 1 argument; got %u",
            (int)len, nm, (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;
    uint32_t width = 1;
    WGSLTypeInfo *vec_type = NULL;
    if (!componentwise_shape(cev, call, nm, len, &arg, 1, -1,
                             &width, &vec_type))
    {
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue v = value_component(&arg, i);
        if (!fold_integer_bits_scalar(cev, call, nm, len, &v, &tmp[i])) {
            return -1;
        }
    }
    return make_componentwise_result(cev, call, width, vec_type, tmp, out) ? 1 : -1;
}

int eval_call_select_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "select")) return 0;
    if (call->child_count != 4) {
        cev_error(cev, call,
            "builtin 'select' expects 3 arguments; got %u",
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
    uint32_t width = 1;
    WGSLTypeInfo *vec_type = NULL;
    if (!componentwise_shape(cev, call, nm, len, vals, 2, -1,
                             &width, &vec_type))
    {
        return -1;
    }
    int cond_vec = vals[2].kind == WGSL_VAL_VEC;
    if (cond_vec && vals[2].u.agg.count != width) {
        cev_error(cev, call, "builtin 'select' condition shape mismatch");
        return -1;
    }
    if (!cond_vec && vals[2].kind != WGSL_VAL_BOOL) {
        cev_error(cev, call, "builtin 'select' requires bool condition");
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue f = value_component(&vals[0], i);
        WGSLValue t = value_component(&vals[1], i);
        if (!promote_pair(cev, &f, &t, call)) return -1;
        WGSLValue c = cond_vec ? vals[2].u.agg.elems[i] : vals[2];
        if (c.kind != WGSL_VAL_BOOL) {
            cev_error(cev, call, "builtin 'select' requires bool condition");
            return -1;
        }
        tmp[i] = c.u.b ? t : f;
    }
    return make_componentwise_result(cev, call, width, vec_type, tmp, out) ? 1 : -1;
}

int eval_call_decompose_float_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int is_frexp = name_eq(nm, len, "frexp");
    int is_modf = name_eq(nm, len, "modf");
    if (!is_frexp && !is_modf) return 0;
    if (call->child_count != 2) {
        cev_error(cev, call,
            "builtin '%.*s' expects 1 argument; got %u",
            (int)len, nm, (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;
    uint32_t width = 1;
    WGSLTypeInfo *vec_type = NULL;
    if (!componentwise_shape(cev, call, nm, len, &arg, 1, -1,
                             &width, &vec_type))
    {
        return -1;
    }

    WGSLValue fract_tmp[4] = {{0}};
    WGSLValue aux_tmp[4] = {{0}};
    WGSLTypeInfo *float_type = NULL;
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue x = value_component(&arg, i);
        if (!require_float_scalar(cev, call, nm, len, &x)) return -1;
        if (!float_type) float_type = x.type;
        if (!wgsl_consteval_materialize(cev, &x, float_type, call)) return -1;

        if (is_frexp) {
            int exp = 0;
            double frac = frexp(x.u.f, &exp);
            if (!finish_float_builtin_result_relaxed_range(
                    cev, call, nm, len, float_type, frac, &fract_tmp[i]))
            {
                return -1;
            }
            aux_tmp[i].kind = WGSL_VAL_INT;
            aux_tmp[i].type = cev->types->t_i32;
            aux_tmp[i].u.i = exp;
        } else {
            double whole = trunc(x.u.f);
            double frac = x.u.f - whole;
            if (!finish_float_builtin_result_relaxed_range(
                    cev, call, nm, len, float_type, frac, &fract_tmp[i]) ||
                !finish_float_builtin_result_relaxed_range(
                    cev, call, nm, len, float_type, whole, &aux_tmp[i]))
            {
                return -1;
            }
        }
    }

    WGSLValue fields[2] = {{0}};
    WGSLTypeInfo *result_shape = float_type;
    if (vec_type) {
        if (!make_vector_result(cev, call, width, float_type, fract_tmp, &fields[0])) {
            return -1;
        }
        WGSLTypeInfo *aux_elem = is_frexp ? cev->types->t_i32 : float_type;
        if (!make_vector_result(cev, call, width, aux_elem, aux_tmp, &fields[1])) {
            return -1;
        }
        result_shape = fields[0].type;
    } else {
        fields[0] = fract_tmp[0];
        fields[1] = aux_tmp[0];
    }

    WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *stored * 2);
    if (!stored) {
        cev_error(cev, call, "out of memory while folding builtin '%.*s'",
                  (int)len, nm);
        return -1;
    }
    stored[0] = fields[0];
    stored[1] = fields[1];
    out->kind = WGSL_VAL_STRUCT;
    out->type = is_frexp
        ? wgsl_type_frexp_result(cev->types, result_shape)
        : wgsl_type_modf_result(cev->types, result_shape);
    out->u.agg.count = 2;
    out->u.agg.elems = stored;
    return 1;
}

int eval_call_any_all_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int is_any = name_eq(nm, len, "any");
    int is_all = name_eq(nm, len, "all");
    if (!is_any && !is_all) return 0;
    if (call->child_count != 2) {
        cev_error(cev, call,
            "builtin '%.*s' expects 1 argument; got %u",
            (int)len, nm, (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;
    if (arg.kind == WGSL_VAL_BOOL) {
        out->kind = WGSL_VAL_BOOL;
        out->type = cev->types->t_bool;
        out->u.b = arg.u.b;
        return 1;
    }
    if (arg.kind != WGSL_VAL_VEC || !arg.type ||
        arg.type->kind != WGSL_TYPE_VEC ||
        (WGSLTypeInfo *)arg.type->ref != cev->types->t_bool)
    {
        cev_error(cev, call,
            "builtin '%.*s' requires a vecN<bool> argument", (int)len, nm);
        return -1;
    }
    bool r = is_all ? true : false;
    for (uint32_t i = 0; i < arg.u.agg.count; i++) {
        WGSLValue c = arg.u.agg.elems[i];
        if (c.kind != WGSL_VAL_BOOL) {
            cev_error(cev, call,
                "builtin '%.*s' requires a vecN<bool> argument", (int)len, nm);
            return -1;
        }
        if (is_all) r = r && c.u.b;
        else        r = r || c.u.b;
    }
    out->kind = WGSL_VAL_BOOL;
    out->type = cev->types->t_bool;
    out->u.b = r;
    return 1;
}

/* Def-generated name→fold-kind table (tools/gen-wgsl-builtins.py).
 * Numeric codes match WGSLConstFoldKind ordinals in exprs_priv.h. */
#include "check/builtins_cfold.gen.h"

static int cfold_name_cmp(const char *name, uint32_t name_len, const char *key) {
    size_t kl = strlen(key);
    size_t n = name_len < kl ? name_len : kl;
    int c = memcmp(name, key, n);
    if (c != 0) return c;
    if (name_len < kl) return -1;
    if (name_len > kl) return 1;
    return 0;
}

static int lookup_cfold(const char *nm, uint32_t len) {
    size_t lo = 0, hi = (size_t)WGSL_BUILTIN_CFOLD_COUNT;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        const WGSLBuiltinCFoldEntry *e = &kBuiltinCFoldTable[mid];
        int c = cfold_name_cmp(nm, len, e->name);
        if (c == 0) return (int)e->cfold;
        if (c < 0) hi = mid;
        else lo = mid + 1;
    }
    return 0; /* WGSL_CFOLD_NONE */
}

int eval_call_const_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    /* Dispatch by def-generated cfold; no top-level strcmp cascade. */
    switch (lookup_cfold(nm, len)) {
    case 0: /* NONE */
        return 0;
    case 1: /* BITCAST */
        return eval_call_bitcast_builtin(cev, call, nm, len, out);
    case 2: /* PACK */
        return eval_call_pack_builtin(cev, call, nm, len, out);
    case 3: /* UNPACK */
        return eval_call_unpack_builtin(cev, call, nm, len, out);
    case 4: /* DOT4 */
        return eval_call_dot4_packed_builtin(cev, call, nm, len, out);
    case 5: /* ANY_ALL */
        return eval_call_any_all_builtin(cev, call, nm, len, out);
    case 6: /* UNARY_FLOAT */
        return eval_call_unary_float_builtin(cev, call, nm, len, out);
    case 7: /* ABS_SIGN */
        return eval_call_abs_sign_builtin(cev, call, nm, len, out);
    case 8: /* BINARY_NUMERIC */
        return eval_call_binary_numeric_builtin(cev, call, nm, len, out);
    case 9: /* TERNARY_NUMERIC */
        return eval_call_ternary_numeric_builtin(cev, call, nm, len, out);
    case 10: /* LDEXP */
        return eval_call_ldexp_builtin(cev, call, nm, len, out);
    case 11: /* LENGTH */
        return eval_call_length_builtin(cev, call, nm, len, out);
    case 12: /* DISTANCE_DOT */
        return eval_call_distance_dot_builtin(cev, call, nm, len, out);
    case 13: /* CROSS */
        return eval_call_cross_builtin(cev, call, nm, len, out);
    case 14: /* NORMALIZE */
        return eval_call_normalize_builtin(cev, call, nm, len, out);
    case 15: /* REFLECT */
        return eval_call_reflect_builtin(cev, call, nm, len, out);
    case 16: /* FACE_FORWARD */
        return eval_call_face_forward_builtin(cev, call, nm, len, out);
    case 17: /* REFRACT */
        return eval_call_refract_builtin(cev, call, nm, len, out);
    case 18: /* TRANSPOSE */
        return eval_call_transpose_builtin(cev, call, nm, len, out);
    case 19: /* DETERMINANT */
        return eval_call_determinant_builtin(cev, call, nm, len, out);
    case 20: /* BITFIELD */
        return eval_call_bitfield_builtin(cev, call, nm, len, out);
    case 21: /* INTEGER_BITS */
        return eval_call_integer_bits_builtin(cev, call, nm, len, out);
    case 22: /* SELECT */
        return eval_call_select_builtin(cev, call, nm, len, out);
    case 23: /* DECOMPOSE_FLOAT */
        return eval_call_decompose_float_builtin(cev, call, nm, len, out);
    default:
        return 0;
    }
}
