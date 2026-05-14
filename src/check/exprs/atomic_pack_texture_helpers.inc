/* §17.8 — atomic builtins take a pointer to `atomic<T>` as the first
 * argument; subsequent value args must match `T`.  Shape table:
 *   atomicLoad(ptr)                              → T
 *   atomicStore(ptr, value:T)                    → ()
 *   atomicAdd / Sub / Max / Min / And / Or / Xor (ptr, value:T) → T
 *   atomicExchange(ptr, value:T)                 → T
 *   atomicCompareExchangeWeak(ptr, cmp:T, val:T) → __atomic_cx_result<T>
 *
 * The pointer's address space is restricted to `storage` or `workgroup`
 * by §6.2.8 + §14.3 (already enforced at the decl site).  Here we only
 * verify (1) pointee is `atomic<T>`, (2) value args have type `T`. */
typedef enum {
    AT_KIND_LOAD,             /* (ptr) → T */
    AT_KIND_STORE,            /* (ptr, T) → () */
    AT_KIND_RMW,              /* (ptr, T) → T  (Add/Sub/Max/Min/.../Exchange) */
    AT_KIND_CX,               /* (ptr, T, T) → __atomic_cx_result<T> */
} WGSLAtomicKind;

static int classify_atomic(const char *name, uint32_t len, WGSLAtomicKind *out) {
    if (len < 6 || memcmp(name, "atomic", 6) != 0) return 0;
    if (len == 10 && memcmp(name, "atomicLoad", 10) == 0)               { *out = AT_KIND_LOAD;  return 1; }
    if (len == 11 && memcmp(name, "atomicStore", 11) == 0)              { *out = AT_KIND_STORE; return 1; }
    if (len == 25 && memcmp(name, "atomicCompareExchangeWeak", 25) == 0){ *out = AT_KIND_CX;    return 1; }
    /* atomicAdd / Sub / Max / Min / And / Or / Xor / Exchange */
    *out = AT_KIND_RMW;
    return 1;
}

/* Pull the atomic<T>'s element T plus pointer qualifiers from the first arg.
 * Atomic builtins require a pointer value, not an atomic<T> value. */
static int atomic_ptr_info(
    WGSLTypeInfo *arg0,
    WGSLTypeInfo **out_T,
    WGSLAddressSpace *out_as,
    WGSLAccessMode *out_am)
{
    if (!arg0 || arg0->kind != WGSL_TYPE_PTR) return 0;
    WGSLTypeInfo *pointee = (WGSLTypeInfo *)arg0->ref;
    if (!pointee || pointee->kind != WGSL_TYPE_ATOMIC) return 0;
    if (out_T)  *out_T  = (WGSLTypeInfo *)pointee->ref;
    if (out_as) *out_as = (WGSLAddressSpace)(arg0->flags & 0xFFu);
    if (out_am) *out_am = (WGSLAccessMode)((arg0->flags >> 8) & 0xFFu);
    return 1;
}

static int atomic_ptr_access_ok(WGSLAddressSpace as, WGSLAccessMode am) {
    return (as == WGSL_AS_WORKGROUP || as == WGSL_AS_STORAGE) &&
           am == WGSL_ACCESS_READ_WRITE;
}

static WGSLTypeInfo *atomic_pointee_T(WGSLTypeInfo *arg0) {
    WGSLTypeInfo *T = NULL;
    WGSLAddressSpace as = WGSL_AS_NONE;
    WGSLAccessMode am = WGSL_ACCESS_NONE;
    if (!atomic_ptr_info(arg0, &T, &as, &am)) return NULL;
    return atomic_ptr_access_ok(as, am) ? T : NULL;
}

static const char *type_label(WGSLTypeInfo *t, char *buf, size_t cap) {
    if (!t) return "(no type)";
    if (!buf || cap == 0) return wgsl_type_kind_name((WGSLTypeKind)t->kind);
    size_t n = wgsl_type_format(t, buf, cap);
    if (n >= cap) n = cap - 1;
    buf[n] = '\0';
    return buf;
}

static int bitcast_scalar_bits(const WGSLTypeInfo *t) {
    if (!t) return 0;
    switch ((WGSLTypeKind)t->kind) {
    case WGSL_TYPE_I32:
    case WGSL_TYPE_U32:
    case WGSL_TYPE_F32:
        return 32;
    case WGSL_TYPE_F16:
        return 16;
    default:
        return 0;
    }
}

static int bitcast_type_bits(const WGSLTypeInfo *t) {
    if (!t) return 0;
    int scalar_bits = 0;
    if (wgsl_type_is_scalar(t)) {
        scalar_bits = bitcast_scalar_bits(t);
        return (scalar_bits % 32) == 0 ? scalar_bits : 0;
    }
    if (t->kind == WGSL_TYPE_VEC) {
        scalar_bits = bitcast_scalar_bits((const WGSLTypeInfo *)t->ref);
        int total = scalar_bits * (int)t->width;
        return (total % 32) == 0 ? total : 0;
    }
    return 0;
}

static void check_atomic_call(
    WGSLTypeChecker *tc, WGSLNode *call,
    const char *name, uint32_t name_len,
    WGSLTypeInfo *arg0_type)
{
    WGSLAtomicKind kind;
    if (!classify_atomic(name, name_len, &kind)) return;

    /* Expected arg count: 1 for Load; 2 for Store/RMW; 3 for CX. */
    int want_args =
        kind == AT_KIND_LOAD  ? 1 :
        kind == AT_KIND_STORE ? 2 :
        kind == AT_KIND_CX    ? 3 : 2;
    int got_args = (int)call->child_count - 1;   /* minus callee */
    if (got_args != want_args) {
        wgsl_tc_error(tc, call,
            "'%.*s' expects %d argument%s; got %d",
            (int)name_len, name, want_args, want_args == 1 ? "" : "s", got_args);
        return;
    }

    /* Arg 0 must be a read_write pointer to atomic<T> in a shared address
     * space. */
    WGSLTypeInfo *T = NULL;
    WGSLAddressSpace as = WGSL_AS_NONE;
    WGSLAccessMode am = WGSL_ACCESS_NONE;
    if (!atomic_ptr_info(arg0_type, &T, &as, &am)) {
        char got[96];
        wgsl_tc_error(tc, call,
            "'%.*s' requires a pointer to atomic<T> as the first "
            "argument; got %s",
            (int)name_len, name,
            type_label(arg0_type, got, sizeof got));
        return;
    }
    if (!atomic_ptr_access_ok(as, am)) {
        char got[96];
        wgsl_tc_error(tc, call,
            "'%.*s' first argument must be ptr<workgroup, atomic<T>> "
            "or ptr<storage, atomic<T>, read_write>; got %s",
            (int)name_len, name,
            type_label(arg0_type, got, sizeof got));
        return;
    }

    /* Validate value args (positions depend on kind).  For each, the
     * arg's type must be feasibly convertible to T. */
    int n_values =
        kind == AT_KIND_LOAD ? 0 :
        kind == AT_KIND_CX   ? 2 : 1;
    for (int i = 0; i < n_values; i++) {
        WGSLNode *arg = call->children[2 + i];
        WGSLTypeInfo *vt = wgsl_tc_store_type_of(tc, arg);
        if (!vt) continue;
        if (wgsl_tc_init_type_mismatch(vt, T)) {
            char got[96], want[96];
            wgsl_tc_error(tc, arg,
                "'%.*s' value argument has type %s; expected %s "
                "(matching the atomic's element type)",
                (int)name_len, name,
                type_label(vt, got, sizeof got),
                type_label(T, want, sizeof want));
        }
    }
}

/* §17.9 / §17.10 — pack* / unpack* builtins take a fixed-shape arg.
 * Return 1 if the call's first arg matches the expected shape, 0
 * otherwise (after emitting a diagnostic with the expected shape).
 * Shapes:
 *   pack4x8snorm/unorm           → vec4<f32>
 *   pack4xI8 / pack4xI8Clamp     → vec4<i32>
 *   pack4xU8 / pack4xU8Clamp     → vec4<u32>
 *   pack2x16snorm/unorm/float    → vec2<f32>
 *   unpack* (all 7)              → u32                                */
static int check_pack_unpack_arg(
    WGSLTypeChecker *tc, WGSLNode *call,
    const char *name, uint32_t name_len,
    WGSLTypeInfo *arg0)
{
    if (!arg0) return 1;   /* arity-mismatch handled elsewhere */

    /* Determine expected shape from name. */
    const WGSLTypeInfo *want_vec_elem = NULL;
    int want_vec_w = 0;
    int want_u32   = 0;

    /* All unpack* take u32. */
    if (name_len >= 6 && memcmp(name, "unpack", 6) == 0) {
        want_u32 = 1;
    }
    /* pack4x8snorm / pack4x8unorm → vec4<f32> */
    else if (name_len == 12 &&
             (memcmp(name, "pack4x8snorm", 12) == 0 ||
              memcmp(name, "pack4x8unorm", 12) == 0))
    {
        want_vec_elem = tc->types->t_f32;
        want_vec_w    = 4;
    }
    /* pack4xI8 / pack4xI8Clamp → vec4<i32> */
    else if ((name_len == 8  && memcmp(name, "pack4xI8", 8) == 0) ||
             (name_len == 13 && memcmp(name, "pack4xI8Clamp", 13) == 0))
    {
        want_vec_elem = tc->types->t_i32;
        want_vec_w    = 4;
    }
    /* pack4xU8 / pack4xU8Clamp → vec4<u32> */
    else if ((name_len == 8  && memcmp(name, "pack4xU8", 8) == 0) ||
             (name_len == 13 && memcmp(name, "pack4xU8Clamp", 13) == 0))
    {
        want_vec_elem = tc->types->t_u32;
        want_vec_w    = 4;
    }
    /* pack2x16{s,u}norm / pack2x16float → vec2<f32> */
    else if ((name_len == 13 && memcmp(name, "pack2x16snorm", 13) == 0) ||
             (name_len == 13 && memcmp(name, "pack2x16unorm", 13) == 0) ||
             (name_len == 13 && memcmp(name, "pack2x16float", 13) == 0))
    {
        want_vec_elem = tc->types->t_f32;
        want_vec_w    = 2;
    } else {
        return 1;   /* not a pack/unpack name */
    }

    /* Validate. */
    if (want_u32) {
        if (arg0 != tc->types->t_u32 &&
            arg0 != tc->types->t_abstract_int)
        {
            wgsl_tc_error(tc, call,
                "'%.*s' requires a u32 argument; got %s",
                (int)name_len, name,
                wgsl_type_kind_name((WGSLTypeKind)arg0->kind));
            return 0;
        }
        return 1;
    }

    if (want_vec_w > 0 && want_vec_elem) {
        WGSLTypeInfo *want = wgsl_type_vec(
            tc->types, (uint8_t)want_vec_w, (WGSLTypeInfo *)want_vec_elem);
        int ok = arg0 && want && wgsl_type_conversion_rank(arg0, want) >= 0;
        if (!ok) {
            const char *elem_name =
                want_vec_elem == tc->types->t_f32 ? "f32" :
                want_vec_elem == tc->types->t_i32 ? "i32" :
                want_vec_elem == tc->types->t_u32 ? "u32" : "?";
            wgsl_tc_error(tc, call,
                "'%.*s' requires a vec%d<%s> argument; got %s",
                (int)name_len, name, want_vec_w, elem_name,
                wgsl_type_kind_name((WGSLTypeKind)arg0->kind));
            return 0;
        }
    }
    return 1;
}

static int builtin_name_eq(const char *name, uint32_t len, const char *lit) {
    size_t n = strlen(lit);
    return len == n && memcmp(name, lit, n) == 0;
}

static int builtin_arg_count(const WGSLNode *call) {
    return call && call->child_count > 0 ? (int)call->child_count - 1 : 0;
}

static int builtin_arg_count_is(WGSLNode *call, int n) {
    return builtin_arg_count(call) == n;
}

static WGSLTypeInfo *builtin_arg_type(
    WGSLTypeChecker *tc, WGSLNode *call, int arg_index)
{
    if (!call || arg_index < 0) return NULL;
    uint32_t child_index = (uint32_t)arg_index + 1u;
    if (child_index >= call->child_count) return NULL;
    return wgsl_tc_store_type_of(tc, call->children[child_index]);
}

static int shape_width(const WGSLTypeInfo *t) {
    if (!t) return 0;
    if (t->kind == WGSL_TYPE_VEC) return (int)t->width;
    if (wgsl_type_is_scalar(t)) return 1;
    return 0;
}

static WGSLTypeInfo *shape_elem(const WGSLTypeInfo *t) {
    if (!t) return NULL;
    if (t->kind == WGSL_TYPE_VEC) return (WGSLTypeInfo *)t->ref;
    if (wgsl_type_is_scalar(t)) return (WGSLTypeInfo *)(uintptr_t)t;
    return NULL;
}

static int type_converts_to(WGSLTypeInfo *from, WGSLTypeInfo *to) {
    return from && to && wgsl_type_conversion_rank(from, to) >= 0;
}

static int expect_arg_count(
    WGSLTypeChecker *tc, WGSLNode *call,
    const char *name, uint32_t name_len, int min_args, int max_args)
{
    int got = builtin_arg_count(call);
    if (got >= min_args && got <= max_args) return 1;
    if (min_args == max_args) {
        wgsl_tc_error(tc, call,
            "'%.*s' expects %d argument%s; got %d",
            (int)name_len, name, min_args, min_args == 1 ? "" : "s", got);
    } else {
        wgsl_tc_error(tc, call,
            "'%.*s' expects %d-%d arguments; got %d",
            (int)name_len, name, min_args, max_args, got);
    }
    return 0;
}

static int expect_sampler_arg(
    WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len,
    WGSLTypeInfo *t, int comparison)
{
    if (t && t->kind == WGSL_TYPE_SAMPLER && ((int)t->width != 0) == comparison) {
        return 1;
    }
    char got[96];
    wgsl_tc_error(tc, at,
        "'%.*s' requires %s as the sampler argument; got %s",
        (int)name_len, name,
        comparison ? "sampler_comparison" : "sampler",
        type_label(t, got, sizeof got));
    return 0;
}

static int expect_float_shape(
    WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len,
    WGSLTypeInfo *t, int width, const char *role)
{
    WGSLTypeInfo *want = width == 1
        ? tc->types->t_f32
        : wgsl_type_vec(tc->types, (uint8_t)width, tc->types->t_f32);
    if (type_converts_to(t, want)) return 1;
    char got[96], exp[96];
    wgsl_tc_error(tc, at,
        "'%.*s' %s has type %s; expected %s",
        (int)name_len, name, role,
        type_label(t, got, sizeof got),
        type_label(want, exp, sizeof exp));
    return 0;
}

static int expect_integer_shape(
    WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len,
    WGSLTypeInfo *t, int width, const char *role)
{
    WGSLTypeInfo *elem = shape_elem(t);
    int ok = elem && wgsl_type_is_integer(elem) && shape_width(t) == width;
    if (ok) return 1;
    char got[96];
    wgsl_tc_error(tc, at,
        "'%.*s' %s has type %s; expected %sinteger %s",
        (int)name_len, name, role,
        type_label(t, got, sizeof got),
        width == 1 ? "an " : "a ",
        width == 1 ? "scalar" : "vector with matching texture dimensionality");
    return 0;
}

static int expect_i32_shape(
    WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len,
    WGSLTypeInfo *t, int width, const char *role)
{
    WGSLTypeInfo *elem = shape_elem(t);
    int ok = elem &&
             (elem == tc->types->t_abstract_int || elem == tc->types->t_i32) &&
             shape_width(t) == width;
    if (ok) return 1;
    char got[96];
    wgsl_tc_error(tc, at,
        "'%.*s' %s has type %s; expected %s",
        (int)name_len, name, role,
        type_label(t, got, sizeof got),
        width == 1 ? "i32 or AbstractInt"
                   : "vecN<i32> or vecN<AbstractInt> with matching offset dimensionality");
    return 0;
}

static int expect_value_convertible(
    WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len,
    WGSLTypeInfo *got_t, WGSLTypeInfo *want_t, const char *role)
{
    if (type_converts_to(got_t, want_t)) return 1;
    char got[96], want[96];
    wgsl_tc_error(tc, at,
        "'%.*s' %s has type %s; expected %s",
        (int)name_len, name, role,
        type_label(got_t, got, sizeof got),
        type_label(want_t, want, sizeof want));
    return 0;
}

static int eval_const_int_silent(
    WGSLTypeChecker *tc, WGSLNode *n, int64_t *out)
{
    if (!tc || !tc->cev || !tc->diag || !n || !out) return 0;
    size_t saved_count = tc->diag->count;
    int saved_errors = tc->diag->error_count;
    int saved_cev = tc->cev->had_error;
    WGSLValue v = {0};
    int ok = wgsl_consteval_expr(tc->cev, n, &v) && v.kind == WGSL_VAL_INT;
    tc->diag->count = saved_count;
    tc->diag->error_count = saved_errors;
    tc->cev->had_error = saved_cev;
    if (!ok) return 0;
    *out = v.u.i;
    return 1;
}

static int collect_const_int_components(
    WGSLTypeChecker *tc, WGSLNode *n, int width, int64_t *out)
{
    if (!n || !out || width < 1 || width > 4) return 0;
    if (width == 1) return eval_const_int_silent(tc, n, &out[0]);
    if (n->kind != WGSL_NODE_EXPR_CALL || n->child_count < 2) return 0;

    int arg_count = (int)n->child_count - 1;
    if (arg_count == 1) {
        int64_t v = 0;
        if (!eval_const_int_silent(tc, n->children[1], &v)) return 0;
        for (int i = 0; i < width; i++) out[i] = v;
        return 1;
    }
    if (arg_count != width) return 0;
    for (int i = 0; i < width; i++) {
        if (!eval_const_int_silent(tc, n->children[1 + i], &out[i])) {
            return 0;
        }
    }
    return 1;
}

static int check_const_int_range(
    WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len,
    int64_t lo, int64_t hi, const char *role)
{
    int64_t v = 0;
    if (!eval_const_int_silent(tc, at, &v)) {
        wgsl_tc_error(tc, at,
            "'%.*s' %s must be a const-expression",
            (int)name_len, name, role);
        return 0;
    }
    if (v < lo || v > hi) {
        wgsl_tc_error(tc, at,
            "'%.*s' %s value %lld is outside [%lld, %lld]",
            (int)name_len, name, role,
            (long long)v, (long long)lo, (long long)hi);
        return 0;
    }
    return 1;
}

static int texture_is_depth(WGSLTextureDim d) {
    return d >= WGSL_TEX_DIM_DEPTH_2D &&
           d <= WGSL_TEX_DIM_DEPTH_MULTISAMPLED_2D;
}

static int texture_is_storage(WGSLTextureDim d) {
    return d >= WGSL_TEX_DIM_STORAGE_1D && d <= WGSL_TEX_DIM_STORAGE_3D;
}

static int texture_is_sampled(WGSLTextureDim d) {
    return d >= WGSL_TEX_DIM_1D && d <= WGSL_TEX_DIM_MULTISAMPLED_2D;
}

static int texture_is_multisampled(WGSLTextureDim d) {
    return d == WGSL_TEX_DIM_MULTISAMPLED_2D ||
           d == WGSL_TEX_DIM_DEPTH_MULTISAMPLED_2D;
}

static int texture_is_arrayed(WGSLTextureDim d) {
    return d == WGSL_TEX_DIM_2D_ARRAY ||
           d == WGSL_TEX_DIM_CUBE_ARRAY ||
           d == WGSL_TEX_DIM_DEPTH_2D_ARRAY ||
           d == WGSL_TEX_DIM_DEPTH_CUBE_ARRAY ||
           d == WGSL_TEX_DIM_STORAGE_2D_ARRAY;
}

static int texture_is_gather_dim(WGSLTextureDim d) {
    return d == WGSL_TEX_DIM_2D ||
           d == WGSL_TEX_DIM_2D_ARRAY ||
           d == WGSL_TEX_DIM_CUBE ||
           d == WGSL_TEX_DIM_CUBE_ARRAY ||
           d == WGSL_TEX_DIM_DEPTH_2D ||
           d == WGSL_TEX_DIM_DEPTH_2D_ARRAY ||
           d == WGSL_TEX_DIM_DEPTH_CUBE ||
           d == WGSL_TEX_DIM_DEPTH_CUBE_ARRAY;
}

static int texture_allows_offset(WGSLTextureDim d) {
    return d == WGSL_TEX_DIM_2D ||
           d == WGSL_TEX_DIM_2D_ARRAY ||
           d == WGSL_TEX_DIM_3D ||
           d == WGSL_TEX_DIM_DEPTH_2D ||
           d == WGSL_TEX_DIM_DEPTH_2D_ARRAY;
}

static int texture_offset_width(WGSLTextureDim d) {
    return d == WGSL_TEX_DIM_3D ? 3 : 2;
}

static int texture_coord_width(WGSLTextureDim d) {
    switch (d) {
    case WGSL_TEX_DIM_1D:
    case WGSL_TEX_DIM_STORAGE_1D:
        return 1;
    case WGSL_TEX_DIM_3D:
    case WGSL_TEX_DIM_CUBE:
    case WGSL_TEX_DIM_CUBE_ARRAY:
    case WGSL_TEX_DIM_DEPTH_CUBE:
    case WGSL_TEX_DIM_DEPTH_CUBE_ARRAY:
    case WGSL_TEX_DIM_STORAGE_3D:
        return 3;
    default:
        return 2;
    }
}

static int texture_dimensions_width(WGSLTextureDim d) {
    switch (d) {
    case WGSL_TEX_DIM_1D:
    case WGSL_TEX_DIM_STORAGE_1D:
        return 1;
    case WGSL_TEX_DIM_3D:
    case WGSL_TEX_DIM_STORAGE_3D:
        return 3;
    default:
        return 2;
    }
}

static WGSLTypeInfo *storage_texture_elem(
    WGSLTypeChecker *tc, WGSLTexelFormat f)
{
    switch (f) {
    case WGSL_TEX_FORMAT_R8UINT:
    case WGSL_TEX_FORMAT_R16UINT:
    case WGSL_TEX_FORMAT_RG8UINT:
    case WGSL_TEX_FORMAT_R32UINT:
    case WGSL_TEX_FORMAT_RG16UINT:
    case WGSL_TEX_FORMAT_RGBA8UINT:
    case WGSL_TEX_FORMAT_RG32UINT:
    case WGSL_TEX_FORMAT_RGBA16UINT:
    case WGSL_TEX_FORMAT_RGBA32UINT:
    case WGSL_TEX_FORMAT_RGB10A2UINT:
        return tc->types->t_u32;
    case WGSL_TEX_FORMAT_R8SINT:
    case WGSL_TEX_FORMAT_R16SINT:
    case WGSL_TEX_FORMAT_RG8SINT:
    case WGSL_TEX_FORMAT_R32SINT:
    case WGSL_TEX_FORMAT_RG16SINT:
    case WGSL_TEX_FORMAT_RGBA8SINT:
    case WGSL_TEX_FORMAT_RG32SINT:
    case WGSL_TEX_FORMAT_RGBA16SINT:
    case WGSL_TEX_FORMAT_RGBA32SINT:
        return tc->types->t_i32;
    default:
        return tc->types->t_f32;
    }
}

static WGSLTypeInfo *texture_value_result(
    WGSLTypeChecker *tc, const char *name, uint32_t name_len, WGSLTypeInfo *tex)
{
    if (!tex || tex->kind != WGSL_TYPE_TEXTURE) return NULL;
    WGSLTextureDim dim = (WGSLTextureDim)tex->width;
    if (builtin_name_eq(name, name_len, "textureSampleCompare") ||
        builtin_name_eq(name, name_len, "textureSampleCompareLevel"))
    {
        return tc->types->t_f32;
    }
    if (builtin_name_eq(name, name_len, "textureGatherCompare"))
    {
        return wgsl_type_vec(tc->types, 4, tc->types->t_f32);
    }
    if (builtin_name_eq(name, name_len, "textureGather")) {
        WGSLTypeInfo *elem = texture_is_storage(dim)
            ? storage_texture_elem(tc, (WGSLTexelFormat)tex->array_len)
            : (WGSLTypeInfo *)tex->ref;
        if (!elem || texture_is_depth(dim) || dim == WGSL_TEX_DIM_EXTERNAL) {
            elem = tc->types->t_f32;
        }
        return wgsl_type_vec(tc->types, 4, elem);
    }
    if (texture_is_depth(dim)) {
        return tc->types->t_f32;
    }
    WGSLTypeInfo *elem = NULL;
    if (texture_is_storage(dim)) {
        elem = storage_texture_elem(tc, (WGSLTexelFormat)tex->array_len);
    } else if (dim == WGSL_TEX_DIM_EXTERNAL) {
        elem = tc->types->t_f32;
    } else {
        elem = (WGSLTypeInfo *)tex->ref;
        if (!elem) elem = tc->types->t_f32;
    }
    return wgsl_type_vec(tc->types, 4, elem);
}

static WGSLTypeInfo *texture_builtin_result(
    WGSLTypeChecker *tc, const char *name, uint32_t name_len, WGSLTypeInfo *tex)
{
    if (builtin_name_eq(name, name_len, "textureStore")) {
        return tc->types->t_void;
    }
    if (!tex || tex->kind != WGSL_TYPE_TEXTURE) return NULL;
    WGSLTextureDim dim = (WGSLTextureDim)tex->width;
    if (builtin_name_eq(name, name_len, "textureDimensions")) {
        int w = texture_dimensions_width(dim);
        return w == 1 ? tc->types->t_u32
                      : wgsl_type_vec(tc->types, (uint8_t)w, tc->types->t_u32);
    }
    if (builtin_name_eq(name, name_len, "textureNumLayers") ||
        builtin_name_eq(name, name_len, "textureNumLevels") ||
        builtin_name_eq(name, name_len, "textureNumSamples"))
    {
        return tc->types->t_u32;
    }
    return texture_value_result(tc, name, name_len, tex);
}

static WGSLTypeInfo *texture_arg_type_for_call(
    WGSLTypeChecker *tc, WGSLNode *call,
    const char *name, uint32_t name_len, WGSLTypeInfo *arg0_type)
{
    if (builtin_name_eq(name, name_len, "textureGather") &&
        (!arg0_type || arg0_type->kind != WGSL_TYPE_TEXTURE))
    {
        return builtin_arg_type(tc, call, 1);
    }
    return arg0_type;
}

static void check_ldexp_call(
    WGSLTypeChecker *tc, WGSLNode *call,
    const char *name, uint32_t name_len, WGSLTypeInfo *arg0_type)
{
    if (!expect_arg_count(tc, call, name, name_len, 2, 2)) return;
    WGSLTypeInfo *x = arg0_type;
    WGSLTypeInfo *e = builtin_arg_type(tc, call, 1);
    WGSLTypeInfo *xe = shape_elem(x);
    WGSLTypeInfo *ee = shape_elem(e);
    int xw = shape_width(x);
    int ew = shape_width(e);
    if (!xe || !wgsl_type_is_float(xe) || xw == 0) {
        char got[96];
        wgsl_tc_error(tc, call,
            "'%.*s' first argument has type %s; expected a floating-point "
            "scalar or vector",
            (int)name_len, name, type_label(x, got, sizeof got));
        return;
    }
    if (!ee || !(ee == tc->types->t_abstract_int || ee == tc->types->t_i32) ||
        ew != xw)
    {
        char got[96];
        wgsl_tc_error(tc, call,
            "'%.*s' exponent argument has type %s; expected %s",
            (int)name_len, name, type_label(e, got, sizeof got),
            xw == 1 ? "AbstractInt or i32"
                    : "vecN<AbstractInt> or vecN<i32> matching the first argument");
        return;
    }

    int64_t exps[4] = {0};
    int max_exp = xe == tc->types->t_f16 ? 16 :
                  xe == tc->types->t_f32 ? 128 : 1024;
    if (collect_const_int_components(tc, call->children[2], xw, exps)) {
        for (int i = 0; i < xw; i++) {
            if (exps[i] > max_exp) {
                wgsl_tc_error(tc, call,
                    "builtin 'ldexp' domain error: exponent too large");
                return;
            }
        }
    }
}

static void check_mix_call(
    WGSLTypeChecker *tc, WGSLNode *call,
    const char *name, uint32_t name_len, WGSLTypeInfo *resolved_shape)
{
    if (!expect_arg_count(tc, call, name, name_len, 3, 3)) return;
    WGSLTypeInfo *shape = resolved_shape ? resolved_shape : builtin_arg_type(tc, call, 0);
    WGSLTypeInfo *factor = builtin_arg_type(tc, call, 2);
    WGSLTypeInfo *elem = shape_elem(shape);
    if (!elem || !wgsl_type_is_float(elem)) return;

    WGSLTypeInfo *same_shape = shape;
    WGSLTypeInfo *scalar_shape = elem;
    if (type_converts_to(factor, same_shape)) return;
    if (shape && shape->kind == WGSL_TYPE_VEC && type_converts_to(factor, scalar_shape)) {
        return;
    }

    char got[96], want[96];
    wgsl_tc_error(tc, call,
        "'%.*s' mix factor has type %s; expected %s%s",
        (int)name_len, name,
        type_label(factor, got, sizeof got),
        type_label(same_shape, want, sizeof want),
        shape && shape->kind == WGSL_TYPE_VEC
            ? " or the matching floating-point scalar" : "");
}

static uint32_t const_value_width_tc(const WGSLValue *v) {
    return v && v->kind == WGSL_VAL_VEC ? v->u.agg.count : 1;
}

static const WGSLValue *const_value_component_tc(const WGSLValue *v, uint32_t i) {
    if (!v) return NULL;
    if (v->kind == WGSL_VAL_VEC) {
        return i < v->u.agg.count ? &v->u.agg.elems[i] : NULL;
    }
    return i == 0 ? v : NULL;
}

static int const_value_gt_tc(
    WGSLTypeChecker *tc, const WGSLValue *a, const WGSLValue *b, int *out)
{
    if (!a || !b || !out) return 0;
    if (a->kind == WGSL_VAL_INT && b->kind == WGSL_VAL_INT) {
        if (a->type == tc->types->t_u32 || b->type == tc->types->t_u32) {
            *out = (uint32_t)a->u.i > (uint32_t)b->u.i;
        } else {
            *out = a->u.i > b->u.i;
        }
        return 1;
    }
    if (a->kind == WGSL_VAL_FLOAT && b->kind == WGSL_VAL_FLOAT) {
        *out = a->u.f > b->u.f;
        return 1;
    }
    return 0;
}

static int const_value_eq_tc(const WGSLValue *a, const WGSLValue *b, int *out)
{
    if (!a || !b || !out) return 0;
    if (a->kind == WGSL_VAL_INT && b->kind == WGSL_VAL_INT) {
        *out = a->u.i == b->u.i;
        return 1;
    }
    if (a->kind == WGSL_VAL_FLOAT && b->kind == WGSL_VAL_FLOAT) {
        *out = a->u.f == b->u.f;
        return 1;
    }
    return 0;
}

static void check_clamp_low_high_call(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len)
{
    if (!builtin_name_eq(name, name_len, "clamp") || builtin_arg_count(call) != 3) {
        return;
    }
    WGSLValue lo = {0};
    WGSLValue hi = {0};
    if (try_consteval_expr_soft(tc, call->children[2], &lo) != 1 ||
        try_consteval_expr_soft(tc, call->children[3], &hi) != 1)
    {
        return;
    }
    uint32_t width = const_value_width_tc(&lo);
    if (const_value_width_tc(&hi) != width) return;
    for (uint32_t i = 0; i < width; i++) {
        int gt = 0;
        if (const_value_gt_tc(
                tc, const_value_component_tc(&lo, i),
                const_value_component_tc(&hi, i), &gt) && gt)
        {
            wgsl_tc_error(tc, call,
                "builtin 'clamp' domain error: low must be <= high");
            return;
        }
    }
}

static void check_smoothstep_edges_call(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len)
{
    if (!builtin_name_eq(name, name_len, "smoothstep") ||
        builtin_arg_count(call) != 3)
    {
        return;
    }
    WGSLValue edge0 = {0};
    WGSLValue edge1 = {0};
    if (try_consteval_expr_soft(tc, call->children[1], &edge0) != 1 ||
        try_consteval_expr_soft(tc, call->children[2], &edge1) != 1)
    {
        return;
    }
    uint32_t width = const_value_width_tc(&edge0);
    if (const_value_width_tc(&edge1) != width) return;
    for (uint32_t i = 0; i < width; i++) {
        int eq = 0;
        if (const_value_eq_tc(
                const_value_component_tc(&edge0, i),
                const_value_component_tc(&edge1, i), &eq) && eq)
        {
            wgsl_tc_error(tc, call,
                "builtin 'smoothstep' domain error: edge0 and edge1 must differ");
            return;
        }
    }
}

