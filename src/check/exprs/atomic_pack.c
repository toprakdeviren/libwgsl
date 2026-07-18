/**
 * @file atomic_pack.c — atomic + pack/unpack call checking
 */
#include "internal/exprs_priv.h"

int classify_atomic(const char *name, uint32_t len, WGSLAtomicKind *out) {
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
int atomic_ptr_info(
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

int atomic_ptr_access_ok(WGSLAddressSpace as, WGSLAccessMode am) {
    return (as == WGSL_AS_WORKGROUP || as == WGSL_AS_STORAGE) &&
           am == WGSL_ACCESS_READ_WRITE;
}

WGSLTypeInfo *atomic_pointee_T(WGSLTypeInfo *arg0) {
    WGSLTypeInfo *T = NULL;
    WGSLAddressSpace as = WGSL_AS_NONE;
    WGSLAccessMode am = WGSL_ACCESS_NONE;
    if (!atomic_ptr_info(arg0, &T, &as, &am)) return NULL;
    return atomic_ptr_access_ok(as, am) ? T : NULL;
}

const char *type_label(WGSLTypeInfo *t, char *buf, size_t cap) {
    if (!t) return "(no type)";
    if (!buf || cap == 0) return wgsl_type_kind_name((WGSLTypeKind)t->kind);
    size_t n = wgsl_type_format(t, buf, cap);
    if (n >= cap) n = cap - 1;
    buf[n] = '\0';
    return buf;
}

int bitcast_scalar_bits(const WGSLTypeInfo *t) {
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

int bitcast_type_bits(const WGSLTypeInfo *t) {
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

void check_atomic_call(
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
int check_pack_unpack_arg(
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

