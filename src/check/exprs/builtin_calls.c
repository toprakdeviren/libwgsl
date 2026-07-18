/**
 * @file builtin_calls.c — expression typing for predeclared builtins.
 *
 * The def-driven generated table
 * (`resolve_generated_builtin_overload` / kBuiltinOverloads) is the single
 * owner for builtin overload resolution.  Return types come from the matched
 * overload row.  bitcast is template-driven (no BOP row).
 * texture builtins use kTextureOverloads via BS_TEXTURE.
 */
#include "internal/exprs_priv.h"

/* Type a builtin call.  The callee may be an EXPR_IDENT (`workgroupBarrier()`)
 * or an EXPR_TEMPLATED_IDENT (`bitcast<f32>(x)`). */
int type_call_builtin(
    WGSLTypeChecker *tc, WGSLNode *call,
    WGSLNode *callee, WGSLSymbol *sym)
{
    const WGSLBuiltinEntry *e = find_builtin_entry(sym->name, sym->name_len);

    /* Conservative first-arg type used as a non-NULL fallback so a missing
     * signature cannot cascade into NULL-type crashes downstream. */
    WGSLTypeInfo *fallback = NULL;
    if (call->child_count >= 2) {
        fallback = wgsl_tc_store_type_of(tc, call->children[1]);
    }
    if (!fallback) fallback = tc->types->t_abstract_float;

    if (!e) {
        /* This function is only reached for WGSL_SYM_PREDECLARED_FN callees
         * (see constructors.inc), i.e. the name IS in kPredeclaredBuiltinNames.
         * A missing kBuiltinTable row therefore means the two generated tables
         * (both from def/wgsl.def) have drifted — a build bug, not user error. */
        wgsl_tc_error(tc, call,
            "internal error: builtin '%.*s' is predeclared but has no "
            "signature table entry (def/wgsl.def registration drift)",
            (int)sym->name_len, sym->name);
        return wgsl_tc_set_type(tc, call, fallback, 0);
    }

    /* §4.1.1 / §17.12 / §17.13: subgroup and quad builtin functions
     * require `enable subgroups;`. */
    if (sym->name_len >= 8 && memcmp(sym->name, "subgroup", 8) == 0) {
        if (!(tc->enabled_extensions & WGSL_EXT_SUBGROUPS)) {
            wgsl_tc_error(tc, call,
                "use of '%.*s' requires 'enable subgroups;'",
                (int)sym->name_len, sym->name);
        }
    }
    if (sym->name_len >= 4 && memcmp(sym->name, "quad", 4) == 0) {
        if (!(tc->enabled_extensions & WGSL_EXT_SUBGROUPS)) {
            wgsl_tc_error(tc, call,
                "use of '%.*s' requires 'enable subgroups;'",
                (int)sym->name_len, sym->name);
        }
    }

    WGSLTypeInfo *arg0_type = NULL;
    if (call->child_count >= 2) {
        WGSLNode *arg0 = call->children[1];
        arg0_type = wgsl_tc_store_type_of(tc, arg0);
    }

    if ((WGSLBuiltinSpecial)e->special == BS_TEXTURE) {
        WGSLTypeInfo *texture_result = NULL;
        int tex_match = resolve_generated_texture_overload(
            tc, call, sym->name, sym->name_len, &texture_result);
        int gather_component_form =
            builtin_name_eq(sym->name, sym->name_len, "textureGather") &&
            (!arg0_type || arg0_type->kind != WGSL_TYPE_TEXTURE);
        WGSLTypeInfo *tex_arg = texture_arg_type_for_call(
            tc, call, sym->name, sym->name_len, arg0_type);
        check_texture_call(
            tc, call, sym->name, sym->name_len, tex_arg,
            gather_component_form);
        if (tex_match < 0) {
            wgsl_tc_error(tc, call, "no matching overload for '%.*s'",
                          (int)sym->name_len, sym->name);
            WGSLTypeInfo *diagnostic_result = texture_builtin_result(
                tc, sym->name, sym->name_len, tex_arg);
            return wgsl_tc_set_type(
                tc, call, diagnostic_result ? diagnostic_result : fallback, 0);
        }
        return wgsl_tc_set_type(
            tc, call, texture_result ? texture_result : fallback, 0);
    }

    /* §17.9 / §17.10 — pack* / unpack* arg shape. */
    if ((sym->name_len >= 4 && memcmp(sym->name, "pack", 4) == 0) ||
        (sym->name_len >= 6 && memcmp(sym->name, "unpack", 6) == 0))
    {
        check_pack_unpack_arg(tc, call, sym->name, sym->name_len, arg0_type);
    }

    /* §17.8 — atomic* arg shapes. */
    if (sym->name_len >= 6 && memcmp(sym->name, "atomic", 6) == 0) {
        check_atomic_call(tc, call, sym->name, sym->name_len, arg0_type);
    }

    /* §17.2 bitcast<T> — template-driven; no generated BOP row. */
    if ((WGSLBuiltinReturn)e->ret == BR_TEMPLATE) {
        if (callee && callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
            callee->child_count >= 2)
        {
            WGSLTypeInfo *t = wgsl_tc_resolve_type_spec(tc, callee->children[1]);
            if (t) {
                const WGSLTypeInfo *scalar = (t->kind == WGSL_TYPE_VEC)
                    ? (const WGSLTypeInfo *)t->ref : t;
                int ok =
                    scalar == tc->types->t_i32 ||
                    scalar == tc->types->t_u32 ||
                    scalar == tc->types->t_f32 ||
                    scalar == tc->types->t_f16;
                if (!ok) {
                    wgsl_tc_error(tc, callee->children[1],
                        "bitcast<T> requires T to be a concrete numeric "
                        "scalar or vector (i32 / u32 / f32 / f16, "
                        "scalar or vecN); got %s",
                        wgsl_type_kind_name((WGSLTypeKind)t->kind));
                }
                int target_bits = bitcast_type_bits(t);
                int source_bits = bitcast_type_bits(arg0_type);
                if (target_bits == 0) {
                    char got[96];
                    wgsl_tc_error(tc, callee->children[1],
                        "bitcast<T> target type %s has no valid 32-bit "
                        "word representation",
                        type_label(t, got, sizeof got));
                }
                if (source_bits == 0) {
                    char got[96];
                    wgsl_tc_error(tc, call->child_count >= 2 ? call->children[1] : call,
                        "bitcast source type must be a concrete numeric "
                        "scalar or vector with a 32-bit word representation; got %s",
                        type_label(arg0_type, got, sizeof got));
                } else if (target_bits != 0 && target_bits != source_bits) {
                    char from[96], to[96];
                    wgsl_tc_error(tc, call,
                        "bitcast source type %s (%d bits) and target type "
                        "%s (%d bits) must have the same bit width",
                        type_label(arg0_type, from, sizeof from), source_bits,
                        type_label(t, to, sizeof to), target_bits);
                }
                check_const_builtin_call(tc, call, e);
                return wgsl_tc_set_type(tc, call, t, 0);
            }
        }
        check_const_builtin_call(tc, call, e);
        return wgsl_tc_set_type(tc, call, fallback, 0);
    }

    /* Primary overload path: def-generated pattern rows. */
    WGSLTypeInfo *generated_result = NULL;
    if (resolve_generated_builtin_overload(
            tc, call, sym->name, sym->name_len, arg0_type, fallback,
            &generated_result))
    {
        check_clamp_low_high_call(tc, call, sym->name, sym->name_len);
        check_smoothstep_edges_call(tc, call, sym->name, sym->name_len);
        if (builtin_name_eq(sym->name, sym->name_len, "ldexp")) {
            check_ldexp_call(tc, call, sym->name, sym->name_len,
                             generated_result ? generated_result : arg0_type);
        }
        if ((WGSLBuiltinSpecial)e->special == BS_MIX) {
            check_mix_call(tc, call, sym->name, sym->name_len,
                           generated_result ? generated_result : arg0_type);
        }
        check_const_builtin_call(tc, call, e);
        return wgsl_tc_set_type(tc, call,
            generated_result ? generated_result : fallback, 0);
    }

    /* No overload index and not bitcast/texture — table drift. */
    wgsl_tc_error(tc, call,
        "internal error: builtin '%.*s' has no generated overload rows",
        (int)sym->name_len, sym->name);
    return wgsl_tc_set_type(tc, call, fallback, 0);
}
