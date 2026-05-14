/* Type a builtin call.  The callee may be an EXPR_IDENT (`workgroupBarrier()`)
 * or an EXPR_TEMPLATED_IDENT (`bitcast<f32>(x)`).  The matched entry's
 * return-rule decides the result type. */
static int type_call_builtin(
    WGSLTypeChecker *tc, WGSLNode *call,
    WGSLNode *callee, WGSLSymbol *sym)
{
    const WGSLBuiltinEntry *e = find_builtin_entry(sym->name, sym->name_len);

    /* Unknown builtin → fall through to a best-effort first-arg type
     * (keeps unfamiliar names from blocking the corpus). */
    WGSLTypeInfo *fallback = NULL;
    if (call->child_count >= 2) {
        fallback = wgsl_tc_store_type_of(tc, call->children[1]);
    }
    if (!fallback) fallback = tc->types->t_abstract_float;

    if (!e) return wgsl_tc_set_type(tc, call, fallback, 0);

    /* §4.1.1 / §17.12 / §17.13: subgroup and quad builtin functions
     * require `enable subgroups;`.  Gate them here so the diagnostic
     * fires at the call site (same pattern as @builtin gating in
     * validate.c). */
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

    /* §17.9 / §17.10 — pack* / unpack* arg shape (vec4<f32> /
     * vec4<i32> / vec4<u32> / vec2<f32> / u32). */
    if ((sym->name_len >= 4 && memcmp(sym->name, "pack", 4) == 0) ||
        (sym->name_len >= 6 && memcmp(sym->name, "unpack", 6) == 0))
    {
        check_pack_unpack_arg(tc, call, sym->name, sym->name_len, arg0_type);
    }

    /* §17.8 — atomic* arg shapes: ptr<…, atomic<T>, …> + value args of T. */
    if (sym->name_len >= 6 && memcmp(sym->name, "atomic", 6) == 0) {
        check_atomic_call(tc, call, sym->name, sym->name_len, arg0_type);
    }

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
        check_const_builtin_call(tc, call, e);
        return wgsl_tc_set_type(tc, call,
            generated_result ? generated_result : fallback, 0);
    }

    /* §6.1.3 — overload resolution.  When this entry advertises a param
     * family, build the candidate set parametrically and run the §6.1.3
     * algorithm.  The chosen candidate's `result` shape supplies the
     * builtin's static type (BR_ARG0 / BR_ARG0_ELEM read from it instead
     * of from the raw arg type, so e.g. `log2(32)` yields AbstractFloat
     * rather than AbstractInt). */
    WGSLTypeInfo *resolved_shape = NULL;
    if (e->fam != PF_NONE && e->arity > 0 && call->child_count >= 1 + e->arity) {
        WGSLTypeInfo *args[WGSL_OL_MAX_PARAMS] = {0};
        int arg_const[WGSL_OL_MAX_PARAMS] = {0};
        int arity = (int)e->arity;
        if (arity > WGSL_OL_MAX_PARAMS) arity = WGSL_OL_MAX_PARAMS;

        int ok = 1;
        for (int i = 0; i < arity; i++) {
            WGSLNode *arg = call->children[1 + i];
            WGSLTypeInfo *t;
            if (arg->kind == WGSL_NODE_EXPR_ADDR_OF && arg->child_count >= 1) {
                t = wgsl_tc_store_type_of(tc, arg->children[0]);
            } else {
                t = wgsl_tc_store_type_of(tc, arg);
            }
            if (!t) { ok = 0; break; }
            args[i] = t;
            arg_const[i] = is_const_expr(arg);
        }

        if (ok) {
            WGSLOLCandidate cands[WGSL_OL_MAX_CANDS];
            int nc = build_t_candidates(tc, e->fam, args, arity, cands);
            if (nc > 0) {
                int picked = resolve_overload(
                    tc, cands, nc, args, arg_const, arity, e->name, call);
                if (picked >= 0) {
                    resolved_shape = cands[picked].result;
                }
            }
        }
    }

    WGSLTypeInfo *special_result = NULL;
    switch ((WGSLBuiltinSpecial)e->special) {
    case BS_LDEXP:
        check_ldexp_call(tc, call, sym->name, sym->name_len,
                         resolved_shape ? resolved_shape : arg0_type);
        break;
    case BS_MIX:
        check_mix_call(tc, call, sym->name, sym->name_len,
                       resolved_shape ? resolved_shape : arg0_type);
        break;
    case BS_TEXTURE: {
        int gather_component_form =
            builtin_name_eq(sym->name, sym->name_len, "textureGather") &&
            (!arg0_type || arg0_type->kind != WGSL_TYPE_TEXTURE);
        WGSLTypeInfo *tex_arg = texture_arg_type_for_call(
            tc, call, sym->name, sym->name_len, arg0_type);
        check_texture_call(
            tc, call, sym->name, sym->name_len, tex_arg,
            gather_component_form);
        special_result = texture_builtin_result(
            tc, sym->name, sym->name_len, tex_arg);
        break;
    }
    case BS_NONE:
    default:
        break;
    }

    switch (e->ret) {
    case BR_VOID:
        return wgsl_tc_set_type(tc, call, tc->types->t_void, 0);
    case BR_BOOL:
        return wgsl_tc_set_type(tc, call, tc->types->t_bool, 0);
    case BR_I32:
        return wgsl_tc_set_type(tc, call, tc->types->t_i32, 0);
    case BR_U32:
        if (special_result) return wgsl_tc_set_type(tc, call, special_result, 0);
        /* §17.4 — `arrayLength(p)` requires p to be a pointer to a
         * runtime-sized array in the storage address space.  We've
         * already collected `arg0_type` above; check the shape here
         * so a `let arr: array<f32,3>; arrayLength(&arr);` gets a
         * crisp diagnostic rather than silently returning u32. */
        if (sym->name_len == 11 &&
            memcmp(sym->name, "arrayLength", 11) == 0)
        {
            if (!array_length_arg_ok(tc, call, arg0_type)) {
                wgsl_tc_error(tc, call,
                    "arrayLength requires a pointer to a runtime-sized "
                    "array in the storage address space");
            }
        }
        return wgsl_tc_set_type(tc, call, tc->types->t_u32, 0);
    case BR_ARG0: {
        WGSLTypeInfo *t = resolved_shape ? resolved_shape :
                          (arg0_type ? arg0_type : fallback);
        return wgsl_tc_set_type(tc, call, t, 0);
    }
    case BR_ARG0_ELEM: {
        WGSLTypeInfo *src = resolved_shape ? resolved_shape :
                            (arg0_type ? arg0_type : fallback);
        WGSLTypeInfo *elem = element_type_of(src);
        return wgsl_tc_set_type(tc, call, elem ? elem : src, 0);
    }
    case BR_VEC4F:
        if (special_result) return wgsl_tc_set_type(tc, call, special_result, 0);
        return wgsl_tc_set_type(tc, call,
            wgsl_type_vec(tc->types, 4, tc->types->t_f32), 0);
    case BR_VEC4U:
        return wgsl_tc_set_type(tc, call,
            wgsl_type_vec(tc->types, 4, tc->types->t_u32), 0);
    case BR_VEC4I:
        return wgsl_tc_set_type(tc, call,
            wgsl_type_vec(tc->types, 4, tc->types->t_i32), 0);
    case BR_VEC2F:
        return wgsl_tc_set_type(tc, call,
            wgsl_type_vec(tc->types, 2, tc->types->t_f32), 0);
    case BR_TEMPLATE: {
        /* `bitcast<T>(x)` — pull T from the templated-callee form.
         * §17.2 restricts T to a concrete numeric scalar (i32 / u32 /
         * f32) or a concrete numeric vector; reject anything else. */
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
                return wgsl_tc_set_type(tc, call, t, 0);
            }
        }
        return wgsl_tc_set_type(tc, call, fallback, 0);
    }
    case BR_ATOMIC_CX_RESULT: {
        if (!arg0_type) return wgsl_tc_set_type(tc, call, fallback, 0);
        WGSLTypeInfo *elem = element_type_of(arg0_type);
        if (!elem) elem = arg0_type;
        return wgsl_tc_set_type(tc, call, wgsl_type_atomic_cx_result(tc->types, elem), 0);
    }
    case BR_FREXP_RESULT: {
        /* §17 — `frexp(T)` → `__frexp_result<T>` where T is the arg's
         * post-concretization type (scalar f / vecN<f>).  `resolved_shape`
         * carries the post-overload-resolution shape; fall back to the
         * raw arg type if the overload resolver couldn't pick. */
        WGSLTypeInfo *T = resolved_shape ? resolved_shape : arg0_type;
        if (!T) return wgsl_tc_set_type(tc, call, fallback, 0);
        return wgsl_tc_set_type(tc, call,
            wgsl_type_frexp_result(tc->types, T), 0);
    }
    case BR_MODF_RESULT: {
        WGSLTypeInfo *T = resolved_shape ? resolved_shape : arg0_type;
        if (!T) return wgsl_tc_set_type(tc, call, fallback, 0);
        return wgsl_tc_set_type(tc, call,
            wgsl_type_modf_result(tc->types, T), 0);
    }
    }
    return wgsl_tc_set_type(tc, call, fallback, 0);
}

