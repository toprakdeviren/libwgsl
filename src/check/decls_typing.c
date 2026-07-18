/**
 * @file decls_typing.c — const/let/var/override/alias/struct decl typing
 */
#include "internal/check_priv.h"

#include <stdlib.h>
#include <string.h>

/* Decl typing. */

typedef struct {
    const char *name;
    uint32_t    len;
    uint8_t     used;
} MemberNameSlot;

static uint64_t member_name_hash(const char *name, uint32_t len) {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t i = 0; i < len; i++) {
        h ^= (uint8_t)name[i];
        h *= 1099511628211ull;
    }
    h ^= h >> 33; h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ull;
    h ^= h >> 33;
    return h;
}

static size_t member_name_cap(uint32_t n) {
    size_t need = (size_t)n * 2u;
    size_t cap = 16u;
    while (cap < need) {
        if (cap > ((size_t)-1) / 2u) return 0;
        cap *= 2u;
    }
    return cap;
}

static int member_name_seen_or_insert(
    MemberNameSlot *tab, size_t cap, const char *name, uint32_t len)
{
    size_t mask = cap - 1u;
    size_t slot = (size_t)member_name_hash(name, len) & mask;
    while (tab[slot].used) {
        if (tab[slot].len == len &&
            memcmp(tab[slot].name, name, len) == 0)
        {
            return 1;
        }
        slot = (slot + 1u) & mask;
    }
    tab[slot].used = 1;
    tab[slot].name = name;
    tab[slot].len = len;
    return 0;
}

int wgsl_tc_type_decl_const_or_let(WGSLTypeChecker *tc, WGSLNode *n) {
    int has_type = wgsl_letlike_has_type(n);
    WGSLNode *type_spec = has_type ? n->children[0] : NULL;
    WGSLNode *init      = (n->child_count >= (uint32_t)(has_type ? 2 : 1))
                            ? n->children[has_type ? 1 : 0] : NULL;

    WGSLTypeInfo *declared = type_spec ? wgsl_tc_resolve_type_spec(tc, type_spec) : NULL;
    if (init) {
        /* §4.3 — thread declared type into the initializer so abstract
         * literals / constructors materialize in context. */
        wgsl_tc_type_expr_exp(tc, init, declared);
        WGSLTypeInfo *itype = wgsl_tc_store_type_of(tc, init);
        /* §6.2.8 — an expression must not evaluate to an atomic type.
         * Atomics can only be accessed through atomic builtins (which
         * take ptr<…,atomic<T>,…> and return T, never atomic<T>). */
        if (itype && itype->kind == WGSL_TYPE_ATOMIC) {
            wgsl_tc_error(tc, init,
                "expression must not evaluate to an atomic type");
        }
        if (declared && itype && wgsl_tc_init_type_mismatch(itype, declared)) {
            wgsl_tc_error(tc, init, "initializer type %s cannot be converted to declared type %s",
                     wgsl_type_kind_name((WGSLTypeKind)itype->kind),
                     wgsl_type_kind_name((WGSLTypeKind)declared->kind));
        }
        if (!declared && itype) {
            declared = (n->kind == WGSL_NODE_DECL_CONST)
                ? itype
                : wgsl_type_concretize(tc->types, itype);
        }
        /* `const` usually runs through the consteval module pass first.
         * If consteval could resolve the declared target, the cached
         * value already has that type and we skip to avoid duplicate
         * diagnostics.  When consteval intentionally leaves a typed const
         * unmaterialized (for example a user alias target it cannot
         * resolve yet), the type checker owns the missing range check. */
        if (n->kind == WGSL_NODE_DECL_LET) {
            wgsl_tc_check_init_concretization_range(tc, init, declared);
        } else if (n->kind == WGSL_NODE_DECL_CONST && declared) {
            WGSLSymbol *sym = wgsl_tc_find_decl_symbol(tc, n);
            const WGSLValue *bound =
                (tc->cev && sym) ? wgsl_consteval_value_of(tc->cev, sym) : NULL;
            if (bound && bound->type != declared) {
                wgsl_tc_check_init_concretization_range(tc, init, declared);
            }
        }
    }
    WGSLSymbol *sym = wgsl_tc_find_decl_symbol(tc, n);
    if (sym && declared) sym->type = declared;

    /* §7.2.3 — a `let`-declaration's effective value type must be a
     * concrete constructible type, a concrete pointer type, or (when
     * the extension is enabled) a texture / sampler type.  The atomic
     * case already fires in the init-type check above; this gate
     * catches `let x: array<f32>` and similar non-constructible kinds
     * even when there's no init type mismatch. */
    if (n->kind == WGSL_NODE_DECL_LET && declared) {
        int ok = (declared->kind == WGSL_TYPE_PTR) ||
                 (declared->kind == WGSL_TYPE_TEXTURE) ||
                 (declared->kind == WGSL_TYPE_SAMPLER) ||
                 (wgsl_type_is_concrete(declared) &&
                  wgsl_tc_is_constructible_deep(tc, declared));
        if (!ok) {
            wgsl_tc_error(tc, n,
                "'let' type must be a concrete constructible type or a "
                "concrete pointer type (got %s)",
                wgsl_type_kind_name((WGSLTypeKind)declared->kind));
        }
    }

    /* §4.1.2 texture_and_sampler_let: `let` with a texture or sampler
     * type requires implementation support for the language extension.
     * `requires` only documents that dependency; it does not enable the
     * feature.  `const` is excluded since textures/samplers can never be
     * const-evaluated. */
    if (n->kind == WGSL_NODE_DECL_LET && declared &&
        (declared->kind == WGSL_TYPE_TEXTURE || declared->kind == WGSL_TYPE_SAMPLER))
    {
        if (!(tc->supported_features & WGSL_FEAT_TEXTURE_AND_SAMPLER_LET)) {
            wgsl_tc_error(tc, n,
                "'let' with a texture or sampler type requires support for "
                "language feature 'texture_and_sampler_let'");
        }
    }
    return 1;
}

int wgsl_tc_type_decl_var(WGSLTypeChecker *tc, WGSLNode *n) {
    /* children: [attrs (count A)][tpl_args (count T)][type?][init?]
     * payload[1] = A | T<<32; payload[2] = has_type | has_init<<32. */
    uint32_t A = wgsl_varlike_attr_count(n);
    uint32_t T = wgsl_varlike_tpl_count(n);
    int has_type = wgsl_varlike_has_type(n);
    int has_init = wgsl_varlike_has_init(n);

    if (!has_type && !has_init) {
        wgsl_tc_error(tc, n, "variable declaration must specify a type or an initializer");
    }
    if (T > 2) {
        wgsl_tc_error(tc, n,
            "var template list accepts at most address space and access mode");
    }

    /* §14.3 address space and access mode from template args. */
    WGSLAddressSpace as = WGSL_AS_NONE;
    WGSLAccessMode  am = WGSL_ACCESS_NONE;
    if (T >= 1 && (A + 0) < n->child_count) {
        as = wgsl_tc_resolve_addr_space_arg(tc, n->children[A + 0]);
    }
    if (T >= 2 && (A + 1) < n->child_count) {
        am = wgsl_tc_resolve_access_mode_arg(tc, n->children[A + 1]);
    }

    uint32_t i = A + T;
    WGSLTypeInfo *declared = NULL;
    if (has_type && i < n->child_count) {
        declared = wgsl_tc_resolve_type_spec(tc, n->children[i++]);
    }
    if (as == WGSL_AS_NONE) {
        if (declared &&
            (declared->kind == WGSL_TYPE_TEXTURE ||
             declared->kind == WGSL_TYPE_SAMPLER))
        {
            as = WGSL_AS_HANDLE;
        } else {
            as = tc->in_function ? WGSL_AS_FUNCTION : WGSL_AS_PRIVATE;
        }
    }
    if (am == WGSL_ACCESS_NONE) {
        if (as == WGSL_AS_FUNCTION || as == WGSL_AS_PRIVATE ||
            as == WGSL_AS_WORKGROUP)
        {
            am = WGSL_ACCESS_READ_WRITE;
        } else if (as == WGSL_AS_UNIFORM || as == WGSL_AS_STORAGE ||
                   as == WGSL_AS_HANDLE || as == WGSL_AS_PUSH_CONSTANT)
        {
            am = WGSL_ACCESS_READ;
        }
    }
    if (has_init && i < n->child_count) {
        WGSLNode *init = n->children[i++];
        wgsl_tc_type_expr_exp(tc, init, declared); /* §4.3 */
        WGSLTypeInfo *itype = wgsl_tc_store_type_of(tc, init);
        /* §6.2.8: atomic-valued initializer is illegal. */
        if (itype && itype->kind == WGSL_TYPE_ATOMIC) {
            wgsl_tc_error(tc, init,
                "expression must not evaluate to an atomic type");
        }
        if (declared && itype && wgsl_tc_init_type_mismatch(itype, declared)) {
            wgsl_tc_error(tc, init, "initializer type %s cannot be converted to declared type %s",
                     wgsl_type_kind_name((WGSLTypeKind)itype->kind),
                     wgsl_type_kind_name((WGSLTypeKind)declared->kind));
        }
        if (!declared) {
            if (itype) declared = wgsl_type_concretize(tc->types, itype);
        }
        /* §6.1.2: range-check abstract → concrete narrowing. */
        wgsl_tc_check_init_concretization_range(tc, init, declared);
        /* §7.3 — module-scope `var<private>` initializer must be a
         * const- or override-expression.  fn-scope `var<function>`
         * initializers are unrestricted (any value-expression). */
        if (as == WGSL_AS_PRIVATE && !wgsl_tc_is_override_expr(init)) {
            wgsl_tc_error(tc, init,
                "var<private> initializer must be a const- or "
                "override-expression");
        }
    }
    WGSLSymbol *sym = wgsl_tc_find_decl_symbol(tc, n);
    if (sym) {
        if (declared) sym->type = declared;
        sym->as = (uint8_t)as;
        sym->am = (uint8_t)am;
    }

    /* §7.3: Access mode MUST NOT appear except for `storage`. */
    if (T >= 2 && as != WGSL_AS_STORAGE) {
        wgsl_tc_error(tc, n, "access mode can only be specified for 'storage' address space");
    }

    /* §7.3: var<storage, write> is forbidden — only read or read_write. */
    if (as == WGSL_AS_STORAGE && am == WGSL_ACCESS_WRITE) {
        wgsl_tc_error(tc, n, "storage variables do not support 'write' access mode; use 'read' or 'read_write'");
    }

    /* §7.3: Initializers are disallowed for non-private module variables. */
    if (has_init &&
        (as == WGSL_AS_WORKGROUP || as == WGSL_AS_UNIFORM ||
         as == WGSL_AS_STORAGE || as == WGSL_AS_HANDLE))
    {
        wgsl_tc_error(tc, n, "variables in '%s' address space cannot have an initializer",
                 as == WGSL_AS_WORKGROUP ? "workgroup" :
                 as == WGSL_AS_UNIFORM   ? "uniform"   :
                 as == WGSL_AS_STORAGE   ? "storage"   : "handle");
    }

    /* §7.3: Scope placement — function ↔ fn-scope, everything else ↔ module. */
    if (as == WGSL_AS_FUNCTION && !tc->in_function) {
        wgsl_tc_error(tc, n, "var<function> can only appear inside a function body");
    }
    if ((as == WGSL_AS_PRIVATE || as == WGSL_AS_WORKGROUP ||
         as == WGSL_AS_UNIFORM || as == WGSL_AS_STORAGE) && tc->in_function)
    {
        wgsl_tc_error(tc, n, "var<%s> must be at module scope",
                 as == WGSL_AS_PRIVATE   ? "private"   :
                 as == WGSL_AS_WORKGROUP ? "workgroup" :
                 as == WGSL_AS_UNIFORM   ? "uniform"   : "storage");
    }

    /* §7.3: texture / sampler vars are module-scope only (handle AS). */
    if (declared && tc->in_function &&
        (declared->kind == WGSL_TYPE_TEXTURE || declared->kind == WGSL_TYPE_SAMPLER))
    {
        wgsl_tc_error(tc, n, "texture and sampler variables must be at module scope");
    }

    /* §6.2.8 / §14.3 — atomic types may only appear in workgroup or
     * storage address space (atomic is meaningful only with shared
     * memory).  This applies through arrays / structs too. */
    int has_atomic_store_type =
        declared && decl_type_contains_atomic(tc, declared);
    if (has_atomic_store_type &&
        as != WGSL_AS_WORKGROUP && as != WGSL_AS_STORAGE)
    {
        wgsl_tc_error(tc, n,
            "atomic store type requires the workgroup or storage "
            "address space");
    }
    if (has_atomic_store_type &&
        as == WGSL_AS_STORAGE && am != WGSL_ACCESS_READ_WRITE)
    {
        wgsl_tc_error(tc, n,
            "atomic store type in storage address space requires "
            "read_write access mode");
    }
    if (declared && as != WGSL_AS_STORAGE &&
        decl_type_contains_runtime_array(tc, declared))
    {
        wgsl_tc_error(tc, n,
            "runtime-sized array store type requires the storage "
            "address space");
    }
    if (declared && as != WGSL_AS_WORKGROUP &&
        decl_type_contains_override_array(tc, declared))
    {
        wgsl_tc_error(tc, n,
            "override-sized array store type requires the workgroup "
            "address space");
    }

    /* §7.3 store-type per AS row.
     *   - private / function: store type must be storable.
     *   - workgroup:          store type must be storable AND have a
     *                         generation-fixed footprint.
     *   - uniform / storage:  host-shareable (gated separately in the
     *                         validator's resource-binding pass).
     *   - handle:             texture / sampler kinds (gated by
     *                         `wgsl_tc_resolve_type_spec`).
     * Texture / sampler types are excluded from storable so they're
     * naturally rejected for private / function / workgroup. */
    if (declared) {
        if ((as == WGSL_AS_PRIVATE || as == WGSL_AS_FUNCTION ||
             as == WGSL_AS_WORKGROUP) &&
            !wgsl_type_is_storable(declared))
        {
            wgsl_tc_error(tc, n,
                "var<%s> store type must be storable (got %s)",
                as == WGSL_AS_PRIVATE   ? "private"   :
                as == WGSL_AS_FUNCTION  ? "function"  : "workgroup",
                wgsl_type_kind_name((WGSLTypeKind)declared->kind));
        }
        /* §7.3 workgroup additionally requires a fixed footprint
         * (generation-fixed OR override-sized array).  Override-sized
         * identity is encoded on the array type; the deep workgroup gate is
         * handled where checker context is available. */
    }

    /* §12.2 / §12.7 — `@group` and `@binding` apply only to resource
     * variables (uniform / storage / texture / sampler).  Reject them
     * on private / workgroup / function vars.  Texture/sampler types
     * always live in the handle AS, so they're inherently resources. */
    {
        int is_resource =
            (as == WGSL_AS_UNIFORM) || (as == WGSL_AS_STORAGE) ||
            (as == WGSL_AS_HANDLE)  || (as == WGSL_AS_PUSH_CONSTANT) ||
            (declared && (declared->kind == WGSL_TYPE_TEXTURE ||
                          declared->kind == WGSL_TYPE_SAMPLER));
        if (!is_resource) {
            for (uint32_t k = 0; k < A; k++) {
                WGSLNode *attr = n->children[k];
                if (!attr || attr->kind != WGSL_NODE_ATTRIBUTE) continue;
                if (attr->child_count == 0) continue;
                WGSLNode *attr_name = attr->children[0];
                if (!attr_name || attr_name->kind != WGSL_NODE_EXPR_IDENT) continue;
                uint32_t off = wgsl_node_name_span(attr_name).offset;
                uint32_t nl  = wgsl_node_name_span(attr_name).length;
                const char *txt = tc->src->bytes + off;
                if ((nl == 5 && memcmp(txt, "group",   5) == 0) ||
                    (nl == 7 && memcmp(txt, "binding", 7) == 0))
                {
                    wgsl_tc_error(tc, attr,
                        "@%.*s is only allowed on resource variables "
                        "(uniform / storage / texture / sampler)",
                        (int)nl, txt);
                }
            }
        }
    }

    return 1;
}

int wgsl_tc_type_decl_override(WGSLTypeChecker *tc, WGSLNode *n) {
    /* children: [attrs (count A)][type?][init?]
     * payload[1] = A; payload[2] = has_type | has_init<<32. */
    uint32_t A = wgsl_varlike_attr_count(n);
    int has_type = wgsl_varlike_has_type(n);
    int has_init = wgsl_varlike_has_init(n);

    if (!has_type && !has_init) {
        wgsl_tc_error(tc, n,
            "override declaration must specify a type or an initializer");
    }

    uint32_t i = A;
    WGSLTypeInfo *declared = NULL;
    if (has_type && i < n->child_count) {
        declared = wgsl_tc_resolve_type_spec(tc, n->children[i++]);
    }
    if (has_init && i < n->child_count) {
        WGSLNode *init = n->children[i++];
        wgsl_tc_type_expr_exp(tc, init, declared); /* §4.3 */
        WGSLTypeInfo *itype = wgsl_tc_store_type_of(tc, init);
        /* §6.2.8: atomic-valued initializer is illegal. */
        if (itype && itype->kind == WGSL_TYPE_ATOMIC) {
            wgsl_tc_error(tc, init,
                "expression must not evaluate to an atomic type");
        }
        if (declared && itype && wgsl_tc_init_type_mismatch(itype, declared)) {
            wgsl_tc_error(tc, init, "initializer type %s cannot be converted to declared type %s",
                     wgsl_type_kind_name((WGSLTypeKind)itype->kind),
                     wgsl_type_kind_name((WGSLTypeKind)declared->kind));
        }
        if (!declared) {
            if (itype) declared = wgsl_type_concretize(tc->types, itype);
        }
        /* §6.1.2: range-check abstract → concrete narrowing of an
         * override-expression initializer. */
        wgsl_tc_check_init_concretization_range(tc, init, declared);
        /* §7.2.2: the initializer must be an override-expression —
         * literals, parens, unary / binary ops, and references to
         * `const` or `override` symbols.  References to `let`, `var`,
         * function calls (incl. value constructors today), or fn
         * parameters are rejected. */
        if (!wgsl_tc_is_override_expr(init)) {
            wgsl_tc_error(tc, init,
                "override initializer must be an override-expression "
                "(literals, parens, unary / binary ops, and references "
                "to other 'const' or 'override' values)");
        }
    }
    WGSLSymbol *sym = wgsl_tc_find_decl_symbol(tc, n);
    if (sym && declared) sym->type = declared;

    /* §7.2.2: Effective-value-type of override must be concrete scalar
     * (bool, i32, u32, f32, f16). */
    if (declared && !wgsl_type_is_scalar(declared)) {
        wgsl_tc_error(tc, n, "override type must be a scalar (bool, i32, u32, f32, or f16)");
    }
    if (declared && wgsl_type_is_abstract(declared)) {
        wgsl_tc_error(tc, n, "override type must be concrete, not abstract");
    }

    /* §7.2.2: @id(N) range check + uniqueness across overrides. */
    for (uint32_t k = 0; k < A; k++) {
        WGSLNode *attr = n->children[k];
        if (!attr || attr->kind != WGSL_NODE_ATTRIBUTE) continue;
        if (attr->child_count < 2) continue;
        WGSLNode *attr_name = attr->children[0];
        if (!attr_name || attr_name->kind != WGSL_NODE_EXPR_IDENT) continue;
        uint32_t off = wgsl_node_name_span(attr_name).offset;
        uint32_t nl  = wgsl_node_name_span(attr_name).length;
        if (!(nl == 2 && memcmp(tc->src->bytes + off, "id", 2) == 0)) continue;

        WGSLNode *id_val = attr->children[1];
        if (!id_val) continue;

        /* The previous version pulled offset/length out of `payload[0]`
         * — that's the IDENT encoding, not the LITERAL_INT encoding
         * (LITERAL_INT's payload[0] carries only the numeric suffix
         * bits).  Drive the value through `wgsl_consteval_expr` so any
         * const-expression form (literal, unary minus, paren, named
         * const) lands the same code path. */
        int64_t id_num = -1;
        int     have_id = 0;
        if (tc->cev) {
            WGSLValue v = {0};
            size_t saved_count = tc->diag->count;
            int    saved_errs  = tc->diag->error_count;
            int    saved_cev   = tc->cev->store.had_error;
            if (wgsl_consteval_expr(tc->cev, id_val, &v) &&
                v.kind == WGSL_VAL_INT)
            {
                id_num  = v.u.i;
                have_id = 1;
            } else {
                /* Not const-evaluable — drop the "not a constant" diag
                 * to avoid noise; a separate spec rule (id must be a
                 * const-expression) lives in §12.8 and isn't this
                 * bullet's territory. */
                tc->diag->count       = saved_count;
                tc->diag->error_count = saved_errs;
                tc->cev->store.had_error    = saved_cev;
            }
        }
        if (!have_id) continue;

        if (id_num < 0 || id_num > 65535) {
            wgsl_tc_error(tc, id_val, "@id value must be in [0, 65535]");
            continue;
        }
        /* Uniqueness — heap-grown set on the type checker. */
        int dup = 0;
        for (int s = 0; s < tc->override_id_count; s++) {
            if (tc->override_ids[s] == (int32_t)id_num) { dup = 1; break; }
        }
        if (dup) {
            wgsl_tc_error(tc, id_val,
                "@id(%lld) must be unique across overrides",
                (long long)id_num);
            continue;
        }
        if (tc->override_id_count == tc->override_id_capacity) {
            if (tc->override_id_capacity > INT32_MAX / 2) continue;
            int cap = tc->override_id_capacity ? tc->override_id_capacity * 2 : 8;
            if ((size_t)cap > SIZE_MAX / sizeof *tc->override_ids) continue;
            int32_t *g = (int32_t *)realloc(tc->override_ids,
                                            (size_t)cap * sizeof *g);
            if (!g) continue;
            tc->override_ids = g;
            tc->override_id_capacity = cap;
        }
        tc->override_ids[tc->override_id_count++] = (int32_t)id_num;
    }

    return 1;
}

int wgsl_tc_type_decl_alias(WGSLTypeChecker *tc, WGSLNode *n) {
    if (n->child_count == 0) return 1;
    WGSLTypeInfo *t = wgsl_tc_resolve_type_spec(tc, n->children[0]);
    WGSLSymbol *sym = wgsl_tc_find_decl_symbol(tc, n);
    if (sym && t) sym->type = t;
    return 1;
}

int wgsl_tc_type_decl_struct(WGSLTypeChecker *tc, WGSLNode *n) {
    /* Bind a struct TypeInfo whose `ref` slot points at the AST node.
     * Member type-specs are walked too so any nested vec<T>/array<T,N>
     * gets its TypeInfo materialised in the type store. */
    WGSLSymbol *sym = wgsl_tc_find_decl_symbol(tc, n);
    if (sym && !sym->type) {
        WGSLTypeInfo *st = (WGSLTypeInfo *)wgsl_arena_calloc(
            tc->types->arena, 1, sizeof(WGSLTypeInfo));
        if (st) {
            st->kind = (uint16_t)WGSL_TYPE_STRUCT;
            st->ref  = n;
            sym->type = st;
        }
    }
    
    uint32_t member_count = 0;
    WGSLNode *last_m = NULL;

    /* §6.2.10 — every member name within a struct must be unique. */
    size_t member_cap = member_name_cap(n->child_count);
    MemberNameSlot *member_names = member_cap
        ? (MemberNameSlot *)calloc(member_cap, sizeof *member_names)
        : NULL;
    if (n->child_count > 0 && !member_names) {
        wgsl_tc_error(tc, n,
            "out of memory while checking struct member names");
    }

    for (uint32_t i = 0; i < n->child_count; i++) {
        WGSLNode *m = n->children[i];
        if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
        member_count++;

        /* Member-name dup check.  Member name encoding (offset|length)
         * lives in payload[0] (same as the IDENT shape). */
        uint32_t mn_off = wgsl_node_name_span(m).offset;
        uint32_t mn_len = wgsl_node_name_span(m).length;
        if (member_names && mn_len > 0) {
            const char *mn = tc->src->bytes + mn_off;
            if (member_name_seen_or_insert(
                    member_names, member_cap, mn, mn_len))
            {
                wgsl_tc_error(tc, m,
                    "struct member name '%.*s' must be unique within "
                    "the struct", (int)mn_len, mn);
            }
        }
        
        /* If we already found a runtime-sized array, it must have been the last member. */
        if (last_m) {
            uint32_t last_attrs = wgsl_member_attr_count(last_m);
            if (last_attrs < last_m->child_count) {
                WGSLTypeInfo *last_t = wgsl_tc_store_type_of(tc, last_m->children[last_attrs]);
                if (last_t && last_t->kind == WGSL_TYPE_ARRAY && last_t->array_len == 0) {
                    wgsl_tc_error(tc, last_m, "runtime-sized array must be the last member of a struct");
                }
            }
        }
        last_m = m;

        uint32_t attrs = wgsl_member_attr_count(m);
        if (attrs < m->child_count) {
            WGSLNode *tspec = m->children[attrs];
            WGSLTypeInfo *mt = wgsl_tc_resolve_type_spec(tc, tspec);
            if (mt) {
                /* Member type must be a plain type, and cannot be a pointer/reference. */
                if (mt->kind == WGSL_TYPE_SAMPLER || mt->kind == WGSL_TYPE_TEXTURE ||
                    mt->kind == WGSL_TYPE_PTR || mt->kind == WGSL_TYPE_REF) {
                    wgsl_tc_error(tc, tspec, "struct member type cannot be a %s",
                             wgsl_type_kind_name((WGSLTypeKind)mt->kind));
                }
                if (decl_type_contains_runtime_array(tc, mt) &&
                    !(mt->kind == WGSL_TYPE_ARRAY &&
                      mt->array_len == 0 && mt->flags == 0 && mt->pad_ == 0))
                {
                    wgsl_tc_error(tc, tspec,
                        "struct member type must not contain a nested "
                        "runtime-sized array");
                }
                if (decl_type_contains_override_array(tc, mt)) {
                    wgsl_tc_error(tc, tspec,
                        "struct member type must not contain an "
                        "override-sized array");
                }
            }
        }
    }

    if (member_count == 0) {
        wgsl_tc_error(tc, n, "struct must have at least one member");
    }

    free(member_names);
    return 1;
}
