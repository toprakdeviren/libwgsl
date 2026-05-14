/**
 * @file types.c — Phase 7 type-specifier resolution.
 *
 * Resolves an AST type-spec subtree (EXPR_IDENT / EXPR_TEMPLATED_IDENT)
 * down to a `WGSLTypeInfo *` using the type interner.  Handles all
 * predeclared types, predeclared aliases, the eight type-generators
 * (`vec*`, `mat*x*`, `atomic`, `array`, `ptr`, all texture / sampler
 * families), and override-sized array length.
 *
 * Co-extensive with §6.2 (Plain Types) + §6.5 (Texture/Sampler types)
 * + the §6.8 type-specifier grammar.
 */
#include "internal/check.h"
#include "internal/check_priv.h"

#include <stdint.h>
#include <string.h>

static WGSLTypeInfo *resolve_type_spec_impl(WGSLTypeChecker *tc, WGSLNode *spec);

static int sym_is_type_symbol(const WGSLSymbol *sym) {
    if (!sym) return 0;
    return sym->kind == WGSL_SYM_PREDECLARED_TYPE ||
           sym->kind == WGSL_SYM_PREDECLARED_TYPEGEN ||
           sym->kind == WGSL_SYM_PREDECLARED_TYPE_ALIAS ||
           sym->kind == WGSL_SYM_TYPE_ALIAS ||
           sym->kind == WGSL_SYM_STRUCT;
}

static int sym_name_eq(
    const WGSLSymbol *sym, const char *name, uint32_t name_len)
{
    return sym && sym->name_len == name_len &&
           memcmp(sym->name, name, name_len) == 0;
}

static void ident_name(
    WGSLTypeChecker *tc, const WGSLNode *ident,
    const char **name, uint32_t *name_len)
{
    if (!tc || !tc->src || !ident || !name || !name_len) return;
    uint32_t off = (uint32_t)(ident->payload[0] & 0xFFFFFFFFu);
    uint32_t len = (uint32_t)(ident->payload[0] >> 32);
    if (off > tc->src->length) return;
    *name = tc->src->bytes + off;
    *name_len = len;
}

static WGSLSymbol *find_predeclared_symbol(
    WGSLTypeChecker *tc, const char *name, uint32_t name_len)
{
    if (!tc || !tc->res || !name) return NULL;
    WGSLScope *scope = &tc->res->predeclared;
    for (size_t i = 0; i < scope->count; i++) {
        WGSLSymbol *sym = scope->items[i];
        if (sym_name_eq(sym, name, name_len)) return sym;
    }
    return NULL;
}

static WGSLSymbol *find_type_symbol_for_ident(
    WGSLTypeChecker *tc, const WGSLNode *ident)
{
    if (!tc || !ident || ident->kind != WGSL_NODE_EXPR_IDENT) return NULL;
    const char *name = NULL;
    uint32_t name_len = 0;
    ident_name(tc, ident, &name, &name_len);
    if (!name) return NULL;

    WGSLSymbol *resolved = wgsl_node_resolved_symbol(ident);
    if (sym_is_type_symbol(resolved)) return resolved;
    if (resolved && resolved->kind != WGSL_SYM_PARAM) return NULL;

    if (tc->res) {
        for (size_t i = 0; i < tc->res->all_decl_count; i++) {
            WGSLSymbol *sym = tc->res->all_decls[i];
            if ((sym->kind == WGSL_SYM_TYPE_ALIAS ||
                 sym->kind == WGSL_SYM_STRUCT) &&
                sym_name_eq(sym, name, name_len))
            {
                return sym;
            }
        }
    }

    WGSLSymbol *pred = find_predeclared_symbol(tc, name, name_len);
    return sym_is_type_symbol(pred) ? pred : NULL;
}

static WGSLTypeInfo *ensure_type_symbol_resolved(
    WGSLTypeChecker *tc, WGSLSymbol *sym)
{
    if (!sym) return NULL;
    switch ((WGSLSymKind)sym->kind) {
    case WGSL_SYM_PREDECLARED_TYPE:
    case WGSL_SYM_PREDECLARED_TYPE_ALIAS:
        return sym->type;
    case WGSL_SYM_TYPE_ALIAS:
        if (sym->type) return sym->type;
        if (sym->flags & WGSL_SYM_FLAG_TYPE_RESOLVING) return NULL;
        sym->flags |= WGSL_SYM_FLAG_TYPE_RESOLVING;
        if (sym->ast) wgsl_tc_type_decl_alias(tc, sym->ast);
        sym->flags &= (uint16_t)~WGSL_SYM_FLAG_TYPE_RESOLVING;
        return sym->type;
    case WGSL_SYM_STRUCT:
        if (sym->type) return sym->type;
        if (sym->flags & WGSL_SYM_FLAG_TYPE_RESOLVING) return NULL;
        sym->flags |= WGSL_SYM_FLAG_TYPE_RESOLVING;
        if (sym->ast) wgsl_tc_type_decl_struct(tc, sym->ast);
        sym->flags &= (uint16_t)~WGSL_SYM_FLAG_TYPE_RESOLVING;
        return sym->type;
    default:
        return NULL;
    }
}

static WGSLSymbol *find_override_symbol_in_expr(WGSLNode *expr) {
    if (!expr) return NULL;
    if (expr->kind == WGSL_NODE_EXPR_IDENT) {
        WGSLSymbol *sym = wgsl_node_resolved_symbol(expr);
        if (sym && sym->kind == WGSL_SYM_OVERRIDE) return sym;
    }
    for (uint32_t i = 0; i < expr->child_count; i++) {
        WGSLSymbol *sym = find_override_symbol_in_expr(expr->children[i]);
        if (sym) return sym;
    }
    return NULL;
}

static int type_has_override_array(const WGSLTypeInfo *t) {
    if (!t) return 0;
    if (t->kind == WGSL_TYPE_ARRAY) {
        if (t->array_len == 0 && (t->flags != 0 || t->pad_ != 0)) return 1;
        return type_has_override_array((const WGSLTypeInfo *)t->ref);
    }
    return 0;
}

static int type_contains_runtime_array_deep(
    const WGSLTypeChecker *tc, const WGSLTypeInfo *t)
{
    if (!t) return 0;
    if (t->kind == WGSL_TYPE_ARRAY) {
        if (t->array_len == 0 && t->flags == 0 && t->pad_ == 0) return 1;
        return type_contains_runtime_array_deep(
            tc, (const WGSLTypeInfo *)t->ref);
    }
    if (t->kind == WGSL_TYPE_STRUCT) {
        const WGSLNode *sd = (const WGSLNode *)t->ref;
        if (!sd) return 0;
        for (uint32_t i = 0; i < sd->child_count; i++) {
            const WGSLNode *m = sd->children[i];
            if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            uint32_t attrs = (uint32_t)(m->payload[1] & 0xFFFFFFFFu);
            if (attrs >= m->child_count) continue;
            const WGSLTypeInfo *mt =
                wgsl_tc_store_type_of(tc, m->children[attrs]);
            if (type_contains_runtime_array_deep(tc, mt)) return 1;
        }
    }
    return 0;
}

static int type_contains_atomic_deep(
    const WGSLTypeChecker *tc, const WGSLTypeInfo *t)
{
    if (!t) return 0;
    if (t->kind == WGSL_TYPE_ATOMIC) return 1;
    if (t->kind == WGSL_TYPE_ARRAY) {
        return type_contains_atomic_deep(tc, (const WGSLTypeInfo *)t->ref);
    }
    if (t->kind == WGSL_TYPE_STRUCT) {
        const WGSLNode *sd = (const WGSLNode *)t->ref;
        if (!sd) return 0;
        for (uint32_t i = 0; i < sd->child_count; i++) {
            const WGSLNode *m = sd->children[i];
            if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            uint32_t attrs = (uint32_t)(m->payload[1] & 0xFFFFFFFFu);
            if (attrs >= m->child_count) continue;
            const WGSLTypeInfo *mt =
                wgsl_tc_store_type_of(tc, m->children[attrs]);
            if (type_contains_atomic_deep(tc, mt)) return 1;
        }
    }
    return 0;
}

static WGSLSymbol *resolve_predeclared_template_arg(
    WGSLTypeChecker *tc, WGSLNode *arg, WGSLSymKind kind,
    const char *expected)
{
    if (!arg) return NULL;
    if (arg->kind != WGSL_NODE_EXPR_IDENT) {
        wgsl_tc_error(tc, arg, "expected %s", expected);
        return NULL;
    }
    const char *name = NULL;
    uint32_t name_len = 0;
    ident_name(tc, arg, &name, &name_len);
    WGSLSymbol *sym = wgsl_node_resolved_symbol(arg);
    if (sym) {
        if (sym->kind == kind) return sym;
        wgsl_tc_error(tc, arg, "expected %s", expected);
        return NULL;
    }
    if (name) {
        WGSLSymbol *pred = find_predeclared_symbol(tc, name, name_len);
        if (pred && pred->kind == kind) sym = pred;
    }
    if (!sym || sym->kind != kind) {
        wgsl_tc_error(tc, arg, "expected %s", expected);
        return NULL;
    }
    return sym;
}

/* Public wrapper: resolves AND records the result in the side-table
 * so downstream consumers (validator, module-summary JSON) can pull
 * the resolved type back out via `wgsl_typecheck_type_of(spec)`. */
WGSLTypeInfo *wgsl_tc_resolve_type_spec(WGSLTypeChecker *tc, WGSLNode *spec) {
    WGSLTypeInfo *t = resolve_type_spec_impl(tc, spec);
    if (t && spec) wgsl_tc_set_type(tc, spec, t, 0);
    return t;
}

/* Try to evaluate an array-length expression as a u32 const.  Returns
 * 0 length to indicate "use a runtime-sized array".  Sets
 * `*out_override_sym` to the override symbol when the length is an
 * `override`-decl reference (§6.2.9 override-sized arrays); the caller
 * uses that pointer to interner a distinct array type per override
 * symbol so two `array<f32, N>` and `array<f32, M>` (different
 * overrides) compare unequal. */
static uint32_t eval_array_length(
    WGSLTypeChecker *tc, WGSLNode *expr, WGSLSymbol **out_override_sym)
{
    if (out_override_sym) *out_override_sym = NULL;
    if (!expr || !tc->cev) return 0;

    /* §6.2.9 override-sized arrays: the count may be an
     * override-expression, not just a bare override identifier.  It is
     * shader-valid even when the default override value is <= 0 because
     * pipeline constants can specialize it later. */
    WGSLSymbol *override_sym = find_override_symbol_in_expr(expr);
    if (override_sym && wgsl_tc_is_override_expr(expr)) {
        if (out_override_sym) *out_override_sym = override_sym;
        return 0;
    }

    WGSLValue v = {0};
    if (!wgsl_consteval_expr(tc->cev, expr, &v)) return 0;
    if (v.kind != WGSL_VAL_INT) return 0;
    if (v.u.i <= 0) return 0;
    return (uint32_t)v.u.i;
}

/* Map a texel-format identifier (an EXPR_IDENT template arg whose
 * resolver pass landed on a PREDECLARED_TEXEL_FORMAT symbol) to its
 * enum value.  Returns WGSL_TEX_FORMAT_NONE on mismatch (with an
 * error emitted). */
static WGSLTexelFormat resolve_texel_format_arg(
    WGSLTypeChecker *tc, WGSLNode *arg)
{
    WGSLSymbol *s = resolve_predeclared_template_arg(
        tc, arg, WGSL_SYM_PREDECLARED_TEXEL_FORMAT, "a texel format");
    if (!s) return WGSL_TEX_FORMAT_NONE;
    const char *n = s->name; uint32_t l = s->name_len;
    static const struct { const char *name; WGSLTexelFormat f; int tier1; } map[] = {
        /* Core texel formats (always available). */
        { "rgba8unorm",      WGSL_TEX_FORMAT_RGBA8UNORM,      0 },
        { "rgba8snorm",      WGSL_TEX_FORMAT_RGBA8SNORM,      0 },
        { "rgba8uint",       WGSL_TEX_FORMAT_RGBA8UINT,       0 },
        { "rgba8sint",       WGSL_TEX_FORMAT_RGBA8SINT,       0 },
        { "rgba16uint",      WGSL_TEX_FORMAT_RGBA16UINT,      0 },
        { "rgba16sint",      WGSL_TEX_FORMAT_RGBA16SINT,      0 },
        { "rgba16float",     WGSL_TEX_FORMAT_RGBA16FLOAT,     0 },
        { "r32uint",         WGSL_TEX_FORMAT_R32UINT,         0 },
        { "r32sint",         WGSL_TEX_FORMAT_R32SINT,         0 },
        { "r32float",        WGSL_TEX_FORMAT_R32FLOAT,        0 },
        { "rg32uint",        WGSL_TEX_FORMAT_RG32UINT,        0 },
        { "rg32sint",        WGSL_TEX_FORMAT_RG32SINT,        0 },
        { "rg32float",       WGSL_TEX_FORMAT_RG32FLOAT,       0 },
        { "rgba32uint",      WGSL_TEX_FORMAT_RGBA32UINT,      0 },
        { "rgba32sint",      WGSL_TEX_FORMAT_RGBA32SINT,      0 },
        { "rgba32float",     WGSL_TEX_FORMAT_RGBA32FLOAT,     0 },
        { "bgra8unorm",      WGSL_TEX_FORMAT_BGRA8UNORM,      0 },
        /* texture_formats_tier1 texel formats (§6.5.10). */
        { "r8unorm",         WGSL_TEX_FORMAT_R8UNORM,         1 },
        { "r8snorm",         WGSL_TEX_FORMAT_R8SNORM,         1 },
        { "r8uint",          WGSL_TEX_FORMAT_R8UINT,          1 },
        { "r8sint",          WGSL_TEX_FORMAT_R8SINT,          1 },
        { "r16uint",         WGSL_TEX_FORMAT_R16UINT,         1 },
        { "r16sint",         WGSL_TEX_FORMAT_R16SINT,         1 },
        { "r16float",        WGSL_TEX_FORMAT_R16FLOAT,        1 },
        { "r16unorm",        WGSL_TEX_FORMAT_R16UNORM,        1 },
        { "r16snorm",        WGSL_TEX_FORMAT_R16SNORM,        1 },
        { "rg8unorm",        WGSL_TEX_FORMAT_RG8UNORM,        1 },
        { "rg8snorm",        WGSL_TEX_FORMAT_RG8SNORM,        1 },
        { "rg8uint",         WGSL_TEX_FORMAT_RG8UINT,         1 },
        { "rg8sint",         WGSL_TEX_FORMAT_RG8SINT,         1 },
        { "rg16uint",        WGSL_TEX_FORMAT_RG16UINT,        1 },
        { "rg16sint",        WGSL_TEX_FORMAT_RG16SINT,        1 },
        { "rg16float",       WGSL_TEX_FORMAT_RG16FLOAT,       1 },
        { "rg16unorm",       WGSL_TEX_FORMAT_RG16UNORM,       1 },
        { "rg16snorm",       WGSL_TEX_FORMAT_RG16SNORM,       1 },
        { "rgba16unorm",     WGSL_TEX_FORMAT_RGBA16UNORM,     1 },
        { "rgba16snorm",     WGSL_TEX_FORMAT_RGBA16SNORM,     1 },
        { "rgb10a2unorm",    WGSL_TEX_FORMAT_RGB10A2UNORM,    1 },
        { "rgb10a2uint",     WGSL_TEX_FORMAT_RGB10A2UINT,     1 },
        { "rg11b10float",    WGSL_TEX_FORMAT_RG11B10FLOAT,    1 },
        { "rg11b10ufloat",   WGSL_TEX_FORMAT_RG11B10UFLOAT,   1 },
        { "rgba8unorm_srgb", WGSL_TEX_FORMAT_RGBA8UNORM_SRGB, 1 },
        { "bgra8unorm_srgb", WGSL_TEX_FORMAT_BGRA8UNORM_SRGB, 1 },
    };
    for (size_t i = 0; i < sizeof map / sizeof map[0]; i++) {
        size_t ml = strlen(map[i].name);
        if (ml == l && memcmp(map[i].name, n, l) == 0) {
            if (map[i].tier1 && !(tc->supported_features & WGSL_FEAT_TEXTURE_FORMATS_TIER1)) {
                wgsl_tc_error(tc, arg,
                    "use of texel format '%s' requires support for "
                    "language feature 'texture_formats_tier1'",
                    map[i].name);
                return WGSL_TEX_FORMAT_NONE;
            }
            return map[i].f;
        }
    }
    wgsl_tc_error(tc, arg, "unknown texel format '%.*s'", (int)l, n);
    return WGSL_TEX_FORMAT_NONE;
}

/* Same idea for the access-mode template arg.  read / write /
 * read_write resolve to PREDECLARED_ACCESS_MODE during Phase 5. */
WGSLAccessMode wgsl_tc_resolve_access_mode_arg(
    WGSLTypeChecker *tc, WGSLNode *arg)
{
    WGSLSymbol *s = resolve_predeclared_template_arg(
        tc, arg, WGSL_SYM_PREDECLARED_ACCESS_MODE, "an access mode");
    if (!s) return WGSL_ACCESS_NONE;
    const char *n = s->name; uint32_t l = s->name_len;
    if (l == 4  && memcmp(n, "read",        4) == 0) return WGSL_ACCESS_READ;
    if (l == 5  && memcmp(n, "write",       5) == 0) return WGSL_ACCESS_WRITE;
    if (l == 10 && memcmp(n, "read_write", 10) == 0) return WGSL_ACCESS_READ_WRITE;
    wgsl_tc_error(tc, arg, "unknown access mode '%.*s'", (int)l, n);
    return WGSL_ACCESS_NONE;
}

/* Same for address space.  §14.3. */
WGSLAddressSpace wgsl_tc_resolve_addr_space_arg(
    WGSLTypeChecker *tc, WGSLNode *arg)
{
    WGSLSymbol *s = resolve_predeclared_template_arg(
        tc, arg, WGSL_SYM_PREDECLARED_ADDR_SPACE, "an address space");
    if (!s) return WGSL_AS_NONE;
    const char *n = s->name; uint32_t l = s->name_len;
    if (l == 8  && memcmp(n, "function", 8) == 0) return WGSL_AS_FUNCTION;
    if (l == 7  && memcmp(n, "private",  7) == 0) return WGSL_AS_PRIVATE;
    if (l == 9  && memcmp(n, "workgroup", 9) == 0) return WGSL_AS_WORKGROUP;
    if (l == 7  && memcmp(n, "uniform",   7) == 0) return WGSL_AS_UNIFORM;
    if (l == 7  && memcmp(n, "storage",   7) == 0) return WGSL_AS_STORAGE;
    if (l == 6  && memcmp(n, "handle",    6) == 0) return WGSL_AS_HANDLE;
    if (l == 13 && memcmp(n, "push_constant", 13) == 0) return WGSL_AS_PUSH_CONSTANT;
    if (l == 9  && memcmp(n, "immediate", 9) == 0) return WGSL_AS_IMMEDIATE;
    wgsl_tc_error(tc, arg, "unknown address space '%.*s'", (int)l, n);
    return WGSL_AS_NONE;
}

static WGSLTypeInfo *resolve_templated_type(
    WGSLTypeChecker *tc, WGSLNode *spec)
{
    /* spec is EXPR_TEMPLATED_IDENT.  children[0] is the head ident,
     * children[1..] are template args. */
    if (spec->child_count == 0) return NULL;
    WGSLNode *head = spec->children[0];
    if (!head || head->kind != WGSL_NODE_EXPR_IDENT) return NULL;
    WGSLSymbol *resolved_head = wgsl_node_resolved_symbol(head);
    if (resolved_head && !sym_is_type_symbol(resolved_head)) {
        wgsl_tc_error(tc, head,
            "'%.*s' is not a type generator",
            (int)resolved_head->name_len, resolved_head->name);
        return NULL;
    }
    WGSLSymbol *sym = find_type_symbol_for_ident(tc, head);
    if (!sym) return NULL;

    /* If the head is itself a fully-typed alias / scalar / scalar-alias,
     * the template list is bogus — Phase 8 territory.  For Iter A we
     * just return the head type. */
    if (sym->kind == WGSL_SYM_PREDECLARED_TYPE ||
        sym->kind == WGSL_SYM_PREDECLARED_TYPE_ALIAS)
    {
        return sym->type;
    }
    if (sym->kind != WGSL_SYM_PREDECLARED_TYPEGEN) {
        wgsl_tc_error(tc, spec, "'%.*s' is not a type generator",
                 (int)sym->name_len, sym->name);
        return NULL;
    }

    const char *nm = sym->name;
    uint32_t    nl = sym->name_len;

    /* vec2 / vec3 / vec4 — single elem-type template arg. */
    if (nl == 4 && memcmp(nm, "vec", 3) == 0) {
        int w = nm[3] - '0';
        if (w < 2 || w > 4) return NULL;
        if (spec->child_count < 2) {
            wgsl_tc_error(tc, spec, "vec%d requires an element type", w);
            return NULL;
        }
        WGSLTypeInfo *elem = wgsl_tc_resolve_type_spec(tc, spec->children[1]);
        if (!elem) return NULL;
        if (!wgsl_type_is_scalar(elem)) {
            wgsl_tc_error(tc, spec->children[1], "vector component type must be a scalar");
            return NULL;
        }
        if (elem == tc->types->t_f16 && !(tc->enabled_extensions & WGSL_EXT_F16)) {
            wgsl_tc_error(tc, spec->children[1], "use of 'f16' requires 'enable f16;'");
            return NULL;
        }
        return wgsl_type_vec(tc->types, (uint8_t)w, elem);
    }

    /* mat2x2 .. mat4x4 — single elem-type template arg. */
    if (nl == 6 && memcmp(nm, "mat", 3) == 0 && nm[4] == 'x') {
        int c = nm[3] - '0';
        int r = nm[5] - '0';
        if (c < 2 || c > 4 || r < 2 || r > 4) return NULL;
        if (spec->child_count < 2) {
            wgsl_tc_error(tc, spec, "mat%dx%d requires an element type", c, r);
            return NULL;
        }
        WGSLTypeInfo *elem = wgsl_tc_resolve_type_spec(tc, spec->children[1]);
        if (!elem) {
            WGSLNode *arg = spec->children[1];
            if (arg &&
                arg->kind != WGSL_NODE_EXPR_IDENT &&
                arg->kind != WGSL_NODE_EXPR_TEMPLATED_IDENT)
            {
                wgsl_tc_error(tc, arg, "array element type must be a type");
            }
            return NULL;
        }
        if (!wgsl_type_is_float(elem)) {
            wgsl_tc_error(tc, spec->children[1], "matrix element type must be a floating-point scalar");
            return NULL;
        }
        if (elem == tc->types->t_f16 && !(tc->enabled_extensions & WGSL_EXT_F16)) {
            wgsl_tc_error(tc, spec->children[1], "use of 'f16' requires 'enable f16;'");
            return NULL;
        }
        return wgsl_type_mat(tc->types, (uint8_t)c, (uint8_t)r, elem);
    }

    /* atomic<T> */
    if (nl == 6 && memcmp(nm, "atomic", 6) == 0) {
        if (spec->child_count < 2) {
            wgsl_tc_error(tc, spec, "atomic requires an element type");
            return NULL;
        }
        WGSLTypeInfo *elem = wgsl_tc_resolve_type_spec(tc, spec->children[1]);
        if (!elem) return NULL;
        if (elem != tc->types->t_i32 && elem != tc->types->t_u32) {
            wgsl_tc_error(tc, spec->children[1], "atomic element type must be i32 or u32");
            return NULL;
        }
        return wgsl_type_atomic(tc->types, elem);
    }

    /* array<T> or array<T, N> */
    if (nl == 5 && memcmp(nm, "array", 5) == 0) {
        if (spec->child_count < 2) {
            wgsl_tc_error(tc, spec, "array requires an element type");
            return NULL;
        }
        WGSLTypeInfo *elem = wgsl_tc_resolve_type_spec(tc, spec->children[1]);
        if (!elem) {
            WGSLNode *arg = spec->children[1];
            if (arg &&
                arg->kind != WGSL_NODE_EXPR_IDENT &&
                arg->kind != WGSL_NODE_EXPR_TEMPLATED_IDENT)
            {
                wgsl_tc_error(tc, arg, "array element type must be a type");
            }
            return NULL;
        }

        /* Element type must be a plain type (not pointer/texture/sampler). */
        if (elem->kind == WGSL_TYPE_SAMPLER || elem->kind == WGSL_TYPE_TEXTURE) {
            wgsl_tc_error(tc, spec->children[1], "array element type must be a plain type");
            return NULL;
        }
        if (elem->kind == WGSL_TYPE_PTR || elem->kind == WGSL_TYPE_REF) {
            wgsl_tc_error(tc, spec->children[1], "array element type must be a plain type");
            return NULL;
        }
        /* §6.2.9 — array element type must have a generation-fixed footprint.
         * In particular, a runtime-sized array (array_len == 0) cannot be
         * an array element type. */
        if (elem->kind == WGSL_TYPE_ARRAY && elem->array_len == 0) {
            wgsl_tc_error(tc, spec->children[1],
                "array element type must have a generation-fixed footprint "
                "(runtime-sized arrays cannot be array elements)");
            return NULL;
        }

        uint32_t length = 0;
        WGSLSymbol *override_sym = NULL;
        if (spec->child_count >= 3) {
            length = eval_array_length(tc, spec->children[2], &override_sym);
            if (length == 0 && !override_sym) {
                wgsl_tc_error(tc, spec->children[2], "array length must be greater than zero");
                return NULL;
            }
        }
        /* §6.2.9 — override-sized arrays interner via the override
         * symbol, so different overrides yield distinct types
         * (`array<f32, N>` ≠ `array<f32, M>`); fixed-size + runtime-
         * sized arrays use the plain `wgsl_type_array` path. */
        if (override_sym) {
            return wgsl_type_array_override(tc->types, elem, override_sym);
        }
        return wgsl_type_array(tc->types, elem, length);
    }

    /* ptr<AS, T, AM> */
    if (nl == 3 && memcmp(nm, "ptr", 3) == 0) {
        if (spec->child_count < 3) {
            wgsl_tc_error(tc, spec, "ptr requires at least an address space and an element type");
            return NULL;
        }
        WGSLAddressSpace as = wgsl_tc_resolve_addr_space_arg(tc, spec->children[1]);
        if (!as) return NULL;
        WGSLTypeInfo *elem = wgsl_tc_resolve_type_spec(tc, spec->children[2]);
        if (!elem) {
            WGSLNode *arg = spec->children[2];
            if (arg &&
                arg->kind != WGSL_NODE_EXPR_IDENT &&
                arg->kind != WGSL_NODE_EXPR_TEMPLATED_IDENT)
            {
                wgsl_tc_error(tc, arg, "ptr store type must be a type");
            }
            return NULL;
        }
        if (as == WGSL_AS_HANDLE ||
            as == WGSL_AS_PUSH_CONSTANT ||
            as == WGSL_AS_IMMEDIATE)
        {
            wgsl_tc_error(tc, spec->children[1],
                "ptr address space must be function, private, workgroup, "
                "uniform, or storage");
            return NULL;
        }
        if (!wgsl_type_is_storable(elem)) {
            wgsl_tc_error(tc, spec->children[2],
                "ptr store type must be storable");
            return NULL;
        }
        if (type_contains_atomic_deep(tc, elem) &&
            as != WGSL_AS_WORKGROUP && as != WGSL_AS_STORAGE)
        {
            wgsl_tc_error(tc, spec->children[2],
                "ptr store type containing atomic requires the workgroup "
                "or storage address space");
            return NULL;
        }
        if (as != WGSL_AS_STORAGE &&
            type_contains_runtime_array_deep(tc, elem))
        {
            wgsl_tc_error(tc, spec->children[2],
                "ptr store type with a runtime-sized array requires "
                "the storage address space");
            return NULL;
        }
        if (as != WGSL_AS_WORKGROUP && type_has_override_array(elem)) {
            wgsl_tc_error(tc, spec->children[2],
                "ptr store type with an override-sized array requires "
                "the workgroup address space");
            return NULL;
        }

        WGSLAccessMode am = WGSL_ACCESS_NONE;
        if (spec->child_count >= 4) {
            am = wgsl_tc_resolve_access_mode_arg(tc, spec->children[3]);
            if (as != WGSL_AS_STORAGE) {
                wgsl_tc_error(tc, spec->children[3],
                    "ptr access mode can only be specified for "
                    "'storage' address space");
                return NULL;
            }
            if (am == WGSL_ACCESS_WRITE) {
                wgsl_tc_error(tc, spec->children[3],
                    "ptr access mode must be 'read' or 'read_write'");
                return NULL;
            }
        } else {
            if (as == WGSL_AS_FUNCTION || as == WGSL_AS_PRIVATE || as == WGSL_AS_WORKGROUP) {
                am = WGSL_ACCESS_READ_WRITE;
            } else {
                am = WGSL_ACCESS_READ;
            }
        }
        if (type_contains_atomic_deep(tc, elem) &&
            as == WGSL_AS_STORAGE && am != WGSL_ACCESS_READ_WRITE)
        {
            wgsl_tc_error(tc, spec->children[2],
                "ptr store type containing atomic in storage address "
                "space requires read_write access mode");
            return NULL;
        }
        return wgsl_type_ptr(tc->types, as, elem, am);
    }

    /* Sampled / multisampled textures: one template arg = sampled type. */
    {
        WGSLTextureDim dim;
        int matched = 1;
        if      (nl == 10 && memcmp(nm, "texture_1d", 10) == 0)
            dim = WGSL_TEX_DIM_1D;
        else if (nl == 10 && memcmp(nm, "texture_2d", 10) == 0)
            dim = WGSL_TEX_DIM_2D;
        else if (nl == 16 && memcmp(nm, "texture_2d_array", 16) == 0)
            dim = WGSL_TEX_DIM_2D_ARRAY;
        else if (nl == 10 && memcmp(nm, "texture_3d", 10) == 0)
            dim = WGSL_TEX_DIM_3D;
        else if (nl == 12 && memcmp(nm, "texture_cube", 12) == 0)
            dim = WGSL_TEX_DIM_CUBE;
        else if (nl == 18 && memcmp(nm, "texture_cube_array", 18) == 0)
            dim = WGSL_TEX_DIM_CUBE_ARRAY;
        else if (nl == 23 && memcmp(nm, "texture_multisampled_2d", 23) == 0)
            dim = WGSL_TEX_DIM_MULTISAMPLED_2D;
        else matched = 0;
        if (matched) {
            if (spec->child_count < 2) {
                wgsl_tc_error(tc, spec,
                    "%.*s requires a sampled type", (int)nl, nm);
                return NULL;
            }
            WGSLTypeInfo *elem = wgsl_tc_resolve_type_spec(tc, spec->children[1]);
            if (!elem) return NULL;
            if (elem != tc->types->t_f32 && elem != tc->types->t_i32 && elem != tc->types->t_u32) {
                wgsl_tc_error(tc, spec->children[1], "sampled texture type must be f32, i32, or u32");
                return NULL;
            }
            return wgsl_type_texture(tc->types, dim, elem,
                WGSL_TEX_FORMAT_NONE, WGSL_ACCESS_NONE);
        }
    }

    /* Storage textures: <texel_format, access_mode>. */
    {
        WGSLTextureDim dim;
        int matched = 1;
        if      (nl == 18 && memcmp(nm, "texture_storage_1d", 18) == 0)
            dim = WGSL_TEX_DIM_STORAGE_1D;
        else if (nl == 18 && memcmp(nm, "texture_storage_2d", 18) == 0)
            dim = WGSL_TEX_DIM_STORAGE_2D;
        else if (nl == 24 && memcmp(nm, "texture_storage_2d_array", 24) == 0)
            dim = WGSL_TEX_DIM_STORAGE_2D_ARRAY;
        else if (nl == 18 && memcmp(nm, "texture_storage_3d", 18) == 0)
            dim = WGSL_TEX_DIM_STORAGE_3D;
        else matched = 0;
        if (matched) {
            if (spec->child_count < 3) {
                wgsl_tc_error(tc, spec,
                    "%.*s requires a texel format and access mode",
                    (int)nl, nm);
                return NULL;
            }
            WGSLTexelFormat fmt = resolve_texel_format_arg(tc, spec->children[1]);
            WGSLAccessMode  acc = wgsl_tc_resolve_access_mode_arg (tc, spec->children[2]);
            if (fmt == WGSL_TEX_FORMAT_NONE) return NULL;
            if (acc == WGSL_ACCESS_NONE)     return NULL;
            if (acc != WGSL_ACCESS_WRITE &&
                !(tc->supported_features & WGSL_FEAT_RO_RW_STORAGE_TEXTURES))
            {
                wgsl_tc_error(tc, spec->children[2],
                    "use of 'read' or 'read_write' on storage textures requires "
                    "support for language feature "
                    "'readonly_and_readwrite_storage_textures'");
                return NULL;
            }
            return wgsl_type_texture(tc->types, dim, NULL, fmt, acc);
        }
    }

    /* Anything else (e.g. user types templated by mistake) — leave the
     * silent NULL fall-through to Phase 8. */
    return NULL;
}

static WGSLTypeInfo *resolve_type_spec_impl(WGSLTypeChecker *tc, WGSLNode *spec) {
    if (!spec) return NULL;

    if (spec->kind == WGSL_NODE_EXPR_IDENT) {
        WGSLSymbol *sym = find_type_symbol_for_ident(tc, spec);
        if (!sym) {
            WGSLSymbol *resolved = wgsl_node_resolved_symbol(spec);
            if (resolved) {
                wgsl_tc_error(tc, spec,
                    "'%.*s' is not a type",
                    (int)resolved->name_len, resolved->name);
            }
            return NULL;
        }
        switch ((WGSLSymKind)sym->kind) {
        case WGSL_SYM_PREDECLARED_TYPE:
        case WGSL_SYM_PREDECLARED_TYPE_ALIAS:
            if (sym->type == tc->types->t_f16 ||
                (sym->type->kind == WGSL_TYPE_VEC && (WGSLTypeInfo *)sym->type->ref == tc->types->t_f16) ||
                (sym->type->kind == WGSL_TYPE_MAT && (WGSLTypeInfo *)sym->type->ref == tc->types->t_f16))
            {
                if (!(tc->enabled_extensions & WGSL_EXT_F16)) {
                    wgsl_tc_error(tc, spec, "use of '%.*s' requires 'enable f16;'", (int)sym->name_len, sym->name);
                    return NULL;
                }
            }
            return sym->type;
        case WGSL_SYM_TYPE_ALIAS:
        case WGSL_SYM_STRUCT:
            return ensure_type_symbol_resolved(tc, sym);
        default:
            wgsl_tc_error(tc, spec,
                "'%.*s' is not a type",
                (int)sym->name_len, sym->name);
            return NULL;
        }
    }

    if (spec->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT) {
        return resolve_templated_type(tc, spec);
    }
    return NULL;
}
