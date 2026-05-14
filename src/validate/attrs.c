/**
 * @file attrs.c — §12 attribute conformance + builtin-name table.
 *
 *   - `kAttrTable` — every spec §12.x attribute's placement context +
 *     argument arity.
 *   - `kBuiltinTable` — every spec §13.3.1.1 builtin name's stage /
 *     direction / required-type matrix.  Exposed cross-file via
 *     `wgsl_val_find_builtin_entry` for io.c's `check_io_slot`.
 *   - Per-attribute argument validation (`check_attr`,
 *     `check_builtin_attr`, `check_interpolate_attr`).
 *   - Cross-attribute co-presence rules (`check_attrs_range`):
 *     `@blend_src` needs `@location`, `@interpolate` needs `@location`,
 *     `@invariant` needs `@builtin(position)`, dedup, etc.
 *   - Top-level `validate_attributes` driver.
 *   - Attribute lookup / integer-arg helpers exposed to other validator
 *     files via priv.h: `wgsl_val_find_attr`, `wgsl_val_attr_int_arg`,
 *     `wgsl_val_attr_int_arg_typed`.
 *
 * Co-extensive with WGSL §12 + §13.3.1.1.
 */
#include "internal/validate.h"
#include "internal/validate_priv.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal/resolver.h"
#include "internal/types.h"
#include "internal/ast.h"

/* ── Attribute conformance (§12) ───────────────────────────────────── */

/* Bitmask of contexts an attribute may legally appear in. */
enum {
    AC_FN_DECL       = 1u << 0,   /* `fn` declaration                       */
    AC_FN_PARAM      = 1u << 1,   /* function parameter                     */
    AC_FN_RETURN     = 1u << 2,   /* function return type slot              */
    AC_VAR_MODULE    = 1u << 3,   /* module-scope `var<…>` decl             */
    AC_OVERRIDE      = 1u << 4,   /* `override` decl                        */
    AC_STRUCT_MEMBER = 1u << 5,   /* struct member                          */
    AC_MODULE        = 1u << 6,   /* module-level (e.g., `@diagnostic`)     */
    AC_STMT          = 1u << 7,   /* statements with attribute lists         */
};

typedef struct {
    const char *name;
    int         min_args;
    int         max_args;
    uint32_t    contexts;
} AttrEntry;

/* Per WGSL §12.  Attribute names are context-keyed identifiers, not
 * keywords — the parser leaves them as IDENTs.  Entries cover every
 * §12.1–12.15 attribute. */
static const AttrEntry kAttrTable[] = {
    /* Stage attributes (§12.15). */
    { "vertex",         0, 0, AC_FN_DECL },
    { "fragment",       0, 0, AC_FN_DECL },
    { "compute",        0, 0, AC_FN_DECL },

    /* Compute (§12.14). */
    { "workgroup_size", 1, 3, AC_FN_DECL },

    /* Resources (§12.4 / §12.5). */
    { "binding",        1, 1, AC_VAR_MODULE },
    { "group",          1, 1, AC_VAR_MODULE },

    /* Entry-point IO (§12.7 / §12.6 / §12.10 / §12.11 / §12.13). */
    { "location",       1, 1, AC_FN_PARAM | AC_FN_RETURN | AC_STRUCT_MEMBER },
    { "builtin",        1, 1, AC_FN_PARAM | AC_FN_RETURN | AC_STRUCT_MEMBER },
    { "interpolate",    1, 2, AC_FN_PARAM | AC_FN_RETURN | AC_STRUCT_MEMBER },
    { "invariant",      0, 0, AC_FN_PARAM | AC_FN_RETURN | AC_STRUCT_MEMBER },
    { "blend_src",      1, 1, AC_FN_PARAM | AC_FN_RETURN | AC_STRUCT_MEMBER },

    /* Layout (§12.1 / §12.12). */
    { "align",          1, 1, AC_STRUCT_MEMBER },
    { "size",           1, 1, AC_STRUCT_MEMBER },

    /* Function (§12.2 / §12.9). */
    { "must_use",       0, 0, AC_FN_DECL },
    { "const",          0, 0, AC_FN_DECL },

    /* Override (§12.8). */
    { "id",             1, 1, AC_OVERRIDE },

/* Diagnostic filter (§4.2). */
    { "diagnostic",     2, 3, AC_MODULE | AC_FN_DECL | AC_STMT },
};
 
/* Per WGSL §13.3.1.1. */
/* §13.3.1.1 builtin matrix.  `stage_mask` is a bitset over
 * STAGE_VX/FR/CO; `dir_in` / `dir_out` are 1 when the builtin is legal
 * in that direction.  `type_hint` is a small enum the IO check
 * compares against the user's declared type. */
/* BuiltinTypeHint / BuiltinEntry moved to internal/validate_priv.h so
 * io.c's `check_io_slot` can read the entry's `stage_mask` / `dir_in` /
 * `dir_out` / `type` fields. */

/* STAGE_* aren't visible yet; we forward-define their bit values
 * locally so the table compiles where it lives.  The bit values match
 * STAGE_VX/FR/CO further down (1<<0, 1<<1, 1<<2). */
#define BSTG_VX 0x1
#define BSTG_FR 0x2
#define BSTG_CO 0x4

static const BuiltinEntry kBuiltinTable[] = {
    /* name                   ext           feat                            stage           in  out  type        */
    { "vertex_index",         0,            0,                              BSTG_VX,        1,  0,   BT_U32 },
    { "instance_index",       0,            0,                              BSTG_VX,        1,  0,   BT_U32 },
    { "position",             0,            0,                              BSTG_VX|BSTG_FR,1,  1,   BT_VEC4_F32 },
    { "front_facing",         0,            0,                              BSTG_FR,        1,  0,   BT_BOOL },
    { "frag_depth",           0,            0,                              BSTG_FR,        0,  1,   BT_F32 },
    { "local_invocation_id",  0,            0,                              BSTG_CO,        1,  0,   BT_VEC3_U32 },
    { "local_invocation_index", 0,          0,                              BSTG_CO,        1,  0,   BT_U32 },
    { "global_invocation_id", 0,            0,                              BSTG_CO,        1,  0,   BT_VEC3_U32 },
    { "workgroup_id",         0,            0,                              BSTG_CO,        1,  0,   BT_VEC3_U32 },
    { "num_workgroups",       0,            0,                              BSTG_CO,        1,  0,   BT_VEC3_U32 },
    { "sample_index",         0,            0,                              BSTG_FR,        1,  0,   BT_U32 },
    { "sample_mask",          0,            0,                              BSTG_FR,        1,  1,   BT_U32 },
    { "subgroup_invocation_id", WGSL_EXT_SUBGROUPS, 0,                      BSTG_FR|BSTG_CO,1,  0,   BT_U32 },
    { "subgroup_size",        WGSL_EXT_SUBGROUPS, 0,                       BSTG_FR|BSTG_CO,1,  0,   BT_U32 },
    { "clip_distances",       WGSL_EXT_CLIP_DISTANCES, 0,                  BSTG_VX,        0,  1,   BT_ARRAY_F32_1_8 },
    { "primitive_index",      WGSL_EXT_PRIMITIVE_INDEX, 0,                 BSTG_FR,        1,  0,   BT_U32 },
    { "primitive_id",         WGSL_EXT_PRIMITIVE_INDEX, 0,                 BSTG_FR,        1,  0,   BT_U32 },
    /* linear_indexing feature (§4.1.2). */
    { "global_invocation_index", 0,         WGSL_FEAT_LINEAR_INDEXING,     BSTG_CO,        1,  0,   BT_U32 },
    { "workgroup_index",      0,            WGSL_FEAT_LINEAR_INDEXING,     BSTG_CO,        1,  0,   BT_U32 },
    /* subgroup_id feature (§4.1.2). */
    { "subgroup_id",          0,            WGSL_FEAT_SUBGROUP_ID,         BSTG_CO,        1,  0,   BT_U32 },
    { "num_subgroups",        0,            WGSL_FEAT_SUBGROUP_ID,         BSTG_CO,        1,  0,   BT_U32 },
};
 
static void check_builtin_attr(
    WGSLValidator *v, WGSLNode *attr)
{
    if (attr->child_count < 2) return;
    WGSLNode *arg = attr->children[1];
    if (arg->kind != WGSL_NODE_EXPR_IDENT) return;
    uint32_t off = (uint32_t)(arg->payload[0] & 0xFFFFFFFFu);
    uint32_t len = (uint32_t)(arg->payload[0] >> 32);
    const char *nm = v->src->bytes + off;
 
    const BuiltinEntry *found = NULL;
    for (size_t i = 0; i < sizeof kBuiltinTable / sizeof kBuiltinTable[0]; i++) {
        if (strlen(kBuiltinTable[i].name) == len &&
            memcmp(kBuiltinTable[i].name, nm, len) == 0)
        {
            found = &kBuiltinTable[i];
            break;
        }
    }
 
    if (!found) {
        wgsl_val_error(v, arg, "unknown builtin '%.*s'", (int)len, nm);
        return;
    }
 
    if (found->ext != 0 && !(v->tc->enabled_extensions & found->ext)) {
        const char *ext_name = "unknown";
        if (found->ext == WGSL_EXT_SUBGROUPS) ext_name = "subgroups";
        if (found->ext == WGSL_EXT_CLIP_DISTANCES) ext_name = "clip_distances";
        if (found->ext == WGSL_EXT_PRIMITIVE_INDEX) ext_name = "primitive_index";
        wgsl_val_error(v, arg, "use of builtin '%.*s' requires 'enable %s;'", (int)len, nm, ext_name);
    }
    if (found->feat != 0 && !(v->tc->supported_features & found->feat)) {
        const char *feat_name = "unknown";
        if (found->feat == WGSL_FEAT_LINEAR_INDEXING) feat_name = "linear_indexing";
        if (found->feat == WGSL_FEAT_SUBGROUP_ID)     feat_name = "subgroup_id";
        wgsl_val_error(v, arg,
            "use of builtin '%.*s' requires support for language feature '%s'",
            (int)len, nm, feat_name);
    }
}

/* WGSL §3.8.6 / §3.8.7 + §13.3.1.4.  `@interpolate(type)` and
 * `@interpolate(type, sampling)` accept only the spec-listed
 * context-dependent names. */
static void check_interpolate_attr(WGSLValidator *v, WGSLNode *attr) {
    /* Validate the first argument (interpolation type, §3.8.6). */
    if (attr->child_count < 2) return;
    {
        WGSLNode *arg = attr->children[1];
        if (!arg || arg->kind != WGSL_NODE_EXPR_IDENT) {
            wgsl_val_error(v, arg ? arg : attr,
                "interpolation type must be an identifier");
        } else {
            uint32_t off = (uint32_t)(arg->payload[0] & 0xFFFFFFFFu);
            uint32_t len = (uint32_t)(arg->payload[0] >> 32);
            const char *nm = v->src->bytes + off;
            int valid =
                (len == 11 && memcmp(nm, "perspective", 11) == 0) ||
                (len == 6  && memcmp(nm, "linear", 6) == 0) ||
                (len == 4  && memcmp(nm, "flat", 4) == 0);
            if (!valid) {
                wgsl_val_error(v, arg,
                    "unknown interpolation type '%.*s' (expected "
                    "'perspective', 'linear', or 'flat')",
                    (int)len, nm);
            }
        }
    }
    /* Validate the optional second argument (interpolation sampling,
     * §3.8.7). */
    if (attr->child_count < 3) return;
    {
        WGSLNode *arg = attr->children[2];
        if (!arg || arg->kind != WGSL_NODE_EXPR_IDENT) {
            wgsl_val_error(v, arg ? arg : attr,
                "interpolation sampling must be an identifier");
        } else {
            uint32_t off = (uint32_t)(arg->payload[0] & 0xFFFFFFFFu);
            uint32_t len = (uint32_t)(arg->payload[0] >> 32);
            const char *nm = v->src->bytes + off;
            int valid =
                (len == 6 && memcmp(nm, "center",   6) == 0) ||
                (len == 8 && memcmp(nm, "centroid", 8) == 0) ||
                (len == 6 && memcmp(nm, "sample",   6) == 0) ||
                (len == 5 && memcmp(nm, "first",    5) == 0) ||
                (len == 6 && memcmp(nm, "either",   6) == 0);
            if (!valid) {
                wgsl_val_error(v, arg,
                    "unknown interpolation sampling '%.*s' (expected "
                    "'center', 'centroid', 'sample', 'first', or "
                    "'either')",
                    (int)len, nm);
            }
        }
    }
}

static const AttrEntry *attr_lookup(
    const char *name, uint32_t name_len)
{
    for (size_t i = 0; i < sizeof kAttrTable / sizeof kAttrTable[0]; i++) {
        const AttrEntry *e = &kAttrTable[i];
        size_t l = strlen(e->name);
        if (l == name_len && memcmp(e->name, name, l) == 0) return e;
    }
    return NULL;
}

static const char *ctx_name(uint32_t bit) {
    switch (bit) {
    case AC_FN_DECL:       return "function declarations";
    case AC_FN_PARAM:      return "function parameters";
    case AC_FN_RETURN:     return "function return types";
    case AC_VAR_MODULE:    return "module-scope `var` declarations";
    case AC_OVERRIDE:      return "`override` declarations";
    case AC_STRUCT_MEMBER: return "struct members";
    case AC_MODULE:        return "the module level";
    case AC_STMT:          return "statements";
    default:               return "this context";
    }
}

static int diagnostic_severity_value(
    const WGSLValidator *v, const WGSLNode *sev,
    const char **out_name, uint32_t *out_len)
{
    if (out_name) *out_name = NULL;
    if (out_len) *out_len = 0;
    if (!sev || sev->kind != WGSL_NODE_EXPR_IDENT) return -1;
    uint32_t off = (uint32_t)(sev->payload[0] & 0xFFFFFFFFu);
    uint32_t len = (uint32_t)(sev->payload[0] >> 32);
    const char *nm = v->src->bytes + off;
    if (out_name) *out_name = nm;
    if (out_len) *out_len = len;
    if (len == 3 && memcmp(nm, "off", 3) == 0) return 0;
    if (len == 4 && memcmp(nm, "info", 4) == 0) return WGSL_DIAG_INFO;
    if (len == 7 && memcmp(nm, "warning", 7) == 0) return WGSL_DIAG_WARNING;
    if (len == 5 && memcmp(nm, "error", 5) == 0) return WGSL_DIAG_ERROR;
    return -1;
}

static void check_diagnostic_severity(
    WGSLValidator *v, const WGSLNode *sev)
{
    const char *nm = NULL;
    uint32_t len = 0;
    if (diagnostic_severity_value(v, sev, &nm, &len) >= 0) return;
    if (nm) {
        wgsl_val_error(v, sev,
            "invalid diagnostic severity '%.*s' (expected 'off', "
            "'info', 'warning', or 'error')",
            (int)len, nm);
    }
}

long long wgsl_val_attr_int_arg(const WGSLValidator *v, const WGSLNode *attr);
int wgsl_val_attr_int_arg_typed(
    WGSLValidator *v, const WGSLNode *attr, long long *out);

/* Validate one ATTRIBUTE node found in `ctx_bit` placement. */
static void check_attr(
    WGSLValidator *v, WGSLNode *attr, uint32_t ctx_bit)
{
    if (!attr || attr->child_count == 0) return;
    WGSLNode *name_node = attr->children[0];
    if (!name_node) return;
    uint32_t off = (uint32_t)(name_node->payload[0] & 0xFFFFFFFFu);
    uint32_t len = (uint32_t)(name_node->payload[0] >> 32);
    const char *name = v->src->bytes + off;

    const AttrEntry *e = attr_lookup(name, len);
    if (!e) {
        wgsl_val_error(v, attr,
            "unknown attribute '@%.*s'", (int)len, name);
        return;
    }
    if ((e->contexts & ctx_bit) == 0) {
        wgsl_val_error(v, attr,
            "attribute '@%.*s' is not allowed on %s",
            (int)len, name, ctx_name(ctx_bit));
        return;
    }
    /* Argument-count check: child[0] is the name; rest are args. */
    int argc = (int)attr->child_count - 1;
    if (argc < e->min_args || argc > e->max_args) {
        wgsl_val_error(v, attr,
            "attribute '@%.*s' expects %d-%d arguments, got %d",
            (int)len, name, e->min_args, e->max_args, argc);
        return;
    }
    if (argc == 0 && e->max_args == 0 &&
        (attr->flags & WGSL_FLAG_ATTR_PARENS))
    {
        wgsl_val_error(v, attr,
            "attribute '@%.*s' must not use parentheses", (int)len, name);
        return;
    }
 
    if (len == 7 && memcmp(name, "builtin", 7) == 0) {
        check_builtin_attr(v, attr);
    }
    if (len == 11 && memcmp(name, "interpolate", 11) == 0) {
        check_interpolate_attr(v, attr);
    }
    if (len == 9 && memcmp(name, "blend_src", 9) == 0) {
        if (!(v->tc->enabled_extensions & WGSL_EXT_DUAL_SOURCE_BLENDING)) {
            wgsl_val_error(v, attr, "use of '@blend_src' requires 'enable dual_source_blending;'");
        }
        /* §12.3 — n must be 0 or 1. */
        long long n;
        if (!wgsl_val_attr_int_arg_typed(v, attr, &n)) {
            wgsl_val_error(v, attr,
                "@blend_src value must be a const-expression of type i32 or u32");
        } else if (n != 0 && n != 1) {
            wgsl_val_error(v, attr, "@blend_src value must be 0 or 1");
        }
    }
    /* §12.1 — @align(n): n must be a const-expression of type i32 or
     * u32, positive, and a power of 2.  The "n ≥ RequiredAlignOf(T,AS)"
     * rule depends on the §14.4.5 table, which is still pending. */
    if (len == 5 && memcmp(name, "align", 5) == 0) {
        long long n;
        if (!wgsl_val_attr_int_arg_typed(v, attr, &n)) {
            wgsl_val_error(v, attr,
                "@align value must be a const-expression of type i32 or u32");
        } else if (n <= 0) {
            wgsl_val_error(v, attr, "@align value must be positive");
        } else if (n > INT32_MAX) {
            wgsl_val_error(v, attr, "@align value must fit in i32");
        } else if ((n & (n - 1)) != 0) {
            wgsl_val_error(v, attr, "@align value must be a power of 2");
        }
    }
    /* §12.13 — @size(n) must be a const-expression of type i32 or u32,
     * positive.  The "n ≥ SizeOf(T)" rule is enforced separately in
     * validate_layouts. */
    if (len == 4 && memcmp(name, "size", 4) == 0) {
        long long n;
        if (!wgsl_val_attr_int_arg_typed(v, attr, &n)) {
            wgsl_val_error(v, attr,
                "@size value must be a const-expression of type i32 or u32");
        } else if (n <= 0) {
            wgsl_val_error(v, attr, "@size value must be positive");
        }
    }
    /* §12.2 / §12.7 / §12.11 — @group / @binding / @location must be
     * non-negative const-expression integers. */
    if ((len == 5 && memcmp(name, "group", 5) == 0) ||
        (len == 7 && memcmp(name, "binding", 7) == 0) ||
        (len == 8 && memcmp(name, "location", 8) == 0))
    {
        long long n;
        if (!wgsl_val_attr_int_arg_typed(v, attr, &n)) {
            wgsl_val_error(v, attr,
                "@%.*s value must be a const-expression of type i32 or u32",
                (int)len, name);
        } else if (n < 0) {
            wgsl_val_error(v, attr,
                "@%.*s value must be a non-negative integer",
                (int)len, name);
        }
    }
    /* §12.8 — @id(N) takes a const-expression integer specialization
     * constant id in the inclusive range [0, 65535]. */
    if (len == 2 && memcmp(name, "id", 2) == 0) {
        long long n;
        if (!wgsl_val_attr_int_arg_typed(v, attr, &n)) {
            wgsl_val_error(v, attr,
                "@id value must be a const-expression of type i32 or u32");
        } else if (n < 0 || n > 65535) {
            wgsl_val_error(v, attr, "@id value must be in [0, 65535]");
        }
    }
    /* §12.5: `@const` is a spec notation for built-in functions only;
     * it must not be applied to user-defined functions.  Every fn
     * declaration in WGSL source is user-defined, so any sighting is
     * an error. */
    if (len == 5 && memcmp(name, "const", 5) == 0) {
        wgsl_val_error(v, attr,
            "attribute '@const' cannot be applied to user-defined functions");
    }
    /* §12.6 / §3.8.3: `@diagnostic(severity, rule)` — validate the
     * rule-name argument.  Same logic as the directive form in check.c:
     * single-token rules must be spec-listed; multi-token rules
     * ("vendor.foo") are silently accepted per §2.3.2. */
    if (len == 10 && memcmp(name, "diagnostic", 10) == 0) {
        /* children: [name_ident, severity_ident, rule_ident, ...] */
        if (attr->child_count >= 2) {
            check_diagnostic_severity(v, attr->children[1]);
        }
        if (attr->child_count >= 3) {
            WGSLNode *rule = attr->children[2];
            if (rule && rule->kind == WGSL_NODE_EXPR_IDENT) {
                uint32_t roff = (uint32_t)(rule->payload[0] & 0xFFFFFFFFu);
                uint32_t rlen = (uint32_t)(rule->payload[0] >> 32);
                const char *rnm = v->src->bytes + roff;
                /* Check whether this is a single-token or multi-token rule.
                 * If there is a fourth child and it's also an IDENT, it's
                 * multi-token ("vendor.foo") — silently accept.  Otherwise
                 * validate the single-token name. */
                int is_multi = (attr->child_count >= 4 &&
                                attr->children[3] &&
                                attr->children[3]->kind == WGSL_NODE_EXPR_IDENT);
                if (!is_multi) {
                    int valid =
                        (rlen == 21 && memcmp(rnm, "derivative_uniformity", 21) == 0) ||
                        (rlen == 19 && memcmp(rnm, "subgroup_uniformity", 19) == 0);
                    if (!valid) {
                        wgsl_diag_emit_at(v->diag, v->src, WGSL_DIAG_WARNING,
                            rule->span_offset, rule->span_length, NULL,
                            "unknown diagnostic rule '%.*s' (expected "
                            "'derivative_uniformity' or "
                            "'subgroup_uniformity')",
                            (int)rlen, rnm);
                    }
                }
            }
        }
    }
}

/* Pull the attribute name span from an ATTRIBUTE node.  Empty span on
 * a malformed attribute. */
static void attr_name_span(
    const WGSLValidator *v, const WGSLNode *attr,
    const char **out_name, uint32_t *out_len)
{
    *out_name = NULL;
    *out_len  = 0;
    if (!attr || attr->child_count == 0) return;
    const WGSLNode *an = attr->children[0];
    if (!an || an->kind != WGSL_NODE_EXPR_IDENT) return;
    uint32_t off = (uint32_t)(an->payload[0] & 0xFFFFFFFFu);
    uint32_t len = (uint32_t)(an->payload[0] >> 32);
    *out_name = v->src->bytes + off;
    *out_len  = len;
}

/* True iff attr is `@builtin(position)`. */
static int attr_is_builtin_position(
    const WGSLValidator *v, const WGSLNode *attr)
{
    const char *nm; uint32_t nl;
    attr_name_span(v, attr, &nm, &nl);
    if (!(nl == 7 && memcmp(nm, "builtin", 7) == 0)) return 0;
    if (attr->child_count < 2) return 0;
    const WGSLNode *av = attr->children[1];
    if (!av || av->kind != WGSL_NODE_EXPR_IDENT) return 0;
    uint32_t voff = (uint32_t)(av->payload[0] & 0xFFFFFFFFu);
    uint32_t vlen = (uint32_t)(av->payload[0] >> 32);
    return vlen == 8 && memcmp(v->src->bytes + voff, "position", 8) == 0;
}

static int attr_name_is(
    const WGSLValidator *v, const WGSLNode *attr, const char *want)
{
    const char *nm; uint32_t nl;
    attr_name_span(v, attr, &nm, &nl);
    size_t wl = strlen(want);
    return nl == wl && memcmp(nm, want, wl) == 0;
}

static int attr_is_entry_point_io(
    const WGSLValidator *v, const WGSLNode *attr)
{
    return attr_name_is(v, attr, "location") ||
           attr_name_is(v, attr, "builtin") ||
           attr_name_is(v, attr, "interpolate") ||
           attr_name_is(v, attr, "invariant") ||
           attr_name_is(v, attr, "blend_src");
}

static void check_non_entry_io_attrs(
    WGSLValidator *v, WGSLNode *parent, uint32_t first, uint32_t count)
{
    for (uint32_t i = 0; i < count && first + i < parent->child_count; i++) {
        WGSLNode *a = parent->children[first + i];
        if (!a || a->kind != WGSL_NODE_ATTRIBUTE) continue;
        if (!attr_is_entry_point_io(v, a)) continue;
        const char *nm; uint32_t nl;
        attr_name_span(v, a, &nm, &nl);
        wgsl_val_error(v, a,
            "entry-point IO attribute '@%.*s' is only valid on entry "
            "point parameters, return types, or IO struct members",
            (int)nl, nm);
    }
}

static int function_has_stage_attr(
    const WGSLValidator *v, const WGSLNode *fn, uint32_t attr_count)
{
    for (uint32_t i = 0; i < attr_count && i < fn->child_count; i++) {
        WGSLNode *a = fn->children[i];
        if (!a || a->kind != WGSL_NODE_ATTRIBUTE) continue;
        if (attr_name_is(v, a, "vertex") ||
            attr_name_is(v, a, "fragment") ||
            attr_name_is(v, a, "compute"))
        {
            return 1;
        }
    }
    return 0;
}

/* Walk every attribute on a decl-shaped node and check it against
 * `ctx_bit`.  Layout-driven helpers know which children are attrs.
 *
 * Beyond per-attribute name / arg checks, this pass also enforces:
 *   - §12 general — an attribute name must not be repeated on the same
 *     object (`@location(0) @location(1)` is invalid).
 *   - §12.9 — `@interpolate` must accompany an `@location` attribute.
 *   - §12.10 — `@invariant` only on a member also carrying
 *     `@builtin(position)`. */
static void check_attrs_range(
    WGSLValidator *v, WGSLNode *parent,
    uint32_t first, uint32_t count, uint32_t ctx_bit)
{
    int seen_location    = 0;
    int seen_interpolate = 0;
    WGSLNode *invariant_attr = NULL;
    WGSLNode *blend_src_attr = NULL;
    int seen_builtin_position = 0;

    /* Track names seen so duplicates can be flagged with the second
     * occurrence pointed at.  Names are stored as (offset, length)
     * pairs into v->src so we don't need a hash table. */
    typedef struct { const char *name; uint32_t len; } SeenAttr;
    SeenAttr seen[16];
    int seen_n = 0;

    /* §12.6 — `@diagnostic(severity, rule)` may repeat as long as the
     * rule names differ.  Track (rule_name, rule_len) pairs separately
     * from the general attribute-name dedup. */
    SeenAttr seen_diag_rules[16];
    int seen_diag_n = 0;

    for (uint32_t i = 0; i < count && first + i < parent->child_count; i++) {
        WGSLNode *a = parent->children[first + i];
        if (!a || a->kind != WGSL_NODE_ATTRIBUTE) continue;

        const char *nm; uint32_t nl;
        attr_name_span(v, a, &nm, &nl);
        if (nm && nl > 0) {
            int is_diag = (nl == 10 && memcmp(nm, "diagnostic", 10) == 0);
            if (is_diag) {
                /* Pull the rule-name span.  In attribute form the rule
                 * lives at children[2]; for `@diagnostic(off, vendor.foo)`
                 * it's an EXPR_MEMBER, otherwise an EXPR_IDENT.  In both
                 * cases the node's `span_offset`/`span_length` cover the
                 * whole rule text, including any '.' between vendor and
                 * sub-name.  Use that span as the dedup key. */
                const char *rnm = NULL; uint32_t rlen = 0;
                if (a->child_count >= 3 && a->children[2]) {
                    const WGSLNode *rule = a->children[2];
                    if (rule->kind == WGSL_NODE_EXPR_IDENT ||
                        rule->kind == WGSL_NODE_EXPR_MEMBER)
                    {
                        rnm  = v->src->bytes + rule->span_offset;
                        rlen = rule->span_length;
                    }
                }
                if (rnm && rlen > 0) {
                    int dup_rule = 0;
                    for (int s = 0; s < seen_diag_n; s++) {
                        if (seen_diag_rules[s].len == rlen &&
                            memcmp(seen_diag_rules[s].name, rnm, rlen) == 0)
                        {
                            dup_rule = 1; break;
                        }
                    }
                    if (dup_rule) {
                        wgsl_val_error(v, a,
                            "multiple @diagnostic attributes on the same "
                            "object must differ on the rule name "
                            "('%.*s' repeated)",
                            (int)rlen, rnm);
                    } else if (seen_diag_n <
                               (int)(sizeof seen_diag_rules /
                                     sizeof seen_diag_rules[0]))
                    {
                        seen_diag_rules[seen_diag_n].name = rnm;
                        seen_diag_rules[seen_diag_n].len  = rlen;
                        seen_diag_n += 1;
                    }
                }
            } else {
                int dup = 0;
                for (int s = 0; s < seen_n; s++) {
                    if (seen[s].len == nl && memcmp(seen[s].name, nm, nl) == 0) {
                        dup = 1; break;
                    }
                }
                if (dup) {
                    wgsl_val_error(v, a,
                        "attribute '@%.*s' must not be specified more than "
                        "once on the same object", (int)nl, nm);
                } else if (seen_n < (int)(sizeof seen / sizeof seen[0])) {
                    seen[seen_n].name = nm;
                    seen[seen_n].len  = nl;
                    seen_n += 1;
                }
            }

            if (nl == 8 && memcmp(nm, "location", 8) == 0)    seen_location    = 1;
            if (nl == 11 && memcmp(nm, "interpolate", 11) == 0) seen_interpolate = 1;
            if (nl == 9 && memcmp(nm, "invariant", 9) == 0)   invariant_attr   = a;
            if (nl == 9 && memcmp(nm, "blend_src", 9) == 0)   blend_src_attr   = a;
            if (attr_is_builtin_position(v, a))               seen_builtin_position = 1;
        }

        check_attr(v, a, ctx_bit);
    }

    if (seen_interpolate && !seen_location) {
        /* Find an @interpolate attr to point the diagnostic at. */
        for (uint32_t i = 0; i < count && first + i < parent->child_count; i++) {
            WGSLNode *a = parent->children[first + i];
            if (!a || a->kind != WGSL_NODE_ATTRIBUTE) continue;
            const char *nm; uint32_t nl;
            attr_name_span(v, a, &nm, &nl);
            if (nl == 11 && memcmp(nm, "interpolate", 11) == 0) {
                wgsl_val_error(v, a,
                    "@interpolate may only appear on a declaration that "
                    "also has @location");
                break;
            }
        }
    }
    if (invariant_attr && !seen_builtin_position) {
        wgsl_val_error(v, invariant_attr,
            "@invariant may only appear on a declaration with "
            "@builtin(position)");
    }
    if (blend_src_attr && !seen_location) {
        wgsl_val_error(v, blend_src_attr,
            "@blend_src may only appear on a declaration that "
            "also has @location");
    }
}

static void check_attrs_array(
    WGSLValidator *v, WGSLNode **attrs, uint32_t count, uint32_t ctx_bit)
{
    typedef struct { const char *name; uint32_t len; } SeenAttr;
    SeenAttr seen_diag_rules[16];
    int seen_diag_n = 0;
    WGSLNode *prev_attr = NULL;

    for (uint32_t i = 0; i < count; i++) {
        WGSLNode *a = attrs ? attrs[i] : NULL;
        if (!a || a->kind != WGSL_NODE_ATTRIBUTE) continue;
        if (prev_attr) {
            uint32_t prev_end = prev_attr->span_offset + prev_attr->span_length;
            int same_list = a->span_offset >= prev_end;
            for (uint32_t p = prev_end; same_list && p < a->span_offset; p++) {
                char c = v->src->bytes[p];
                if (!(c == ' ' || c == '\t' || c == '\n' || c == '\r')) {
                    same_list = 0;
                }
            }
            if (!same_list) seen_diag_n = 0;
        }
        const char *nm; uint32_t nl;
        attr_name_span(v, a, &nm, &nl);
        if (nl == 10 && memcmp(nm, "diagnostic", 10) == 0) {
            const char *rnm = NULL;
            uint32_t rlen = 0;
            if (a->child_count >= 3 && a->children[2]) {
                const WGSLNode *rule = a->children[2];
                if (rule->kind == WGSL_NODE_EXPR_IDENT ||
                    rule->kind == WGSL_NODE_EXPR_MEMBER)
                {
                    rnm = v->src->bytes + rule->span_offset;
                    rlen = rule->span_length;
                }
            }
            if (rnm && rlen > 0) {
                int dup_rule = 0;
                for (int s = 0; s < seen_diag_n; s++) {
                    if (seen_diag_rules[s].len == rlen &&
                        memcmp(seen_diag_rules[s].name, rnm, rlen) == 0)
                    {
                        dup_rule = 1;
                        break;
                    }
                }
                if (dup_rule) {
                    wgsl_val_error(v, a,
                        "multiple @diagnostic attributes on the same "
                        "object must differ on the rule name "
                        "('%.*s' repeated)",
                        (int)rlen, rnm);
                } else if (seen_diag_n <
                           (int)(sizeof seen_diag_rules /
                                 sizeof seen_diag_rules[0]))
                {
                    seen_diag_rules[seen_diag_n].name = rnm;
                    seen_diag_rules[seen_diag_n].len = rlen;
                    seen_diag_n += 1;
                }
            }
        }
        check_attr(v, a, ctx_bit);
        prev_attr = a;
    }
}

static int stmt_kind_accepts_attrs(WGSLNodeKind k) {
    return k == WGSL_NODE_STMT_COMPOUND ||
           k == WGSL_NODE_STMT_IF ||
           k == WGSL_NODE_STMT_WHILE ||
           k == WGSL_NODE_STMT_FOR ||
           k == WGSL_NODE_STMT_LOOP ||
           k == WGSL_NODE_STMT_CONTINUING ||
           k == WGSL_NODE_STMT_SWITCH;
}

static void validate_statement_attrs_node(WGSLValidator *v, WGSLNode *n) {
    if (!n) return;
    if ((n->flags & WGSL_FLAG_HAS_ATTRS) &&
        stmt_kind_accepts_attrs((WGSLNodeKind)n->kind))
    {
        uint32_t attr_count = (uint32_t)(n->payload[1] & 0xFFFFFFFFu);
        WGSLNode **attrs = (WGSLNode **)(uintptr_t)n->payload[2];
        check_attrs_array(v, attrs, attr_count, AC_STMT);
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        validate_statement_attrs_node(v, n->children[i]);
    }
}

static int diagnostic_rule_name(
    const WGSLValidator *v, const WGSLNode *dir,
    const char **out_name, uint32_t *out_len)
{
    *out_name = NULL;
    *out_len = 0;
    if (!dir || dir->kind != WGSL_NODE_DIR_DIAGNOSTIC ||
        dir->child_count < 2)
    {
        return 0;
    }
    uint32_t rule_parts = (uint32_t)(dir->payload[0] & 0xFFFFFFFFu);
    if (rule_parts != 1) return 0;
    const WGSLNode *rule = dir->children[1];
    if (!rule || rule->kind != WGSL_NODE_EXPR_IDENT) return 0;
    uint32_t off = (uint32_t)(rule->payload[0] & 0xFFFFFFFFu);
    uint32_t len = (uint32_t)(rule->payload[0] >> 32);
    const char *nm = v->src->bytes + off;
    if (!((len == 21 && memcmp(nm, "derivative_uniformity", 21) == 0) ||
          (len == 19 && memcmp(nm, "subgroup_uniformity", 19) == 0)))
    {
        return 0;
    }
    *out_name = nm;
    *out_len = len;
    return 1;
}

void validate_attributes(WGSLValidator *v, WGSLNode *tu) {
    if (!tu) return;
    typedef struct { const char *rule; uint32_t len; int severity; } SeenDirective;
    SeenDirective seen_dirs[16];
    int seen_dir_n = 0;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n) continue;
        switch ((WGSLNodeKind)n->kind) {

        case WGSL_NODE_DECL_FUNCTION: {
            uint32_t FA = (uint32_t)(n->payload[1] & 0xFFFFFFFFu);
            uint32_t P  = (uint32_t)(n->payload[1] >> 32);
            uint32_t RA = (uint32_t)(n->payload[2] & 0xFFFFFFFFu);
            int is_entry = function_has_stage_attr(v, n, FA);
            check_attrs_range(v, n, 0,        FA, AC_FN_DECL);
            /* Return-attrs sit between params and (optional) ret_type. */
            check_attrs_range(v, n, FA + P,   RA, AC_FN_RETURN);
            if (!is_entry) check_non_entry_io_attrs(v, n, FA + P, RA);

            /* Per-param attrs. */
            for (uint32_t k = 0; k < P; k++) {
                uint32_t pi = FA + k;
                if (pi >= n->child_count) break;
                WGSLNode *p = n->children[pi];
                if (!p || p->kind != WGSL_NODE_DECL_PARAM) continue;
                uint32_t PA = (uint32_t)(p->payload[1] & 0xFFFFFFFFu);
                check_attrs_range(v, p, 0, PA, AC_FN_PARAM);
                if (!is_entry) check_non_entry_io_attrs(v, p, 0, PA);
            }
            break;
        }

        case WGSL_NODE_DECL_VAR: {
            uint32_t A = (uint32_t)(n->payload[1] & 0xFFFFFFFFu);
            check_attrs_range(v, n, 0, A, AC_VAR_MODULE);
            break;
        }

        case WGSL_NODE_DECL_OVERRIDE: {
            uint32_t A = (uint32_t)(n->payload[1] & 0xFFFFFFFFu);
            check_attrs_range(v, n, 0, A, AC_OVERRIDE);
            break;
        }

        case WGSL_NODE_DECL_STRUCT: {
            for (uint32_t k = 0; k < n->child_count; k++) {
                WGSLNode *m = n->children[k];
                if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
                uint32_t MA = (uint32_t)(m->payload[1] & 0xFFFFFFFFu);
                check_attrs_range(v, m, 0, MA, AC_STRUCT_MEMBER);
            }
            break;
        }

        case WGSL_NODE_DIR_DIAGNOSTIC:
            if (n->child_count >= 1) {
                check_diagnostic_severity(v, n->children[0]);
            }
            {
                const char *rule = NULL;
                uint32_t rule_len = 0;
                int sev = n->child_count >= 1
                    ? diagnostic_severity_value(v, n->children[0], NULL, NULL)
                    : -1;
                if (sev >= 0 && diagnostic_rule_name(v, n, &rule, &rule_len)) {
                    int found = 0;
                    for (int s = 0; s < seen_dir_n; s++) {
                        if (seen_dirs[s].len == rule_len &&
                            memcmp(seen_dirs[s].rule, rule, rule_len) == 0)
                        {
                            found = 1;
                            if (seen_dirs[s].severity != sev) {
                                wgsl_val_error(v, n,
                                    "conflicting diagnostic directive for "
                                    "'%.*s'",
                                    (int)rule_len, rule);
                            }
                            break;
                        }
                    }
                    if (!found && seen_dir_n <
                        (int)(sizeof seen_dirs / sizeof seen_dirs[0]))
                    {
                        seen_dirs[seen_dir_n].rule = rule;
                        seen_dirs[seen_dir_n].len = rule_len;
                        seen_dirs[seen_dir_n].severity = sev;
                        seen_dir_n += 1;
                    }
                }
            }
            break;

        default:
            break;
        }
        validate_statement_attrs_node(v, n);
    }
}


WGSLNode *wgsl_val_find_attr(
    const WGSLValidator *v, const WGSLNode *parent,
    uint32_t first, uint32_t count, const char *want)
{
    size_t wl = strlen(want);
    for (uint32_t i = 0; i < count && first + i < parent->child_count; i++) {
        WGSLNode *a = parent->children[first + i];
        if (!a || a->kind != WGSL_NODE_ATTRIBUTE) continue;
        if (a->child_count == 0) continue;
        WGSLNode *nm = a->children[0];
        if (!nm) continue;
        uint32_t off = (uint32_t)(nm->payload[0] & 0xFFFFFFFFu);
        uint32_t len = (uint32_t)(nm->payload[0] >> 32);
        if (len == wl && memcmp(v->src->bytes + off, want, wl) == 0) {
            return a;
        }
    }
    return NULL;
}

long long wgsl_val_attr_int_arg(const WGSLValidator *v, const WGSLNode *attr) {
    if (!attr || attr->child_count < 2) return 0;
    WGSLValue val = {0};
    if (wgsl_consteval_expr(v->cev, attr->children[1], &val)) {
        if (val.kind == WGSL_VAL_INT) return val.u.i;
    }
    return 0;
}

/** Like `wgsl_val_attr_int_arg`, but distinguishes "ok" from "missing /
 *  non-int / non-const".  On failure the consteval-bag noise is rolled
 *  back so the caller can emit a single clean diagnostic. */
int wgsl_val_attr_int_arg_typed(
    WGSLValidator *v, const WGSLNode *attr, long long *out)
{
    *out = 0;
    if (!attr || attr->child_count < 2) return 0;
    size_t saved_count = v->diag->count;
    int    saved_errs  = v->diag->error_count;
    int    saved_cev   = v->cev->had_error;
    WGSLValue val = {0};
    int ok = wgsl_consteval_expr(v->cev, attr->children[1], &val) &&
             val.kind == WGSL_VAL_INT;
    if (!ok) {
        v->diag->count       = saved_count;
        v->diag->error_count = saved_errs;
        v->cev->had_error    = saved_cev;
        return 0;
    }
    *out = val.u.i;
    return 1;
}

/* IOState typedef moved to io.c (only used by check_io_slot / check_io_type). */

const BuiltinEntry *wgsl_val_find_builtin_entry(const char *nm, uint32_t nl) {
    for (size_t i = 0; i < sizeof kBuiltinTable / sizeof kBuiltinTable[0]; i++) {
        if (strlen(kBuiltinTable[i].name) == nl &&
            memcmp(kBuiltinTable[i].name, nm, nl) == 0)
        {
            return &kBuiltinTable[i];
        }
    }
    return NULL;
}
