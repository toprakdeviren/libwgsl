/**
 * @file attrs_table.c — attribute name tables + per-attr checkers
 */
#include "internal/validate_priv.h"
#include "internal/validate.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal/resolver.h"
#include "internal/types.h"
#include "internal/ast.h"
#include "internal/token.h"
#include "internal/consteval.h"
#include "internal/diag.h"

/* Attribute conformance (§12). */

/* AC_* / AttrEntry: validate_priv.h */

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
 
void check_builtin_attr(
    WGSLValidator *v, WGSLNode *attr)
{
    if (attr->child_count < 2) return;
    WGSLNode *arg = attr->children[1];
    if (arg->kind != WGSL_NODE_EXPR_IDENT) return;
    uint32_t off = wgsl_node_name_span(arg).offset;
    uint32_t len = wgsl_node_name_span(arg).length;
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
void check_interpolate_attr(WGSLValidator *v, WGSLNode *attr) {
    /* Validate the first argument (interpolation type, §3.8.6). */
    if (attr->child_count < 2) return;
    {
        WGSLNode *arg = attr->children[1];
        if (!arg || arg->kind != WGSL_NODE_EXPR_IDENT) {
            wgsl_val_error(v, arg ? arg : attr,
                "interpolation type must be an identifier");
        } else {
            uint32_t off = wgsl_node_name_span(arg).offset;
            uint32_t len = wgsl_node_name_span(arg).length;
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
            uint32_t off = wgsl_node_name_span(arg).offset;
            uint32_t len = wgsl_node_name_span(arg).length;
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

const AttrEntry *attr_lookup(
    const char *name, uint32_t name_len)
{
    for (size_t i = 0; i < sizeof kAttrTable / sizeof kAttrTable[0]; i++) {
        const AttrEntry *e = &kAttrTable[i];
        size_t l = strlen(e->name);
        if (l == name_len && memcmp(e->name, name, l) == 0) return e;
    }
    return NULL;
}

const char *ctx_name(uint32_t bit) {
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

int diagnostic_severity_value(
    const WGSLValidator *v, const WGSLNode *sev,
    const char **out_name, uint32_t *out_len)
{
    if (out_name) *out_name = NULL;
    if (out_len) *out_len = 0;
    if (!sev || sev->kind != WGSL_NODE_EXPR_IDENT) return -1;
    uint32_t off = wgsl_node_name_span(sev).offset;
    uint32_t len = wgsl_node_name_span(sev).length;
    const char *nm = v->src->bytes + off;
    if (out_name) *out_name = nm;
    if (out_len) *out_len = len;
    if (len == 3 && memcmp(nm, "off", 3) == 0) return 0;
    if (len == 4 && memcmp(nm, "info", 4) == 0) return WGSL_DIAG_INFO;
    if (len == 7 && memcmp(nm, "warning", 7) == 0) return WGSL_DIAG_WARNING;
    if (len == 5 && memcmp(nm, "error", 5) == 0) return WGSL_DIAG_ERROR;
    return -1;
}

void check_diagnostic_severity(
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
void check_attr(
    WGSLValidator *v, WGSLNode *attr, uint32_t ctx_bit)
{
    if (!attr || attr->child_count == 0) return;
    WGSLNode *name_node = attr->children[0];
    if (!name_node) return;
    uint32_t off = wgsl_node_name_span(name_node).offset;
    uint32_t len = wgsl_node_name_span(name_node).length;
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
                uint32_t roff = wgsl_node_name_span(rule).offset;
                uint32_t rlen = wgsl_node_name_span(rule).length;
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
