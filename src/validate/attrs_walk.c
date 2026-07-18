/**
 * @file attrs_walk.c — attribute range walk + statement attr validation
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

void attr_name_span(
    const WGSLValidator *v, const WGSLNode *attr,
    const char **out_name, uint32_t *out_len)
{
    *out_name = NULL;
    *out_len  = 0;
    if (!attr || attr->child_count == 0) return;
    const WGSLNode *an = attr->children[0];
    if (!an || an->kind != WGSL_NODE_EXPR_IDENT) return;
    uint32_t off = wgsl_node_name_span(an).offset;
    uint32_t len = wgsl_node_name_span(an).length;
    *out_name = v->src->bytes + off;
    *out_len  = len;
}

/* True iff attr is `@builtin(position)`. */
int attr_is_builtin_position(
    const WGSLValidator *v, const WGSLNode *attr)
{
    const char *nm; uint32_t nl;
    attr_name_span(v, attr, &nm, &nl);
    if (!(nl == 7 && memcmp(nm, "builtin", 7) == 0)) return 0;
    if (attr->child_count < 2) return 0;
    const WGSLNode *av = attr->children[1];
    if (!av || av->kind != WGSL_NODE_EXPR_IDENT) return 0;
    uint32_t voff = wgsl_node_name_span(av).offset;
    uint32_t vlen = wgsl_node_name_span(av).length;
    return vlen == 8 && memcmp(v->src->bytes + voff, "position", 8) == 0;
}

int attr_name_is(
    const WGSLValidator *v, const WGSLNode *attr, const char *want)
{
    const char *nm; uint32_t nl;
    attr_name_span(v, attr, &nm, &nl);
    size_t wl = strlen(want);
    return nl == wl && memcmp(nm, want, wl) == 0;
}

int attr_is_entry_point_io(
    const WGSLValidator *v, const WGSLNode *attr)
{
    return attr_name_is(v, attr, "location") ||
           attr_name_is(v, attr, "builtin") ||
           attr_name_is(v, attr, "interpolate") ||
           attr_name_is(v, attr, "invariant") ||
           attr_name_is(v, attr, "blend_src");
}

void check_non_entry_io_attrs(
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

int function_has_stage_attr(
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
void check_attrs_range(
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

void check_attrs_array(
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

int val_stmt_kind_accepts_attrs(WGSLNodeKind k) {
    return k == WGSL_NODE_STMT_COMPOUND ||
           k == WGSL_NODE_STMT_IF ||
           k == WGSL_NODE_STMT_WHILE ||
           k == WGSL_NODE_STMT_FOR ||
           k == WGSL_NODE_STMT_LOOP ||
           k == WGSL_NODE_STMT_CONTINUING ||
           k == WGSL_NODE_STMT_SWITCH;
}

void validate_statement_attrs_node(WGSLValidator *v, WGSLNode *n) {
    if (!n) return;
    if ((n->flags & WGSL_FLAG_HAS_ATTRS) &&
        val_stmt_kind_accepts_attrs((WGSLNodeKind)n->kind))
    {
        uint32_t attr_count = wgsl_stmt_attr_count(n);
        WGSLNode **attrs = wgsl_stmt_attrs(n);
        check_attrs_array(v, attrs, attr_count, AC_STMT);
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        validate_statement_attrs_node(v, n->children[i]);
    }
}

int diagnostic_rule_name(
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
    uint32_t rule_parts = wgsl_diag_rule_parts(dir);
    if (rule_parts != 1) return 0;
    const WGSLNode *rule = dir->children[1];
    if (!rule || rule->kind != WGSL_NODE_EXPR_IDENT) return 0;
    uint32_t off = wgsl_node_name_span(rule).offset;
    uint32_t len = wgsl_node_name_span(rule).length;
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

