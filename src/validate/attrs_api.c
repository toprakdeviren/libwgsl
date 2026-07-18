/**
 * @file attrs_api.c — validate_attributes entry + public attr helpers
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
            uint32_t FA = wgsl_fn_attr_count(n);
            uint32_t P  = wgsl_fn_param_count(n);
            uint32_t RA = wgsl_fn_ret_attr_count(n);
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
                uint32_t PA = wgsl_param_attr_count(p);
                check_attrs_range(v, p, 0, PA, AC_FN_PARAM);
                if (!is_entry) check_non_entry_io_attrs(v, p, 0, PA);
            }
            break;
        }

        case WGSL_NODE_DECL_VAR: {
            uint32_t A = wgsl_varlike_attr_count(n);
            check_attrs_range(v, n, 0, A, AC_VAR_MODULE);
            break;
        }

        case WGSL_NODE_DECL_OVERRIDE: {
            uint32_t A = wgsl_varlike_attr_count(n);
            check_attrs_range(v, n, 0, A, AC_OVERRIDE);
            break;
        }

        case WGSL_NODE_DECL_STRUCT: {
            for (uint32_t k = 0; k < n->child_count; k++) {
                WGSLNode *m = n->children[k];
                if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
                uint32_t MA = wgsl_member_attr_count(m);
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
        uint32_t off = wgsl_node_name_span(nm).offset;
        uint32_t len = wgsl_node_name_span(nm).length;
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
    int    saved_cev   = v->cev->store.had_error;
    WGSLValue val = {0};
    int ok = wgsl_consteval_expr(v->cev, attr->children[1], &val) &&
             val.kind == WGSL_VAL_INT;
    if (!ok) {
        v->diag->count       = saved_count;
        v->diag->error_count = saved_errs;
        v->cev->store.had_error    = saved_cev;
        return 0;
    }
    *out = val.u.i;
    return 1;
}

/* IOState typedef moved to io.c (only used by check_io_slot / check_io_type). */


