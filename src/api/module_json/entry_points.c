/**
 * @file entry_points.c — entry IO reflection + entry_points[] emission
 */
#include "internal/module_json_priv.h"

/* IO reflection.
 *
 * Per-entry vertex inputs / fragment outputs / builtins with their
 * @location / @builtin / @interpolate / @blend_src / @invariant — the
 * data a host needs to build GPUVertexState / fragment target layout.
 * Nothing is persisted by validation (io.c is a transient walk), so we
 * re-walk the entry's params + return, expanding struct-typed IO into
 * its members (where the per-slot attributes live). */

void jb_put_ident_name(WGSLResult *r, JsonBuf *j, const WGSLNode *id) {
    if (id && id->kind == WGSL_NODE_EXPR_IDENT) {
        uint32_t o = wgsl_node_name_span(id).offset;
        uint32_t l = wgsl_node_name_span(id).length;
        jb_put_span(j, &r->source, o, l);
    } else {
        jb_puts(j, "\"\"");
    }
}

/* Emit one IO slot: its type + any location/builtin/interpolate/
 * blend_src/invariant attributes found on `ap[afirst .. afirst+acount)`. */
static void emit_io_slot(WGSLResult *r, JsonBuf *j, int *first,
                         const WGSLNode *ap, uint32_t afirst, uint32_t acount,
                         int have_name, uint32_t noff, uint32_t nlen,
                         WGSLTypeInfo *type) {
    if (!*first) jb_putc(j, ',');
    *first = 0;
    jb_putc(j, '{');
    if (have_name) {
        jb_puts(j, "\"name\":");
        jb_put_span(j, &r->source, noff, nlen);
        jb_putc(j, ',');
    }
    jb_puts(j, "\"type\":");
    jb_put_type(j, type);

    WGSLNode *loc = json_find_attr(r, ap, afirst, acount, "location");
    if (loc && loc->child_count >= 2) {
        jb_puts(j, ",\"location\":");
        jb_put_int(j, wgsl_attr_int_arg(r, loc->children[1]));
    }
    WGSLNode *blt = json_find_attr(r, ap, afirst, acount, "builtin");
    if (blt && blt->child_count >= 2) {
        jb_puts(j, ",\"builtin\":");
        jb_put_ident_name(r, j, blt->children[1]);
    }
    WGSLNode *itp = json_find_attr(r, ap, afirst, acount, "interpolate");
    if (itp && itp->child_count >= 2) {
        jb_puts(j, ",\"interpolate\":{\"type\":");
        jb_put_ident_name(r, j, itp->children[1]);
        if (itp->child_count >= 3) {
            jb_puts(j, ",\"sampling\":");
            jb_put_ident_name(r, j, itp->children[2]);
        }
        jb_putc(j, '}');
    }
    WGSLNode *bs = json_find_attr(r, ap, afirst, acount, "blend_src");
    if (bs && bs->child_count >= 2) {
        jb_puts(j, ",\"blend_src\":");
        jb_put_int(j, wgsl_attr_int_arg(r, bs->children[1]));
    }
    if (json_find_attr(r, ap, afirst, acount, "invariant"))
        jb_puts(j, ",\"invariant\":true");

    jb_putc(j, '}');
}

/* Emit IO for a param/return `type` and attribute range; struct-typed
 * IO fans out into its members (each member carries its own attrs). */
static void emit_io_for_type(WGSLResult *r, JsonBuf *j, int *first,
                             WGSLTypeInfo *type,
                             const WGSLNode *ap, uint32_t afirst, uint32_t acount,
                             int have_name, uint32_t noff, uint32_t nlen) {
    if (type && type->kind == WGSL_TYPE_STRUCT && type->ref) {
        WGSLNode *sd = (WGSLNode *)type->ref;
        for (uint32_t k = 0; k < sd->child_count; k++) {
            WGSLNode *m = sd->children[k];
            if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            uint32_t MA = wgsl_member_attr_count(m);
            WGSLTypeInfo *mt = (MA < m->child_count)
                ? wgsl_typecheck_type_of(&r->tc, m->children[MA]) : NULL;
            uint32_t mno = wgsl_node_name_span(m).offset;
            uint32_t mnl = wgsl_node_name_span(m).length;
            emit_io_slot(r, j, first, m, 0, MA, 1, mno, mnl, mt);
        }
    } else {
        emit_io_slot(r, j, first, ap, afirst, acount, have_name, noff, nlen, type);
    }
}

void emit_entry_io(WGSLResult *r, JsonBuf *j, const WGSLNode *fn) {
    uint32_t FA = wgsl_fn_attr_count(fn);
    uint32_t P  = wgsl_fn_param_count(fn);
    uint32_t RA = wgsl_fn_ret_attr_count(fn);
    uint32_t HR = wgsl_fn_has_return_type(fn);

    jb_puts(j, ",\"inputs\":[");
    int first = 1;
    for (uint32_t k = 0; k < P; k++) {
        uint32_t ci = FA + k;
        if (ci >= fn->child_count) break;
        WGSLNode *p = fn->children[ci];
        if (!p || p->kind != WGSL_NODE_DECL_PARAM) continue;
        uint32_t PA = wgsl_param_attr_count(p);
        WGSLTypeInfo *pt = (PA < p->child_count)
            ? wgsl_typecheck_type_of(&r->tc, p->children[PA]) : NULL;
        uint32_t pno = wgsl_node_name_span(p).offset;
        uint32_t pnl = wgsl_node_name_span(p).length;
        emit_io_for_type(r, j, &first, pt, p, 0, PA, 1, pno, pnl);
    }
    jb_putc(j, ']');

    jb_puts(j, ",\"outputs\":[");
    first = 1;
    if (HR) {
        uint32_t rti = FA + P + RA;
        WGSLTypeInfo *rt = (rti < fn->child_count)
            ? wgsl_typecheck_type_of(&r->tc, fn->children[rti]) : NULL;
        /* return value has no name; its attrs sit on the fn node at
         * [FA+P, FA+P+RA). */
        emit_io_for_type(r, j, &first, rt, fn, FA + P, RA, 0, 0, 0);
    }
    jb_putc(j, ']');
}

void emit_entry_points(WGSLResult *r, JsonBuf *j) {
    jb_puts(j, "\"entry_points\":[");
    int first = 1;
    if (!r->ast.root) { jb_puts(j, "]"); return; }
    WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n || n->kind != WGSL_NODE_DECL_FUNCTION) continue;

        uint32_t FA = wgsl_fn_attr_count(n);
        const char *stage = json_stage_for(r, n, FA);
        if (!stage) continue;

        if (!first) jb_putc(j, ',');
        first = 0;
        jb_puts(j, "{\"name\":");
        uint32_t no = wgsl_node_name_span(n).offset;
        uint32_t nl = wgsl_node_name_span(n).length;
        jb_put_span(j, &r->source, no, nl);
        jb_puts(j, ",\"stage\":\"");
        jb_puts(j, stage);
        jb_putc(j, '"');

        if (strcmp(stage, "compute") == 0) {
            WGSLNode *wgs = json_find_attr(r, n, 0, FA, "workgroup_size");
            if (wgs) {
                jb_puts(j, ",\"workgroup_size\":[");
                /* wgs->children: [name_ident, arg1, arg2?, arg3?] */
                jb_put_wgs_dim(r, j, wgs->child_count > 1 ? wgs->children[1] : NULL);
                jb_putc(j, ',');
                jb_put_wgs_dim(r, j, wgs->child_count > 2 ? wgs->children[2] : NULL);
                jb_putc(j, ',');
                jb_put_wgs_dim(r, j, wgs->child_count > 3 ? wgs->children[3] : NULL);
                jb_putc(j, ']');
            }
        }
        emit_entry_usage(r, j, n, stage);
        emit_entry_io(r, j, n);
        jb_putc(j, '}');
    }
    jb_putc(j, ']');
}
