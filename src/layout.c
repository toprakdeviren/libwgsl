/**
 * @file layout.c — Memory layout calculator (WGSL §14.4).
 *
 * Implements the per-AS layout rules from spec §14.4.2 (struct member
 * offsets), §14.4.3 (array stride), and §14.4.5 (uniform AS extra
 * alignment).  See `internal/layout.h` for the API contract.
 */
#include "internal/layout.h"
#include "internal/check.h"
#include "internal/consteval.h"
#include "internal/source.h"
#include "internal/ast.h"
#include "internal/diag.h"

#include <stdint.h>
#include <string.h>

static uint32_t round_up(uint32_t align, uint32_t value) {
    if (align == 0) return value;
    return ((value + align - 1) / align) * align;
}

/* Walk a member node's attribute range looking for `@<want>(...)`. */
static const WGSLNode *find_attr_named(
    const WGSLLayoutCtx *cx, const WGSLNode *parent,
    uint32_t first, uint32_t count, const char *want)
{
    size_t wl = strlen(want);
    for (uint32_t i = 0; i < count && first + i < parent->child_count; i++) {
        const WGSLNode *a = parent->children[first + i];
        if (!a || a->kind != WGSL_NODE_ATTRIBUTE) continue;
        if (a->child_count == 0) continue;
        const WGSLNode *nm = a->children[0];
        if (!nm) continue;
        uint32_t off = wgsl_node_name_span(nm).offset;
        uint32_t len = wgsl_node_name_span(nm).length;
        if (len == wl && memcmp(cx->src->bytes + off, want, wl) == 0) {
            return a;
        }
    }
    return NULL;
}

/* Const-eval the attribute's first argument as an int.  Returns 0 on
 * failure (callers default to the type's natural alignment / size). */
static long long attr_int_arg(
    const WGSLLayoutCtx *cx, const WGSLNode *attr)
{
    if (!attr || attr->child_count < 2) return 0;
    WGSLValue val = {0};
    if (wgsl_consteval_expr(cx->cev, attr->children[1], &val)) {
        if (val.kind == WGSL_VAL_INT) return val.u.i;
    }
    return 0;
}

uint32_t wgsl_layout_align_of(const WGSLLayoutCtx *cx, const WGSLTypeInfo *t, WGSLAddressSpace as) {
    if (!t) return 1;
    switch ((WGSLTypeKind)t->kind) {
    case WGSL_TYPE_BOOL: return 1;
    case WGSL_TYPE_F16:  return 2;
    case WGSL_TYPE_I32: case WGSL_TYPE_U32: case WGSL_TYPE_F32: return 4;
    case WGSL_TYPE_VEC: {
        uint32_t e = wgsl_layout_align_of(cx, (const WGSLTypeInfo *)t->ref, as);
        return (t->width == 2) ? 2 * e : 4 * e;
    }
    case WGSL_TYPE_MAT: {
        uint32_t e = wgsl_layout_align_of(cx, (const WGSLTypeInfo *)t->ref, as);
        return (t->rows == 2) ? 2 * e : 4 * e;
    }
    case WGSL_TYPE_ARRAY:
        return wgsl_layout_align_of(cx, (const WGSLTypeInfo *)t->ref, as);
    case WGSL_TYPE_ATOMIC:
        return wgsl_layout_align_of(cx, (const WGSLTypeInfo *)t->ref, as);
    case WGSL_TYPE_STRUCT: {
        uint32_t a = 0, s = 0;
        uint32_t offs[64];
        if (!wgsl_layout_struct(cx, t, as, offs, 64, &a, &s)) return 4;
        return a;
    }
    default: return 4;
    }
}

uint32_t wgsl_layout_size_of(const WGSLLayoutCtx *cx, const WGSLTypeInfo *t, WGSLAddressSpace as) {
    if (!t) return 0;
    switch ((WGSLTypeKind)t->kind) {
    case WGSL_TYPE_BOOL: return 1;
    case WGSL_TYPE_F16:  return 2;
    case WGSL_TYPE_I32: case WGSL_TYPE_U32: case WGSL_TYPE_F32: return 4;
    case WGSL_TYPE_VEC: {
        uint32_t e = wgsl_layout_size_of(cx, (const WGSLTypeInfo *)t->ref, as);
        return t->width * e;
    }
    case WGSL_TYPE_MAT: {
        const WGSLTypeInfo *elem = (const WGSLTypeInfo *)t->ref;
        uint32_t es = wgsl_layout_size_of(cx, elem, as);
        uint32_t ea = wgsl_layout_align_of(cx, elem, as);
        uint32_t col_align = (t->rows == 2) ? 2 * ea : 4 * ea;
        uint32_t col_stride = round_up(col_align, t->rows * es);
        return (t->width - 1) * col_stride + t->rows * es;
    }
    case WGSL_TYPE_ARRAY: {
        const WGSLTypeInfo *elem = (const WGSLTypeInfo *)t->ref;
        if (t->array_len == 0) return 0;   /* runtime-sized */
        uint32_t stride = wgsl_layout_array_stride(cx, elem, as);
        uint32_t es = wgsl_layout_size_of(cx, elem, as);
        return (t->array_len - 1) * stride + es;
    }
    case WGSL_TYPE_ATOMIC:
        return wgsl_layout_size_of(cx, (const WGSLTypeInfo *)t->ref, as);
    case WGSL_TYPE_STRUCT: {
        uint32_t a = 0, s = 0;
        uint32_t offs[64];
        if (!wgsl_layout_struct(cx, t, as, offs, 64, &a, &s)) return 0;
        return s;
    }
    default: return 0;
    }
}

uint32_t wgsl_layout_required_align_of(const WGSLLayoutCtx *cx, const WGSLTypeInfo *t, WGSLAddressSpace as) {
    uint32_t a = wgsl_layout_align_of(cx, t, as);
    if (as == WGSL_AS_UNIFORM &&
        t && (t->kind == WGSL_TYPE_STRUCT || t->kind == WGSL_TYPE_ARRAY))
    {
        uint32_t bumped = round_up(16, a);
        if (bumped > a) a = bumped;
    }
    return a;
}

uint32_t wgsl_layout_array_stride(const WGSLLayoutCtx *cx, const WGSLTypeInfo *elem, WGSLAddressSpace as) {
    if (!elem) return 0;
    uint32_t es = wgsl_layout_size_of(cx, elem, as);
    uint32_t ea = wgsl_layout_required_align_of(cx, elem, as);
    return round_up(ea, es);
}

int wgsl_layout_struct(
    const WGSLLayoutCtx *cx,
    const WGSLTypeInfo *st, WGSLAddressSpace as,
    uint32_t *offsets_out, uint32_t cap,
    uint32_t *out_align, uint32_t *out_size)
{
    if (!st || st->kind != WGSL_TYPE_STRUCT) return 0;
    const WGSLNode *sd = (const WGSLNode *)st->ref;
    if (!sd) return 0;
    /* Bound recursion on the resolved type graph: size_of/align_of re-enter
     * this function for struct-typed members, so a deep struct chain
     * (struct S1{a:S0}…) would overflow the stack.  cx->depth increments once
     * per struct level; fail (size 0) past the cap.  The over-deep type is
     * rejected by the checker/validator's own type-graph guards. */
    if (cx->depth > WGSL_MAX_AST_DEPTH) return 0;
    WGSLLayoutCtx cxd = *cx;
    cxd.depth = cx->depth + 1;
    uint32_t struct_align = 1;
    uint32_t prev_just_after = 0;
    uint32_t midx = 0;
    for (uint32_t i = 0; i < sd->child_count; i++) {
        WGSLNode *m = sd->children[i];
        if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
        uint32_t MA = wgsl_member_attr_count(m);
        const WGSLTypeInfo *mt = NULL;
        if (MA < m->child_count) mt = wgsl_typecheck_type_of(cx->tc, m->children[MA]);
        if (!mt) return 0;
        /* Per-member alignment + size.  For a struct-typed member compute
         * BOTH from a single wgsl_layout_struct call: doing required-align and
         * size separately would each re-walk the member's whole subtree, i.e.
         * O(2^depth) on a nested-struct chain (align_of and size_of both
         * re-enter wgsl_layout_struct).  The depth-incremented ctx additionally
         * caps the recursion (see cx->depth above). */
        uint32_t a, s;
        if (mt->kind == WGSL_TYPE_STRUCT) {
            uint32_t moffs[64], ma = 1, ms = 0;
            if (!wgsl_layout_struct(&cxd, mt, as, moffs, 64, &ma, &ms)) return 0;
            a = ma;
            if (as == WGSL_AS_UNIFORM) {   /* §14.4.5 roundUp(16, AlignOf) */
                uint32_t bumped = round_up(16, ma);
                if (bumped > a) a = bumped;
            }
            s = ms;
        } else {
            a = wgsl_layout_required_align_of(&cxd, mt, as);
            s = wgsl_layout_size_of(&cxd, mt, as);
        }
        const WGSLNode *align_attr = find_attr_named(cx, m, 0, MA, "align");
        if (align_attr) {
            long long an = attr_int_arg(cx, align_attr);
            if (an > 0) a = (uint32_t)an;
        }
        const WGSLNode *size_attr = find_attr_named(cx, m, 0, MA, "size");
        if (size_attr) {
            long long sn = attr_int_arg(cx, size_attr);
            if (sn > 0) s = (uint32_t)sn;
        }
        uint32_t off = round_up(a, prev_just_after);
        /* Store the offset only if it fits the caller's array; align/size are
         * still accumulated for every member so SizeOf/AlignOf stay correct
         * even for a struct with more members than `cap` (callers that need
         * all offsets pass an array sized to the member count). */
        if (offsets_out && midx < cap) offsets_out[midx] = off;
        prev_just_after = off + s;
        if (a > struct_align) struct_align = a;
        midx += 1;
    }
    if (out_align) *out_align = struct_align;
    if (out_size)  *out_size  = round_up(struct_align, prev_just_after);
    return 1;
}
