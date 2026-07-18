/**
 * @file exec_leaf.c — expression/lvalue eval and straight-line leaf actions
 *
 * Value model: first-class WGSL_VAL_PTR {root, path, lane}.  All shared
 * stores funnel through mem_store (race_check; coarse whole-root when path
 * precision is missing — never skip).  Lane is baked at `&` and asserted on
 * function/private deref.
 */
#include "internal/interp_priv.h"
#include "internal/layout.h"

/* expression + lvalue evaluation. */

int eval(Interp *ip, WGSLNode *e, WGSLValue *out) {
    memset(out, 0, sizeof *out);
    return wgsl_consteval_expr(ip->cev, e, out);
}

/* Declaration nodes don't stash their symbol in payload[2] (that slot is
 * reused per node kind — e.g. DECL_FUNCTION keeps return-attr counts there),
 * so a decl → symbol lookup goes through the resolver's flat table.  Only
 * IDENT *references* carry the symbol in payload[2] (→ resolved_symbol). */
WGSLSymbol *decl_symbol(WGSLConstEvaluator *cev, const WGSLNode *decl) {
    if (!cev || !cev->res) return NULL;
    return wgsl_resolver_symbol_for_decl(cev->res, decl);
}

/* Byte size of a scalar/vector/matrix leaf (cost-model store accounting). */
uint32_t interp_leaf_bytes(const WGSLTypeInfo *t) {
    if (!t) return 0;
    switch ((WGSLTypeKind)t->kind) {
    case WGSL_TYPE_F16: return 2;
    case WGSL_TYPE_BOOL: case WGSL_TYPE_I32: case WGSL_TYPE_U32:
    case WGSL_TYPE_F32:  case WGSL_TYPE_ABSTRACT_INT: case WGSL_TYPE_ABSTRACT_FLOAT: return 4;
    case WGSL_TYPE_VEC: return (uint32_t)t->width * interp_leaf_bytes((const WGSLTypeInfo *)t->ref);
    case WGSL_TYPE_MAT: return (uint32_t)t->width * t->rows * interp_leaf_bytes((const WGSLTypeInfo *)t->ref);
    case WGSL_TYPE_ATOMIC:
        return interp_leaf_bytes((const WGSLTypeInfo *)t->ref);
    default: return 0;
    }
}

/* Walk an lvalue down to its root declaration symbol (address-space check). */
WGSLSymbol *lvalue_root_symbol(WGSLNode *lhs) {
    while (lhs) {
        if (lhs->kind == WGSL_NODE_EXPR_IDENT) return wgsl_node_resolved_symbol(lhs);
        if (lhs->kind == WGSL_NODE_EXPR_PAREN || lhs->kind == WGSL_NODE_EXPR_INDEX ||
            lhs->kind == WGSL_NODE_EXPR_MEMBER || lhs->kind == WGSL_NODE_EXPR_INDIRECTION)
            lhs = lhs->child_count ? lhs->children[0] : NULL;
        else return NULL;
    }
    return NULL;
}

static int ptr_root_is_shared(const WGSLSymbol *root) {
    if (!root) return 0;
    return root->as == WGSL_AS_STORAGE || root->as == WGSL_AS_UNIFORM ||
           root->as == WGSL_AS_WORKGROUP;
}

static int ptr_root_is_per_lane(const WGSLSymbol *root) {
    return root && wgsl_consteval_sym_is_per_lane(root);
}

/* Unwrap REF (and PTR) to the store type. */
static WGSLTypeInfo *ptr_store_type(const WGSLSymbol *root) {
    if (!root || !root->type) return NULL;
    WGSLTypeInfo *t = root->type;
    while (t && (t->kind == WGSL_TYPE_REF || t->kind == WGSL_TYPE_PTR))
        t = (WGSLTypeInfo *)t->ref;
    return t;
}

/* Atomic locations are defined under concurrent access — not data races. */
static int ptr_is_atomic_loc(const WGSLValue *ptr) {
    if (!ptr || ptr->kind != WGSL_VAL_PTR || !ptr->u.ptr.root) return 0;
    WGSLTypeInfo *t = ptr_store_type(ptr->u.ptr.root);
    if (!t) return 0;
    if (t->kind == WGSL_TYPE_ATOMIC) return 1;
    if (ptr->u.ptr.has_index && t->kind == WGSL_TYPE_ARRAY) {
        WGSLTypeInfo *e = (WGSLTypeInfo *)t->ref;
        if (e && e->kind == WGSL_TYPE_ATOMIC) return 1;
    }
    return 0;
}

/* Lane isolation: function/private addresses bake cur_lane; deref must match. */
static int ptr_lane_ok(Interp *ip, const WGSLValue *ptr) {
    if (!ptr || ptr->kind != WGSL_VAL_PTR || !ptr->u.ptr.root) return 0;
    if (!ptr_root_is_per_lane(ptr->u.ptr.root)) return 1;
    if (ptr->u.ptr.lane != ip->cev->simt.cur_lane) {
        /* Loud fail — a scheduler bug must not silently touch another lane. */
        return 0;
    }
    return 1;
}

/* Race offset for a PTR: precise 1-D index×stride when known, else coarse 0
 * (whole-root cell).  Never returns a "skip" sentinel — under-report is banned. */
static uint32_t ptr_race_offset(const WGSLValue *ptr, uint32_t elem_bytes) {
    if (!ptr || ptr->kind != WGSL_VAL_PTR) return 0;
    if (ptr->u.ptr.race_coarse || !ptr->u.ptr.has_index) return 0;
    if (ptr->u.ptr.index < 0) return 0;
    uint32_t eb = elem_bytes ? elem_bytes : 1u;
    return (uint32_t)ptr->u.ptr.index * eb;
}

static void ptr_race(Interp *ip, const WGSLValue *ptr, int is_write, uint32_t line,
                     uint32_t elem_bytes) {
    if (!ip || !ptr || ptr->kind != WGSL_VAL_PTR || !ptr->u.ptr.root) return;
    if (!ptr_root_is_shared(ptr->u.ptr.root)) return;
    /* Atomics: concurrent RMW is well-defined, not a data race. */
    if (ptr_is_atomic_loc(ptr)) return;
    race_check(ip, ptr->u.ptr.root, ptr_race_offset(ptr, elem_bytes), is_write, line);
}

static void ptr_init(WGSLValue *out, const WGSLSymbol *root, uint32_t lane,
                     int has_index, int32_t index, int race_coarse) {
    memset(out, 0, sizeof *out);
    out->kind = WGSL_VAL_PTR;
    out->u.ptr.root = root;
    out->u.ptr.lane = lane;
    out->u.ptr.has_index = has_index ? 1u : 0u;
    out->u.ptr.index = has_index ? index : 0;
    out->u.ptr.race_coarse = race_coarse ? 1u : 0u;
    if (root && root->type) {
        WGSLTypeInfo *t = root->type;
        if (t->kind == WGSL_TYPE_REF) t = (WGSLTypeInfo *)t->ref;
        /* Pointer value's type is not required for resolve; leave NULL. */
        (void)t;
    }
}

/* Build a WGSL_VAL_PTR from an AST lvalue.  The compact path stores an empty
 * path or one 1-D index off the root ident.  Deeper chains (member, nested
 * index) set race_coarse so race still fires on the whole root. */
int ptr_from_lvalue(Interp *ip, WGSLNode *lhs, WGSLValue *out_ptr) {
    memset(out_ptr, 0, sizeof *out_ptr);
    if (!lhs) return 0;
    while (lhs && lhs->kind == WGSL_NODE_EXPR_PAREN)
        lhs = lhs->child_count ? lhs->children[0] : NULL;
    if (!lhs) return 0;

    /* *p = …  →  the pointer value itself is the address. */
    if (lhs->kind == WGSL_NODE_EXPR_INDIRECTION) {
        WGSLValue p;
        if (!lhs->child_count || !eval(ip, lhs->children[0], &p) ||
            p.kind != WGSL_VAL_PTR)
            return 0;
        *out_ptr = p;
        return 1;
    }

    WGSLSymbol *root = lvalue_root_symbol(lhs);
    if (!root) return 0;
    uint32_t lane = ip->cev->simt.cur_lane;

    if (lhs->kind == WGSL_NODE_EXPR_IDENT) {
        ptr_init(out_ptr, root, lane, 0, 0, 0);
        return 1;
    }

    /* Direct 1-D: root[idx] */
    if (lhs->kind == WGSL_NODE_EXPR_INDEX && lhs->child_count >= 2) {
        WGSLNode *base = lhs->children[0];
        while (base && base->kind == WGSL_NODE_EXPR_PAREN)
            base = base->child_count ? base->children[0] : NULL;
        if (base && base->kind == WGSL_NODE_EXPR_IDENT) {
            WGSLValue idx;
            if (!eval(ip, lhs->children[1], &idx) || idx.kind != WGSL_VAL_INT)
                return 0;
            if (idx.u.i < 0) return 0;
            ptr_init(out_ptr, root, lane, 1, (int32_t)idx.u.i, 0);
            return 1;
        }
        /* Nested index / pointer index: coarse whole-root race. */
        ptr_init(out_ptr, root, lane, 0, 0, 1);
        return 1;
    }

    /* Member / swizzle / other: slot resolve still works; race is coarse. */
    if (lhs->kind == WGSL_NODE_EXPR_MEMBER) {
        ptr_init(out_ptr, root, lane, 0, 0, 1);
        return 1;
    }
    return 0;
}

/* Resolve PTR to the mutable arena-owned slot it names. */
WGSLValue *mem_slot(Interp *ip, const WGSLValue *ptr) {
    if (!ip || !ptr || ptr->kind != WGSL_VAL_PTR || !ptr->u.ptr.root) return NULL;
    if (!ptr_lane_ok(ip, ptr)) return NULL;
    WGSLValue *base = (WGSLValue *)wgsl_consteval_value_of(ip->cev, ptr->u.ptr.root);
    if (!base) return NULL;
    if (!ptr->u.ptr.has_index) return base;
    if (base->kind != WGSL_VAL_ARRAY && base->kind != WGSL_VAL_VEC) return NULL;
    if (ptr->u.ptr.index < 0 || (uint64_t)ptr->u.ptr.index >= base->u.agg.count)
        return NULL;
    return &base->u.agg.elems[(uint32_t)ptr->u.ptr.index];
}

int mem_load(Interp *ip, const WGSLValue *ptr, WGSLValue *out, uint32_t line) {
    memset(out, 0, sizeof *out);
    WGSLValue *slot = mem_slot(ip, ptr);
    if (!slot) return 0;
    *out = *slot;
    uint32_t eb = interp_leaf_bytes(slot->type ? slot->type : out->type);
    if (ptr->u.ptr.root &&
        (ptr->u.ptr.root->as == WGSL_AS_STORAGE || ptr->u.ptr.root->as == WGSL_AS_UNIFORM))
        ip->cev->cost.bytes_loaded += eb;
    ptr_race(ip, ptr, 0, line, eb ? eb : 1u);
    return 1;
}

int mem_store(Interp *ip, const WGSLValue *ptr, const WGSLValue *v, uint32_t line) {
    if (!ip || !ptr || !v || ptr->kind != WGSL_VAL_PTR || !ptr->u.ptr.root) return 0;
    if (!ptr_lane_ok(ip, ptr)) return 0;

    /* Empty path → rebind whole root (deep-copy into environment). */
    if (!ptr->u.ptr.has_index && !ptr->u.ptr.race_coarse) {
        WGSLValue tmp = *v;
        WGSLTypeInfo *dt = ptr_store_type(ptr->u.ptr.root);
        /* atomic<T> stores hold a T value; materialize against T, not atomic. */
        if (dt && dt->kind == WGSL_TYPE_ATOMIC) dt = (WGSLTypeInfo *)dt->ref;
        if (dt) wgsl_consteval_materialize(ip->cev, &tmp, dt, NULL);
        if (!wgsl_consteval_set_value(ip->cev, ptr->u.ptr.root, &tmp)) return 0;
        uint32_t eb = interp_leaf_bytes(dt ? dt : v->type);
        if (ptr->u.ptr.root->as == WGSL_AS_STORAGE || ptr->u.ptr.root->as == WGSL_AS_UNIFORM)
            ip->cev->cost.bytes_stored += eb;
        ptr_race(ip, ptr, 1, line, eb ? eb : 1u);
        return 1;
    }

    WGSLValue *slot = mem_slot(ip, ptr);
    if (!slot) return 0;
    WGSLValue tmp = *v;
    WGSLTypeInfo *et = slot->type;
    if (et) wgsl_consteval_materialize(ip->cev, &tmp, et, NULL);
    *slot = tmp;
    uint32_t eb = interp_leaf_bytes(et ? et : v->type);
    if (ptr->u.ptr.root->as == WGSL_AS_STORAGE || ptr->u.ptr.root->as == WGSL_AS_UNIFORM)
        ip->cev->cost.bytes_stored += eb;
    ptr_race(ip, ptr, 1, line, eb ? eb : 1u);
    return 1;
}

/* Full AST walk for member/swizzle (and nested) write targets.  PTR path for
 * race is filled via ptr_from_lvalue (coarse when not empty/1-D). */
static WGSLValue *lvalue_slot_walk(Interp *ip, WGSLNode *lhs) {
    if (!lhs) return NULL;
    switch ((WGSLNodeKind)lhs->kind) {
    case WGSL_NODE_EXPR_IDENT: {
        WGSLSymbol *s = wgsl_node_resolved_symbol(lhs);
        if (!s) return NULL;
        return (WGSLValue *)wgsl_consteval_value_of(ip->cev, s);
    }
    case WGSL_NODE_EXPR_PAREN:
        return lhs->child_count ? lvalue_slot_walk(ip, lhs->children[0]) : NULL;
    case WGSL_NODE_EXPR_INDIRECTION: {
        WGSLValue p;
        if (!lhs->child_count || !eval(ip, lhs->children[0], &p) ||
            p.kind != WGSL_VAL_PTR)
            return NULL;
        return mem_slot(ip, &p);
    }
    case WGSL_NODE_EXPR_INDEX: {
        WGSLValue *base = lvalue_slot_walk(ip, lhs->children[0]);
        if (!base) return NULL;
        WGSLValue idx;
        if (!eval(ip, lhs->children[1], &idx) || idx.kind != WGSL_VAL_INT) return NULL;
        if (base->kind != WGSL_VAL_ARRAY && base->kind != WGSL_VAL_VEC) return NULL;
        if (idx.u.i < 0 || (uint64_t)idx.u.i >= base->u.agg.count) return NULL;
        return &base->u.agg.elems[(uint32_t)idx.u.i];
    }
    case WGSL_NODE_EXPR_MEMBER: {
        WGSLValue *base = lvalue_slot_walk(ip, lhs->children[0]);
        if (!base) return NULL;
        char m[64]; span_text(ip->src, lhs->children[1], m, sizeof m);
        if (base->kind == WGSL_VAL_VEC && m[0] && !m[1]) {
            int idx = swizzle_index_h(m[0]);
            if (idx < 0 || (uint32_t)idx >= base->u.agg.count) return NULL;
            return &base->u.agg.elems[idx];
        }
        if (base->kind == WGSL_VAL_STRUCT && base->type && base->type->kind == WGSL_TYPE_STRUCT) {
            WGSLNode *sd = (WGSLNode *)base->type->ref;
            uint32_t idx = 0, len = (uint32_t)strlen(m);
            for (uint32_t i = 0; sd && i < sd->child_count; i++) {
                WGSLNode *mm = sd->children[i];
                if (!mm || mm->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
                uint32_t off = wgsl_node_name_span(mm).offset;
                uint32_t ml  = wgsl_node_name_span(mm).length;
                if (ml == len && (size_t)off + ml <= ip->src->length &&
                    memcmp(ip->src->bytes + off, m, len) == 0 && idx < base->u.agg.count)
                    return &base->u.agg.elems[idx];
                idx++;
            }
        }
        return NULL;
    }
    default: return NULL;
    }
}

WGSLValue *lvalue_resolve(Interp *ip, WGSLNode *lhs, WGSLValue *out_ptr) {
    if (out_ptr) {
        if (!ptr_from_lvalue(ip, lhs, out_ptr))
            memset(out_ptr, 0, sizeof *out_ptr);
    }
    return lvalue_slot_walk(ip, lhs);
}

/* Runtime address and indirection helpers for const-eval fallback. */

static Interp *interp_from_cev(WGSLConstEvaluator *cev) {
    if (!cev) return NULL;
    if (cev->simt.mem_access_ud) return (Interp *)cev->simt.mem_access_ud;
    if (cev->simt.builtin_call_ud) return (Interp *)cev->simt.builtin_call_ud;
    if (cev->simt.user_call_ud) return (Interp *)cev->simt.user_call_ud;
    return NULL;
}

int wgsl_runtime_eval_addr_of(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out) {
    memset(out, 0, sizeof *out);
    Interp *ip = interp_from_cev(cev);
    if (!ip || !n || !n->child_count || !n->children[0]) return 0;
    return ptr_from_lvalue(ip, n->children[0], out);
}

int wgsl_runtime_eval_indirection(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out) {
    memset(out, 0, sizeof *out);
    Interp *ip = interp_from_cev(cev);
    if (!ip || !n || !n->child_count) return 0;
    WGSLValue p;
    if (!eval(ip, n->children[0], &p) || p.kind != WGSL_VAL_PTR) return 0;
    return mem_load(ip, &p, out, line_of(ip->src, n));
}

/* statement execution. */

WGSLNode *fn_body(WGSLNode *fn) {
    return (fn && fn->child_count) ? fn->children[fn->child_count - 1] : NULL;
}

/* Unwrap REF/PTR store types for layout walks. */
static WGSLTypeInfo *store_type_of_node(Interp *ip, WGSLNode *n) {
    if (!ip->tc || !n) return NULL;
    WGSLTypeInfo *t = wgsl_typecheck_type_of((WGSLTypeChecker *)ip->tc, n);
    while (t && (t->kind == WGSL_TYPE_REF || t->kind == WGSL_TYPE_PTR ||
                 t->kind == WGSL_TYPE_ATOMIC))
        t = (WGSLTypeInfo *)t->ref;
    return t;
}

/* Byte offset of an AST lvalue for race_check, reusing layout.c for struct
 * members (no hand-rolled offset math).  Returns 1 when precise; 0 → caller
 * must still race at offset 0 (coarse whole-root — never skip). */
static int lvalue_race_byte_offset(Interp *ip, WGSLNode *lhs, uint32_t *out_off) {
    *out_off = 0;
    while (lhs && lhs->kind == WGSL_NODE_EXPR_PAREN)
        lhs = lhs->child_count ? lhs->children[0] : NULL;
    if (!lhs) return 0;

    if (lhs->kind == WGSL_NODE_EXPR_IDENT) return 1;

    if (lhs->kind == WGSL_NODE_EXPR_INDIRECTION) {
        /* Compact pointer path: empty/1-D only; nested through *p is coarse. */
        return 0;
    }

    WGSLSymbol *root = lvalue_root_symbol(lhs);
    WGSLAddressSpace as = root ? (WGSLAddressSpace)root->as : WGSL_AS_FUNCTION;
    WGSLLayoutCtx lcx;
    memset(&lcx, 0, sizeof lcx);
    lcx.tc  = (WGSLTypeChecker *)ip->tc;
    lcx.cev = ip->cev;
    lcx.src = ip->src;

    if (lhs->kind == WGSL_NODE_EXPR_INDEX && lhs->child_count >= 2) {
        uint32_t base_off = 0;
        if (!lvalue_race_byte_offset(ip, lhs->children[0], &base_off)) return 0;
        WGSLValue idx;
        if (!eval(ip, lhs->children[1], &idx) || idx.kind != WGSL_VAL_INT ||
            idx.u.i < 0)
            return 0;
        WGSLTypeInfo *bt = store_type_of_node(ip, lhs->children[0]);
        if (!bt) return 0;
        uint32_t stride = 0;
        if (bt->kind == WGSL_TYPE_ARRAY) {
            WGSLTypeInfo *elem = (WGSLTypeInfo *)bt->ref;
            stride = wgsl_layout_array_stride(&lcx, elem, as);
            if (!stride) stride = interp_leaf_bytes(elem);
        } else if (bt->kind == WGSL_TYPE_VEC) {
            stride = interp_leaf_bytes((WGSLTypeInfo *)bt->ref);
        } else {
            return 0;
        }
        if (!stride) stride = 1;
        *out_off = base_off + (uint32_t)idx.u.i * stride;
        return 1;
    }

    if (lhs->kind == WGSL_NODE_EXPR_MEMBER && lhs->child_count >= 2) {
        uint32_t base_off = 0;
        if (!lvalue_race_byte_offset(ip, lhs->children[0], &base_off)) return 0;
        WGSLTypeInfo *bt = store_type_of_node(ip, lhs->children[0]);
        if (!bt) return 0;

        char m[64];
        span_text(ip->src, lhs->children[1], m, sizeof m);

        /* Vector component: logical index × elem size (not layout struct). */
        if (bt->kind == WGSL_TYPE_VEC && m[0] && !m[1]) {
            int ci = swizzle_index_h(m[0]);
            if (ci < 0) return 0;
            uint32_t es = interp_leaf_bytes((WGSLTypeInfo *)bt->ref);
            if (!es) es = 4;
            *out_off = base_off + (uint32_t)ci * es;
            return 1;
        }

        if (bt->kind != WGSL_TYPE_STRUCT) return 0;
        uint32_t offs[64];
        uint32_t al = 0, sz = 0;
        if (!wgsl_layout_struct(&lcx, bt, as, offs, 64, &al, &sz)) return 0;

        WGSLNode *sd = (WGSLNode *)bt->ref;
        uint32_t mi = 0, len = (uint32_t)strlen(m);
        for (uint32_t i = 0; sd && i < sd->child_count; i++) {
            WGSLNode *mm = sd->children[i];
            if (!mm || mm->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            uint32_t off = wgsl_node_name_span(mm).offset;
            uint32_t ml  = wgsl_node_name_span(mm).length;
            if (ml == len && (size_t)off + ml <= ip->src->length &&
                memcmp(ip->src->bytes + off, m, len) == 0) {
                if (mi >= 64) return 0;
                *out_off = base_off + offs[mi];
                return 1;
            }
            mi++;
        }
        return 0;
    }
    return 0;
}

/* Store an already-computed value into an lvalue through mem_store when the
 * compact pointer path is exact (ident / root[i] / *p); otherwise write the walked
 * slot and race with layout-precise offset when available, else coarse
 * whole-root (never skip). */
void store_lvalue(Interp *ip, WGSLNode *lhs, WGSLValue *v) {
    if (!lhs || !v) return;
    uint32_t line = line_of(ip->src, lhs);
    WGSLValue ptr;
    if (!ptr_from_lvalue(ip, lhs, &ptr)) return;

    /* Precise compact path (empty or 1-D) and not a member/swizzle: pure mem_store. */
    if (lhs->kind == WGSL_NODE_EXPR_IDENT ||
        lhs->kind == WGSL_NODE_EXPR_INDIRECTION ||
        (lhs->kind == WGSL_NODE_EXPR_INDEX && ptr.u.ptr.has_index && !ptr.u.ptr.race_coarse) ||
        (lhs->kind == WGSL_NODE_EXPR_PAREN)) {
        /* Paren may wrap member — fall through to walk if needed. */
        WGSLNode *inner = lhs;
        while (inner && inner->kind == WGSL_NODE_EXPR_PAREN)
            inner = inner->child_count ? inner->children[0] : NULL;
        if (inner && (inner->kind == WGSL_NODE_EXPR_IDENT ||
                      inner->kind == WGSL_NODE_EXPR_INDIRECTION ||
                      (inner->kind == WGSL_NODE_EXPR_INDEX && ptr.u.ptr.has_index &&
                       !ptr.u.ptr.race_coarse))) {
            mem_store(ip, &ptr, v, line);
            /* Warp coalescing/bank analysis still wants the AST site. */
            if (inner->kind == WGSL_NODE_EXPR_INDEX && ip->cev->simt.mem_access &&
                ptr.u.ptr.has_index && ptr_root_is_shared(ptr.u.ptr.root)) {
                uint32_t eb = interp_leaf_bytes(v->type);
                if (!eb) eb = 4;
                int fw = (ptr.u.ptr.root->as == WGSL_AS_WORKGROUP);
                ip->cev->simt.mem_access(ip->cev, inner, (uint32_t)ptr.u.ptr.index, eb,
                                    1, fw, ip->cev->simt.mem_access_ud);
            }
            return;
        }
    }

    /* Member / nested / swizzle: mutate the walked slot; race with layout
     * offset when reusable, else coarse whole-root — never skip. */
    WGSLValue *slot = lvalue_slot_walk(ip, lhs);
    if (!slot) return;
    WGSLTypeInfo *et = slot->type;
    WGSLValue tmp = *v;
    if (et) wgsl_consteval_materialize(ip->cev, &tmp, et, lhs);
    *slot = tmp;
    uint32_t eb = interp_leaf_bytes(et ? et : v->type);
    if (ptr.u.ptr.root &&
        (ptr.u.ptr.root->as == WGSL_AS_STORAGE || ptr.u.ptr.root->as == WGSL_AS_UNIFORM))
        ip->cev->cost.bytes_stored += eb;
    if (ptr.u.ptr.root && ptr_root_is_shared(ptr.u.ptr.root)) {
        uint32_t off = 0;
        if (!lvalue_race_byte_offset(ip, lhs, &off))
            off = 0; /* coarse whole-root cell */
        race_check(ip, ptr.u.ptr.root, off, 1, line);
    }
}

/* The arithmetic operator underlying a compound-assignment token
 * (`+=` → `+`, `>>=` → `>>`, …).  Returns the input unchanged if it is not a
 * compound-assign token. */
WGSLTokenKind base_binop_of_compound(WGSLTokenKind t) {
    switch (t) {
    case WGSL_TOK_PLUS_EQUAL:            return WGSL_TOK_PLUS;
    case WGSL_TOK_MINUS_EQUAL:           return WGSL_TOK_MINUS;
    case WGSL_TOK_STAR_EQUAL:            return WGSL_TOK_STAR;
    case WGSL_TOK_SLASH_EQUAL:           return WGSL_TOK_SLASH;
    case WGSL_TOK_PERCENT_EQUAL:         return WGSL_TOK_PERCENT;
    case WGSL_TOK_AMP_EQUAL:             return WGSL_TOK_AMP;
    case WGSL_TOK_PIPE_EQUAL:            return WGSL_TOK_PIPE;
    case WGSL_TOK_CARET_EQUAL:           return WGSL_TOK_CARET;
    case WGSL_TOK_LESS_LESS_EQUAL:       return WGSL_TOK_LESS_LESS;
    case WGSL_TOK_GREATER_GREATER_EQUAL: return WGSL_TOK_GREATER_GREATER;
    default:                             return t;
    }
}

/* Point the shared const-evaluator at lane `laneid`'s per-lane environment,
 * and gate step recording to the single lane whose trace we serialise. */
void set_lane(Interp *ip, uint32_t laneid) {
    ip->cev->simt.cur_lane = laneid;
    ip->recording = (laneid == ip->focus_lane);
}

/* leaf actions - the non-control-flow effect of a statement.
 *
 * These are the straight-line pieces the bytecode VM (below) executes for
 * every active lane at a given program counter; control flow itself is lowered
 * to explicit branches by the compiler.  Expression evaluation still tree-walks
 * through the const-evaluator, which now reads/writes the *current lane's*
 * bindings (set via `set_lane`).  Semantics are identical to the previous
 * recursive tree-walker — only the driver changed. */

static void act_decl(Interp *ip, WGSLNode *n) {
    char nm[96];
    WGSLSymbol *s = decl_symbol(ip->cev, n);
    WGSLValue v; memset(&v, 0, sizeof v);
    WGSLNode *init = n->child_count ? n->children[n->child_count - 1] : NULL;
    int have = init && eval(ip, init, &v);
    /* Materialize the initializer to the variable's declared/inferred type so a
     * later `x << 40u` or `x / -1` operates in the concrete type (i32/u32) and
     * follows runtime wrap semantics, instead of computing in 64-bit AbstractInt
     * and tripping a range error at the next store (§1.8). */
    if (have && s && s->type) {
        WGSLTypeInfo *dt = s->type;
        if (dt->kind == WGSL_TYPE_REF) dt = (WGSLTypeInfo *)dt->ref;
        if (dt) wgsl_consteval_materialize(ip->cev, &v, dt, init);
    }
    /* No initializer (the last child is the type node, not an expr) → WGSL
     * default-initializes the var to zero.  Seeding it is essential so a later
     * partial store (`v.x = …`, `v[i] = …`) has a slot to land in. */
    if (!have && s && s->type) {
        WGSLTypeInfo *dt = s->type;
        if (dt->kind == WGSL_TYPE_REF) dt = (WGSLTypeInfo *)dt->ref;
        v = zero_value(ip->cev, ip->tc, dt, ip->default_len);
        have = v.kind != WGSL_VAL_INVALID;
    }
    if (s && have) wgsl_consteval_set_value(ip->cev, s, &v);
    name_text(ip->src, n, nm, sizeof nm);
    record_step(ip, line_of(ip->src, n),
                n->kind == WGSL_NODE_DECL_LET ? "let" : "var",
                nm, have ? &v : NULL, have ? anomaly_tag(ip, &v) : NULL);
}

static void act_assign(Interp *ip, WGSLNode *n) {
    char nm[96];
    WGSLNode *lhs = n->children[0], *rhs = n->children[1];
    WGSLValue v;
    if (!eval(ip, rhs, &v)) return;
    const char *tag = anomaly_tag(ip, &v);
    store_lvalue(ip, lhs, &v);
    span_text(ip->src, lhs, nm, sizeof nm);
    record_step(ip, line_of(ip->src, lhs), "store", nm, &v, tag);
}

static void act_compound_assign(Interp *ip, WGSLNode *n) {
    /* `lhs OP= rhs` → read lhs (DRAM load billed inside eval), combine with rhs
     * under the base operator, store back.  Reuses the const evaluator's full
     * binary-op rules (promotion, shifts, signedness). */
    char nm[96];
    WGSLNode *lhs = n->children[0], *rhs = n->children[1];
    WGSLValue a, b, v;
    if (!eval(ip, lhs, &a) || !eval(ip, rhs, &b)) return;
    WGSLTokenKind op = base_binop_of_compound((WGSLTokenKind)wgsl_node_op_kind_u32(n));
    if (!wgsl_consteval_binop(ip->cev, n, op, &a, &b, &v)) return;
    const char *tag = anomaly_tag(ip, &v);
    store_lvalue(ip, lhs, &v);
    span_text(ip->src, lhs, nm, sizeof nm);
    record_step(ip, line_of(ip->src, lhs), "store", nm, &v, tag);
}

static void act_incdec(Interp *ip, WGSLNode *n) {
    /* `lhs++` / `lhs--` → lhs = lhs ± 1 (integer, same type as lhs). */
    char nm[96];
    WGSLNode *lhs = n->child_count ? n->children[0] : NULL;
    if (!lhs) return;
    WGSLValue a, one, v;
    if (!eval(ip, lhs, &a)) return;
    one = a; one.u.i = 1;    /* integer 1 of lhs's own type (u32/i32/abstract) */
    WGSLTokenKind op = (n->kind == WGSL_NODE_STMT_INCREMENT) ? WGSL_TOK_PLUS : WGSL_TOK_MINUS;
    if (!wgsl_consteval_binop(ip->cev, n, op, &a, &one, &v)) return;
    const char *tag = anomaly_tag(ip, &v);
    store_lvalue(ip, lhs, &v);
    span_text(ip->src, lhs, nm, sizeof nm);
    record_step(ip, line_of(ip->src, lhs), "store", nm, &v, tag);
}

/* A bare call statement (`foo();`, `workgroupBarrier();`) — evaluate for its
 * side effects / cost; the value is discarded. */
static void act_eval(Interp *ip, WGSLNode *n) {
    WGSLValue tmp;
    if (n->child_count) eval(ip, n->children[0], &tmp);
}

/* Execute one straight-line statement for the current lane. */
void exec_leaf(Interp *ip, WGSLNode *n) {
    if (!n) return;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_DECL_VAR:
    case WGSL_NODE_DECL_LET:            act_decl(ip, n);            break;
    case WGSL_NODE_STMT_ASSIGN:         act_assign(ip, n);          break;
    case WGSL_NODE_STMT_COMPOUND_ASSIGN:act_compound_assign(ip, n); break;
    case WGSL_NODE_STMT_INCREMENT:
    case WGSL_NODE_STMT_DECREMENT:      act_incdec(ip, n);          break;
    case WGSL_NODE_STMT_FN_CALL:
    case WGSL_NODE_STMT_PHONY_ASSIGN:   act_eval(ip, n);            break;
    default:                            /* const decls, … */        break;
    }
}
