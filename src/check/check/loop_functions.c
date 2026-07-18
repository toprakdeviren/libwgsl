/**
 * @file loop_functions.c — typecheck AST walk.
 */
#include "internal/check_walk_priv.h"

/* §9.4.8 — for loop bodies whose continuing block references a decl D
 * declared lexically inside the same loop body, every `continue`
 * appearing before D (in source order) is invalid: when the continue
 * jumps to the continuing block, D wouldn't have been declared yet.
 *
 * Implemented as two scans:
 *   1. collect_continuing_uses: walk the continuing block, gather the
 *      set of LET/VAR/CONST symbols that are declared inside this loop
 *      body (skipping refs to outer / module-scope decls).
 *   2. scan_continues_before: walk the loop body in source order
 *      (skipping nested loops — their continues target the inner loop
 *      and are checked by their own pass), reporting any continue
 *      whose source span starts before any used decl's span.
 *
 * Decl-source-position is the AST span_offset; same for continue.
 * The use-set grows on the heap (doubling from 8), so a continuing block
 * referencing many loop-body decls no longer silently drops the excess
 * (which would miss their §9.4.8 diagnostics). */
/* ContUse: check_walk_priv.h */
int loop_decls_contain(
    const ContUse *uses, int n_uses, WGSLSymbol *sym)
{
    for (int i = 0; i < n_uses; i++) if (uses[i].sym == sym) return 1;
    return 0;
}

/* True iff `sym` is a let / var / const declared lexically inside the
 * loop body whose statement-list spans [body_start, body_end). */
int sym_in_loop_body(
    const WGSLSymbol *sym, uint32_t body_start, uint32_t body_end)
{
    if (!sym || !sym->ast) return 0;
    if (sym->kind != WGSL_SYM_LET && sym->kind != WGSL_SYM_VAR &&
        sym->kind != WGSL_SYM_CONST)
        return 0;
    uint32_t off = sym->ast->span_offset;
    return off >= body_start && off < body_end;
}

/* Growable set of continuing-block uses (heap; owned by the caller).  A
 * fixed cap would silently drop the 9th+ referenced decl and miss its
 * §9.4.8 error on a wide continuing block. */
void collect_continuing_uses(
    const WGSLNode *n, uint32_t body_start, uint32_t body_end,
    ContUse **uses, int *n_uses, int *cap)
{
    if (!n) return;
    if (n->kind == WGSL_NODE_EXPR_IDENT) {
        WGSLSymbol *s = wgsl_node_resolved_symbol((WGSLNode *)n);
        if (s && sym_in_loop_body(s, body_start, body_end) &&
            !loop_decls_contain(*uses, *n_uses, s))
        {
            if (*n_uses == *cap) {
                int nc = *cap ? *cap * 2 : 8;
                ContUse *g = (ContUse *)realloc(*uses, (size_t)nc * sizeof *g);
                if (!g) return;   /* OOM: stop growing (conservative) */
                *uses = g;
                *cap  = nc;
            }
            (*uses)[*n_uses].sym         = s;
            (*uses)[*n_uses].decl_offset = s->ast->span_offset;
            (*uses)[*n_uses].decl_node   = s->ast;
            *n_uses += 1;
        }
        return;
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        collect_continuing_uses(n->children[i], body_start, body_end, uses, n_uses, cap);
    }
}

void scan_continues_before(
    WGSLTypeChecker *tc, WGSLNode *n,
    const ContUse *uses, int n_uses)
{
    if (!n || n_uses == 0) return;
    /* Don't descend into a nested loop / for / while: continues there
     * target that inner loop, not the outer one. */
    if (n->kind == WGSL_NODE_STMT_LOOP ||
        n->kind == WGSL_NODE_STMT_FOR  ||
        n->kind == WGSL_NODE_STMT_WHILE)
    {
        return;
    }
    if (n->kind == WGSL_NODE_STMT_CONTINUE) {
        uint32_t cont_off = n->span_offset;
        for (int i = 0; i < n_uses; i++) {
            if (cont_off < uses[i].decl_offset) {
                wgsl_tc_error(tc, n,
                    "'continue' here would re-enter the loop's "
                    "continuing block, which uses an identifier "
                    "declared later in the loop body (spec §9.4.8)");
                return;     /* one diag per continue is enough */
            }
        }
        return;
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        scan_continues_before(tc, n->children[i], uses, n_uses);
    }
}

void check_continue_past_decl(WGSLTypeChecker *tc, WGSLNode *loop) {
    if (!loop || loop->child_count == 0) return;
    /* Find the continuing block (last child if STMT_CONTINUING).  Loop
     * body is the [0, continuing_idx) prefix; if no continuing, no
     * decls are observable from one and the rule trivially holds. */
    WGSLNode *cont = NULL;
    uint32_t cont_idx = loop->child_count;
    for (uint32_t i = 0; i < loop->child_count; i++) {
        if (loop->children[i] &&
            loop->children[i]->kind == WGSL_NODE_STMT_CONTINUING)
        {
            cont = loop->children[i];
            cont_idx = i;
            break;
        }
    }
    if (!cont) return;
    /* Body span = from loop's start to continuing's start; the source
     * offsets gate `sym_in_loop_body`. */
    uint32_t body_start = loop->span_offset;
    uint32_t body_end   = cont->span_offset;
    ContUse *uses = NULL;
    int n_uses = 0, cap = 0;
    collect_continuing_uses(cont, body_start, body_end, &uses, &n_uses, &cap);
    if (n_uses == 0) { free(uses); return; }
    /* Walk the loop body's statements (children before continuing) and
     * surface any continue that comes before a referenced decl. */
    for (uint32_t i = 0; i < cont_idx; i++) {
        scan_continues_before(tc, loop->children[i], uses, n_uses);
    }
    free(uses);
}

/* True iff this function's body is on the partial-typecheck skip list. */
static int fn_body_is_skippable(const WGSLTypeChecker *tc, const WGSLNode *fn) {
    if (!tc || !fn || tc->skip_body_count <= 0) return 0;
    if (!tc->skip_body_names || !tc->skip_body_name_lens) return 0;
    uint32_t no = wgsl_node_name_span(fn).offset;
    uint32_t nl = wgsl_node_name_span(fn).length;
    if (nl == 0 || !tc->src || (size_t)no + (size_t)nl > tc->src->length)
        return 0;
    const char *nm = tc->src->bytes + no;
    for (int i = 0; i < tc->skip_body_count; i++) {
        if (tc->skip_body_name_lens[i] != nl) continue;
        const char *s = tc->skip_body_names[i];
        if (s && memcmp(s, nm, nl) == 0) return 1;
    }
    return 0;
}

static int ast_node_count(const WGSLNode *n) {
    if (!n) return 0;
    int total = 1;
    for (uint32_t i = 0; i < n->child_count; i++) {
        int child = ast_node_count(n->children[i]);
        if (child > INT32_MAX - total) return INT32_MAX;
        total += child;
    }
    return total;
}

/* Walk a function body that was previously skipped.  Signature typing
 * is assumed already done (params / return / stage flags on the symbol). */
static int type_function_body_only(WGSLTypeChecker *tc, WGSLNode *fn) {
    if (!tc || !fn) return 0;
    uint32_t FA = wgsl_fn_attr_count(fn);
    uint32_t P  = wgsl_fn_param_count(fn);
    uint32_t RA = wgsl_fn_ret_attr_count(fn);
    uint32_t HR = wgsl_fn_has_return_type(fn);
    uint32_t i = FA + P + RA;
    if (HR) i += 1;
    if (i >= fn->child_count) return 1;

    WGSLSymbol *fsym = wgsl_tc_find_decl_symbol(tc, fn);
    WGSLTypeInfo *ret = fsym && fsym->type ? fsym->type : tc->types->t_void;

    tc->in_function       = 1;
    tc->current_fn_return = HR ? ret : NULL;
    walk_stmt(tc, fn->children[i]);
    tc->current_fn_return = NULL;
    tc->in_function       = 0;
    if (fsym) fsym->flags = (uint16_t)(fsym->flags & ~WGSL_SYM_FLAG_BODY_SKIPPED);
    return 1;
}

int wgsl_typecheck_ensure_function_body(WGSLTypeChecker *tc, WGSLNode *fn) {
    if (!tc || !fn || fn->kind != WGSL_NODE_DECL_FUNCTION) return 0;
    WGSLSymbol *fsym = wgsl_tc_find_decl_symbol(tc, fn);
    if (!fsym || !(fsym->flags & WGSL_SYM_FLAG_BODY_SKIPPED)) return 1;
    return type_function_body_only(tc, fn);
}

int type_function(WGSLTypeChecker *tc, WGSLNode *fn) {
    /* children: [fn_attrs (FA)][params (P)][ret_attrs (RA)][ret_type?][body] */
    uint32_t FA = wgsl_fn_attr_count(fn);
    uint32_t P  = wgsl_fn_param_count(fn);
    uint32_t RA = wgsl_fn_ret_attr_count(fn);
    uint32_t HR = wgsl_fn_has_return_type(fn);

    uint32_t i = FA;

    /* Type each param. */
    for (uint32_t k = 0; k < P && i < fn->child_count; k++, i++) {
        WGSLNode *p = fn->children[i];
        if (!p || p->kind != WGSL_NODE_DECL_PARAM) continue;
        uint32_t pattrs = wgsl_param_attr_count(p);
        if (pattrs < p->child_count) {
            WGSLNode *tspec = p->children[pattrs];
            WGSLTypeInfo *t = wgsl_tc_resolve_type_spec(tc, tspec);
            WGSLSymbol *sym = wgsl_tc_find_decl_symbol(tc, p);
            if (sym && t) sym->type = t;
            /* §11.1 — function parameter type must be constructible,
             * pointer, texture, or sampler.  Rejects `fn f(a: atomic<i32>)`
             * and `fn f(a: array<f32>)` (runtime-sized). */
            if (t) {
                int ok = (t->kind == WGSL_TYPE_PTR) ||
                         (t->kind == WGSL_TYPE_TEXTURE) ||
                         (t->kind == WGSL_TYPE_SAMPLER) ||
                         wgsl_tc_is_constructible_deep(tc, t);
                if (!ok) {
                    wgsl_tc_error(tc, tspec,
                        "function parameter type must be constructible, "
                        "pointer, texture, or sampler (got %s)",
                        wgsl_type_kind_name((WGSLTypeKind)t->kind));
                }
            }
        }
    }
    uint32_t ret_attr_start = i;  /* index of first @ret-attr child */
    i += RA;
    WGSLTypeInfo *ret = tc->types->t_void;
    WGSLNode *ret_spec_node = NULL;
    if (HR && i < fn->child_count) {
        WGSLNode *rspec = fn->children[i];
        ret_spec_node = rspec;
        WGSLTypeInfo *r = wgsl_tc_resolve_type_spec(tc, rspec);
        if (r) {
            ret = r;
            /* §11.1 — function return type must be constructible (or
             * void, when no return type is written, which lands in the
             * `else` branch above).  Rejects `fn f() -> array<f32>`,
             * `fn f() -> texture_2d<f32>`, etc. */
            if (!wgsl_tc_is_constructible_deep(tc, r)) {
                wgsl_tc_error(tc, rspec,
                    "function return type must be constructible (got %s)",
                    wgsl_type_kind_name((WGSLTypeKind)r->kind));
            }
        }
        i += 1;
    }

    /* §11.4 — a `@vertex` entry point must return a value that
     * includes `@builtin(position)`.  Either:
     *   - the return itself carries `@builtin(position)` (a non-struct
     *     return), OR
     *   - the return type is a struct one of whose members carries
     *     `@builtin(position)`.
     * We only run this for fn-decls marked `@vertex`; the stage flag
     * on the symbol isn't set yet (this is the first pass that walks
     * the attributes), so re-scan the fn attrs locally. */
    int has_vertex_attr = 0;
    for (uint32_t k = 0; k < FA; k++) {
        WGSLNode *attr = fn->children[k];
        if (!attr || attr->kind != WGSL_NODE_ATTRIBUTE) continue;
        if (attr->child_count == 0) continue;
        WGSLNode *an = attr->children[0];
        if (!an || an->kind != WGSL_NODE_EXPR_IDENT) continue;
        uint32_t aoff = wgsl_node_name_span(an).offset;
        uint32_t alen = wgsl_node_name_span(an).length;
        if (alen == 6 && memcmp(tc->src->bytes + aoff, "vertex", 6) == 0) {
            has_vertex_attr = 1;
            break;
        }
    }
    if (has_vertex_attr) {
        int has_position = 0;
        /* Direct return-attribute walk: @builtin(position) on the
         * function return itself. */
        for (uint32_t k = 0; k < RA && (ret_attr_start + k) < fn->child_count; k++) {
            WGSLNode *attr = fn->children[ret_attr_start + k];
            if (!attr || attr->kind != WGSL_NODE_ATTRIBUTE) continue;
            if (attr->child_count < 2) continue;
            WGSLNode *an = attr->children[0];
            if (!an || an->kind != WGSL_NODE_EXPR_IDENT) continue;
            uint32_t aoff = wgsl_node_name_span(an).offset;
            uint32_t alen = wgsl_node_name_span(an).length;
            if (!(alen == 7 && memcmp(tc->src->bytes + aoff, "builtin", 7) == 0))
                continue;
            WGSLNode *av = attr->children[1];
            if (!av || av->kind != WGSL_NODE_EXPR_IDENT) continue;
            uint32_t voff = wgsl_node_name_span(av).offset;
            uint32_t vlen = wgsl_node_name_span(av).length;
            if (vlen == 8 &&
                memcmp(tc->src->bytes + voff, "position", 8) == 0)
            {
                has_position = 1;
                break;
            }
        }
        /* Struct return: walk members looking for @builtin(position). */
        if (!has_position && ret && ret->kind == WGSL_TYPE_STRUCT) {
            WGSLNode *sd = (WGSLNode *)ret->ref;
            if (sd) {
                for (uint32_t mi = 0; mi < sd->child_count && !has_position; mi++) {
                    WGSLNode *m = sd->children[mi];
                    if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
                    uint32_t attrs = wgsl_member_attr_count(m);
                    for (uint32_t ai = 0; ai < attrs; ai++) {
                        WGSLNode *attr = m->children[ai];
                        if (!attr || attr->kind != WGSL_NODE_ATTRIBUTE) continue;
                        if (attr->child_count < 2) continue;
                        WGSLNode *an = attr->children[0];
                        if (!an || an->kind != WGSL_NODE_EXPR_IDENT) continue;
                        uint32_t aoff = wgsl_node_name_span(an).offset;
                        uint32_t alen = wgsl_node_name_span(an).length;
                        if (!(alen == 7 && memcmp(tc->src->bytes + aoff,
                                                  "builtin", 7) == 0))
                            continue;
                        WGSLNode *av = attr->children[1];
                        if (!av || av->kind != WGSL_NODE_EXPR_IDENT) continue;
                        uint32_t voff = wgsl_node_name_span(av).offset;
                        uint32_t vlen = wgsl_node_name_span(av).length;
                        if (vlen == 8 &&
                            memcmp(tc->src->bytes + voff, "position", 8) == 0)
                        {
                            has_position = 1;
                            break;
                        }
                    }
                }
            }
        }
        if (!has_position) {
            wgsl_tc_error(tc, ret_spec_node ? ret_spec_node : fn,
                "vertex shader must return a value that includes "
                "@builtin(position)");
        }
    }
    /* Stash the function's return type on its symbol.  `type_call` reads
     * this back when typing user-function call sites.
     * Also record `@must_use` / `@vertex` / `@fragment` / `@compute`
     * on the symbol — downstream call-statement (§9.5) and
     * entry-point-call (§11.4) gates read these flags back. */
    WGSLSymbol *fsym = wgsl_tc_find_decl_symbol(tc, fn);
    if (fsym) {
        fsym->type = ret;
        for (uint32_t k = 0; k < FA; k++) {
            WGSLNode *attr = fn->children[k];
            if (!attr || attr->kind != WGSL_NODE_ATTRIBUTE) continue;
            if (attr->child_count == 0) continue;
            WGSLNode *attr_name = attr->children[0];
            if (!attr_name || attr_name->kind != WGSL_NODE_EXPR_IDENT) continue;
            uint32_t off = wgsl_node_name_span(attr_name).offset;
            uint32_t nl  = wgsl_node_name_span(attr_name).length;
            const char *txt = tc->src->bytes + off;
            if (nl == 8 && memcmp(txt, "must_use", 8) == 0) {
                fsym->flags |= WGSL_SYM_FLAG_MUST_USE;
            } else if (nl == 6 && memcmp(txt, "vertex", 6) == 0) {
                fsym->flags |= WGSL_SYM_FLAG_VERTEX;
            } else if (nl == 8 && memcmp(txt, "fragment", 8) == 0) {
                fsym->flags |= WGSL_SYM_FLAG_FRAGMENT;
            } else if (nl == 7 && memcmp(txt, "compute", 7) == 0) {
                fsym->flags |= WGSL_SYM_FLAG_COMPUTE;
            }
        }
    }

    /* Session partial typecheck: skip the body walk when this function is
     * content-stable under a stable module interface.  Signature typing
     * above still runs (call sites / entry IO need it). */
    if (fn_body_is_skippable(tc, fn)) {
        if (fsym) fsym->flags |= WGSL_SYM_FLAG_BODY_SKIPPED;
        tc->bodies_skipped += 1;
        if (i < fn->child_count) {
            int skipped = ast_node_count(fn->children[i]);
            if (skipped > INT32_MAX - tc->skipped_nodes)
                tc->skipped_nodes = INT32_MAX;
            else
                tc->skipped_nodes += skipped;
        }
        return 1;
    }

    if (i < fn->child_count) {
        tc->in_function       = 1;
        tc->current_fn_return = HR ? ret : NULL;
        walk_stmt(tc, fn->children[i]);
        tc->current_fn_return = NULL;
        tc->in_function       = 0;
        tc->bodies_typed += 1;
    } else {
        tc->bodies_typed += 1;
    }
    return 1;
}

int wgsl_tc_type_module_decl(WGSLTypeChecker *tc, WGSLNode *n) {
    WGSLSymbol *sym = wgsl_tc_find_decl_symbol(tc, n);

    /* §7.2.1 — `const` type must be constructible (concrete or abstract).
     * Done eagerly so the gate fires even when the consteval module
     * pass has already populated sym->type with the value's type
     * (which can mask an unresolvable declared type like `atomic<i32>`
     * — the consteval typespec table doesn't know atomic, so it falls
     * back to the value's concretization instead). */
    if (n && n->kind == WGSL_NODE_DECL_CONST) {
        int has_type = wgsl_letlike_has_type(n);
        if (has_type && n->child_count >= 1) {
            WGSLTypeInfo *decl = wgsl_tc_resolve_type_spec(tc, n->children[0]);
            if (decl && !wgsl_tc_is_constructible_deep(tc, decl)) {
                wgsl_tc_error(tc, n,
                    "'const' type must be constructible (got %s)",
                    wgsl_type_kind_name((WGSLTypeKind)decl->kind));
            }
        }
    }

    if (sym && sym->type && n->kind != WGSL_NODE_DECL_CONST) return 1;

    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_DECL_CONST:
    case WGSL_NODE_DECL_LET:        return wgsl_tc_type_decl_const_or_let(tc, n);
    case WGSL_NODE_DECL_VAR:        return wgsl_tc_type_decl_var          (tc, n);
    case WGSL_NODE_DECL_OVERRIDE:   return wgsl_tc_type_decl_override     (tc, n);
    case WGSL_NODE_DECL_ALIAS:      return wgsl_tc_type_decl_alias        (tc, n);
    case WGSL_NODE_DECL_STRUCT:     return wgsl_tc_type_decl_struct       (tc, n);
    case WGSL_NODE_DECL_FUNCTION:   return type_function          (tc, n);
    case WGSL_NODE_DECL_CONST_ASSERT:
        if (n->child_count >= 1) {
            wgsl_tc_type_expr(tc, n->children[0]);
            WGSLTypeInfo *t = wgsl_tc_store_type_of(tc, n->children[0]);
            if (t && t != tc->types->t_bool) {
                wgsl_tc_error(tc, n->children[0], "const_assert condition must be a bool");
            }
        }
        return 1;
    default:
        return 1;
    }
}
