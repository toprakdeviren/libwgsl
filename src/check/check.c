/**
 * @file check.c — WGSL type checker driver (Phase 7).
 *
 * Slim driver that ties together the type checker's four translation
 * units: this file (driver / side-table / walk_stmt / function-decl /
 * directives / entry point), types.c (type-spec resolution),
 * decls.c (decl typing + init-vs-decl compatibility), and
 * exprs.c (expression typing + builtin overload resolution).
 *
 * Cross-file sharing happens through `internal/check_priv.h`.
 */
#include "internal/check.h"
#include "internal/check_priv.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal/token.h"

/* ── Diagnostics ───────────────────────────────────────────────────── */

void wgsl_tc_error(
    WGSLTypeChecker *tc, const WGSLNode *at, const char *fmt, ...)
{
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    uint32_t off = at ? at->span_offset : 0;
    uint32_t len = at ? at->span_length : 0;
    wgsl_diag_emit_at(tc->diag, tc->src, WGSL_DIAG_ERROR,
                      off, len, NULL, "%s", buf);
    tc->had_error = 1;
}

/* ── Side-table ────────────────────────────────────────────────────── */

int wgsl_tc_set_type(
    WGSLTypeChecker *tc, WGSLNode *n, WGSLTypeInfo *t, int is_ref)
{
    if (!n || !t) return 0;
    WGSLTypeInfo *stored = t;
    if (is_ref && t->kind != WGSL_TYPE_REF) {
        WGSLAddressSpace as = WGSL_AS_NONE;
        WGSLAccessMode   am = WGSL_ACCESS_NONE;
        wgsl_tc_get_ref_space_and_mode(tc, n, &as, &am);
        stored = wgsl_type_ref(tc->types, (uint8_t)as, t, am);
        if (!stored) return 0;
    }
    if (tc->count == tc->capacity) {
        if (tc->capacity > SIZE_MAX / 2) return 0;
        size_t cap = tc->capacity ? tc->capacity * 2 : 64;
        if (cap > SIZE_MAX / sizeof *tc->table) return 0;
        WGSLNodeType *g = (WGSLNodeType *)realloc(
            tc->table, cap * sizeof *g);
        if (!g) return 0;
        tc->table = g;
        tc->capacity = cap;
    }
    tc->table[tc->count].node  = n;
    tc->table[tc->count].type  = stored;
    tc->table[tc->count].flags = is_ref ? WGSL_FLAG_IS_REF : 0;
    tc->count += 1;
    n->flags |= WGSL_FLAG_TYPED;
    if (is_ref) n->flags |= WGSL_FLAG_IS_REF;
    else        n->flags &= (uint16_t)~WGSL_FLAG_IS_REF;
    return 1;
}

WGSLTypeInfo *wgsl_typecheck_type_of(
    const WGSLTypeChecker *tc, const WGSLNode *n)
{
    if (!tc || !n) return NULL;
    for (size_t i = 0; i < tc->count; i++) {
        if (tc->table[i].node == n) return tc->table[i].type;
    }
    if (n->kind == WGSL_NODE_EXPR_IDENT) {
        WGSLSymbol *s = wgsl_node_resolved_symbol(n);
        if (s && s->type) return s->type;
    }
    return NULL;
}

WGSLTypeInfo *wgsl_tc_unwrap_ref_type(WGSLTypeInfo *t) {
    if (t && t->kind == WGSL_TYPE_REF) return (WGSLTypeInfo *)t->ref;
    return t;
}

WGSLTypeInfo *wgsl_tc_store_type_of(
    const WGSLTypeChecker *tc, const WGSLNode *n)
{
    return wgsl_tc_unwrap_ref_type(wgsl_typecheck_type_of(tc, n));
}

int wgsl_typecheck_is_ref(
    const WGSLTypeChecker *tc, const WGSLNode *n)
{
    if (!tc || !n) return 0;
    if (n->flags & WGSL_FLAG_IS_REF) return 1;
    WGSLTypeInfo *t = wgsl_typecheck_type_of(tc, n);
    return t && t->kind == WGSL_TYPE_REF;
}

static WGSLTypeInfo *tc_elem_type(WGSLTypeInfo *t) {
    if (!t) return NULL;
    if (wgsl_type_is_scalar(t)) return t;
    if (t->kind == WGSL_TYPE_VEC ||
        t->kind == WGSL_TYPE_MAT ||
        t->kind == WGSL_TYPE_ARRAY ||
        t->kind == WGSL_TYPE_ATOMIC)
    {
        return (WGSLTypeInfo *)t->ref;
    }
    return NULL;
}

static int tc_same_shape(WGSLTypeInfo *a, WGSLTypeInfo *b) {
    if (!a || !b) return 0;
    if (wgsl_type_is_scalar(a) && wgsl_type_is_scalar(b)) return 1;
    if (a->kind == WGSL_TYPE_VEC && b->kind == WGSL_TYPE_VEC) {
        return a->width == b->width;
    }
    if (a->kind == WGSL_TYPE_MAT && b->kind == WGSL_TYPE_MAT) {
        return a->width == b->width && a->rows == b->rows;
    }
    return 0;
}

static int tc_can_convert(WGSLTypeInfo *from, WGSLTypeInfo *to) {
    return from && to && wgsl_type_conversion_rank(from, to) >= 0;
}

static WGSLTokenKind compound_binary_op(WGSLTokenKind op) {
    switch (op) {
    case WGSL_TOK_PLUS_EQUAL:              return WGSL_TOK_PLUS;
    case WGSL_TOK_MINUS_EQUAL:             return WGSL_TOK_MINUS;
    case WGSL_TOK_STAR_EQUAL:              return WGSL_TOK_STAR;
    case WGSL_TOK_SLASH_EQUAL:             return WGSL_TOK_SLASH;
    case WGSL_TOK_PERCENT_EQUAL:           return WGSL_TOK_PERCENT;
    case WGSL_TOK_AMP_EQUAL:               return WGSL_TOK_AMP;
    case WGSL_TOK_PIPE_EQUAL:              return WGSL_TOK_PIPE;
    case WGSL_TOK_CARET_EQUAL:             return WGSL_TOK_CARET;
    case WGSL_TOK_LESS_LESS_EQUAL:         return WGSL_TOK_LESS_LESS;
    case WGSL_TOK_GREATER_GREATER_EQUAL:   return WGSL_TOK_GREATER_GREATER;
    default:                               return WGSL_TOK_INVALID;
    }
}

static int validate_compound_assignment(
    WGSLTypeChecker *tc, WGSLNode *stmt, WGSLTypeInfo *lt, WGSLTypeInfo *rt)
{
    if (!tc || !stmt || !lt || !rt) return 0;
    WGSLTokenKind bin = compound_binary_op((WGSLTokenKind)stmt->payload[0]);
    WGSLTypeInfo *le = tc_elem_type(lt);
    WGSLTypeInfo *re = tc_elem_type(rt);
    if (!le || !re) {
        wgsl_tc_error(tc, stmt, "compound assignment requires typed operands");
        return 0;
    }

    switch (bin) {
    case WGSL_TOK_PLUS:
    case WGSL_TOK_MINUS:
    case WGSL_TOK_SLASH:
    case WGSL_TOK_PERCENT:
        if (!(tc_same_shape(lt, rt) ||
              ((lt->kind == WGSL_TYPE_VEC || lt->kind == WGSL_TYPE_MAT) &&
               wgsl_type_is_scalar(rt))))
        {
            wgsl_tc_error(tc, stmt,
                "compound assignment operands must have compatible shapes");
            return 0;
        }
        if (!wgsl_type_is_numeric(le) || !tc_can_convert(re, le)) {
            wgsl_tc_error(tc, stmt,
                "compound assignment operator requires numeric operands");
            return 0;
        }
        if ((bin == WGSL_TOK_SLASH || bin == WGSL_TOK_PERCENT) &&
            stmt->child_count >= 2)
        {
            wgsl_tc_check_constexpr_divisor_nonzero(
                tc, stmt->children[1], lt,
                bin == WGSL_TOK_SLASH ? "division by zero" : "modulo by zero");
        }
        return 1;
    case WGSL_TOK_STAR: {
        int shape_ok = tc_same_shape(lt, rt) ||
            ((lt->kind == WGSL_TYPE_VEC || lt->kind == WGSL_TYPE_MAT) &&
             wgsl_type_is_scalar(rt));
        if (!shape_ok) {
            wgsl_tc_error(tc, stmt,
                "compound assignment operands must have compatible shapes");
            return 0;
        }
        if (!wgsl_type_is_numeric(le) || !tc_can_convert(re, le)) {
            wgsl_tc_error(tc, stmt,
                "compound assignment operator requires numeric operands");
            return 0;
        }
        return 1;
    }
    case WGSL_TOK_AMP:
    case WGSL_TOK_PIPE:
    case WGSL_TOK_CARET:
        if (!tc_same_shape(lt, rt)) {
            wgsl_tc_error(tc, stmt,
                "compound assignment operands must have matching shapes");
            return 0;
        }
        if (le == tc->types->t_bool || re == tc->types->t_bool) {
            if (bin == WGSL_TOK_CARET ||
                le != tc->types->t_bool || re != tc->types->t_bool)
            {
                wgsl_tc_error(tc, stmt,
                    "compound bitwise assignment requires integer operands");
                return 0;
            }
            return 1;
        }
        if (!wgsl_type_is_integer(le) || !tc_can_convert(re, le)) {
            wgsl_tc_error(tc, stmt,
                "compound bitwise assignment requires integer operands");
            return 0;
        }
        return 1;
    case WGSL_TOK_LESS_LESS:
    case WGSL_TOK_GREATER_GREATER: {
        int lhs_ok = wgsl_type_is_integer(le);
        int rhs_ok = re == tc->types->t_u32 || re == tc->types->t_abstract_int;
        if (!lhs_ok || !rhs_ok || !tc_same_shape(lt, rt)) {
            wgsl_tc_error(tc, stmt,
                "compound shift assignment requires integer lhs and u32 rhs");
            return 0;
        }
        return 1;
    }
    default:
        wgsl_tc_error(tc, stmt, "unsupported compound assignment operator");
        return 0;
    }
}

void wgsl_typecheck_destroy(WGSLTypeChecker *tc) {
    if (!tc) return;
    free(tc->table);
    free(tc->override_ids);
    memset(tc, 0, sizeof *tc);
}

/* ── Symbol lookup helper ──────────────────────────────────────────── */

WGSLSymbol *wgsl_tc_find_decl_symbol(
    const WGSLTypeChecker *tc, const WGSLNode *decl)
{
    if (!tc->res) return NULL;
    for (size_t i = 0; i < tc->res->all_decl_count; i++) {
        if (tc->res->all_decls[i]->ast == decl) {
            return tc->res->all_decls[i];
        }
    }
    return NULL;
}

static void walk_stmt(WGSLTypeChecker *tc, WGSLNode *n);

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
 * `cont_decl_caps` keeps the working set tiny (8 entries), enough for
 * realistic continuing blocks.  Excess decls are silently dropped — the
 * over-approximation only matters when a continuing references > 8
 * decls (unrealistic for shaders). */
typedef struct {
    WGSLSymbol *sym;
    uint32_t    decl_offset;
    const WGSLNode *decl_node;
} ContUse;

static int loop_decls_contain(
    const ContUse *uses, int n_uses, WGSLSymbol *sym)
{
    for (int i = 0; i < n_uses; i++) if (uses[i].sym == sym) return 1;
    return 0;
}

/* True iff `sym` is a let / var / const declared lexically inside the
 * loop body whose statement-list spans [body_start, body_end). */
static int sym_in_loop_body(
    const WGSLSymbol *sym, uint32_t body_start, uint32_t body_end)
{
    if (!sym || !sym->ast) return 0;
    if (sym->kind != WGSL_SYM_LET && sym->kind != WGSL_SYM_VAR &&
        sym->kind != WGSL_SYM_CONST)
        return 0;
    uint32_t off = sym->ast->span_offset;
    return off >= body_start && off < body_end;
}

static void collect_continuing_uses(
    const WGSLNode *n, uint32_t body_start, uint32_t body_end,
    ContUse *uses, int *n_uses, int cap)
{
    if (!n) return;
    if (n->kind == WGSL_NODE_EXPR_IDENT) {
        WGSLSymbol *s = wgsl_node_resolved_symbol((WGSLNode *)n);
        if (s && sym_in_loop_body(s, body_start, body_end) &&
            !loop_decls_contain(uses, *n_uses, s) && *n_uses < cap)
        {
            uses[*n_uses].sym         = s;
            uses[*n_uses].decl_offset = s->ast->span_offset;
            uses[*n_uses].decl_node   = s->ast;
            *n_uses += 1;
        }
        return;
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        collect_continuing_uses(n->children[i], body_start, body_end, uses, n_uses, cap);
    }
}

static void scan_continues_before(
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

static void check_continue_past_decl(WGSLTypeChecker *tc, WGSLNode *loop) {
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
    ContUse uses[8];
    int n_uses = 0;
    collect_continuing_uses(cont, body_start, body_end,
                            uses, &n_uses, (int)(sizeof uses / sizeof uses[0]));
    if (n_uses == 0) return;
    /* Walk the loop body's statements (children before continuing) and
     * surface any continue that comes before a referenced decl. */
    for (uint32_t i = 0; i < cont_idx; i++) {
        scan_continues_before(tc, loop->children[i], uses, n_uses);
    }
}

static int type_function(WGSLTypeChecker *tc, WGSLNode *fn) {
    /* children: [fn_attrs (FA)][params (P)][ret_attrs (RA)][ret_type?][body] */
    uint32_t FA = (uint32_t)(fn->payload[1] & 0xFFFFFFFFu);
    uint32_t P  = (uint32_t)(fn->payload[1] >> 32);
    uint32_t RA = (uint32_t)(fn->payload[2] & 0xFFFFFFFFu);
    uint32_t HR = (uint32_t)(fn->payload[2] >> 32);

    uint32_t i = FA;

    /* Type each param. */
    for (uint32_t k = 0; k < P && i < fn->child_count; k++, i++) {
        WGSLNode *p = fn->children[i];
        if (!p || p->kind != WGSL_NODE_DECL_PARAM) continue;
        uint32_t pattrs = (uint32_t)(p->payload[1] & 0xFFFFFFFFu);
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
        uint32_t aoff = (uint32_t)(an->payload[0] & 0xFFFFFFFFu);
        uint32_t alen = (uint32_t)(an->payload[0] >> 32);
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
            uint32_t aoff = (uint32_t)(an->payload[0] & 0xFFFFFFFFu);
            uint32_t alen = (uint32_t)(an->payload[0] >> 32);
            if (!(alen == 7 && memcmp(tc->src->bytes + aoff, "builtin", 7) == 0))
                continue;
            WGSLNode *av = attr->children[1];
            if (!av || av->kind != WGSL_NODE_EXPR_IDENT) continue;
            uint32_t voff = (uint32_t)(av->payload[0] & 0xFFFFFFFFu);
            uint32_t vlen = (uint32_t)(av->payload[0] >> 32);
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
                    uint32_t attrs = (uint32_t)(m->payload[1] & 0xFFFFFFFFu);
                    for (uint32_t ai = 0; ai < attrs; ai++) {
                        WGSLNode *attr = m->children[ai];
                        if (!attr || attr->kind != WGSL_NODE_ATTRIBUTE) continue;
                        if (attr->child_count < 2) continue;
                        WGSLNode *an = attr->children[0];
                        if (!an || an->kind != WGSL_NODE_EXPR_IDENT) continue;
                        uint32_t aoff = (uint32_t)(an->payload[0] & 0xFFFFFFFFu);
                        uint32_t alen = (uint32_t)(an->payload[0] >> 32);
                        if (!(alen == 7 && memcmp(tc->src->bytes + aoff,
                                                  "builtin", 7) == 0))
                            continue;
                        WGSLNode *av = attr->children[1];
                        if (!av || av->kind != WGSL_NODE_EXPR_IDENT) continue;
                        uint32_t voff = (uint32_t)(av->payload[0] & 0xFFFFFFFFu);
                        uint32_t vlen = (uint32_t)(av->payload[0] >> 32);
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
    /* Stash the function's return type on its symbol.  Iter B's
     * `type_call` reads this back when typing user-fn call sites.
     * Also record `@must_use` / `@vertex` / `@fragment` / `@compute`
     * on the symbol — downstream call-statement (§9.5) and
     * entry-point-call (§11.4) gates read these flags back. */
    {
        WGSLSymbol *fsym = wgsl_tc_find_decl_symbol(tc, fn);
        if (fsym) {
            fsym->type = ret;
            for (uint32_t k = 0; k < FA; k++) {
                WGSLNode *attr = fn->children[k];
                if (!attr || attr->kind != WGSL_NODE_ATTRIBUTE) continue;
                if (attr->child_count == 0) continue;
                WGSLNode *attr_name = attr->children[0];
                if (!attr_name || attr_name->kind != WGSL_NODE_EXPR_IDENT) continue;
                uint32_t off = (uint32_t)(attr_name->payload[0] & 0xFFFFFFFFu);
                uint32_t nl  = (uint32_t)(attr_name->payload[0] >> 32);
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
    }
    if (i < fn->child_count) {
        tc->in_function       = 1;
        tc->current_fn_return = HR ? ret : NULL;
        walk_stmt(tc, fn->children[i]);
        tc->current_fn_return = NULL;
        tc->in_function       = 0;
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
        int has_type = (int)(n->payload[1] & 1u);
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

static void walk_stmt(WGSLTypeChecker *tc, WGSLNode *n) {
    if (!n) return;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_DECL_LET:
    case WGSL_NODE_DECL_CONST:
        wgsl_tc_type_decl_const_or_let(tc, n);
        return;
    case WGSL_NODE_DECL_VAR:
        wgsl_tc_type_decl_var(tc, n);
        return;
    case WGSL_NODE_DECL_CONST_ASSERT:
        if (n->child_count >= 1) {
            wgsl_tc_type_expr(tc, n->children[0]);
            WGSLTypeInfo *t = wgsl_tc_store_type_of(tc, n->children[0]);
            if (t && t != tc->types->t_bool) {
                wgsl_tc_error(tc, n->children[0], "const_assert condition must be a bool");
            }
        }
        return;
    case WGSL_NODE_STMT_RETURN: {
        /* §9.4.10 — bare `return` only legal in a void function;
         * `return expr;` only legal in a non-void function whose
         * return type accepts the expr's type. */
        if (tc->continuing_depth > 0) {
            /* §9.4.9 — return is not allowed inside a continuing block. */
            wgsl_tc_error(tc, n,
                "'return' is not allowed inside a continuing block");
        }
        if (n->child_count == 0) {
            if (tc->current_fn_return) {
                wgsl_tc_error(tc, n,
                    "'return;' requires an expression in a function "
                    "with return type %s",
                    wgsl_type_kind_name((WGSLTypeKind)
                        tc->current_fn_return->kind));
            }
        } else {
            wgsl_tc_type_expr(tc, n->children[0]);
            WGSLTypeInfo *rt = wgsl_tc_store_type_of(tc, n->children[0]);
            if (!tc->current_fn_return) {
                wgsl_tc_error(tc, n->children[0],
                    "'return' with expression in a void function");
            } else if (rt && wgsl_tc_init_type_mismatch(rt, tc->current_fn_return)) {
                wgsl_tc_error(tc, n->children[0],
                    "return value of type %s does not match function "
                    "return type %s",
                    wgsl_type_kind_name((WGSLTypeKind)rt->kind),
                    wgsl_type_kind_name((WGSLTypeKind)
                        tc->current_fn_return->kind));
            }
        }
        return;
    }
    case WGSL_NODE_STMT_FN_CALL:
        if (n->child_count >= 1) {
            WGSLNode *expr = n->children[0];
            wgsl_tc_type_expr(tc, expr);
            if (expr && expr->kind != WGSL_NODE_EXPR_CALL) {
                wgsl_tc_error(tc, n,
                    "expression statement must be a function call");
            }
            /* §9.5 — call statement discards the result, so the callee
             * must not carry `@must_use`. */
            WGSLNode *call = expr;
            if (call && call->kind == WGSL_NODE_EXPR_CALL &&
                call->child_count >= 1)
            {
                WGSLNode *callee = call->children[0];
                WGSLSymbol *csym = NULL;
                if (callee && callee->kind == WGSL_NODE_EXPR_IDENT) {
                    csym = wgsl_node_resolved_symbol(callee);
                } else if (callee &&
                           callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
                           callee->child_count >= 1)
                {
                    csym = wgsl_node_resolved_symbol(callee->children[0]);
                }
                int constructor =
                    csym && (csym->kind == WGSL_SYM_PREDECLARED_TYPE ||
                             csym->kind == WGSL_SYM_PREDECLARED_TYPEGEN ||
                             csym->kind == WGSL_SYM_PREDECLARED_TYPE_ALIAS ||
                             csym->kind == WGSL_SYM_TYPE_ALIAS ||
                             csym->kind == WGSL_SYM_STRUCT);
                int must_use = csym &&
                    ((csym->flags & WGSL_SYM_FLAG_MUST_USE) ||
                     constructor ||
                     (csym->kind == WGSL_SYM_PREDECLARED_FN &&
                      wgsl_tc_builtin_must_use(csym->name, csym->name_len)));
                if (must_use) {
                    wgsl_tc_error(tc, n,
                        "call to '@must_use' function '%.*s' must not "
                        "discard its result",
                        (int)csym->name_len, csym->name);
                }
            }
        }
        return;
    case WGSL_NODE_STMT_PHONY_ASSIGN:
        if (n->child_count >= 1) {
            wgsl_tc_type_expr(tc, n->children[0]);
            /* §9.2.2 — RHS must be constructible / pointer / texture /
             * sampler.  Same shape as the function-arg gate. */
            WGSLTypeInfo *rt = wgsl_tc_store_type_of(tc, n->children[0]);
            if (rt) {
                int ok = (rt->kind == WGSL_TYPE_PTR) ||
                         (rt->kind == WGSL_TYPE_TEXTURE) ||
                         (rt->kind == WGSL_TYPE_SAMPLER) ||
                         wgsl_tc_is_constructible_deep(tc, rt);
                if (!ok) {
                    wgsl_tc_error(tc, n->children[0],
                        "right-hand side of '_ = …' must be a "
                        "constructible, pointer, texture, or sampler "
                        "value (got %s)",
                        wgsl_type_kind_name((WGSLTypeKind)rt->kind));
                }
            }
        }
        if (n->child_count >= 2) wgsl_tc_type_expr(tc, n->children[1]);
        return;
    case WGSL_NODE_STMT_ASSIGN:
    case WGSL_NODE_STMT_COMPOUND_ASSIGN:
    case WGSL_NODE_STMT_INCREMENT:
    case WGSL_NODE_STMT_DECREMENT: {
        for (uint32_t i = 0; i < n->child_count; i++) wgsl_tc_type_expr(tc, n->children[i]);
        if (n->child_count > 0 && n->children[0]) {
            WGSLNode *lhs = n->children[0];
            if (!wgsl_typecheck_is_ref(tc, lhs)) {
                wgsl_tc_error(tc, lhs, "cannot assign to a value; left-hand side must be a reference");
            } else {
                WGSLAddressSpace as = WGSL_AS_NONE;
                WGSLAccessMode   am = WGSL_ACCESS_NONE;
                wgsl_tc_get_ref_space_and_mode(tc, lhs, &as, &am);
                if (am == WGSL_ACCESS_READ) {
                    wgsl_tc_error(tc, lhs, "cannot assign to read-only reference");
                }
            }
            /* §9.2.1 RHS type vs LHS store type. */
            if (n->kind == WGSL_NODE_STMT_ASSIGN && n->child_count >= 2) {
                WGSLTypeInfo *lt = wgsl_tc_store_type_of(tc, lhs);
                WGSLTypeInfo *rt = wgsl_tc_store_type_of(tc, n->children[1]);
                if (lt && rt && wgsl_tc_init_type_mismatch(rt, lt)) {
                    wgsl_tc_error(tc, n->children[1],
                        "value of type %s cannot be converted to the "
                        "assignment target's store type %s",
                        wgsl_type_kind_name((WGSLTypeKind)rt->kind),
                        wgsl_type_kind_name((WGSLTypeKind)lt->kind));
                }
            }
            if (n->kind == WGSL_NODE_STMT_COMPOUND_ASSIGN &&
                n->child_count >= 2)
            {
                WGSLTypeInfo *lt = wgsl_tc_store_type_of(tc, lhs);
                WGSLTypeInfo *rt = wgsl_tc_store_type_of(tc, n->children[1]);
                if (lt && rt) validate_compound_assignment(tc, n, lt, rt);
            }
            /* §9.3 — increment / decrement operand must be a ref to a
             * concrete integer scalar (i32 / u32).  The ref check above
             * already covers the let / param cases. */
            if ((n->kind == WGSL_NODE_STMT_INCREMENT ||
                 n->kind == WGSL_NODE_STMT_DECREMENT))
            {
                WGSLTypeInfo *lt = wgsl_tc_store_type_of(tc, lhs);
                int ok_int_scalar =
                    lt && (lt == tc->types->t_i32 || lt == tc->types->t_u32);
                if (lt && !ok_int_scalar) {
                    wgsl_tc_error(tc, lhs,
                        "'%s' requires a concrete integer scalar (i32 "
                        "or u32) reference; got %s",
                        n->kind == WGSL_NODE_STMT_INCREMENT ? "++" : "--",
                        wgsl_type_kind_name((WGSLTypeKind)lt->kind));
                }
            }
        }
        return;
    }
    case WGSL_NODE_STMT_BREAK:
        /* §9.4.6 — `break` inside a continuing block must target an
         * inner loop / switch; otherwise it would exit the loop's
         * continuing block, which is forbidden. */
        if (tc->continuing_depth > 0 &&
            tc->inner_break_target_depth == 0)
        {
            wgsl_tc_error(tc, n,
                "'break' is not allowed to exit the enclosing loop's "
                "continuing block");
        }
        return;
    case WGSL_NODE_STMT_BREAK_IF:
        for (uint32_t i = 0; i < n->child_count; i++) {
            wgsl_tc_type_expr(tc, n->children[i]);
        }
        if (n->child_count >= 1) {
            WGSLTypeInfo *t = wgsl_tc_store_type_of(tc, n->children[0]);
            if (t && t != tc->types->t_bool) {
                wgsl_tc_error(tc, n->children[0],
                    "'break if' condition must evaluate to bool");
            }
        }
        return;
    case WGSL_NODE_STMT_CONTINUE:
        /* §9.4.8 / §9.7 — `continue` inside a continuing block must
         * target an inner loop; otherwise it would re-enter the
         * enclosing loop's continuing block, which is forbidden. */
        if (tc->continuing_depth > 0 &&
            tc->inner_continue_target_depth == 0)
        {
            wgsl_tc_error(tc, n,
                "'continue' is not allowed inside a continuing block");
        }
        return;
    case WGSL_NODE_STMT_CONTINUING: {
        /* Bump continuing_depth so any lexically-immediate STMT_RETURN
         * encountered during recursion is flagged (return is forbidden
         * regardless of inner loop nesting since there's no inner fn
         * boundary).  Reset both inner-target depths so a bare `break`
         * / `continue` directly in this body (not inside an inner
         * loop / switch) can be flagged.  §9.4.6 / §9.4.8 / §9.4.9. */
        tc->continuing_depth += 1;
        int saved_break = tc->inner_break_target_depth;
        int saved_cont  = tc->inner_continue_target_depth;
        tc->inner_break_target_depth    = 0;
        tc->inner_continue_target_depth = 0;
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if (c->kind >= WGSL_NODE_EXPR_LITERAL_BOOL &&
                c->kind <= WGSL_NODE_EXPR_INDIRECTION)
            {
                wgsl_tc_type_expr(tc, c);
            } else {
                walk_stmt(tc, c);
            }
        }
        tc->inner_break_target_depth    = saved_break;
        tc->inner_continue_target_depth = saved_cont;
        tc->continuing_depth -= 1;
        return;
    }
    case WGSL_NODE_STMT_SWITCH: {
        /* §9.4.2 — selector + case structural checks. */
        /* §9.4.6 — switch is also a `break` target, so bump the inner
         * break-target depth for the duration of the body walk. */
        tc->inner_break_target_depth += 1;
        WGSLTypeInfo *sel_t = NULL;
        if (n->child_count >= 1 && n->children[0]) {
            wgsl_tc_type_expr(tc, n->children[0]);
            sel_t = wgsl_tc_store_type_of(tc, n->children[0]);
            int ok_sel = sel_t &&
                (sel_t == tc->types->t_i32 || sel_t == tc->types->t_u32 ||
                 sel_t == tc->types->t_abstract_int);
            if (sel_t && !ok_sel) {
                wgsl_tc_error(tc, n->children[0],
                    "switch selector must be a concrete integer scalar "
                    "(i32 or u32); got %s",
                    wgsl_type_kind_name((WGSLTypeKind)sel_t->kind));
            }
        }
        int default_count = 0;
        /* Track case-selector const-values seen, to flag duplicates.
         * Stack-cap is fine for realistic shaders. */
        int64_t seen[256];
        int seen_count = 0;
        WGSLTypeInfo *case_common_t = NULL;
        for (uint32_t k = 1; k < n->child_count; k++) {
            WGSLNode *cc = n->children[k];
            if (!cc || cc->kind != WGSL_NODE_STMT_CASE_CLAUSE) continue;
            uint32_t sel_cnt   = (uint32_t)(cc->payload[0] & 0xFFFFFFFFu);
            uint32_t is_def_aln = (uint32_t)((cc->payload[0] >> 32) & 1u);
            uint32_t has_def_in_list = (uint32_t)((cc->payload[0] >> 33) & 1u);
            if (is_def_aln) {
                default_count += 1;
                walk_stmt(tc, cc);
                continue;
            }
            /* `case X, default, Y:` — the default keyword inside the
             * selector list counts toward the switch's default tally
             * (per spec §9.4.2 / §18 grammar). */
            if (has_def_in_list) {
                default_count += 1;
            }
            /* Each selector child first, then the body. */
            for (uint32_t s = 0; s < sel_cnt && s < cc->child_count; s++) {
                WGSLNode *se = cc->children[s];
                if (!se) continue;
                wgsl_tc_type_expr(tc, se);
                WGSLTypeInfo *st = wgsl_tc_store_type_of(tc, se);
                /* Selector ↔ case type compatibility. */
                int case_compatible = sel_t && st &&
                    !wgsl_tc_init_type_mismatch(st, sel_t);
                if (sel_t == tc->types->t_abstract_int &&
                    (st == tc->types->t_i32 ||
                     st == tc->types->t_u32 ||
                     st == tc->types->t_abstract_int))
                {
                    case_compatible = 1;
                }
                if ((sel_t == tc->types->t_i32 ||
                     sel_t == tc->types->t_u32) &&
                    st == tc->types->t_abstract_int)
                {
                    case_compatible = 1;
                }
                if (sel_t && st && !case_compatible) {
                    wgsl_tc_error(tc, se,
                        "case selector type %s does not match switch "
                        "selector type %s",
                        wgsl_type_kind_name((WGSLTypeKind)st->kind),
                        wgsl_type_kind_name((WGSLTypeKind)sel_t->kind));
                }
                if (st == tc->types->t_i32 || st == tc->types->t_u32) {
                    if (!case_common_t) {
                        case_common_t = st;
                    } else if (case_common_t != st) {
                        wgsl_tc_error(tc, se,
                            "case selector type %s does not match earlier "
                            "case selector type %s",
                            wgsl_type_kind_name((WGSLTypeKind)st->kind),
                            wgsl_type_kind_name(
                                (WGSLTypeKind)case_common_t->kind));
                    }
                }
                /* §9.4.2: each case selector must be a const-expression.
                 * Run consteval, then either fold the value into the
                 * duplicate-check set or surface the spec error.  In
                 * either branch, drop the consteval-internal diagnostic
                 * noise so we present a single clean message. */
                if (tc->cev && tc->diag) {
                    size_t sc = tc->diag->count;
                    int    se_e = tc->diag->error_count;
                    int    cv = tc->cev->had_error;
                    WGSLValue v = {0};
                    int eval_ok = wgsl_consteval_expr(tc->cev, se, &v) &&
                                  v.kind == WGSL_VAL_INT;
                    /* Always roll the consteval bag back; we'll emit
                     * the right diagnostic at this layer instead. */
                    tc->diag->count       = sc;
                    tc->diag->error_count = se_e;
                    tc->cev->had_error    = cv;
                    if (eval_ok) {
                        int dup = 0;
                        for (int x = 0; x < seen_count; x++) {
                            if (seen[x] == v.u.i) { dup = 1; break; }
                        }
                        if (dup) {
                            wgsl_tc_error(tc, se,
                                "duplicate case selector value %lld",
                                (long long)v.u.i);
                        } else if (seen_count <
                                   (int)(sizeof seen / sizeof seen[0]))
                        {
                            seen[seen_count++] = v.u.i;
                        }
                    } else {
                        wgsl_tc_error(tc, se,
                            "case selector must be a const-expression");
                    }
                }
            }
            /* Walk the body. */
            if (sel_cnt < cc->child_count) {
                walk_stmt(tc, cc->children[sel_cnt]);
            }
        }
        if (default_count == 0) {
            wgsl_tc_error(tc, n,
                "switch statement must have exactly one default clause");
        } else if (default_count > 1) {
            wgsl_tc_error(tc, n,
                "switch statement must not have more than one default "
                "clause (found %d)", default_count);
        }
        tc->inner_break_target_depth -= 1;
        return;
    }
    case WGSL_NODE_STMT_LOOP:
    case WGSL_NODE_STMT_FOR:
    case WGSL_NODE_STMT_WHILE: {
        /* §9.4.6 / §9.4.8 — these are both `break` and `continue`
         * targets, so bump both inner-target depths.  Switch is a
         * `break` target only (handled in STMT_SWITCH above). */
        tc->inner_break_target_depth    += 1;
        tc->inner_continue_target_depth += 1;
        check_continue_past_decl(tc, n);
        /* §9.4.3 / §9.4.4 — `while` cond and `for` cond must be bool.
         * For `while`, child[0] is the cond.  For `for`, the cond is
         * whichever child is an EXPR node (init / update are stmts). */
        WGSLNode *cond_check = NULL;
        const char *cond_label = NULL;
        if (n->kind == WGSL_NODE_STMT_WHILE && n->child_count >= 1) {
            cond_check = n->children[0];
            cond_label = "while";
        } else if (n->kind == WGSL_NODE_STMT_FOR) {
            for (uint32_t i = 0; i < n->child_count; i++) {
                WGSLNode *c = n->children[i];
                if (c && c->kind >= WGSL_NODE_EXPR_LITERAL_BOOL &&
                       c->kind <= WGSL_NODE_EXPR_INDIRECTION)
                {
                    cond_check = c;
                    cond_label = "for";
                    break;
                }
            }
            uint32_t flags = (uint32_t)n->payload[0];
            if (flags & 4u) {
                uint32_t update_index = 0;
                if (flags & 1u) update_index += 1;
                if (flags & 2u) update_index += 1;
                if (update_index < n->child_count) {
                    WGSLNode *upd = n->children[update_index];
                    if (upd &&
                        (upd->kind == WGSL_NODE_DECL_VAR ||
                         upd->kind == WGSL_NODE_DECL_LET ||
                         upd->kind == WGSL_NODE_DECL_CONST))
                    {
                        wgsl_tc_error(tc, upd,
                            "for update must not be a declaration");
                    }
                }
            }
        }
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if (c->kind >= WGSL_NODE_EXPR_LITERAL_BOOL &&
                c->kind <= WGSL_NODE_EXPR_INDIRECTION)
            {
                wgsl_tc_type_expr(tc, c);
            } else {
                walk_stmt(tc, c);
            }
        }
        if (cond_check) {
            WGSLTypeInfo *t = wgsl_tc_store_type_of(tc, cond_check);
            if (t && t != tc->types->t_bool) {
                wgsl_tc_error(tc, cond_check,
                    "'%s' condition must evaluate to bool", cond_label);
            }
        }
        tc->inner_break_target_depth    -= 1;
        tc->inner_continue_target_depth -= 1;
        return;
    }
    case WGSL_NODE_STMT_IF:
    case WGSL_NODE_STMT_ELSE_IF:
    case WGSL_NODE_STMT_ELSE:
    case WGSL_NODE_STMT_CASE_CLAUSE:
    case WGSL_NODE_STMT_COMPOUND:
        /* Mixed scope: some children are exprs (e.g. if-cond), others are
         * statements (the block body).  Walk blindly — for each child,
         * try expr-typing first by kind, else recurse as stmt. */
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if (c->kind >= WGSL_NODE_EXPR_LITERAL_BOOL &&
                c->kind <= WGSL_NODE_EXPR_INDIRECTION)
            {
                wgsl_tc_type_expr(tc, c);
            } else {
                walk_stmt(tc, c);
            }
        }
        /* §9.4.1 — `if` / `else if` condition (child[0]) must be bool. */
        if ((n->kind == WGSL_NODE_STMT_IF ||
             n->kind == WGSL_NODE_STMT_ELSE_IF) &&
            n->child_count >= 1)
        {
            WGSLNode *c = n->children[0];
            if (c && c->kind >= WGSL_NODE_EXPR_LITERAL_BOOL &&
                   c->kind <= WGSL_NODE_EXPR_INDIRECTION)
            {
                WGSLTypeInfo *t = wgsl_tc_store_type_of(tc, c);
                if (t && t != tc->types->t_bool) {
                    wgsl_tc_error(tc, c,
                        "'if' condition must evaluate to bool");
                }
            }
        }
        return;
    default:
        return;
    }
}

/* ── Entry point ───────────────────────────────────────────────────── */

static void collect_directives(WGSLTypeChecker *tc, WGSLNode *tu) {
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n) continue;
        if (n->kind == WGSL_NODE_DIR_ENABLE) {
            for (uint32_t j = 0; j < n->child_count; j++) {
                WGSLNode *arg = n->children[j];
                if (!arg || arg->kind != WGSL_NODE_EXPR_IDENT) continue;
                /* §4.1.1: enable-extension names are NOT identifiers and
                 * therefore are not resolved against the symbol table.
                 * Read the name directly from source text. */
                uint32_t off = (uint32_t)(arg->payload[0] & 0xFFFFFFFFu);
                uint32_t nl  = (uint32_t)(arg->payload[0] >> 32);
                const char *nm = tc->src->bytes + off;
                uint32_t bit = 0;
                if (nl == 3 && memcmp(nm, "f16", 3) == 0)
                    bit = WGSL_EXT_F16;
                else if (nl == 14 && memcmp(nm, "clip_distances", 14) == 0)
                    bit = WGSL_EXT_CLIP_DISTANCES;
                else if (nl == 20 && memcmp(nm, "dual_source_blending", 20) == 0)
                    bit = WGSL_EXT_DUAL_SOURCE_BLENDING;
                else if (nl == 9 && memcmp(nm, "subgroups", 9) == 0)
                    bit = WGSL_EXT_SUBGROUPS;
                else if (nl == 15 && memcmp(nm, "primitive_index", 15) == 0)
                    bit = WGSL_EXT_PRIMITIVE_INDEX;
                else if (nl == 34 && memcmp(nm, "chromium_experimental_primitive_id", 34) == 0)
                    bit = WGSL_EXT_PRIMITIVE_INDEX;

                if (bit) {
                    /* §4.1.1 example: "A redundant enable directive is
                     * ok." — do not emit a duplicate-extension diagnostic. */
                    tc->enabled_extensions |= bit;
                } else {
                    wgsl_tc_error(tc, arg, "unknown extension '%.*s'", (int)nl, nm);
                }
            }
        } else if (n->kind == WGSL_NODE_DIR_REQUIRES) {
            /* §4.1.2 + §3.8.5: language extension names are NOT
             * identifiers and don't resolve.  Read names directly
             * from source text.  All 11 spec-listed names are
             * recognized; anything else is rejected. */
            static const struct {
                const char *name;
                uint32_t    len;
                uint32_t    bit;
            } kLangFeatures[] = {
                { "readonly_and_readwrite_storage_textures", 39, WGSL_FEAT_RO_RW_STORAGE_TEXTURES },
                { "packed_4x8_integer_dot_product",          30, WGSL_FEAT_PACKED_4X8_INT_DOT },
                { "unrestricted_pointer_parameters",         31, WGSL_FEAT_UNRESTRICTED_PTR_PARAMS },
                { "pointer_composite_access",                24, WGSL_FEAT_POINTER_COMPOSITE_ACCESS },
                { "uniform_buffer_standard_layout",          30, WGSL_FEAT_UNIFORM_BUF_STD_LAYOUT },
                { "subgroup_id",                             11, WGSL_FEAT_SUBGROUP_ID },
                { "subgroup_uniformity",                     19, WGSL_FEAT_SUBGROUP_UNIFORMITY },
                { "texture_and_sampler_let",                 23, WGSL_FEAT_TEXTURE_AND_SAMPLER_LET },
                { "texture_formats_tier1",                   21, WGSL_FEAT_TEXTURE_FORMATS_TIER1 },
                { "linear_indexing",                         15, WGSL_FEAT_LINEAR_INDEXING },
                { "immediate_data",                          14, WGSL_FEAT_IMMEDIATE_DATA },
            };
            for (uint32_t j = 0; j < n->child_count; j++) {
                WGSLNode *arg = n->children[j];
                if (!arg || arg->kind != WGSL_NODE_EXPR_IDENT) continue;
                uint32_t off = (uint32_t)(arg->payload[0] & 0xFFFFFFFFu);
                uint32_t nl  = (uint32_t)(arg->payload[0] >> 32);
                const char *nm = tc->src->bytes + off;
                uint32_t bit = 0;
                for (size_t k = 0; k < sizeof kLangFeatures / sizeof kLangFeatures[0]; k++) {
                    if (kLangFeatures[k].len == nl &&
                        memcmp(kLangFeatures[k].name, nm, nl) == 0) {
                        bit = kLangFeatures[k].bit;
                        break;
                    }
                }
                if (bit) {
                    /* Redundant `requires` is harmless (mirrors §4.1.1
                     * note for `enable`). */
                    tc->required_features |= bit;
                } else {
                    wgsl_tc_error(tc, arg,
                        "unknown language feature '%.*s'", (int)nl, nm);
                }
            }
        } else if (n->kind == WGSL_NODE_DIR_DIAGNOSTIC) {
            /* children: [severity_ident, rule_ident, optional sub_ident]
             * payload[0]: rule_parts (1 or 2). */
            int sev_val = -1;  /* -1 = invalid; 0 = off; 1..4 = WGSLDiagSeverity */
            if (n->child_count >= 1) {
                WGSLNode *sev = n->children[0];
                if (sev && sev->kind == WGSL_NODE_EXPR_IDENT) {
                    uint32_t off = (uint32_t)(sev->payload[0] & 0xFFFFFFFFu);
                    uint32_t len = (uint32_t)(sev->payload[0] >> 32);
                    const char *nm = tc->src->bytes + off;
                    if (len == 3 && memcmp(nm, "off", 3) == 0)     sev_val = 0;
                    if (len == 4 && memcmp(nm, "info", 4) == 0)    sev_val = WGSL_DIAG_INFO;
                    if (len == 7 && memcmp(nm, "warning", 7) == 0) sev_val = WGSL_DIAG_WARNING;
                    if (len == 5 && memcmp(nm, "error", 5) == 0)   sev_val = WGSL_DIAG_ERROR;
                    if (sev_val < 0) {
                        wgsl_tc_error(tc, sev, "invalid diagnostic severity '%.*s' (expected 'off', 'info', 'warning', or 'error')", (int)len, nm);
                    }
                }
            }
            /* §3.8.3 / §2.3.2: single-token rule name must be one of the
             * pre-defined names; unknown multi-token names ("vendor.foo")
             * are permitted (spec: "may itself trigger a diagnostic" —
             * we silently accept). */
            uint32_t rule_parts = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
            const char *rule_nm = NULL;
            uint32_t rule_len = 0;
            if (rule_parts == 1 && n->child_count >= 2) {
                WGSLNode *rule = n->children[1];
                if (rule && rule->kind == WGSL_NODE_EXPR_IDENT) {
                    uint32_t off = (uint32_t)(rule->payload[0] & 0xFFFFFFFFu);
                    uint32_t len = (uint32_t)(rule->payload[0] >> 32);
                    const char *nm = tc->src->bytes + off;
                    int valid =
                        (len == 21 && memcmp(nm, "derivative_uniformity", 21) == 0) ||
                        (len == 19 && memcmp(nm, "subgroup_uniformity", 19) == 0);
                    if (!valid) {
                        wgsl_diag_emit_at(tc->diag, tc->src, WGSL_DIAG_WARNING,
                            rule->span_offset, rule->span_length, NULL,
                            "unknown diagnostic rule '%.*s' (expected "
                            "'derivative_uniformity' or "
                            "'subgroup_uniformity')",
                            (int)len, nm);
                    } else {
                        rule_nm = nm;
                        rule_len = len;
                    }
                }
            }
            /* §4.2: register the filter on the diagnostic bag so downstream
             * filterable diagnostics honour it. */
            if (sev_val >= 0 && rule_nm) {
                wgsl_diag_set_filter(tc->diag, rule_nm, rule_len, sev_val);
            }
        }
    }
}

int wgsl_typecheck(
    WGSLAst             *ast,
    const WGSLSource    *src,
    WGSLArena           *arena,
    WGSLDiagBag         *diag,
    WGSLTypeStore       *types,
    WGSLResolver        *res,
    WGSLConstEvaluator  *cev,
    WGSLTypeChecker     *out)
{
    if (!ast || !src || !arena || !diag || !types || !res || !out) return 0;

    memset(out, 0, sizeof *out);
    out->arena = arena;
    out->diag  = diag;
    out->types = types;
    out->src   = src;
    out->res   = res;
    out->cev   = cev;
    out->supported_features = WGSL_FEAT_ALL_SUPPORTED;

    if (!ast->root) return 0;
    WGSLNode *tu = ast->root;

    collect_directives(out, tu);

    /* Pre-pass — type all module-scope structs and aliases first so
     * downstream type-specs that reference them resolve.  Source order
     * is good enough for v1; cycle detection lives in Phase 8. */
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n) continue;
        if (n->kind == WGSL_NODE_DECL_STRUCT)  wgsl_tc_type_decl_struct(out, n);
        if (n->kind == WGSL_NODE_DECL_ALIAS)   wgsl_tc_type_decl_alias(out, n);
    }

    /* Main pass — every other module-scope decl. */
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n) continue;
        if (n->kind == WGSL_NODE_DECL_STRUCT) continue;
        if (n->kind == WGSL_NODE_DECL_ALIAS)  continue;
        wgsl_tc_type_module_decl(out, n);
    }

    return out->had_error ? 0 : 1;
}
