/**
 * @file user_fn.c — §11.3 memoizing const-evaluation of pure user functions.
 *
 * WGSL's `@const` attribute is only a notation for built-ins (§12.5); it
 * cannot appear on user-declared functions.  Still, a pure user function
 * whose body is only `const`/`let` bindings plus a `return` of a
 * const-expression is evaluable at shader-creation time when all arguments
 * are const-expressions — the same shape Naga's constant evaluator folds.
 *
 * This module:
 *   - classifies pure function bodies (static AST walk)
 *   - evaluates pure calls on the shared const-eval path (`!runtime`)
 *   - memoizes (fn, args) → result so diamond call graphs are O(unique)
 */
#include "internal/consteval_priv.h"

#include <string.h>

#define USER_FN_MAX_DEPTH  64
#define USER_FN_MAX_ARGS   16

/* value equality (memo key). */

static uint64_t mix64(uint64_t h) {
    h ^= h >> 33; h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ull;
    h ^= h >> 33;
    return h;
}

static uint64_t hash_combine(uint64_t h, uint64_t x) {
    return mix64(h ^ (x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2)));
}

static uint64_t value_hash(const WGSLValue *v) {
    if (!v) return 0x9ae16a3b2f90404full;
    uint64_t h = hash_combine(0x243f6a8885a308d3ull, v->kind);
    switch ((WGSLValKind)v->kind) {
    case WGSL_VAL_INVALID:
        return h;
    case WGSL_VAL_BOOL:
        return hash_combine(h, v->u.b ? 1u : 0u);
    case WGSL_VAL_INT:
        return hash_combine(h, (uint64_t)v->u.i);
    case WGSL_VAL_FLOAT: {
        uint64_t bits = 0;
        if (v->u.f != v->u.f) {
            bits = 0x7ff8000000000000ull;
        } else if (v->u.f == 0.0) {
            bits = 0;
        } else {
            memcpy(&bits, &v->u.f, sizeof bits);
        }
        return hash_combine(h, bits);
    }
    case WGSL_VAL_VEC:
    case WGSL_VAL_MAT:
    case WGSL_VAL_ARRAY:
    case WGSL_VAL_STRUCT:
        h = hash_combine(h, v->u.agg.count);
        for (uint32_t i = 0; i < v->u.agg.count; i++)
            h = hash_combine(h, value_hash(&v->u.agg.elems[i]));
        return h;
    default:
        return h;
    }
}

static int value_eq(const WGSLValue *a, const WGSLValue *b) {
    if (!a || !b) return a == b;
    if (a->kind != b->kind) return 0;
    switch ((WGSLValKind)a->kind) {
    case WGSL_VAL_INVALID: return 1;
    case WGSL_VAL_BOOL:    return a->u.b == b->u.b;
    case WGSL_VAL_INT:     return a->u.i == b->u.i;
    case WGSL_VAL_FLOAT:
        if (a->u.f == b->u.f) return 1;
        /* NaN keys match NaN (any) so repeated NaN results hit the cache. */
        return (a->u.f != a->u.f) && (b->u.f != b->u.f);
    case WGSL_VAL_VEC:
    case WGSL_VAL_MAT:
    case WGSL_VAL_ARRAY:
    case WGSL_VAL_STRUCT:
        if (a->u.agg.count != b->u.agg.count) return 0;
        for (uint32_t i = 0; i < a->u.agg.count; i++)
            if (!value_eq(&a->u.agg.elems[i], &b->u.agg.elems[i])) return 0;
        return 1;
    default:
        return 0;
    }
}

/* body purity (static). */

static int expr_ok_for_const_fn(const WGSLNode *n, int depth);
static int stmt_ok_for_const_fn(const WGSLNode *n, int depth);

static int fn_decl_is_const_form(const WGSLNode *fn, int depth) {
    if (!fn || fn->kind != WGSL_NODE_DECL_FUNCTION) return 0;
    if (depth > USER_FN_MAX_DEPTH) return 0;
    if (fn->child_count == 0) return 0;
    WGSLNode *body = fn->children[fn->child_count - 1];
    if (!body || body->kind != WGSL_NODE_STMT_COMPOUND) return 0;
    return stmt_ok_for_const_fn(body, depth);
}

static int call_callee_is_const_form(const WGSLNode *call, int depth) {
    if (!call || call->kind != WGSL_NODE_EXPR_CALL || call->child_count == 0)
        return 0;
    WGSLNode *callee = call->children[0];
    if (callee && callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
        callee->child_count)
        callee = callee->children[0];
    WGSLSymbol *sym = callee ? wgsl_node_resolved_symbol(callee) : NULL;
    if (!sym) return 0;
    if (sym->kind == WGSL_SYM_PREDECLARED_FN) {
        /* Built-in: allowed only if marked @const in the overload table.
         * Conservative without the table: allow; eval will reject non-@const. */
        return 1;
    }
    if (sym->kind == WGSL_SYM_PREDECLARED_TYPEGEN ||
        sym->kind == WGSL_SYM_STRUCT ||
        sym->kind == WGSL_SYM_TYPE_ALIAS)
        return 1;   /* constructor-like */
    if (sym->kind != WGSL_SYM_FUNCTION || !sym->ast) return 0;
    uint16_t stage = (uint16_t)(WGSL_SYM_FLAG_VERTEX | WGSL_SYM_FLAG_FRAGMENT |
                                WGSL_SYM_FLAG_COMPUTE);
    if (sym->flags & stage) return 0;
    return fn_decl_is_const_form(sym->ast, depth + 1);
}

static int expr_ok_for_const_fn(const WGSLNode *n, int depth) {
    if (!n) return 0;
    if (depth > USER_FN_MAX_DEPTH) return 0;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_LITERAL_BOOL:
    case WGSL_NODE_EXPR_LITERAL_INT:
    case WGSL_NODE_EXPR_LITERAL_FLOAT:
        return 1;
    case WGSL_NODE_EXPR_PAREN:
    case WGSL_NODE_EXPR_UNARY:
        return n->child_count >= 1 && expr_ok_for_const_fn(n->children[0], depth + 1);
    case WGSL_NODE_EXPR_BINARY:
        return n->child_count >= 2 &&
               expr_ok_for_const_fn(n->children[0], depth + 1) &&
               expr_ok_for_const_fn(n->children[1], depth + 1);
    case WGSL_NODE_EXPR_IDENT: {
        WGSLSymbol *s = wgsl_node_resolved_symbol((WGSLNode *)n);
        if (!s) return 0;
        /* Params, local const/let (bound at eval), and module const. */
        return s->kind == WGSL_SYM_CONST || s->kind == WGSL_SYM_LET ||
               s->kind == WGSL_SYM_PARAM || s->kind == WGSL_SYM_OVERRIDE;
    }
    case WGSL_NODE_EXPR_CALL: {
        if (!call_callee_is_const_form(n, depth)) return 0;
        for (uint32_t i = 1; i < n->child_count; i++)
            if (!expr_ok_for_const_fn(n->children[i], depth + 1)) return 0;
        return 1;
    }
    case WGSL_NODE_EXPR_INDEX:
        return n->child_count >= 2 &&
               expr_ok_for_const_fn(n->children[0], depth + 1) &&
               expr_ok_for_const_fn(n->children[1], depth + 1);
    case WGSL_NODE_EXPR_MEMBER:
        return n->child_count >= 1 &&
               expr_ok_for_const_fn(n->children[0], depth + 1);
    default:
        return 0;
    }
}

static int stmt_ok_for_const_fn(const WGSLNode *n, int depth) {
    if (!n) return 1;
    if (depth > USER_FN_MAX_DEPTH) return 0;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_STMT_COMPOUND:
        for (uint32_t i = 0; i < n->child_count; i++)
            if (!stmt_ok_for_const_fn(n->children[i], depth + 1)) return 0;
        return 1;
    case WGSL_NODE_DECL_CONST:
    case WGSL_NODE_DECL_LET: {
        int has_type = wgsl_letlike_has_type(n);
        WGSLNode *init = (n->child_count >= (uint32_t)(has_type ? 2 : 1))
            ? n->children[has_type ? 1 : 0] : NULL;
        return init && expr_ok_for_const_fn(init, depth + 1);
    }
    case WGSL_NODE_STMT_RETURN:
        if (n->child_count == 0) return 1;   /* void return */
        return expr_ok_for_const_fn(n->children[0], depth + 1);
    case WGSL_NODE_STMT_IF:
    case WGSL_NODE_STMT_ELSE_IF: {
        /* cond is child 0; then/else compounds follow. */
        if (n->child_count < 1 || !expr_ok_for_const_fn(n->children[0], depth + 1))
            return 0;
        for (uint32_t i = 1; i < n->child_count; i++)
            if (!stmt_ok_for_const_fn(n->children[i], depth + 1)) return 0;
        return 1;
    }
    case WGSL_NODE_STMT_ELSE:
        return n->child_count >= 1 && stmt_ok_for_const_fn(n->children[0], depth + 1);
    default:
        /* var, loops, assign, barriers, discard, … — not const-fn material. */
        return 0;
    }
}

int user_fn_is_const_form(const WGSLNode *fn) {
    return fn_decl_is_const_form(fn, 0);
}

int user_fn_call_is_const_expr(const WGSLNode *call) {
    if (!call || call->kind != WGSL_NODE_EXPR_CALL || call->child_count == 0)
        return 0;
    WGSLNode *callee = call->children[0];
    if (callee && callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
        callee->child_count)
        callee = callee->children[0];
    WGSLSymbol *sym = callee ? wgsl_node_resolved_symbol(callee) : NULL;
    if (!sym || sym->kind != WGSL_SYM_FUNCTION || !sym->ast) return 0;
    uint16_t stage = (uint16_t)(WGSL_SYM_FLAG_VERTEX | WGSL_SYM_FLAG_FRAGMENT |
                                WGSL_SYM_FLAG_COMPUTE);
    if (sym->flags & stage) return 0;
    if (!user_fn_is_const_form(sym->ast)) return 0;
    for (uint32_t i = 1; i < call->child_count; i++) {
        /* is_const_expr is in exprs / overloads — declared in consteval_priv
         * as a forward to avoid a cyclic include.  We re-use the public shape
         * via a local walk that matches is_const_expr's const rules plus
         * user-fn recursion already in expr_ok_for_const_fn. */
        if (!expr_ok_for_const_fn(call->children[i], 0)) return 0;
    }
    return 1;
}

/* memo table. */

static uint64_t memo_key_hash(
    const WGSLNode *fn, const WGSLValue *args, uint32_t nargs)
{
    uint64_t h = mix64((uint64_t)(uintptr_t)fn);
    h = hash_combine(h, nargs);
    for (uint32_t i = 0; i < nargs; i++)
        h = hash_combine(h, value_hash(&args[i]));
    return h ? h : 1;
}

static void memo_htab_put_index(WGSLConstEvaluator *cev, size_t idx) {
    if (!cev->store.fn_memo_htab || !cev->store.fn_memo_htab_cap ||
        idx >= cev->store.fn_memo_count)
    {
        return;
    }
    size_t mask = cev->store.fn_memo_htab_cap - 1;
    size_t slot = (size_t)cev->store.fn_memo[idx].hash & mask;
    while (cev->store.fn_memo_htab[slot] != (size_t)-1) {
        if (cev->store.fn_memo_htab[slot] == idx) return;
        slot = (slot + 1) & mask;
    }
    cev->store.fn_memo_htab[slot] = idx;
}

static int memo_htab_reserve(WGSLConstEvaluator *cev, size_t want_count) {
    if (want_count == 0) return 1;
    if (want_count > SIZE_MAX / 10) return 0;
    size_t cap = cev->store.fn_memo_htab_cap ? cev->store.fn_memo_htab_cap : 32;
    while (want_count * 10 >= cap * 7) {
        if (cap > SIZE_MAX / 2) return 0;
        cap *= 2;
    }
    if (cap == cev->store.fn_memo_htab_cap) return 1;

    size_t *nt = (size_t *)malloc(cap * sizeof *nt);
    if (!nt) return 0;
    for (size_t i = 0; i < cap; i++) nt[i] = (size_t)-1;
    size_t *old = cev->store.fn_memo_htab;
    cev->store.fn_memo_htab = nt;
    cev->store.fn_memo_htab_cap = cap;
    for (size_t i = 0; i < cev->store.fn_memo_count; i++)
        memo_htab_put_index(cev, i);
    free(old);
    return 1;
}

static int memo_find(WGSLConstEvaluator *cev, const WGSLNode *fn,
                     const WGSLValue *args, uint32_t nargs, WGSLValue *out) {
    if (!cev || !cev->store.fn_memo_htab || !cev->store.fn_memo_htab_cap) return 0;
    uint64_t hash = memo_key_hash(fn, args, nargs);
    size_t mask = cev->store.fn_memo_htab_cap - 1;
    size_t slot = (size_t)hash & mask;
    for (size_t n = 0; n < cev->store.fn_memo_htab_cap; n++) {
        size_t idx = cev->store.fn_memo_htab[slot];
        if (idx == (size_t)-1) break;
        if (idx >= cev->store.fn_memo_count) {
            slot = (slot + 1) & mask;
            continue;
        }
        WGSLConstFnMemo *m = &cev->store.fn_memo[idx];
        if (m->hash != hash || m->fn != fn || m->nargs != nargs) {
            slot = (slot + 1) & mask;
            continue;
        }
        int match = 1;
        for (uint32_t a = 0; a < nargs; a++) {
            if (!value_eq(&m->args[a], &args[a])) { match = 0; break; }
        }
        if (!match) {
            slot = (slot + 1) & mask;
            continue;
        }
        if (!m->ok) return -1;   /* cached failure */
        *out = deep_copy_value(cev, &m->result);
        return 1;
    }
    return 0;
}

static void memo_store(WGSLConstEvaluator *cev, const WGSLNode *fn,
                       const WGSLValue *args, uint32_t nargs,
                       const WGSLValue *result, int ok) {
    uint64_t hash = memo_key_hash(fn, args, nargs);
    if (!memo_htab_reserve(cev, cev->store.fn_memo_count + 1)) return;
    if (cev->store.fn_memo_count == cev->store.fn_memo_cap) {
        size_t nc = cev->store.fn_memo_cap ? cev->store.fn_memo_cap * 2 : 8;
        WGSLConstFnMemo *g = (WGSLConstFnMemo *)realloc(
            cev->store.fn_memo, nc * sizeof *g);
        if (!g) return;
        cev->store.fn_memo = g;
        cev->store.fn_memo_cap = nc;
    }
    WGSLConstFnMemo *m = &cev->store.fn_memo[cev->store.fn_memo_count++];
    memset(m, 0, sizeof *m);
    m->fn = fn;
    m->hash = hash;
    m->nargs = nargs;
    m->ok = ok;
    if (nargs) {
        m->args = (WGSLValue *)wgsl_arena_alloc(cev->arena, nargs * sizeof(WGSLValue));
        if (!m->args) {
            cev->store.fn_memo_count--;
            memset(m, 0, sizeof *m);
            return;
        }
        for (uint32_t i = 0; i < nargs; i++)
            m->args[i] = deep_copy_value(cev, &args[i]);
    }
    if (ok && result)
        m->result = deep_copy_value(cev, result);
    memo_htab_put_index(cev, cev->store.fn_memo_count - 1);
}

/* body evaluation. */

static int eval_const_fn_stmt(WGSLConstEvaluator *cev, WGSLNode *n,
                              WGSLValue *ret_out, int *did_return);

static int eval_const_fn_block(WGSLConstEvaluator *cev, WGSLNode *block,
                               WGSLValue *ret_out, int *did_return) {
    if (!block) return 1;
    for (uint32_t i = 0; i < block->child_count; i++) {
        if (!eval_const_fn_stmt(cev, block->children[i], ret_out, did_return))
            return 0;
        if (*did_return) return 1;
    }
    return 1;
}

static int eval_const_fn_stmt(WGSLConstEvaluator *cev, WGSLNode *n,
                              WGSLValue *ret_out, int *did_return) {
    if (!n) return 1;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_STMT_COMPOUND:
        return eval_const_fn_block(cev, n, ret_out, did_return);
    case WGSL_NODE_DECL_CONST:
        return eval_decl_const(cev, n);
    case WGSL_NODE_DECL_LET: {
        int has_type = wgsl_letlike_has_type(n);
        WGSLNode *init = (n->child_count >= (uint32_t)(has_type ? 2 : 1))
            ? n->children[has_type ? 1 : 0] : NULL;
        if (!init) {
            cev_error(cev, n, "let in const-function requires an initializer");
            return 0;
        }
        WGSLValue v = {0};
        if (!eval_expr(cev, init, &v)) return 0;
        WGSLSymbol *sym = find_decl_symbol(cev, n);
        if (sym) {
            if (sym->type)
                wgsl_consteval_materialize(cev, &v, sym->type, init);
            if (!wgsl_consteval_set_value(cev, sym, &v)) {
                cev_error(cev, n, "out of memory binding let in const-function");
                return 0;
            }
        }
        return 1;
    }
    case WGSL_NODE_STMT_RETURN: {
        memset(ret_out, 0, sizeof *ret_out);
        if (n->child_count && n->children[0]) {
            if (!eval_expr(cev, n->children[0], ret_out)) return 0;
        }
        *did_return = 1;
        return 1;
    }
    case WGSL_NODE_STMT_IF:
    case WGSL_NODE_STMT_ELSE_IF: {
        if (n->child_count < 1) return 0;
        WGSLValue c = {0};
        if (!eval_expr(cev, n->children[0], &c)) return 0;
        int truth = 0;
        if (c.kind == WGSL_VAL_BOOL) truth = c.u.b;
        else if (c.kind == WGSL_VAL_INT) truth = c.u.i != 0;
        else {
            cev_error(cev, n->children[0],
                      "const-function 'if' condition must be bool");
            return 0;
        }
        if (truth) {
            if (n->child_count >= 2)
                return eval_const_fn_stmt(cev, n->children[1], ret_out, did_return);
            return 1;
        }
        /* else / else-if chain: remaining children after then-body */
        for (uint32_t i = 2; i < n->child_count; i++) {
            if (!eval_const_fn_stmt(cev, n->children[i], ret_out, did_return))
                return 0;
            if (*did_return) return 1;
        }
        return 1;
    }
    case WGSL_NODE_STMT_ELSE:
        if (n->child_count >= 1)
            return eval_const_fn_stmt(cev, n->children[0], ret_out, did_return);
        return 1;
    default:
        cev_error(cev, n,
                  "statement not allowed in a const-evaluable user function");
        return 0;
    }
}

int eval_user_fn_const_call(WGSLConstEvaluator *cev, WGSLNode *call,
                            WGSLSymbol *sym, WGSLValue *out) {
    if (!cev || !call || !sym || !sym->ast) return 0;
    /* Runtime / interpreter path keeps the SIMT hook — not shader-creation.
     * Also skip once N-lane mode is live: param bindings must stay per-lane. */
    if (cev->simt.runtime || cev->simt.nlanes >= 1) return 0;

    WGSLNode *fn = sym->ast;
    if (fn->kind != WGSL_NODE_DECL_FUNCTION) return 0;

    uint16_t stage = (uint16_t)(WGSL_SYM_FLAG_VERTEX | WGSL_SYM_FLAG_FRAGMENT |
                                WGSL_SYM_FLAG_COMPUTE);
    if (sym->flags & stage) {
        cev_error(cev, call, "cannot call entry point in a const-expression");
        return 0;
    }
    if (!user_fn_is_const_form(fn)) return 0;

    if (cev->store.fn_call_depth >= USER_FN_MAX_DEPTH) {
        cev_error(cev, call, "const-function call nested too deeply (max %d)",
                  USER_FN_MAX_DEPTH);
        return 0;
    }

    /* Evaluate arguments. */
    uint32_t nargs = call->child_count > 0 ? call->child_count - 1 : 0;
    if (nargs > USER_FN_MAX_ARGS) {
        cev_error(cev, call, "const-function has too many arguments");
        return 0;
    }
    WGSLValue args[USER_FN_MAX_ARGS];
    memset(args, 0, sizeof args);
    for (uint32_t i = 0; i < nargs; i++) {
        if (!eval_expr(cev, call->children[i + 1], &args[i])) return 0;
    }

    int hit = memo_find(cev, fn, args, nargs, out);
    if (hit > 0) return 1;
    if (hit < 0) {
        cev_error(cev, call, "const-function evaluation failed (cached)");
        return 0;
    }

    /* Bind parameters in declaration order. */
    uint32_t pi = 0;
    for (uint32_t c = 0; c < fn->child_count && pi < nargs; c++) {
        WGSLNode *p = fn->children[c];
        if (!p || p->kind != WGSL_NODE_DECL_PARAM) continue;
        WGSLSymbol *ps = find_decl_symbol(cev, p);
        if (ps) {
            if (ps->type)
                wgsl_consteval_materialize(cev, &args[pi], ps->type,
                                           call->children[pi + 1]);
            if (!wgsl_consteval_set_value(cev, ps, &args[pi])) {
                cev_error(cev, call, "out of memory binding const-function param");
                return 0;
            }
        }
        pi++;
    }

    WGSLNode *body = fn->children[fn->child_count - 1];
    WGSLValue ret;
    memset(&ret, 0, sizeof ret);
    int did_return = 0;
    cev->store.fn_call_depth++;
    int ok = eval_const_fn_block(cev, body, &ret, &did_return);
    cev->store.fn_call_depth--;
    if (!ok) {
        memo_store(cev, fn, args, nargs, NULL, 0);
        return 0;
    }
    if (!did_return) {
        /* void-ish fallthrough: invalid result for value context */
        memset(out, 0, sizeof *out);
        memo_store(cev, fn, args, nargs, out, 1);
        return 1;
    }
    *out = ret;
    memo_store(cev, fn, args, nargs, out, 1);
    return 1;
}
