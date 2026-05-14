static int eval_call(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out) {
    if (!n || n->child_count == 0) return 0;
    WGSLNode *callee = n->children[0];
    const char *nm = NULL;
    uint32_t len = 0;
    WGSLSymbol *sym = callee_head_symbol(callee);
    if (sym && sym->kind == WGSL_SYM_PREDECLARED_FN) {
        WGSLNode *name_node =
            callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
            callee->child_count >= 1 ? callee->children[0] : callee;
        if (!node_ident_text(cev, name_node, &nm, &len)) {
            cev_error(cev, n, "malformed builtin call");
            return 0;
        }
        int folded = eval_call_const_builtin(cev, n, nm, len, out);
        if (folded > 0) return 1;
        if (folded < 0) return 0;
        cev_error(cev, n,
            "builtin '%.*s' const-eval not implemented yet (Iter C)",
            (int)len, nm);
        return 0;
    }
    WGSLTypeInfo *target = constructor_target_type(cev, callee);
    if (target) return eval_constructor_call(cev, n, target, out);
    if (sym && sym->kind == WGSL_SYM_PREDECLARED_TYPEGEN &&
        callee->kind == WGSL_NODE_EXPR_IDENT)
    {
        if (eval_inferred_typegen_constructor(cev, n, out)) return 1;
        if (!cev->had_error) {
            cev_error(cev, n, "constructor const-eval not implemented yet (Iter C)");
        }
        return 0;
    }
    cev_error(cev, n, "constructor const-eval not implemented yet (Iter C)");
    return 0;
}

static int eval_index(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out) {
    if (n->child_count != 2 || !n->children[0] || !n->children[1]) {
        cev_error(cev, n, "malformed index expression");
        return 0;
    }
    WGSLValue base = {0}, idx = {0};
    if (!eval_expr(cev, n->children[0], &base)) return 0;
    if (!eval_expr(cev, n->children[1], &idx)) return 0;
    if (idx.kind != WGSL_VAL_INT) {
        cev_error(cev, n->children[1], "constant index must be an integer");
        return 0;
    }
    if (idx.u.i < 0 || idx.u.i > UINT32_MAX) {
        cev_error(cev, n->children[1], "constant index is out of range");
        return 0;
    }
    uint32_t i = (uint32_t)idx.u.i;
    if (base.kind == WGSL_VAL_VEC || base.kind == WGSL_VAL_ARRAY) {
        if (i >= base.u.agg.count) {
            cev_error(cev, n->children[1], "constant index is out of bounds");
            return 0;
        }
        *out = base.u.agg.elems[i];
        return 1;
    }
    if (base.kind == WGSL_VAL_MAT) {
        if (!base.type || base.type->kind != WGSL_TYPE_MAT) return 0;
        if (i >= base.type->width) {
            cev_error(cev, n->children[1], "constant index is out of bounds");
            return 0;
        }
        uint32_t rows = base.type->rows;
        WGSLValue *elems = (WGSLValue *)wgsl_arena_alloc(
            cev->arena, sizeof *elems * rows);
        if (!elems) {
            cev_error(cev, n, "out of memory while folding matrix index");
            return 0;
        }
        for (uint32_t r = 0; r < rows; r++) {
            elems[r] = base.u.agg.elems[i * rows + r];
        }
        out->kind = WGSL_VAL_VEC;
        out->type = wgsl_type_vec(
            cev->types, (uint8_t)rows, (WGSLTypeInfo *)base.type->ref);
        out->u.agg.count = rows;
        out->u.agg.elems = elems;
        return 1;
    }
    cev_error(cev, n, "constant index requires vector, matrix, or array");
    return 0;
}

static int swizzle_index(char c, int *set) {
    switch (c) {
    case 'x': *set = 0; return 0;
    case 'y': *set = 0; return 1;
    case 'z': *set = 0; return 2;
    case 'w': *set = 0; return 3;
    case 'r': *set = 1; return 0;
    case 'g': *set = 1; return 1;
    case 'b': *set = 1; return 2;
    case 'a': *set = 1; return 3;
    default: return -1;
    }
}

static int eval_vector_member(
    WGSLConstEvaluator *cev, WGSLNode *n, const WGSLValue *base,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (len == 0 || len > 4) {
        cev_error(cev, n, "invalid vector swizzle");
        return 0;
    }
    int set0 = -1;
    WGSLValue elems[4] = {{0}};
    for (uint32_t i = 0; i < len; i++) {
        int set = -1;
        int idx = swizzle_index(nm[i], &set);
        if (idx < 0 || (uint32_t)idx >= base->u.agg.count) {
            cev_error(cev, n, "invalid vector swizzle");
            return 0;
        }
        if (set0 < 0) set0 = set;
        if (set != set0) {
            cev_error(cev, n, "vector swizzle must not mix component sets");
            return 0;
        }
        elems[i] = base->u.agg.elems[idx];
    }
    if (len == 1) {
        *out = elems[0];
        return 1;
    }
    WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *stored * len);
    if (!stored) {
        cev_error(cev, n, "out of memory while folding vector swizzle");
        return 0;
    }
    for (uint32_t i = 0; i < len; i++) stored[i] = elems[i];
    out->kind = WGSL_VAL_VEC;
    out->type = wgsl_type_vec(
        cev->types, (uint8_t)len, (WGSLTypeInfo *)base->type->ref);
    out->u.agg.count = len;
    out->u.agg.elems = stored;
    return 1;
}

static int eval_struct_member(
    WGSLConstEvaluator *cev, WGSLNode *n, const WGSLValue *base,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (base->type && base->type->kind == WGSL_TYPE_FREXP_RESULT) {
        if (len == 5 && memcmp(nm, "fract", 5) == 0 &&
            base->u.agg.count >= 1)
        {
            *out = base->u.agg.elems[0];
            return 1;
        }
        if (len == 3 && memcmp(nm, "exp", 3) == 0 &&
            base->u.agg.count >= 2)
        {
            *out = base->u.agg.elems[1];
            return 1;
        }
        cev_error(cev, n, "__frexp_result has no such member");
        return 0;
    }
    if (base->type && base->type->kind == WGSL_TYPE_MODF_RESULT) {
        if (len == 5 && memcmp(nm, "fract", 5) == 0 &&
            base->u.agg.count >= 1)
        {
            *out = base->u.agg.elems[0];
            return 1;
        }
        if (len == 5 && memcmp(nm, "whole", 5) == 0 &&
            base->u.agg.count >= 2)
        {
            *out = base->u.agg.elems[1];
            return 1;
        }
        cev_error(cev, n, "__modf_result has no such member");
        return 0;
    }
    if (!base->type || base->type->kind != WGSL_TYPE_STRUCT) return 0;
    WGSLNode *sd = (WGSLNode *)base->type->ref;
    uint32_t idx = 0;
    for (uint32_t i = 0; sd && i < sd->child_count; i++) {
        WGSLNode *m = sd->children[i];
        if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
        uint32_t off = (uint32_t)(m->payload[0] & 0xFFFFFFFFu);
        uint32_t ml = (uint32_t)(m->payload[0] >> 32);
        if (ml == len && cev->src && off + ml <= cev->src->length &&
            memcmp(cev->src->bytes + off, nm, len) == 0)
        {
            if (idx >= base->u.agg.count) return 0;
            *out = base->u.agg.elems[idx];
            return 1;
        }
        idx++;
    }
    cev_error(cev, n, "unknown struct member in constant expression");
    return 0;
}

static int eval_member(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out) {
    if (n->child_count != 2 || !n->children[0] || !n->children[1]) {
        cev_error(cev, n, "malformed member expression");
        return 0;
    }
    WGSLValue base = {0};
    if (!eval_expr(cev, n->children[0], &base)) return 0;
    const char *nm = NULL;
    uint32_t len = 0;
    if (!node_ident_text(cev, n->children[1], &nm, &len)) {
        cev_error(cev, n->children[1], "malformed member name");
        return 0;
    }
    if (base.kind == WGSL_VAL_VEC) {
        return eval_vector_member(cev, n, &base, nm, len, out);
    }
    if (base.kind == WGSL_VAL_STRUCT) {
        return eval_struct_member(cev, n, &base, nm, len, out);
    }
    cev_error(cev, n, "constant member access requires vector or struct");
    return 0;
}

/* ── Top-level dispatch ────────────────────────────────────────────── */

static int eval_expr(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out) {
    if (!n) return 0;
    memset(out, 0, sizeof *out);
    switch ((WGSLNodeKind)n->kind) {

    case WGSL_NODE_EXPR_LITERAL_INT:   return parse_int_literal  (cev, n, out);
    case WGSL_NODE_EXPR_LITERAL_FLOAT: return parse_float_literal(cev, n, out);
    case WGSL_NODE_EXPR_LITERAL_BOOL:  return eval_literal_bool  (cev, n, out);

    case WGSL_NODE_EXPR_PAREN:
        if (n->child_count == 1 && n->children[0]) {
            return eval_expr(cev, n->children[0], out);
        }
        cev_error(cev, n, "malformed parenthesised expression");
        return 0;

    case WGSL_NODE_EXPR_UNARY:  return eval_unary (cev, n, out);
    case WGSL_NODE_EXPR_BINARY: return eval_binary(cev, n, out);

    case WGSL_NODE_EXPR_IDENT: {
        WGSLSymbol *sym = wgsl_node_resolved_symbol(n);
        if (!sym) {
            cev_error(cev, n,
                "unresolved identifier in constant expression");
            return 0;
        }
        const WGSLValue *bound = wgsl_consteval_value_of(cev, sym);
        if (!bound) {
            /* Lazy evaluation for module-scope consts */
            if (sym->ast && sym->ast->kind == WGSL_NODE_DECL_CONST) {
                if (sym->ast->flags & WGSL_FLAG_CONST_EVALING) {
                    cev_error(cev, n, "cyclic dependency detected for '%.*s'",
                              (int)sym->name_len, sym->name);
                    return 0;
                }
                sym->ast->flags |= WGSL_FLAG_CONST_EVALING;
                int ok = eval_decl_const(cev, sym->ast);
                sym->ast->flags &= ~WGSL_FLAG_CONST_EVALING;
                if (!ok) return 0;
                bound = wgsl_consteval_value_of(cev, sym);
            }
        }
        if (!bound) {
            cev_error(cev, n,
                "'%.*s' is not a constant",
                (int)sym->name_len, sym->name);
            return 0;
        }
        *out = *bound;
        return 1;
    }

    case WGSL_NODE_EXPR_TEMPLATED_IDENT:
        cev_error(cev, n,
            "constructor const-eval not implemented yet (Iter C)");
        return 0;
    case WGSL_NODE_EXPR_CALL:
        return eval_call(cev, n, out);
    case WGSL_NODE_EXPR_INDEX:
        return eval_index(cev, n, out);
    case WGSL_NODE_EXPR_MEMBER:
        return eval_member(cev, n, out);

    default:
        cev_error(cev, n,
            "expression is not a constant expression");
        return 0;
    }
}

int wgsl_consteval_expr(
    WGSLConstEvaluator *cev, WGSLNode *expr, WGSLValue *out)
{
    if (!cev || !expr || !out) return 0;
    return eval_expr(cev, expr, out);
}

/* ── Iter B — module + fn-scope walker for const decls / asserts ───── */

/* Resolve the constructible type-specifier shapes consteval needs before
 * the type-checker pre-pass has populated every user alias / struct type. */
static WGSLTypeInfo *type_from_typespec(
    const WGSLConstEvaluator *cev, const WGSLNode *tnode)
{
    if (!tnode) return NULL;
    if (tnode->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT) {
        if (tnode->child_count == 0 || !tnode->children[0]) return NULL;
        const WGSLNode *head = tnode->children[0];
        if (head->kind != WGSL_NODE_EXPR_IDENT) return NULL;
        WGSLSymbol *sym = wgsl_node_resolved_symbol((WGSLNode *)head);
        if (!sym) return NULL;
        if (sym->kind == WGSL_SYM_PREDECLARED_TYPE ||
            sym->kind == WGSL_SYM_PREDECLARED_TYPE_ALIAS)
        {
            return sym->type;
        }
        if (sym->kind != WGSL_SYM_PREDECLARED_TYPEGEN) return NULL;
        const char *nm = sym->name;
        uint32_t len = sym->name_len;
        int width = vector_width_from_name(nm, len);
        if (width) {
            if (tnode->child_count < 2) return NULL;
            WGSLTypeInfo *elem = type_from_typespec(cev, tnode->children[1]);
            return elem ? wgsl_type_vec(cev->types, (uint8_t)width, elem) : NULL;
        }
        uint8_t cols = 0, rows = 0;
        if (matrix_dims_from_name(nm, len, &cols, &rows)) {
            if (tnode->child_count < 2) return NULL;
            WGSLTypeInfo *elem = type_from_typespec(cev, tnode->children[1]);
            return elem ? wgsl_type_mat(cev->types, cols, rows, elem) : NULL;
        }
        if (len == 5 && memcmp(nm, "array", 5) == 0) {
            if (tnode->child_count < 2) return NULL;
            WGSLTypeInfo *elem = type_from_typespec(cev, tnode->children[1]);
            if (!elem) return NULL;
            uint32_t length = 0;
            if (tnode->child_count >= 3) {
                WGSLValue v = {0};
                if (!eval_expr((WGSLConstEvaluator *)cev,
                               tnode->children[2], &v) ||
                    v.kind != WGSL_VAL_INT || v.u.i <= 0)
                {
                    return NULL;
                }
                length = (uint32_t)v.u.i;
            }
            return wgsl_type_array(cev->types, elem, length);
        }
        return NULL;
    }
    if (tnode->kind != WGSL_NODE_EXPR_IDENT) return NULL;
    WGSLSymbol *sym = wgsl_node_resolved_symbol((WGSLNode *)tnode);
    if (!sym) return NULL;
    if (sym->kind == WGSL_SYM_PREDECLARED_TYPE ||
        sym->kind == WGSL_SYM_PREDECLARED_TYPE_ALIAS)
    {
        return sym->type;
    }
    if (sym->kind == WGSL_SYM_TYPE_ALIAS) {
        if (sym->type) return sym->type;
        if (!sym->ast || sym->ast->kind != WGSL_NODE_DECL_ALIAS ||
            sym->ast->child_count == 0)
        {
            return NULL;
        }
        if (sym->ast->flags & WGSL_FLAG_TYPING) return NULL;
        sym->ast->flags |= WGSL_FLAG_TYPING;
        WGSLTypeInfo *t = type_from_typespec(cev, sym->ast->children[0]);
        sym->ast->flags &= ~WGSL_FLAG_TYPING;
        if (t) sym->type = t;
        return t;
    }
    if (sym->kind == WGSL_SYM_STRUCT) {
        if (sym->type) return sym->type;
        if (!sym->ast || sym->ast->kind != WGSL_NODE_DECL_STRUCT) return NULL;
        WGSLTypeInfo *st = (WGSLTypeInfo *)wgsl_arena_calloc(
            cev->types->arena, 1, sizeof *st);
        if (!st) return NULL;
        st->kind = (uint16_t)WGSL_TYPE_STRUCT;
        st->ref = sym->ast;
        sym->type = st;
        return st;
    }
    return NULL;
}

/* DECL_CONST children: [type?][init].  payload[1] low bit = has_type.
 *
 * An untyped `const` keeps the initializer's raw type.  In particular,
 * `const C = 1;` binds `C` as AbstractInt, not i32; materialization is
 * a use-site concern.  A typed `const` still materializes to its declared
 * type so value-side range and conversion diagnostics fire here. */
static int eval_decl_const(WGSLConstEvaluator *cev, WGSLNode *n) {
    int has_type = (int)(n->payload[1] & 1u);
    WGSLSymbol *sym = find_decl_symbol(cev, n);
    if (sym && wgsl_consteval_value_of(cev, sym)) return 1;

    if (n->child_count < (uint32_t)(has_type ? 2 : 1)) {
        cev_error(cev, n, "const declaration is missing its initializer");
        return 0;
    }
    WGSLNode *type_node = has_type ? n->children[0]                : NULL;
    WGSLNode *init      = has_type ? n->children[1]                : n->children[0];

    WGSLValue v = {0};
    if (!eval_expr(cev, init, &v)) return 0;

    WGSLTypeInfo *target = type_node
        ? type_from_typespec(cev, type_node) : NULL;
    if (target && !wgsl_consteval_materialize(cev, &v, target, init)) {
        return 0;
    }

    if (!sym) return 1;          /* resolver didn't bind — already errored */
    sym->type = v.type;
    if (!store_binding(cev, sym, &v)) {
        cev_error(cev, n, "out of memory while caching const value");
        return 0;
    }
    return 1;
}

/* DECL_OVERRIDE: in v1, we don't fold the default value into the
 * binding store (override values are pipeline-time).  We only validate
 * that the init, if present, is itself an override-expr — and the
 * cheapest way to check that is to attempt a pure const-eval and quietly
 * suppress the diagnostic on miss. */
static int eval_decl_override(WGSLConstEvaluator *cev, WGSLNode *n) {
    (void)cev; (void)n;
    /* No-op for v1.  Phase 7 will validate the override-expr tier with
     * full context. */
    return 1;
}

/* const_assert evaluation is deferred to the validator (Phase 8) so the
 * type checker (Phase 7) can enforce the boolean type constraint first. */

static void walk_stmt(WGSLConstEvaluator *cev, WGSLNode *n);

static void walk_block(WGSLConstEvaluator *cev, WGSLNode *block) {
    if (!block) return;
    for (uint32_t i = 0; i < block->child_count; i++) {
        walk_stmt(cev, block->children[i]);
    }
}

static void walk_stmt(WGSLConstEvaluator *cev, WGSLNode *n) {
    if (!n) return;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_DECL_CONST:
        eval_decl_const(cev, n);
        return;
    case WGSL_NODE_DECL_CONST_ASSERT:
        /* Deferred to Phase 8 */
        return;
    /* Recurse into compound forms so nested `const` / `const_assert`
     * inside `if { … }`, `for { … }`, etc. get evaluated. */
    case WGSL_NODE_STMT_COMPOUND:
    case WGSL_NODE_STMT_IF:
    case WGSL_NODE_STMT_ELSE_IF:
    case WGSL_NODE_STMT_ELSE:
    case WGSL_NODE_STMT_WHILE:
    case WGSL_NODE_STMT_FOR:
    case WGSL_NODE_STMT_LOOP:
    case WGSL_NODE_STMT_SWITCH:
    case WGSL_NODE_STMT_CASE_CLAUSE:
    case WGSL_NODE_STMT_CONTINUING:
        walk_block(cev, n);
        return;
    default:
        return;
    }
}

static void walk_function(WGSLConstEvaluator *cev, WGSLNode *fn) {
    if (!fn || fn->child_count == 0) return;
    /* Last child is the body (STMT_COMPOUND) per parser layout. */
    WGSLNode *body = fn->children[fn->child_count - 1];
    if (body && body->kind == WGSL_NODE_STMT_COMPOUND) {
        walk_block(cev, body);
    }
}

int wgsl_consteval(
    WGSLAst *ast, const WGSLSource *src, WGSLArena *arena,
    WGSLDiagBag *diag, WGSLTypeStore *types, WGSLResolver *res,
    WGSLConstEvaluator *out)
{
    wgsl_consteval_init(out, arena, diag, types, src, res);
    if (!ast || !ast->root) return 0;

    WGSLNode *tu = ast->root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n) continue;
        switch ((WGSLNodeKind)n->kind) {
        case WGSL_NODE_DECL_CONST:        eval_decl_const   (out, n); break;
        case WGSL_NODE_DECL_OVERRIDE:     eval_decl_override(out, n); break;
        case WGSL_NODE_DECL_CONST_ASSERT: /* Deferred to Phase 8 */ break;
        case WGSL_NODE_DECL_FUNCTION:     walk_function     (out, n); break;
        default: break;
        }
    }
    return out->had_error ? 0 : 1;
}
