/* ── Statements ─────────────────────────────────────────────────── */

static WGSLNode *parse_statement(P *p);

static WGSLNode *parse_compound_statement(P *p) {
    const WGSLToken *open = expect(p, WGSL_TOK_LBRACE, "'{'");

    NL kids; nl_init(&kids);
    while (!at(p, WGSL_TOK_RBRACE) && !at(p, WGSL_TOK_EOF)) {
        WGSLNode *s = parse_statement(p);
        if (s && s->kind != WGSL_NODE_INVALID) {
            nl_push(&kids, s);
        } else if (s && s->span_offset == 0 && s->span_length == 0) {
            continue;
        } else {
            recover_to_sync(p);
        }
    }
    const WGSLToken *close = peek(p);
    expect(p, WGSL_TOK_RBRACE, "'}'");

    WGSLSpan span = wgsl_span_make(
        open->span.offset, close->span.offset + close->span.length - open->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_STMT_COMPOUND, span);
    if (node) finalize_children(p, node, &kids);
    nl_destroy(&kids);
    return node;
}

static void parse_discard_attributes(P *p) {
    if (!at(p, WGSL_TOK_AT)) return;
    NL discard; nl_init(&discard);
    parse_attributes(p, &discard);
    nl_destroy(&discard);
}

static WGSLNode *parse_attributed_compound_statement(P *p) {
    NL attrs; nl_init(&attrs);
    parse_attributes(p, &attrs);
    WGSLNode *body = parse_compound_statement(p);
    attach_statement_attrs(p, body, &attrs);
    nl_destroy(&attrs);
    return body;
}

static const WGSLToken *expect_attributed_lbrace(P *p, const char *label, NL *attrs) {
    if (attrs) parse_attributes(p, attrs);
    else       parse_discard_attributes(p);
    return expect(p, WGSL_TOK_LBRACE, label);
}

/* Helper: optionally consume `;` and compute the right span end.
 * If `as_statement == 0` (used by for_init / for_update), skip the `;`.
 * Returns the desired end offset for the span. */
static uint32_t finish_stmt_or_inline(P *p, uint32_t fallback_end, int as_statement) {
    if (!as_statement) return fallback_end;
    const WGSLToken *semi = peek(p);
    expect(p, WGSL_TOK_SEMICOLON, "';'");
    return semi->span.offset + semi->span.length;
}

/* Parse `let IDENT (':' type)? '=' expr` (with or without trailing `;`). */
static WGSLNode *parse_let_decl(P *p, int as_statement) {
    const WGSLToken *kw   = expect(p, WGSL_TOK_KW_LET, "'let'");
    const WGSLToken *name = expect(p, WGSL_TOK_IDENT, "identifier");

    NL kids; nl_init(&kids);
    uint64_t name_pld =
        ((uint64_t)name->span.offset) | ((uint64_t)name->span.length << 32);

    int has_type = 0;
    if (match(p, WGSL_TOK_COLON)) {
        WGSLNode *t = parse_type_specifier(p);
        if (t) { nl_push(&kids, t); has_type = 1; }
    }
    expect(p, WGSL_TOK_EQUAL, "'='");
    WGSLNode *init = parse_expression(p);
    if (init) nl_push(&kids, init);

    uint32_t fallback = init ? init->span_offset + init->span_length
                             : name->span.offset + name->span.length;
    uint32_t end = finish_stmt_or_inline(p, fallback, as_statement);

    WGSLSpan span = wgsl_span_make(kw->span.offset, end - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_DECL_LET, span);
    if (node) {
        node->payload[0] = name_pld;
        node->payload[1] = (uint64_t)(has_type ? 1u : 0u);
        finalize_children(p, node, &kids);
    }
    nl_destroy(&kids);
    return node;
}

/* Parse `var ('<' tpl, … '>')? IDENT (':' type)? ('=' expr)?`.
 *
 * Both function-scope and module-scope use this parser; the caller
 * controls whether attributes (`attrs_or_null`) and a template list
 * (`allow_template`) are accepted.  Function-scope passes `NULL, 0`;
 * module-scope passes the collected `&attrs, 1`.
 *
 * DECL_VAR payload encoding (used by both contexts):
 *   payload[0]: name span (offset | length << 32)
 *   payload[1]: attr_count (low 32) | tpl_count (high 32)
 *   payload[2]: has_type (low 32) | has_init (high 32)
 *
 * Children layout: [attrs…][tpl_args…][type?][init?]. */
static WGSLNode *parse_var_decl(
    P *p, int as_statement, int allow_template, NL *attrs_or_null)
{
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_VAR, "'var'");

    NL kids; nl_init(&kids);
    uint32_t attr_count = 0;
    if (attrs_or_null) {
        for (size_t i = 0; i < attrs_or_null->count; i++)
            nl_push(&kids, attrs_or_null->items[i]);
        attr_count = (uint32_t)attrs_or_null->count;
    }

    /* Optional template list (module-scope only): <addr_space>
     * or <addr_space, access>. */
    uint32_t tpl_count = 0;
    if (allow_template && at(p, WGSL_TOK_TEMPLATE_START)) {
        advance(p);
        do {
            if (at(p, WGSL_TOK_TEMPLATE_END)) break;
            WGSLNode *e = parse_expression(p);
            if (e) { nl_push(&kids, e); tpl_count += 1; }
        } while (match(p, WGSL_TOK_COMMA));
        expect(p, WGSL_TOK_TEMPLATE_END, "'>' (template list end)");
    }

    const WGSLToken *name = expect(p, WGSL_TOK_IDENT, "variable name");
    uint64_t name_pld =
        ((uint64_t)name->span.offset) | ((uint64_t)name->span.length << 32);

    int has_type = 0;
    if (match(p, WGSL_TOK_COLON)) {
        WGSLNode *t = parse_type_specifier(p);
        if (t) { nl_push(&kids, t); has_type = 1; }
    }
    int has_init = 0;
    if (match(p, WGSL_TOK_EQUAL)) {
        WGSLNode *init = parse_expression(p);
        if (init) { nl_push(&kids, init); has_init = 1; }
    }

    uint32_t fallback = (kids.count > 0)
        ? kids.items[kids.count - 1]->span_offset
            + kids.items[kids.count - 1]->span_length
        : name->span.offset + name->span.length;
    uint32_t end = finish_stmt_or_inline(p, fallback, as_statement);

    uint32_t start = (attr_count > 0)
        ? attrs_or_null->items[0]->span_offset
        : kw->span.offset;
    WGSLSpan span = wgsl_span_make(start, end - start);
    WGSLNode *node = make_node(p, WGSL_NODE_DECL_VAR, span);
    if (node) {
        node->payload[0] = name_pld;
        node->payload[1] = ((uint64_t)attr_count) | ((uint64_t)tpl_count << 32);
        node->payload[2] =
            ((uint64_t)(has_type ? 1u : 0u)) |
            ((uint64_t)(has_init ? 1u : 0u) << 32);
        finalize_children(p, node, &kids);
    }
    nl_destroy(&kids);
    return node;
}

/* Parse `const IDENT (':' type)? '=' expr`.
 * (Module-scope `const` has the same shape but lives in TU; Round B
 * Iter 2 reuses this helper.) */
static WGSLNode *parse_const_decl(P *p, int as_statement) {
    const WGSLToken *kw   = expect(p, WGSL_TOK_KW_CONST, "'const'");
    const WGSLToken *name = expect(p, WGSL_TOK_IDENT, "name");

    NL kids; nl_init(&kids);
    uint64_t name_pld =
        ((uint64_t)name->span.offset) | ((uint64_t)name->span.length << 32);

    int has_type = 0;
    if (match(p, WGSL_TOK_COLON)) {
        WGSLNode *t = parse_type_specifier(p);
        if (t) { nl_push(&kids, t); has_type = 1; }
    }
    expect(p, WGSL_TOK_EQUAL, "'='");
    WGSLNode *init = parse_expression(p);
    if (init) nl_push(&kids, init);

    uint32_t fallback = init ? init->span_offset + init->span_length
                             : name->span.offset + name->span.length;
    uint32_t end = finish_stmt_or_inline(p, fallback, as_statement);

    WGSLSpan span = wgsl_span_make(kw->span.offset, end - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_DECL_CONST, span);
    if (node) {
        node->payload[0] = name_pld;
        node->payload[1] = (uint64_t)(has_type ? 1u : 0u);
        finalize_children(p, node, &kids);
    }
    nl_destroy(&kids);
    return node;
}

static WGSLNode *parse_const_assert_statement(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_CONST_ASSERT, "'const_assert'");
    WGSLNode *e = parse_expression(p);
    const WGSLToken *semi = peek(p);
    expect(p, WGSL_TOK_SEMICOLON, "';'");

    WGSLSpan span = wgsl_span_make(
        kw->span.offset, semi->span.offset + semi->span.length - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_DECL_CONST_ASSERT, span);
    if (node && e) {
        WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, sizeof(WGSLNode *));
        if (arr) { arr[0] = e; node->children = arr; node->child_count = 1; }
    }
    return node;
}

static WGSLNode *parse_return_statement(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_RETURN, "'return'");

    NL kids; nl_init(&kids);
    if (!at(p, WGSL_TOK_SEMICOLON)) {
        WGSLNode *e = parse_expression(p);
        if (e) nl_push(&kids, e);
    }
    const WGSLToken *semi = peek(p);
    expect(p, WGSL_TOK_SEMICOLON, "';'");

    WGSLSpan span = wgsl_span_make(
        kw->span.offset, semi->span.offset + semi->span.length - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_STMT_RETURN, span);
    if (node) finalize_children(p, node, &kids);
    nl_destroy(&kids);
    return node;
}

/* Default statement: handles assignment / compound assignment / phony /
 * increment / decrement / function-call statement.  Pass `as_statement=0`
 * for use as a for-init / for-update update.  Returns NULL if there is
 * nothing to parse. */
static WGSLNode *parse_default_statement(P *p, int as_statement) {
    /* Phony assignment: `_` `=` expr ';' */
    if (at(p, WGSL_TOK_UNDERSCORE) && peek_n(p, 1)->kind == WGSL_TOK_EQUAL) {
        const WGSLToken *under = advance(p);
        advance(p);   /* `=` */
        WGSLNode *rhs = parse_expression(p);
        uint32_t fallback = rhs ? rhs->span_offset + rhs->span_length
                                : under->span.offset + under->span.length;
        uint32_t end = finish_stmt_or_inline(p, fallback, as_statement);
        WGSLSpan span = wgsl_span_make(under->span.offset, end - under->span.offset);
        WGSLNode *node = make_node(p, WGSL_NODE_STMT_PHONY_ASSIGN, span);
        if (node && rhs) {
            WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, sizeof(WGSLNode *));
            if (arr) { arr[0] = rhs; node->children = arr; node->child_count = 1; }
        }
        return node;
    }

    WGSLNode *lhs = parse_expression(p);
    if (!lhs) return NULL;

    WGSLTokenKind k = peek(p)->kind;

    /* `++` / `--` */
    if (k == WGSL_TOK_PLUS_PLUS || k == WGSL_TOK_MINUS_MINUS) {
        advance(p);
        uint32_t fallback = lhs->span_offset + lhs->span_length + 2;
        uint32_t end = finish_stmt_or_inline(p, fallback, as_statement);
        WGSLSpan span = wgsl_span_make(lhs->span_offset, end - lhs->span_offset);
        WGSLNodeKind nk = (k == WGSL_TOK_PLUS_PLUS)
            ? WGSL_NODE_STMT_INCREMENT : WGSL_NODE_STMT_DECREMENT;
        WGSLNode *node = make_node(p, nk, span);
        if (node) {
            WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, sizeof(WGSLNode *));
            if (arr) { arr[0] = lhs; node->children = arr; node->child_count = 1; }
        }
        return node;
    }

    /* Assignment forms */
    int is_assign = 0, is_compound = 0;
    switch (k) {
    case WGSL_TOK_EQUAL:                  is_assign = 1; break;
    case WGSL_TOK_PLUS_EQUAL:
    case WGSL_TOK_MINUS_EQUAL:
    case WGSL_TOK_STAR_EQUAL:
    case WGSL_TOK_SLASH_EQUAL:
    case WGSL_TOK_PERCENT_EQUAL:
    case WGSL_TOK_AMP_EQUAL:
    case WGSL_TOK_PIPE_EQUAL:
    case WGSL_TOK_CARET_EQUAL:
    case WGSL_TOK_LESS_LESS_EQUAL:
    case WGSL_TOK_GREATER_GREATER_EQUAL:  is_compound = 1; is_assign = 1; break;
    default: break;
    }

    if (is_assign) {
        const WGSLToken *op = peek(p);
        WGSLTokenKind op_kind = op->kind;
        WGSLSpan op_span = op->span;
        advance(p);
        WGSLNode *rhs = parse_expression(p);
        uint32_t fallback = rhs ? rhs->span_offset + rhs->span_length
                                : op_span.offset + op_span.length;
        uint32_t end = finish_stmt_or_inline(p, fallback, as_statement);
        WGSLSpan span = wgsl_span_make(lhs->span_offset, end - lhs->span_offset);
        WGSLNodeKind nk = is_compound
            ? WGSL_NODE_STMT_COMPOUND_ASSIGN : WGSL_NODE_STMT_ASSIGN;
        WGSLNode *node = make_node(p, nk, span);
        if (node) {
            node->payload[0] = (uint64_t)op_kind;
            node->payload[1] =
                ((uint64_t)op_span.offset) | ((uint64_t)op_span.length << 32);
            WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, 2 * sizeof(WGSLNode *));
            if (arr && rhs) {
                arr[0] = lhs; arr[1] = rhs;
                node->children = arr; node->child_count = 2;
            }
        }
        return node;
    }

    /* Plain expression statement (call, indexing, etc.). */
    uint32_t fallback = lhs->span_offset + lhs->span_length;
    uint32_t end = finish_stmt_or_inline(p, fallback, as_statement);
    WGSLSpan span = wgsl_span_make(lhs->span_offset, end - lhs->span_offset);
    WGSLNode *node = make_node(p, WGSL_NODE_STMT_FN_CALL, span);
    if (node) {
        WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, sizeof(WGSLNode *));
        if (arr) { arr[0] = lhs; node->children = arr; node->child_count = 1; }
    }
    return node;
}

/* — Control flow — */

static WGSLNode *parse_if_statement(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_IF, "'if'");
    WGSLNode *cond  = parse_expression(p);
    WGSLNode *then_ = parse_attributed_compound_statement(p);

    NL kids; nl_init(&kids);
    if (cond)  nl_push(&kids, cond);
    if (then_) nl_push(&kids, then_);

    while (at(p, WGSL_TOK_KW_ELSE)) {
        const WGSLToken *else_kw = advance(p);
        if (at(p, WGSL_TOK_KW_IF)) {
            advance(p);
            WGSLNode *ec = parse_expression(p);
            WGSLNode *eb = parse_attributed_compound_statement(p);
            uint32_t end = eb ? eb->span_offset + eb->span_length
                              : else_kw->span.offset + else_kw->span.length;
            WGSLSpan espan = wgsl_span_make(else_kw->span.offset, end - else_kw->span.offset);
            WGSLNode *enode = make_node(p, WGSL_NODE_STMT_ELSE_IF, espan);
            if (enode) {
                NL ek; nl_init(&ek);
                if (ec) nl_push(&ek, ec);
                if (eb) nl_push(&ek, eb);
                finalize_children(p, enode, &ek);
                nl_destroy(&ek);
                nl_push(&kids, enode);
            }
        } else {
            WGSLNode *eb = parse_attributed_compound_statement(p);
            uint32_t end = eb ? eb->span_offset + eb->span_length
                              : else_kw->span.offset + else_kw->span.length;
            WGSLSpan espan = wgsl_span_make(else_kw->span.offset, end - else_kw->span.offset);
            WGSLNode *enode = make_node(p, WGSL_NODE_STMT_ELSE, espan);
            if (enode) {
                NL ek; nl_init(&ek);
                if (eb) nl_push(&ek, eb);
                finalize_children(p, enode, &ek);
                nl_destroy(&ek);
                nl_push(&kids, enode);
            }
            break;
        }
    }

    uint32_t end_off = (kids.count > 0)
        ? kids.items[kids.count - 1]->span_offset
            + kids.items[kids.count - 1]->span_length
        : kw->span.offset + kw->span.length;
    WGSLSpan span = wgsl_span_make(kw->span.offset, end_off - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_STMT_IF, span);
    if (node) finalize_children(p, node, &kids);
    nl_destroy(&kids);
    return node;
}

static WGSLNode *parse_while_statement(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_WHILE, "'while'");
    WGSLNode *cond = parse_expression(p);
    WGSLNode *body = parse_attributed_compound_statement(p);

    uint32_t end = body ? body->span_offset + body->span_length
                        : kw->span.offset + kw->span.length;
    WGSLSpan span = wgsl_span_make(kw->span.offset, end - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_STMT_WHILE, span);
    if (node) {
        NL kids; nl_init(&kids);
        if (cond) nl_push(&kids, cond);
        if (body) nl_push(&kids, body);
        finalize_children(p, node, &kids);
        nl_destroy(&kids);
    }
    return node;
}

static WGSLNode *parse_statement(P *p);
static WGSLNode *parse_continuing_statement(P *p);

/* `for` `(` for_init? `;` expression? `;` for_update? `)` compound
 * payload[0] low byte = bitmap (bit0=init, bit1=cond, bit2=update).
 * Children layout: [init?, cond?, update?, body] in present order. */
static WGSLNode *parse_for_init_or_update(P *p) {
    WGSLTokenKind k = peek(p)->kind;
    if (k == WGSL_TOK_KW_LET)          return parse_let_decl(p, /*as_statement=*/0);
    if (k == WGSL_TOK_KW_VAR)          return parse_var_decl(p, /*as_statement=*/0,
                                                              /*allow_template=*/1,
                                                              /*attrs=*/NULL);
    if (k == WGSL_TOK_KW_CONST)        return parse_const_decl(p, /*as_statement=*/0);
    return parse_default_statement(p, /*as_statement=*/0);
}

static WGSLNode *parse_for_statement(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_FOR, "'for'");
    expect(p, WGSL_TOK_LPAREN, "'('");

    NL kids; nl_init(&kids);
    uint32_t flags = 0;

    if (!at(p, WGSL_TOK_SEMICOLON)) {
        WGSLNode *init = parse_for_init_or_update(p);
        if (init) { nl_push(&kids, init); flags |= 1u; }
    }
    expect(p, WGSL_TOK_SEMICOLON, "';' (for header)");

    if (!at(p, WGSL_TOK_SEMICOLON)) {
        WGSLNode *cond = parse_expression(p);
        if (cond) { nl_push(&kids, cond); flags |= 2u; }
    }
    expect(p, WGSL_TOK_SEMICOLON, "';' (for header)");

    if (!at(p, WGSL_TOK_RPAREN)) {
        WGSLNode *upd = parse_for_init_or_update(p);
        if (upd) { nl_push(&kids, upd); flags |= 4u; }
    }
    expect(p, WGSL_TOK_RPAREN, "')'");

    WGSLNode *body = parse_attributed_compound_statement(p);
    if (body) nl_push(&kids, body);

    uint32_t end = body ? body->span_offset + body->span_length
                        : kw->span.offset + kw->span.length;
    WGSLSpan span = wgsl_span_make(kw->span.offset, end - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_STMT_FOR, span);
    if (node) {
        node->payload[0] = (uint64_t)flags;
        finalize_children(p, node, &kids);
    }
    nl_destroy(&kids);
    return node;
}

static WGSLNode *parse_loop_statement(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_LOOP, "'loop'");
    NL attrs; nl_init(&attrs);
    expect_attributed_lbrace(p, "'{'", &attrs);

    NL kids; nl_init(&kids);
    while (!at(p, WGSL_TOK_RBRACE) && !at(p, WGSL_TOK_EOF)) {
        if (at(p, WGSL_TOK_KW_CONTINUING)) {
            WGSLNode *c = parse_continuing_statement(p);
            if (c) nl_push(&kids, c);
            break;     /* continuing must be the last statement */
        }
        WGSLNode *s = parse_statement(p);
        if (s && s->kind != WGSL_NODE_INVALID) nl_push(&kids, s);
        else if (s && s->span_offset == 0 && s->span_length == 0) {
            continue;
        } else {
            recover_to_sync(p);
        }
    }
    const WGSLToken *close = peek(p);
    expect(p, WGSL_TOK_RBRACE, "'}'");

    WGSLSpan span = wgsl_span_make(
        kw->span.offset, close->span.offset + close->span.length - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_STMT_LOOP, span);
    if (node) {
        finalize_children(p, node, &kids);
        attach_statement_attrs(p, node, &attrs);
    }
    nl_destroy(&attrs);
    nl_destroy(&kids);
    return node;
}

static WGSLNode *parse_continuing_statement(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_CONTINUING, "'continuing'");
    NL attrs; nl_init(&attrs);
    expect_attributed_lbrace(p, "'{'", &attrs);

    NL kids; nl_init(&kids);
    while (!at(p, WGSL_TOK_RBRACE) && !at(p, WGSL_TOK_EOF)) {
        /* `break if expr ;` is permitted — and only — at the tail. */
        if (at(p, WGSL_TOK_KW_BREAK) && peek_n(p, 1)->kind == WGSL_TOK_KW_IF) {
            const WGSLToken *bk = advance(p);
            advance(p);   /* if */
            WGSLNode *cond = parse_expression(p);
            const WGSLToken *semi = peek(p);
            expect(p, WGSL_TOK_SEMICOLON, "';'");
            WGSLSpan bspan = wgsl_span_make(
                bk->span.offset, semi->span.offset + semi->span.length - bk->span.offset);
            WGSLNode *bi = make_node(p, WGSL_NODE_STMT_BREAK_IF, bspan);
            if (bi && cond) {
                WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, sizeof(WGSLNode *));
                if (arr) { arr[0] = cond; bi->children = arr; bi->child_count = 1; }
            }
            if (bi) nl_push(&kids, bi);
            break;
        }
        WGSLNode *s = parse_statement(p);
        if (s && s->kind != WGSL_NODE_INVALID) nl_push(&kids, s);
        else if (s && s->span_offset == 0 && s->span_length == 0) {
            continue;
        } else {
            recover_to_sync(p);
        }
    }
    const WGSLToken *close = peek(p);
    expect(p, WGSL_TOK_RBRACE, "'}'");

    WGSLSpan span = wgsl_span_make(
        kw->span.offset, close->span.offset + close->span.length - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_STMT_CONTINUING, span);
    if (node) {
        finalize_children(p, node, &kids);
        attach_statement_attrs(p, node, &attrs);
    }
    nl_destroy(&attrs);
    nl_destroy(&kids);
    return node;
}

static WGSLNode *parse_break_statement(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_BREAK, "'break'");
    /* `break if expr ;` is parsed at the tail of `continuing { … }` and
     * not here.  A bare `break ;` always becomes WGSL_NODE_STMT_BREAK. */
    const WGSLToken *semi = peek(p);
    expect(p, WGSL_TOK_SEMICOLON, "';'");
    WGSLSpan span = wgsl_span_make(
        kw->span.offset, semi->span.offset + semi->span.length - kw->span.offset);
    return make_node(p, WGSL_NODE_STMT_BREAK, span);
}

static WGSLNode *parse_continue_statement(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_CONTINUE, "'continue'");
    const WGSLToken *semi = peek(p);
    expect(p, WGSL_TOK_SEMICOLON, "';'");
    WGSLSpan span = wgsl_span_make(
        kw->span.offset, semi->span.offset + semi->span.length - kw->span.offset);
    return make_node(p, WGSL_NODE_STMT_CONTINUE, span);
}

static WGSLNode *parse_discard_statement(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_DISCARD, "'discard'");
    const WGSLToken *semi = peek(p);
    expect(p, WGSL_TOK_SEMICOLON, "';'");
    WGSLSpan span = wgsl_span_make(
        kw->span.offset, semi->span.offset + semi->span.length - kw->span.offset);
    return make_node(p, WGSL_NODE_STMT_DISCARD, span);
}

/* `case` selectors `:`? compound  |  `default` `:`? compound
 * payload[0] low 32 = selector_count
 * payload[0] high 32 = is_default_alone (0/1)
 * children: [selector_exprs..., body] for case;
 *           [body] for default-alone. */
static WGSLNode *parse_case_clause(P *p) {
    WGSLTokenKind k = peek(p)->kind;

    if (k == WGSL_TOK_KW_DEFAULT) {
        const WGSLToken *kw = advance(p);
        match(p, WGSL_TOK_COLON);
        WGSLNode *body = parse_attributed_compound_statement(p);
        uint32_t end = body ? body->span_offset + body->span_length
                            : kw->span.offset + kw->span.length;
        WGSLSpan span = wgsl_span_make(kw->span.offset, end - kw->span.offset);
        WGSLNode *node = make_node(p, WGSL_NODE_STMT_CASE_CLAUSE, span);
        if (node) {
            node->payload[0] = ((uint64_t)0) | ((uint64_t)1u << 32);
            if (body) {
                WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, sizeof(WGSLNode *));
                if (arr) { arr[0] = body; node->children = arr; node->child_count = 1; }
            }
        }
        return node;
    }

    /* `case` selectors — per spec §18 / §9.4.2, the list may include
     * the `default` keyword as one of its selectors (e.g.
     * `case 0, default, 2:`).  Accept that shape and record it via
     * a separate bit so the validator can count default occurrences. */
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_CASE, "'case' or 'default'");
    NL sels; nl_init(&sels);
    int has_default_in_list = 0;
    int default_count_in_list = 0;
    const WGSLToken *first_extra_default = NULL;
    do {
        if (at(p, WGSL_TOK_COLON) || at(p, WGSL_TOK_LBRACE)) break;
        if (at(p, WGSL_TOK_KW_DEFAULT)) {
            const WGSLToken *dtok = advance(p);
            default_count_in_list += 1;
            if (default_count_in_list == 2) first_extra_default = dtok;
            has_default_in_list = 1;
        } else {
            WGSLNode *e = parse_expression(p);
            if (e) nl_push(&sels, e);
        }
    } while (match(p, WGSL_TOK_COMMA));
    /* §9.4.2 — `default` must not appear more than once in a single
     * case selector list (`case 0, default, default, 2:` is invalid).
     * The "more than one default clause overall" check happens later
     * in the typechecker; this gate is the per-list version. */
    if (default_count_in_list > 1 && first_extra_default) {
        parse_error(p,
            first_extra_default->span.offset,
            first_extra_default->span.length,
            "'default' must not appear more than once in the same case "
            "selector list");
    }

    match(p, WGSL_TOK_COLON);
    WGSLNode *body = parse_attributed_compound_statement(p);

    uint32_t end = body ? body->span_offset + body->span_length
                        : kw->span.offset + kw->span.length;
    WGSLSpan span = wgsl_span_make(kw->span.offset, end - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_STMT_CASE_CLAUSE, span);
    if (node) {
        /* payload[0]: low32 = selector_count;
         *             bit 32 = is_default_alone (standalone `default:`);
         *             bit 33 = has_default_in_list (`case X, default, …`). */
        node->payload[0] = (uint64_t)sels.count |
                           ((uint64_t)has_default_in_list << 33);
        NL kids; nl_init(&kids);
        for (size_t i = 0; i < sels.count; i++) nl_push(&kids, sels.items[i]);
        if (body) nl_push(&kids, body);
        finalize_children(p, node, &kids);
        nl_destroy(&kids);
    }
    nl_destroy(&sels);
    return node;
}

static WGSLNode *parse_switch_statement(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_SWITCH, "'switch'");
    WGSLNode *selector = parse_expression(p);
    NL attrs; nl_init(&attrs);
    expect_attributed_lbrace(p, "'{' (switch body)", &attrs);

    NL kids; nl_init(&kids);
    if (selector) nl_push(&kids, selector);

    while (!at(p, WGSL_TOK_RBRACE) && !at(p, WGSL_TOK_EOF)) {
        WGSLNode *clause = parse_case_clause(p);
        if (clause) nl_push(&kids, clause);
        else        recover_to_sync(p);
    }
    const WGSLToken *close = peek(p);
    expect(p, WGSL_TOK_RBRACE, "'}'");

    WGSLSpan span = wgsl_span_make(
        kw->span.offset, close->span.offset + close->span.length - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_STMT_SWITCH, span);
    if (node) {
        finalize_children(p, node, &kids);
        attach_statement_attrs(p, node, &attrs);
    }
    nl_destroy(&attrs);
    nl_destroy(&kids);
    return node;
}

static WGSLNode *parse_statement(P *p) {
    /* §18 grammar permits a leading attribute list on:
     *   - compound_statement      (`attribute* '{' …`)
     *   - if / loop / for / while (`attribute* 'if' …` etc.)
     *   - switch                  (`attribute* 'switch' …`)
     *
     * The semantically meaningful one in this set is `@diagnostic(...)`
     * — which adjusts the diag filter for the block.  The validator's
     * scope-push already handles fn-level @diagnostic; stmt-level
     * scope semantics are handled by the uniformity validator using
     * the statement attribute payload attached below. */
    NL attrs; nl_init(&attrs);
    parse_attributes(p, &attrs);
    WGSLTokenKind k = peek(p)->kind;
    WGSLNode *node = NULL;
    switch (k) {
    case WGSL_TOK_LBRACE:           node = parse_compound_statement(p); break;
    case WGSL_TOK_KW_LET:           node = parse_let_decl(p, 1); break;
    /* §7.4 grammar: `variable_decl` accepts the template list at any
     * scope.  Statement-level var still cannot carry attributes
     * (`@group`/`@binding` are module-only), so `attrs_or_null=NULL`. */
    case WGSL_TOK_KW_VAR:           node = parse_var_decl(p, 1, 1, NULL); break;
    case WGSL_TOK_KW_CONST:         node = parse_const_decl(p, 1); break;
    case WGSL_TOK_KW_CONST_ASSERT:  node = parse_const_assert_statement(p); break;
    case WGSL_TOK_KW_RETURN:        node = parse_return_statement(p); break;
    case WGSL_TOK_KW_IF:            node = parse_if_statement(p); break;
    case WGSL_TOK_KW_WHILE:         node = parse_while_statement(p); break;
    case WGSL_TOK_KW_FOR:           node = parse_for_statement(p); break;
    case WGSL_TOK_KW_LOOP:          node = parse_loop_statement(p); break;
    case WGSL_TOK_KW_SWITCH:        node = parse_switch_statement(p); break;
    case WGSL_TOK_KW_BREAK:         node = parse_break_statement(p); break;
    case WGSL_TOK_KW_CONTINUE:      node = parse_continue_statement(p); break;
    case WGSL_TOK_KW_DISCARD:       node = parse_discard_statement(p); break;
    case WGSL_TOK_SEMICOLON:
        advance(p);
        node = make_node(p, WGSL_NODE_INVALID, wgsl_span_make(0, 0));
        break;
    default:
        node = parse_default_statement(p, /*as_statement=*/1);
        break;
    }
    if (node && node->kind != WGSL_NODE_INVALID &&
        stmt_kind_accepts_attrs((WGSLNodeKind)node->kind))
    {
        attach_statement_attrs(p, node, &attrs);
    } else if (attrs.count > 0) {
        WGSLNode *first = attrs.items[0];
        parse_error(p, first->span_offset, first->span_length,
                    "attributes are not permitted on this statement");
    }
    nl_destroy(&attrs);
    return node;
}

