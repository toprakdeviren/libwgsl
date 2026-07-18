/**
 * @file expressions_attrs.c — Pratt expressions + attributes
 */
#include "internal/parser_priv.h"

/* BinaryOpGroup: parser_priv.h */
BinaryOpGroup binary_group(WGSLTokenKind k) {
    switch (k) {
    case WGSL_TOK_STAR:
    case WGSL_TOK_SLASH:
    case WGSL_TOK_PERCENT:
        return BIN_GROUP_MUL;
    case WGSL_TOK_PLUS:
    case WGSL_TOK_MINUS:
        return BIN_GROUP_ADD;
    case WGSL_TOK_LESS_LESS:
    case WGSL_TOK_GREATER_GREATER:
        return BIN_GROUP_SHIFT;
    case WGSL_TOK_LESS:
    case WGSL_TOK_LESS_EQUAL:
    case WGSL_TOK_GREATER:
    case WGSL_TOK_GREATER_EQUAL:
    case WGSL_TOK_EQUAL_EQUAL:
    case WGSL_TOK_BANG_EQUAL:
        return BIN_GROUP_REL;
    case WGSL_TOK_AMP:
        return BIN_GROUP_BIT_AND;
    case WGSL_TOK_CARET:
        return BIN_GROUP_BIT_XOR;
    case WGSL_TOK_PIPE:
        return BIN_GROUP_BIT_OR;
    case WGSL_TOK_AMP_AMP:
    case WGSL_TOK_PIPE_PIPE:
        return BIN_GROUP_LOGICAL;
    default:
        return BIN_GROUP_NONE;
    }
}

uint32_t can_precede_groups(BinaryOpGroup g) {
    switch (g) {
    case BIN_GROUP_MUL:
        return BIN_GROUP_MUL | BIN_GROUP_ADD | BIN_GROUP_REL;
    case BIN_GROUP_ADD:
        return BIN_GROUP_MUL | BIN_GROUP_ADD | BIN_GROUP_REL;
    case BIN_GROUP_SHIFT:
        return BIN_GROUP_REL | BIN_GROUP_LOGICAL;
    case BIN_GROUP_REL:
        return BIN_GROUP_MUL | BIN_GROUP_ADD | BIN_GROUP_SHIFT |
               BIN_GROUP_LOGICAL;
    case BIN_GROUP_BIT_AND:
        return BIN_GROUP_BIT_AND;
    case BIN_GROUP_BIT_XOR:
        return BIN_GROUP_BIT_XOR;
    case BIN_GROUP_BIT_OR:
        return BIN_GROUP_BIT_OR;
    case BIN_GROUP_LOGICAL:
        return BIN_GROUP_REL;
    default:
        return 0;
    }
}

/* §8.18 — ordered binary operator pairs that require parentheses when
 * written as `a prev b next c`. */
int requires_parens_ordered(WGSLTokenKind prev, WGSLTokenKind next) {
    if ((prev == WGSL_TOK_AMP_AMP || prev == WGSL_TOK_PIPE_PIPE) &&
        (next == WGSL_TOK_AMP_AMP || next == WGSL_TOK_PIPE_PIPE))
    {
        return prev != next;
    }

    BinaryOpGroup pg = binary_group(prev);
    BinaryOpGroup ng = binary_group(next);
    if (pg == BIN_GROUP_NONE || ng == BIN_GROUP_NONE) return 0;
    return (can_precede_groups(pg) & (uint32_t)ng) == 0;
}

/* Binary-expression precedence table.  Higher = tighter.
 * Returns 0 for non-binary tokens. */
int prec_of(WGSLTokenKind k) {
    switch (k) {
    case WGSL_TOK_PIPE_PIPE:        return 1;
    case WGSL_TOK_AMP_AMP:          return 2;
    case WGSL_TOK_PIPE:             return 3;
    case WGSL_TOK_CARET:            return 4;
    case WGSL_TOK_AMP:              return 5;
    case WGSL_TOK_EQUAL_EQUAL:
    case WGSL_TOK_BANG_EQUAL:       return 6;
    case WGSL_TOK_LESS:
    case WGSL_TOK_LESS_EQUAL:
    case WGSL_TOK_GREATER:
    case WGSL_TOK_GREATER_EQUAL:    return 7;
    case WGSL_TOK_LESS_LESS:
    case WGSL_TOK_GREATER_GREATER:  return 8;
    case WGSL_TOK_PLUS:
    case WGSL_TOK_MINUS:            return 9;
    case WGSL_TOK_STAR:
    case WGSL_TOK_SLASH:
    case WGSL_TOK_PERCENT:          return 10;
    default:                        return 0;
    }
}

WGSLNode *parse_binary_rhs(P *p, WGSLNode *lhs, int min_prec) {
    while (1) {
        const WGSLToken *op = peek(p);
        int prec = prec_of(op->kind);
        if (prec == 0 || prec < min_prec) return lhs;

        WGSLTokenKind op_kind = op->kind;
        WGSLSpan op_span = op->span;
        advance(p);

        WGSLNode *rhs = parse_unary(p);
        if (!rhs) return lhs;

        /* Left-assoc: parse right side with higher min_prec. */
        rhs = parse_binary_rhs(p, rhs, prec + 1);

        WGSLSpan span = wgsl_span_make(
            lhs->span_offset,
            rhs->span_offset + rhs->span_length - lhs->span_offset);

        WGSLNode *bin = make_node(p, WGSL_NODE_EXPR_BINARY, span);
        if (!bin) return lhs;

        bin->payload[0] = (uint64_t)op_kind;
        bin->payload[1] = ((uint64_t)op_span.offset) | ((uint64_t)op_span.length << 32);

        WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, 2 * sizeof(WGSLNode *));
        if (!arr) { p->had_error = 1; return lhs; }
        arr[0] = lhs; arr[1] = rhs;
        bin->children    = arr;
        bin->child_count = 2;

        /* §8.18 — flag operator pairs that are mixed without parens.
         * EXPR_PAREN around either operand suppresses the diagnostic
         * (the user disambiguated). */
        if (lhs->kind == WGSL_NODE_EXPR_BINARY) {
            WGSLTokenKind inner = (WGSLTokenKind)wgsl_node_op_kind_u32(lhs);
            if (requires_parens_ordered(inner, op_kind)) {
                parse_error(p, op_span.offset, op_span.length,
                    "operator '%s' requires parentheses when mixed with '%s' "
                    "(spec §8.18: no relative precedence)",
                    wgsl_token_kind_name(op_kind),
                    wgsl_token_kind_name(inner));
            }
        }
        if (rhs->kind == WGSL_NODE_EXPR_BINARY) {
            WGSLTokenKind inner = (WGSLTokenKind)wgsl_node_op_kind_u32(rhs);
            if (requires_parens_ordered(op_kind, inner)) {
                parse_error(p, op_span.offset, op_span.length,
                    "operator '%s' requires parentheses when mixed with '%s' "
                    "(spec §8.18: no relative precedence)",
                    wgsl_token_kind_name(op_kind),
                    wgsl_token_kind_name(inner));
            }
        }

        lhs = bin;
    }
}

WGSLNode *parse_expression(P *p) {
    WGSLNode *lhs = parse_unary(p);
    if (!lhs) return NULL;
    return parse_binary_rhs(p, lhs, 1);
}

/* Depth-guarded entry point.  All expression recursion funnels through
 * here (parse_expression, parse_binary_rhs, prefix chains, and — via
 * parse_primary/parse_postfix — parenthesised, call-argument, index, and
 * template-argument sub-expressions), so a single counter bounds every
 * expression-nesting vector.  See WGSL_PARSE_MAX_DEPTH in parser.c. */
WGSLNode *parse_unary_impl(P *p);

WGSLNode *parse_unary(P *p) {
    if (p->depth >= WGSL_PARSE_MAX_DEPTH) {
        const WGSLToken *t = peek(p);
        parse_error(p, t->span.offset, t->span.length,
                    "expression nested too deeply (max %d) — input rejected "
                    "to avoid stack overflow", WGSL_PARSE_MAX_DEPTH);
        return NULL;
    }
    p->depth++;
    WGSLNode *n = parse_unary_impl(p);
    p->depth--;
    return n;
}

WGSLNode *parse_unary_impl(P *p) {
    const WGSLToken *t = peek(p);
    WGSLTokenKind k = t->kind;
    if (k == WGSL_TOK_MINUS || k == WGSL_TOK_BANG ||
        k == WGSL_TOK_TILDE || k == WGSL_TOK_STAR  ||
        k == WGSL_TOK_AMP)
    {
        WGSLSpan op_span = t->span;
        WGSLTokenKind op_kind = k;
        advance(p);
        WGSLNode *operand = parse_unary(p);
        if (!operand) return NULL;
        WGSLSpan span = wgsl_span_make(op_span.offset,
            operand->span_offset + operand->span_length - op_span.offset);
        WGSLNodeKind nk;
        switch (op_kind) {
        case WGSL_TOK_STAR: nk = WGSL_NODE_EXPR_INDIRECTION; break;
        case WGSL_TOK_AMP:  nk = WGSL_NODE_EXPR_ADDR_OF;     break;
        default:            nk = WGSL_NODE_EXPR_UNARY;       break;
        }
        WGSLNode *u = make_node(p, nk, span);
        if (!u) return NULL;
        u->payload[0] = (uint64_t)op_kind;
        u->payload[1] = ((uint64_t)op_span.offset) | ((uint64_t)op_span.length << 32);
        WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, sizeof(WGSLNode *));
        if (!arr) { p->had_error = 1; return NULL; }
        arr[0] = operand;
        u->children    = arr;
        u->child_count = 1;
        return u;
    }
    return parse_postfix(p, parse_primary(p));
}

/* `vec3<f32>` → emit a TEMPLATED_IDENT with children: ident node + arg expressions.
 * Caller has already consumed the IDENT.  This consumes from TEMPLATE_START
 * to TEMPLATE_END.  Returns a node spanning ident-through-TEMPLATE_END. */
WGSLNode *parse_template_arg_list(P *p, WGSLNode *ident) {
    /* current token is TEMPLATE_START */
    expect(p, WGSL_TOK_TEMPLATE_START, "template arg list opener");

    NL kids; nl_init(&kids);
    nl_push(&kids, ident);

    if (!at(p, WGSL_TOK_TEMPLATE_END)) {
        do {
            if (at(p, WGSL_TOK_TEMPLATE_END)) break;   /* trailing comma */
            WGSLNode *arg = parse_expression(p);
            if (arg) nl_push(&kids, arg);
        } while (match(p, WGSL_TOK_COMMA));
    }
    const WGSLToken *end = peek(p);
    expect(p, WGSL_TOK_TEMPLATE_END, "'>' (template list end)");

    WGSLSpan span = wgsl_span_make(
        ident->span_offset,
        end->span.offset + end->span.length - ident->span_offset);

    WGSLNode *node = make_node(p, WGSL_NODE_EXPR_TEMPLATED_IDENT, span);
    if (!node) { nl_destroy(&kids); return ident; }
    finalize_children(p, node, &kids);
    nl_destroy(&kids);
    return node;
}

WGSLNode *parse_postfix(P *p, WGSLNode *base) {
    if (!base) return NULL;
    while (1) {
        WGSLTokenKind k = peek(p)->kind;
        if (k == WGSL_TOK_LPAREN) {
            advance(p);
            NL kids; nl_init(&kids);
            nl_push(&kids, base);
            if (!at(p, WGSL_TOK_RPAREN)) {
                do {
                    if (at(p, WGSL_TOK_RPAREN)) break;
                    WGSLNode *arg = parse_expression(p);
                    if (arg) nl_push(&kids, arg);
                } while (match(p, WGSL_TOK_COMMA));
            }
            const WGSLToken *rp = peek(p);
            expect(p, WGSL_TOK_RPAREN, "')'");
            WGSLSpan span = wgsl_span_make(
                base->span_offset,
                rp->span.offset + rp->span.length - base->span_offset);
            WGSLNode *call = make_node(p, WGSL_NODE_EXPR_CALL, span);
            if (call) {
                finalize_children(p, call, &kids);
                base = call;
            }
            nl_destroy(&kids);
            continue;
        }
        if (k == WGSL_TOK_LBRACKET) {
            advance(p);
            WGSLNode *idx = parse_expression(p);
            const WGSLToken *rb = peek(p);
            expect(p, WGSL_TOK_RBRACKET, "']'");
            WGSLSpan span = wgsl_span_make(
                base->span_offset,
                rb->span.offset + rb->span.length - base->span_offset);
            WGSLNode *node = make_node(p, WGSL_NODE_EXPR_INDEX, span);
            if (node && idx) {
                WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, 2 * sizeof(WGSLNode *));
                if (arr) {
                    arr[0] = base; arr[1] = idx;
                    node->children    = arr;
                    node->child_count = 2;
                }
                base = node;
            }
            continue;
        }
        if (k == WGSL_TOK_DOT) {
            advance(p);
            const WGSLToken *member = expect(p, WGSL_TOK_IDENT, "member name");
            WGSLSpan member_span = member->span;
            WGSLSpan span = wgsl_span_make(
                base->span_offset,
                member_span.offset + member_span.length - base->span_offset);
            WGSLNode *node = make_node(p, WGSL_NODE_EXPR_MEMBER, span);
            if (node) {
                WGSLNode *member_node = make_node(p, WGSL_NODE_EXPR_IDENT, member_span);
                if (member_node) {
                    member_node->payload[0] =
                        ((uint64_t)member_span.offset) | ((uint64_t)member_span.length << 32);
                    WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, 2 * sizeof(WGSLNode *));
                    if (arr) {
                        arr[0] = base; arr[1] = member_node;
                        node->children    = arr;
                        node->child_count = 2;
                    }
                }
                base = node;
            }
            continue;
        }
        return base;
    }
}

WGSLNode *parse_primary(P *p) {
    const WGSLToken *t = peek(p);
    WGSLSpan span = t->span;

    switch (t->kind) {
    case WGSL_TOK_INT_LIT: {
        advance(p);
        WGSLNode *n = make_node(p, WGSL_NODE_EXPR_LITERAL_INT, span);
        if (n) n->payload[0] = (uint64_t)t->payload;
        return n;
    }
    case WGSL_TOK_FLOAT_LIT: {
        advance(p);
        WGSLNode *n = make_node(p, WGSL_NODE_EXPR_LITERAL_FLOAT, span);
        if (n) n->payload[0] = (uint64_t)t->payload;
        return n;
    }
    case WGSL_TOK_KW_TRUE:
    case WGSL_TOK_KW_FALSE: {
        advance(p);
        WGSLNode *n = make_node(p, WGSL_NODE_EXPR_LITERAL_BOOL, span);
        if (n) n->payload[0] = (t->kind == WGSL_TOK_KW_TRUE) ? 1u : 0u;
        return n;
    }
    case WGSL_TOK_LPAREN: {
        advance(p);
        WGSLNode *inner = parse_expression(p);
        const WGSLToken *rp = peek(p);
        expect(p, WGSL_TOK_RPAREN, "')'");
        WGSLSpan pspan = wgsl_span_make(
            span.offset, rp->span.offset + rp->span.length - span.offset);
        WGSLNode *n = make_node(p, WGSL_NODE_EXPR_PAREN, pspan);
        if (n && inner) {
            WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, sizeof(WGSLNode *));
            if (arr) { arr[0] = inner; n->children = arr; n->child_count = 1; }
        }
        return n;
    }
    case WGSL_TOK_IDENT: {
        advance(p);
        WGSLNode *id = make_node(p, WGSL_NODE_EXPR_IDENT, span);
        if (id) id->payload[0] = ((uint64_t)span.offset) | ((uint64_t)span.length << 32);
        if (at(p, WGSL_TOK_TEMPLATE_START)) {
            return parse_template_arg_list(p, id);
        }
        return id;
    }
    default:
        parse_error(p, t->span.offset, t->span.length,
                    "expected expression, got %s",
                    wgsl_token_kind_name(t->kind));
        advance(p);
        return make_node(p, WGSL_NODE_INVALID, t->span);
    }
}

/* type_specifier := template_elaborated_ident.  An IDENT with optional
 * `<...>`.  We synthesize the same shape as parse_primary's IDENT path. */
WGSLNode *parse_type_specifier(P *p) {
    if (!at(p, WGSL_TOK_IDENT) && !at(p, WGSL_TOK_KW_VAR)) {
        const WGSLToken *t = peek(p);
        parse_error(p, t->span.offset, t->span.length,
                    "expected type specifier, got %s",
                    wgsl_token_kind_name(t->kind));
        advance(p);
        return make_node(p, WGSL_NODE_INVALID, t->span);
    }
    const WGSLToken *t = advance(p);
    WGSLNode *id = make_node(p, WGSL_NODE_EXPR_IDENT, t->span);
    if (id) id->payload[0] = ((uint64_t)t->span.offset) | ((uint64_t)t->span.length << 32);
    if (at(p, WGSL_TOK_TEMPLATE_START)) {
        return parse_template_arg_list(p, id);
    }
    return id;
}

/* Attributes. */

WGSLNode *parse_attribute(P *p) {
    const WGSLToken *at_tok = expect(p, WGSL_TOK_AT, "'@'");
    const WGSLToken *t = peek(p);
    int is_ident_like = (t->kind == WGSL_TOK_IDENT) ||
                        (t->kind >= WGSL_TOK_KW_ALIAS && t->kind <= WGSL_TOK_KW_WHILE);
    if (!is_ident_like) {
        parse_error(p, t->span.offset, t->span.length,
                    "expected attribute name (identifier or keyword), got %s",
                    wgsl_token_kind_name(t->kind));
        return make_node(p, WGSL_NODE_INVALID, at_tok->span);
    }
    const WGSLToken *name = advance(p);

    NL kids; nl_init(&kids);
    WGSLNode *name_node = make_node(p, WGSL_NODE_EXPR_IDENT, name->span);
    if (name_node) {
        name_node->payload[0] =
            ((uint64_t)name->span.offset) | ((uint64_t)name->span.length << 32);
        nl_push(&kids, name_node);
    }

    uint32_t end = name->span.offset + name->span.length;
    int has_parens = 0;
    if (match(p, WGSL_TOK_LPAREN)) {
        has_parens = 1;
        if (!at(p, WGSL_TOK_RPAREN)) {
            do {
                if (at(p, WGSL_TOK_RPAREN)) break;
                WGSLNode *arg = parse_expression(p);
                if (arg) nl_push(&kids, arg);
            } while (match(p, WGSL_TOK_COMMA));
        }
        const WGSLToken *rp = peek(p);
        if (rp->kind == WGSL_TOK_RPAREN) {
            end = rp->span.offset + rp->span.length;
            advance(p);
        } else {
            expect(p, WGSL_TOK_RPAREN, "')'");
        }
    }

    WGSLSpan span = wgsl_span_make(at_tok->span.offset, end - at_tok->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_ATTRIBUTE, span);
    if (!node) { nl_destroy(&kids); return NULL; }
    if (has_parens) node->flags |= WGSL_FLAG_ATTR_PARENS;
    finalize_children(p, node, &kids);
    nl_destroy(&kids);
    return node;
}

int parse_attributes(P *p, NL *out) {
    while (at(p, WGSL_TOK_AT)) {
        WGSLNode *a = parse_attribute(p);
        if (a) nl_push(out, a);
    }
    return 1;
}

int attach_statement_attrs(P *p, WGSLNode *node, const NL *attrs) {
    if (!p || !node || !attrs || attrs->count == 0) return 1;
    uint32_t existing_count = wgsl_stmt_attr_count(node);
    WGSLNode **existing = wgsl_stmt_attrs(node);
    if (attrs->count > UINT32_MAX - existing_count) return 0;
    uint32_t total = existing_count + (uint32_t)attrs->count;
    WGSLNode **items = (WGSLNode **)wgsl_arena_alloc(
        p->arena, (size_t)total * sizeof *items);
    if (!items) return 0;
    for (uint32_t i = 0; i < existing_count; i++) {
        items[i] = existing ? existing[i] : NULL;
    }
    for (size_t i = 0; i < attrs->count; i++) {
        items[existing_count + (uint32_t)i] = attrs->items[i];
    }
    node->payload[1] = (node->payload[1] & 0xFFFFFFFF00000000ull) | (uint64_t)total;
    node->payload[2] = (uint64_t)(uintptr_t)items;
    node->flags |= WGSL_FLAG_HAS_ATTRS;

    uint32_t start = node->span_offset;
    for (size_t i = 0; i < attrs->count; i++) {
        WGSLNode *a = attrs->items[i];
        if (a && a->span_offset < start) start = a->span_offset;
    }
    if (start < node->span_offset) {
        node->span_length += node->span_offset - start;
        node->span_offset = start;
    }
    return 1;
}

int stmt_kind_accepts_attrs(WGSLNodeKind k) {
    return k == WGSL_NODE_STMT_COMPOUND ||
           k == WGSL_NODE_STMT_IF ||
           k == WGSL_NODE_STMT_WHILE ||
           k == WGSL_NODE_STMT_FOR ||
           k == WGSL_NODE_STMT_LOOP ||
           k == WGSL_NODE_STMT_CONTINUING ||
           k == WGSL_NODE_STMT_SWITCH;
}
