/**
 * @file decls_entry.c — global decls + wgsl_parse entry
 */
#include "internal/parser_priv.h"

/* Decls. */

WGSLNode *parse_param(P *p) {
    NL attrs; nl_init(&attrs);
    parse_attributes(p, &attrs);

    const WGSLToken *name = expect(p, WGSL_TOK_IDENT, "parameter name");
    expect(p, WGSL_TOK_COLON, "':'");
    WGSLNode *type = parse_type_specifier(p);

    uint32_t attr_count = (uint32_t)attrs.count;

    NL kids; nl_init(&kids);
    for (size_t i = 0; i < attrs.count; i++) nl_push(&kids, attrs.items[i]);
    if (type) nl_push(&kids, type);
    nl_destroy(&attrs);

    uint32_t end = type ? type->span_offset + type->span_length
                        : name->span.offset + name->span.length;
    uint32_t start = (kids.count > 0 && kids.items[0]->kind == WGSL_NODE_ATTRIBUTE)
                       ? kids.items[0]->span_offset
                       : name->span.offset;

    WGSLSpan span = wgsl_span_make(start, end - start);
    WGSLNode *node = make_node(p, WGSL_NODE_DECL_PARAM, span);
    if (node) {
        node->payload[0] = ((uint64_t)name->span.offset) | ((uint64_t)name->span.length << 32);
        node->payload[1] = (uint64_t)attr_count;
        finalize_children(p, node, &kids);
    }
    nl_destroy(&kids);
    return node;
}

WGSLNode *parse_function_decl(P *p, NL *fn_attrs) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_FN, "'fn'");
    const WGSLToken *name = expect(p, WGSL_TOK_IDENT, "function name");
    expect(p, WGSL_TOK_LPAREN, "'('");

    NL kids; nl_init(&kids);

    /* fn attrs first, source order */
    for (size_t i = 0; i < fn_attrs->count; i++) nl_push(&kids, fn_attrs->items[i]);
    uint32_t fn_attr_count = (uint32_t)fn_attrs->count;

    /* params */
    uint32_t param_count = 0;
    if (!at(p, WGSL_TOK_RPAREN)) {
        do {
            if (at(p, WGSL_TOK_RPAREN)) break;
            WGSLNode *param = parse_param(p);
            if (param) { nl_push(&kids, param); param_count += 1; }
        } while (match(p, WGSL_TOK_COMMA));
    }
    expect(p, WGSL_TOK_RPAREN, "')'");

    /* return attrs + return type */
    uint32_t ret_attr_count = 0;
    uint32_t has_return_type = 0;
    if (match(p, WGSL_TOK_ARROW)) {
        NL ret_attrs; nl_init(&ret_attrs);
        parse_attributes(p, &ret_attrs);
        for (size_t i = 0; i < ret_attrs.count; i++) {
            nl_push(&kids, ret_attrs.items[i]);
            ret_attr_count += 1;
        }
        nl_destroy(&ret_attrs);

        WGSLNode *ret_type = parse_type_specifier(p);
        if (ret_type) { nl_push(&kids, ret_type); has_return_type = 1; }
    }

    /* body */
    WGSLNode *body = parse_attributed_compound_statement(p);
    if (body) nl_push(&kids, body);

    uint32_t start = fn_attr_count > 0 ? fn_attrs->items[0]->span_offset : kw->span.offset;
    uint32_t end   = body ? body->span_offset + body->span_length
                          : kw->span.offset + kw->span.length;

    WGSLSpan span = wgsl_span_make(start, end - start);
    WGSLNode *node = make_node(p, WGSL_NODE_DECL_FUNCTION, span);
    if (node) {
        node->payload[0] = ((uint64_t)name->span.offset) | ((uint64_t)name->span.length << 32);
        node->payload[1] = ((uint64_t)fn_attr_count) | ((uint64_t)param_count << 32);
        node->payload[2] = ((uint64_t)ret_attr_count) | ((uint64_t)has_return_type << 32);
        finalize_children(p, node, &kids);
    }
    nl_destroy(&kids);
    return node;
}

/* Module-scope decls. */

/* Helper: build a child IDENT node for a token's span. */
WGSLNode *make_ident_from(P *p, const WGSLToken *t) {
    if (!t) return NULL;
    WGSLNode *n = make_node(p, WGSL_NODE_EXPR_IDENT, t->span);
    if (n) n->payload[0] =
        ((uint64_t)t->span.offset) | ((uint64_t)t->span.length << 32);
    return n;
}

/* `attribute* IDENT ':' type_specifier`. */
WGSLNode *parse_struct_member(P *p) {
    NL attrs; nl_init(&attrs);
    parse_attributes(p, &attrs);
    uint32_t attr_count = (uint32_t)attrs.count;

    const WGSLToken *name = expect(p, WGSL_TOK_IDENT, "member name");
    expect(p, WGSL_TOK_COLON, "':'");
    WGSLNode *type = parse_type_specifier(p);

    NL kids; nl_init(&kids);
    for (size_t i = 0; i < attrs.count; i++) nl_push(&kids, attrs.items[i]);
    if (type) nl_push(&kids, type);
    nl_destroy(&attrs);

    uint32_t start = (attr_count > 0) ? kids.items[0]->span_offset
                                       : name->span.offset;
    uint32_t end = type ? type->span_offset + type->span_length
                        : name->span.offset + name->span.length;
    WGSLSpan span = wgsl_span_make(start, end - start);
    WGSLNode *node = make_node(p, WGSL_NODE_DECL_STRUCT_MEMBER, span);
    if (node) {
        node->payload[0] = ((uint64_t)name->span.offset) | ((uint64_t)name->span.length << 32);
        node->payload[1] = (uint64_t)attr_count;
        finalize_children(p, node, &kids);
    }
    nl_destroy(&kids);
    return node;
}

/* `'struct' IDENT '{' member (',' member)* ','? '}'`. */
WGSLNode *parse_struct_decl(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_STRUCT, "'struct'");
    const WGSLToken *name = expect(p, WGSL_TOK_IDENT, "struct name");
    expect(p, WGSL_TOK_LBRACE, "'{'");

    NL kids; nl_init(&kids);
    while (!at(p, WGSL_TOK_RBRACE) && !at(p, WGSL_TOK_EOF)) {
        WGSLNode *m = parse_struct_member(p);
        if (m) nl_push(&kids, m);
        if (!match(p, WGSL_TOK_COMMA)) break;
    }
    const WGSLToken *close = peek(p);
    expect(p, WGSL_TOK_RBRACE, "'}'");

    WGSLSpan span = wgsl_span_make(
        kw->span.offset, close->span.offset + close->span.length - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_DECL_STRUCT, span);
    if (node) {
        node->payload[0] = ((uint64_t)name->span.offset) | ((uint64_t)name->span.length << 32);
        node->payload[1] = (uint64_t)kids.count;     /* member count */
        finalize_children(p, node, &kids);
    }
    nl_destroy(&kids);
    return node;
}

/* `'alias' IDENT '=' type_specifier ';'`. */
WGSLNode *parse_alias_decl(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_ALIAS, "'alias'");
    const WGSLToken *name = expect(p, WGSL_TOK_IDENT, "alias name");
    expect(p, WGSL_TOK_EQUAL, "'='");
    WGSLNode *type = parse_type_specifier(p);
    const WGSLToken *semi = peek(p);
    expect(p, WGSL_TOK_SEMICOLON, "';'");

    WGSLSpan span = wgsl_span_make(
        kw->span.offset, semi->span.offset + semi->span.length - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_DECL_ALIAS, span);
    if (node) {
        node->payload[0] = ((uint64_t)name->span.offset) | ((uint64_t)name->span.length << 32);
        if (type) {
            WGSLNode **arr = (WGSLNode **)wgsl_arena_alloc(p->arena, sizeof(WGSLNode *));
            if (arr) { arr[0] = type; node->children = arr; node->child_count = 1; }
        }
    }
    return node;
}

/* `attribute* 'override' IDENT (':' type)? ('=' expr)? ';'`.
 * Payload mirrors the var encoding so walkers can reuse logic.
 *   payload[0]: name span
 *   payload[1]: attr_count (low) | tpl_count = 0 (high)
 *   payload[2]: has_type | has_init << 32
 *   children: [attrs…][type?][init?] */
WGSLNode *parse_override_decl(P *p, NL *attrs) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_OVERRIDE, "'override'");
    const WGSLToken *name = expect(p, WGSL_TOK_IDENT, "override name");

    NL kids; nl_init(&kids);
    for (size_t i = 0; i < attrs->count; i++) nl_push(&kids, attrs->items[i]);
    uint32_t attr_count = (uint32_t)attrs->count;

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
    const WGSLToken *semi = peek(p);
    expect(p, WGSL_TOK_SEMICOLON, "';'");

    uint32_t start = (attr_count > 0) ? attrs->items[0]->span_offset : kw->span.offset;
    WGSLSpan span = wgsl_span_make(
        start, semi->span.offset + semi->span.length - start);
    WGSLNode *node = make_node(p, WGSL_NODE_DECL_OVERRIDE, span);
    if (node) {
        node->payload[0] = ((uint64_t)name->span.offset) | ((uint64_t)name->span.length << 32);
        node->payload[1] = (uint64_t)attr_count;     /* tpl_count = 0 */
        node->payload[2] =
            ((uint64_t)(has_type ? 1u : 0u)) |
            ((uint64_t)(has_init ? 1u : 0u) << 32);
        finalize_children(p, node, &kids);
    }
    nl_destroy(&kids);
    return node;
}

/* Directives. */

/* Generic comma-separated ident-list directive: `kw name (',' name)* ','? ';'`.
 * Each name becomes an EXPR_IDENT child. */
WGSLNode *parse_ident_list_directive(
    P *p, WGSLTokenKind kw_kind, const char *kw_name, WGSLNodeKind node_kind)
{
    const WGSLToken *kw = expect(p, kw_kind, kw_name);
    NL names; nl_init(&names);
    do {
        if (at(p, WGSL_TOK_SEMICOLON)) break;          /* trailing comma */
        const WGSLToken *t = expect(p, WGSL_TOK_IDENT, "extension name");
        WGSLNode *n = make_ident_from(p, t);
        if (n) nl_push(&names, n);
    } while (match(p, WGSL_TOK_COMMA));
    const WGSLToken *semi = peek(p);
    expect(p, WGSL_TOK_SEMICOLON, "';'");

    WGSLSpan span = wgsl_span_make(
        kw->span.offset, semi->span.offset + semi->span.length - kw->span.offset);
    WGSLNode *node = make_node(p, node_kind, span);
    if (node) {
        node->payload[0] = (uint64_t)names.count;
        finalize_children(p, node, &names);
    }
    nl_destroy(&names);
    return node;
}

/* `'diagnostic' '(' severity_name ',' rule_name ('.' sub_name)? ')' ';'`
 *   payload[0] = rule-part-count (1 if `rule`, 2 if `rule.sub`)
 *   children:    [severity_ident, rule_ident, optional sub_ident] */
WGSLNode *parse_diagnostic_directive(P *p) {
    const WGSLToken *kw = expect(p, WGSL_TOK_KW_DIAGNOSTIC, "'diagnostic'");
    expect(p, WGSL_TOK_LPAREN, "'('");

    NL kids; nl_init(&kids);

    const WGSLToken *sev = expect(p, WGSL_TOK_IDENT, "severity name");
    nl_push(&kids, make_ident_from(p, sev));
    expect(p, WGSL_TOK_COMMA, "','");

    const WGSLToken *r1 = expect(p, WGSL_TOK_IDENT, "rule name");
    nl_push(&kids, make_ident_from(p, r1));
    uint32_t rule_parts = 1;
    if (match(p, WGSL_TOK_DOT)) {
        const WGSLToken *r2 = expect(p, WGSL_TOK_IDENT, "rule sub-name");
        nl_push(&kids, make_ident_from(p, r2));
        rule_parts = 2;
    }
    expect(p, WGSL_TOK_RPAREN, "')'");
    const WGSLToken *semi = peek(p);
    expect(p, WGSL_TOK_SEMICOLON, "';'");

    WGSLSpan span = wgsl_span_make(
        kw->span.offset, semi->span.offset + semi->span.length - kw->span.offset);
    WGSLNode *node = make_node(p, WGSL_NODE_DIR_DIAGNOSTIC, span);
    if (node) {
        node->payload[0] = (uint64_t)rule_parts;
        finalize_children(p, node, &kids);
    }
    nl_destroy(&kids);
    return node;
}

/* Global decl dispatch. */

int parse_global_decl(P *p, NL *out) {
    NL attrs; nl_init(&attrs);
    parse_attributes(p, &attrs);

    WGSLTokenKind k = peek(p)->kind;
    WGSLNode *decl = NULL;
    int consumed_attrs = 0;
    int produced = 0;

    WGSLSpan k_span = peek(p)->span;
    switch (k) {
    case WGSL_TOK_KW_FN:
        decl = parse_function_decl(p, &attrs);
        consumed_attrs = 1;
        break;

    case WGSL_TOK_KW_STRUCT:
        decl = parse_struct_decl(p);
        break;

    case WGSL_TOK_KW_ALIAS:
        decl = parse_alias_decl(p);
        break;

    case WGSL_TOK_KW_CONST:
        decl = parse_const_decl(p, /*as_statement=*/1);
        break;

    case WGSL_TOK_KW_CONST_ASSERT:
        decl = parse_const_assert_statement(p);
        break;

    case WGSL_TOK_KW_OVERRIDE:
        decl = parse_override_decl(p, &attrs);
        consumed_attrs = 1;
        break;

    case WGSL_TOK_KW_VAR:
        decl = parse_var_decl(p, /*as_statement=*/1,
                              /*allow_template=*/1, &attrs);
        consumed_attrs = 1;
        break;

    case WGSL_TOK_KW_ENABLE:
        if (p->has_seen_decl) {
            parse_error(p, k_span.offset, k_span.length,
                        "'enable' directive must appear before any declarations");
        }
        decl = parse_ident_list_directive(
            p, WGSL_TOK_KW_ENABLE, "'enable'", WGSL_NODE_DIR_ENABLE);
        break;

    case WGSL_TOK_KW_REQUIRES:
        if (p->has_seen_decl) {
            parse_error(p, k_span.offset, k_span.length,
                        "'requires' directive must appear before any declarations");
        }
        decl = parse_ident_list_directive(
            p, WGSL_TOK_KW_REQUIRES, "'requires'", WGSL_NODE_DIR_REQUIRES);
        break;

    case WGSL_TOK_KW_DIAGNOSTIC:
        if (p->has_seen_decl) {
            parse_error(p, k_span.offset, k_span.length,
                        "'diagnostic' directive must appear before any declarations");
        }
        decl = parse_diagnostic_directive(p);
        break;

    default: {
        const WGSLToken *t = peek(p);
        parse_error(p, t->span.offset, t->span.length,
                    "expected a top-level declaration, got %s",
                    wgsl_token_kind_name(t->kind));
        break;
    }
    }

    /* Diagnose attributes that the declaration kind doesn't accept. */
    if (attrs.count > 0 && !consumed_attrs) {
        WGSLNode *first = attrs.items[0];
        parse_error(p, first->span_offset, first->span_length,
                    "attributes are not permitted on this declaration");
    }
    nl_destroy(&attrs);

    if (decl) {
        if (decl->kind != WGSL_NODE_DIR_ENABLE &&
            decl->kind != WGSL_NODE_DIR_REQUIRES &&
            decl->kind != WGSL_NODE_DIR_DIAGNOSTIC)
        {
            p->has_seen_decl = 1;
        }
        produced = nl_push(out, decl);
    }
    return produced;
}

/* Entry point. */

int wgsl_parse(
    const WGSLLexResult *tokens,
    const WGSLSource    *source,
    WGSLArena           *arena,
    WGSLDiagBag         *diag,
    WGSLAst             *out)
{
    if (!tokens || !source || !arena || !diag || !out) return 0;
    out->root = NULL;

    P p;
    memset(&p, 0, sizeof p);
    p.tokens = tokens->tokens;
    p.count  = tokens->count;
    p.src    = source;
    p.arena  = arena;
    p.diag   = diag;

    if (!p.tokens || p.count == 0 ||
        p.tokens[p.count - 1].kind != WGSL_TOK_EOF)
    {
        parse_error(&p, 0, 0, "internal lexer error: missing EOF token");
        WGSLNode *tu = make_node(&p, WGSL_NODE_TRANSLATION_UNIT,
                                 wgsl_span_make(0, (uint32_t)source->length));
        out->root = tu;
        return 0;
    }

    WGSLNode *tu = make_node(&p, WGSL_NODE_TRANSLATION_UNIT,
                             wgsl_span_make(0, (uint32_t)source->length));
    if (!tu) return 0;

    NL list; nl_init(&list);
    while (!at(&p, WGSL_TOK_EOF)) {
        if (at(&p, WGSL_TOK_SEMICOLON)) { advance(&p); continue; }
        /* Stray `}` at the top level: a malformed inner block has
         * leaked its closer up here.  `parse_global_decl` would emit
         * "expected top-level declaration" without advancing, and
         * `recover_to_sync` stops at `}` — yielding an infinite loop.
         * Consume the `}` here so the recovery makes progress. */
        if (at(&p, WGSL_TOK_RBRACE)) {
            const WGSLToken *t = peek(&p);
            parse_error(&p, t->span.offset, t->span.length,
                        "stray '}' at top level");
            advance(&p);
            continue;
        }
        if (!parse_global_decl(&p, &list)) {
            recover_to_sync(&p);
        }
    }
    finalize_children(&p, tu, &list);
    nl_destroy(&list);

    out->root = tu;
    return p.had_error ? 0 : 1;
}
