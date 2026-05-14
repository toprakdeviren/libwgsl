/* ── Diagnostics ──────────────────────────────────────────────────── */

int wgsl_diagnostic_count(const WGSLResult *r) {
    if (!r) return 0;
    return (int)r->diag.count;
}

const WGSLDiagnostic *wgsl_diagnostic(const WGSLResult *r, int index) {
    if (!r || index < 0 || (size_t)index >= r->diag.count) return NULL;
    return &r->diag.items[index];
}

/* ── Lexer (raw tokens for syntax highlighters) ───────────────────── */

/* Translate `WGSLTokenKind` → `WGSLLexTokenKind` for the public stream.
 * Iter B will additionally promote IDENT tokens to STRUCT / FUNCTION /
 * PARAMETER / FIELD / VAR / LET / CONST / OVERRIDE / TYPE_ALIAS based
 * on resolver bindings. */
static WGSLLexTokenKind classify_raw_token(WGSLTokenKind k) {
    switch (k) {
    case WGSL_TOK_IDENT:           return WGSL_LEX_IDENTIFIER;
    case WGSL_TOK_INT_LIT:
    case WGSL_TOK_FLOAT_LIT:       return WGSL_LEX_NUMBER;

    /* Keywords. */
    case WGSL_TOK_KW_ALIAS:
    case WGSL_TOK_KW_BREAK:
    case WGSL_TOK_KW_CASE:
    case WGSL_TOK_KW_CONST:
    case WGSL_TOK_KW_CONST_ASSERT:
    case WGSL_TOK_KW_CONTINUE:
    case WGSL_TOK_KW_CONTINUING:
    case WGSL_TOK_KW_DEFAULT:
    case WGSL_TOK_KW_DISCARD:
    case WGSL_TOK_KW_ELSE:
    case WGSL_TOK_KW_FN:
    case WGSL_TOK_KW_FOR:
    case WGSL_TOK_KW_IF:
    case WGSL_TOK_KW_LET:
    case WGSL_TOK_KW_LOOP:
    case WGSL_TOK_KW_OVERRIDE:
    case WGSL_TOK_KW_RETURN:
    case WGSL_TOK_KW_STRUCT:
    case WGSL_TOK_KW_SWITCH:
    case WGSL_TOK_KW_VAR:
    case WGSL_TOK_KW_WHILE:        return WGSL_LEX_KEYWORD;

    /* Boolean literals are spec-keywords; treat them as builtin values
     * so editors can render `true` / `false` distinctly from `if`. */
    case WGSL_TOK_KW_TRUE:
    case WGSL_TOK_KW_FALSE:        return WGSL_LEX_BUILTIN_VALUE;

    case WGSL_TOK_KW_ENABLE:
    case WGSL_TOK_KW_REQUIRES:
    case WGSL_TOK_KW_DIAGNOSTIC:   return WGSL_LEX_DIRECTIVE;

    /* Punctuation. */
    case WGSL_TOK_LPAREN:
    case WGSL_TOK_RPAREN:
    case WGSL_TOK_LBRACE:
    case WGSL_TOK_RBRACE:
    case WGSL_TOK_LBRACKET:
    case WGSL_TOK_RBRACKET:
    case WGSL_TOK_COMMA:
    case WGSL_TOK_SEMICOLON:
    case WGSL_TOK_COLON:
    case WGSL_TOK_DOT:
    case WGSL_TOK_UNDERSCORE:      return WGSL_LEX_PUNCT;

    case WGSL_TOK_AT:              return WGSL_LEX_ATTRIBUTE;

    /* Template-list start / end (post §3.9 disambiguation). */
    case WGSL_TOK_TEMPLATE_START:
    case WGSL_TOK_TEMPLATE_END:    return WGSL_LEX_TEMPLATE_DELIM;

    /* Operators (everything else is an operator at lex level). */
    case WGSL_TOK_AMP:
    case WGSL_TOK_AMP_AMP:
    case WGSL_TOK_AMP_EQUAL:
    case WGSL_TOK_PIPE:
    case WGSL_TOK_PIPE_PIPE:
    case WGSL_TOK_PIPE_EQUAL:
    case WGSL_TOK_CARET:
    case WGSL_TOK_CARET_EQUAL:
    case WGSL_TOK_TILDE:
    case WGSL_TOK_BANG:
    case WGSL_TOK_BANG_EQUAL:
    case WGSL_TOK_PLUS:
    case WGSL_TOK_PLUS_PLUS:
    case WGSL_TOK_PLUS_EQUAL:
    case WGSL_TOK_MINUS:
    case WGSL_TOK_MINUS_MINUS:
    case WGSL_TOK_MINUS_EQUAL:
    case WGSL_TOK_ARROW:
    case WGSL_TOK_STAR:
    case WGSL_TOK_STAR_EQUAL:
    case WGSL_TOK_SLASH:
    case WGSL_TOK_SLASH_EQUAL:
    case WGSL_TOK_PERCENT:
    case WGSL_TOK_PERCENT_EQUAL:
    case WGSL_TOK_LESS:
    case WGSL_TOK_LESS_EQUAL:
    case WGSL_TOK_LESS_LESS:
    case WGSL_TOK_LESS_LESS_EQUAL:
    case WGSL_TOK_GREATER:
    case WGSL_TOK_GREATER_EQUAL:
    case WGSL_TOK_GREATER_GREATER:
    case WGSL_TOK_GREATER_GREATER_EQUAL:
    case WGSL_TOK_EQUAL:
    case WGSL_TOK_EQUAL_EQUAL:     return WGSL_LEX_OPERATOR;

    case WGSL_TOK_LINE_COMMENT:
    case WGSL_TOK_BLOCK_COMMENT:   return WGSL_LEX_COMMENT;
    case WGSL_TOK_BLANKSPACE:      return WGSL_LEX_WHITESPACE;

    case WGSL_TOK_INVALID:
    case WGSL_TOK_EOF:
    default:                       return WGSL_LEX_UNKNOWN;
    }
}

/* Internal helper: copy a `WGSLLexResult` into a heap-owned
 * `WGSLLexToken[]`, applying `classify` to upgrade idents. */
typedef WGSLLexTokenKind (*ClassifyFn)(
    const WGSLToken *t, const WGSLSource *src, void *ctx);

static int collect_lex_tokens(
    const WGSLSource *source, const WGSLLexResult *lex,
    ClassifyFn classify, void *ctx,
    WGSLLexToken **out_tokens, int *out_count)
{
    /* Skip the trailing EOF sentinel; emit blankspace + comment + real
     * tokens.  Callers can filter trivia client-side if they want. */
    size_t total = lex->count;
    if (total > 0 && lex->tokens[total - 1].kind == WGSL_TOK_EOF) total -= 1;

    WGSLLexToken *out = NULL;
    if (total > 0) {
        if (total > (size_t)INT32_MAX ||
            total > SIZE_MAX / sizeof *out)
        {
            return 0;
        }
        out = (WGSLLexToken *)malloc(total * sizeof *out);
        if (!out) return 0;
    }

    for (size_t i = 0; i < total; i++) {
        const WGSLToken *t = &lex->tokens[i];
        out[i].kind   = classify(t, source, ctx);
        out[i].offset = t->span.offset;
        out[i].length = t->span.length;
        out[i].line   = t->line;
        out[i].column = t->column;
    }
    *out_tokens = out;
    *out_count  = (int)total;
    return 1;
}

static WGSLLexTokenKind raw_classify(
    const WGSLToken *t, const WGSLSource *src, void *ctx)
{
    (void)src; (void)ctx;
    return classify_raw_token(t->kind);
}

int wgsl_lex(const char *src, WGSLLexToken **out_tokens, int *out_count) {
    if (!out_tokens || !out_count) return 0;
    *out_tokens = NULL;
    *out_count  = 0;

    wgsl_utf8_init();

    WGSLArena   arena;     wgsl_arena_init(&arena);
    WGSLDiagBag diag;      wgsl_diag_init(&diag);
    WGSLSource  source;
    if (!wgsl_source_init(&source, src ? src : "", src ? strlen(src) : 0)) {
        wgsl_diag_destroy(&diag);
        wgsl_arena_destroy(&arena);
        return 0;
    }
    WGSLLexResult lex = {0};
    int lex_ok = wgsl_tokenize(&source, &arena, &diag, &lex);

    int rc = lex_ok ? collect_lex_tokens(
        &source, &lex, raw_classify, NULL, out_tokens, out_count) : 0;

    wgsl_diag_destroy(&diag);
    wgsl_source_destroy(&source);
    wgsl_arena_destroy(&arena);
    return rc;
}

/* ── Semantic-token overlay — Iter B ──────────────────────────────── */

/* Per-decl-name span we emit while walking the AST.  Callers merge
 * these into the raw token stream by exact (offset, length) match. */
typedef struct {
    uint32_t          offset;
    uint32_t          length;
    WGSLLexTokenKind  kind;
} SemSpan;

typedef struct {
    SemSpan *items;
    size_t   count;
    size_t   capacity;
} SemList;

static void sem_push(SemList *L, uint32_t off, uint32_t len, WGSLLexTokenKind k) {
    if (len == 0) return;
    if (L->count == L->capacity) {
        if (L->capacity > SIZE_MAX / 2) return;
        size_t cap = L->capacity ? L->capacity * 2 : 64;
        if (cap > SIZE_MAX / sizeof *L->items) return;
        SemSpan *g = (SemSpan *)realloc(L->items, cap * sizeof *g);
        if (!g) return;
        L->items = g;
        L->capacity = cap;
    }
    L->items[L->count++] = (SemSpan){ off, len, k };
}

static WGSLLexTokenKind kind_for_sym(const WGSLSymbol *s) {
    switch ((WGSLSymKind)s->kind) {
    case WGSL_SYM_PREDECLARED_TYPE:
    case WGSL_SYM_PREDECLARED_TYPE_ALIAS:
    case WGSL_SYM_PREDECLARED_TYPEGEN:    return WGSL_LEX_TYPE;
    case WGSL_SYM_PREDECLARED_FN:         return WGSL_LEX_BUILTIN_FUNC;
    case WGSL_SYM_PREDECLARED_VALUE:      return WGSL_LEX_BUILTIN_VALUE;
    case WGSL_SYM_PREDECLARED_ADDR_SPACE:
    case WGSL_SYM_PREDECLARED_ACCESS_MODE:
    case WGSL_SYM_PREDECLARED_TEXEL_FORMAT: return WGSL_LEX_KEYWORD;
    case WGSL_SYM_TYPE_ALIAS:             return WGSL_LEX_TYPE_ALIAS;
    case WGSL_SYM_STRUCT:                 return WGSL_LEX_STRUCT_NAME;
    case WGSL_SYM_FUNCTION:               return WGSL_LEX_FUNCTION_NAME;
    case WGSL_SYM_VAR:                    return WGSL_LEX_VAR;
    case WGSL_SYM_LET:                    return WGSL_LEX_LET;
    case WGSL_SYM_CONST:                  return WGSL_LEX_CONST;
    case WGSL_SYM_OVERRIDE:               return WGSL_LEX_OVERRIDE;
    case WGSL_SYM_PARAM:                  return WGSL_LEX_PARAMETER;
    default:                              return WGSL_LEX_IDENTIFIER;
    }
}

/* Walk the AST and emit `(offset, length, kind)` triples for:
 *   - every EXPR_IDENT bound to a resolver symbol — uses `payload[2]`
 *     as `WGSLSymbol *`
 *   - every decl-shaped node's name span — `payload[0]` carries
 *     `(offset | length << 32)` for DECL_FUNCTION / PARAM / VAR / LET /
 *     CONST / OVERRIDE / STRUCT / ALIAS / STRUCT_MEMBER. */
static void collect_sem_spans(WGSLNode *n, SemList *L) {
    if (!n) return;
    switch ((WGSLNodeKind)n->kind) {

    case WGSL_NODE_EXPR_IDENT: {
        WGSLSymbol *s = wgsl_node_resolved_symbol(n);
        if (s) {
            sem_push(L, n->span_offset, n->span_length, kind_for_sym(s));
        }
        break;
    }

    case WGSL_NODE_DECL_FUNCTION: {
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        sem_push(L, no, nl, WGSL_LEX_FUNCTION_NAME);
        break;
    }
    case WGSL_NODE_DECL_PARAM: {
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        sem_push(L, no, nl, WGSL_LEX_PARAMETER);
        break;
    }
    case WGSL_NODE_DECL_VAR: {
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        sem_push(L, no, nl, WGSL_LEX_VAR);
        break;
    }
    case WGSL_NODE_DECL_LET: {
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        sem_push(L, no, nl, WGSL_LEX_LET);
        break;
    }
    case WGSL_NODE_DECL_CONST: {
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        sem_push(L, no, nl, WGSL_LEX_CONST);
        break;
    }
    case WGSL_NODE_DECL_OVERRIDE: {
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        sem_push(L, no, nl, WGSL_LEX_OVERRIDE);
        break;
    }
    case WGSL_NODE_DECL_STRUCT: {
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        sem_push(L, no, nl, WGSL_LEX_STRUCT_NAME);
        break;
    }
    case WGSL_NODE_DECL_STRUCT_MEMBER: {
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        sem_push(L, no, nl, WGSL_LEX_FIELD);
        break;
    }
    case WGSL_NODE_DECL_ALIAS: {
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        sem_push(L, no, nl, WGSL_LEX_TYPE_ALIAS);
        break;
    }

    case WGSL_NODE_EXPR_MEMBER: {
        /* `obj.member` — children[0] = obj (recursed below), children[1] =
         * the member-name ident.  Tag the member as FIELD. */
        if (n->child_count >= 2 && n->children[1] &&
            n->children[1]->kind == WGSL_NODE_EXPR_IDENT)
        {
            WGSLNode *m = n->children[1];
            sem_push(L, m->span_offset, m->span_length, WGSL_LEX_FIELD);
        }
        if (n->child_count >= 1) collect_sem_spans(n->children[0], L);
        return;       /* children[1] handled inline; don't recurse into it */
    }

    default:
        break;
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        collect_sem_spans(n->children[i], L);
    }
}

typedef struct {
    const SemSpan *spans;
    size_t         count;
} SemCtx;

static int span_cmp_offset(const void *a, const void *b) {
    const SemSpan *x = (const SemSpan *)a;
    const SemSpan *y = (const SemSpan *)b;
    if (x->offset < y->offset) return -1;
    if (x->offset > y->offset) return  1;
    return 0;
}

static WGSLLexTokenKind sem_classify(
    const WGSLToken *t, const WGSLSource *src, void *vctx)
{
    (void)src;
    SemCtx *ctx = (SemCtx *)vctx;
    /* Binary search by offset for an exact-span match. */
    if (ctx->count > 0) {
        size_t lo = 0, hi = ctx->count;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (ctx->spans[mid].offset < t->span.offset) lo = mid + 1;
            else                                          hi = mid;
        }
        for (size_t i = lo;
             i < ctx->count && ctx->spans[i].offset == t->span.offset;
             i++)
        {
            if (ctx->spans[i].length == t->span.length) {
                return ctx->spans[i].kind;
            }
        }
    }
    return classify_raw_token(t->kind);
}

int wgsl_semantic_tokens(const char *src, WGSLLexToken **out_tokens, int *out_count) {
    if (!out_tokens || !out_count) return 0;
    *out_tokens = NULL;
    *out_count  = 0;

    /* Run the full pipeline so the resolver bindings are in place; if
     * parsing or resolving fails we degrade gracefully to raw lex. */
    WGSLResult *r = wgsl_check(src);
    if (!r) return 0;

    SemList L = {0};
    if (r->ast.root) collect_sem_spans(r->ast.root, &L);
    /* Sort so we can binary-search by offset. */
    if (L.count > 1) qsort(L.items, L.count, sizeof *L.items, span_cmp_offset);

    SemCtx ctx = { L.items, L.count };
    int rc = collect_lex_tokens(
        &r->source, &r->lex, sem_classify, &ctx, out_tokens, out_count);

    free(L.items);
    wgsl_free(r);
    return rc;
}

void wgsl_lex_free(WGSLLexToken *tokens) {
    free(tokens);
}

