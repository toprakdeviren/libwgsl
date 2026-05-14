/**
 * Phase 7 Iter A — type checker scalar foundation.
 *
 * Covers:
 *   - type-spec resolution: scalar / vec<T> / mat<T> / atomic<T> /
 *     array<T> / array<T, N> (with N from a const-eval'd symbol) /
 *     predeclared aliases (vec3f / mat4x4f / …)
 *   - module decl typing: const / let / var / override / alias / struct
 *     all populate `sym->type`
 *   - function param + return type typing
 *   - expression typing: literals, idents, parens, unary `- ! ~`,
 *     binary numeric / comparison / logical / bitwise
 *   - implicit promotion via §6.1.2 conversion-rank table
 *   - structural ref<AS,T,AM> type + is_ref bit on `var` ident uses,
 *     clear on const / let / param
 *   - error cases: u32 unary minus, concrete-type mismatch,
 *     non-numeric operands, non-bool logical operands
 */
#include "internal/check.h"
#include "internal/lexer.h"
#include "internal/parser.h"
#include "internal/resolver.h"
#include "internal/consteval.h"
#include "internal/types.h"
#include "internal/utf8.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int fail = 0;
#define CHECK(cond, msg) do {                                          \
    if (!(cond)) {                                                     \
        fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);  \
        fail += 1;                                                     \
    }                                                                  \
} while (0)

typedef struct {
    WGSLArena          arena;
    WGSLSource         src;
    WGSLDiagBag        diag;
    WGSLLexResult      lex;
    WGSLAst            ast;
    WGSLTypeStore      types;
    WGSLResolver       res;
    WGSLConstEvaluator cev;
    WGSLTypeChecker    tc;
    int                ok;
} P;

static void run(P *p, const char *text) {
    memset(p, 0, sizeof *p);
    wgsl_arena_init(&p->arena);
    wgsl_diag_init(&p->diag);
    wgsl_source_init(&p->src, text, strlen(text));
    int lex_ok   = wgsl_tokenize(&p->src, &p->arena, &p->diag, &p->lex);
    int parse_ok = wgsl_parse(&p->lex, &p->src, &p->arena, &p->diag, &p->ast);
    wgsl_types_init(&p->types, &p->arena);
    int res_ok   = wgsl_resolve(&p->ast, &p->src, &p->arena,
                                &p->diag, &p->types, &p->res);
    int ce_ok    = wgsl_consteval(&p->ast, &p->src, &p->arena,
                                  &p->diag, &p->types, &p->res, &p->cev);
    int tc_ok    = wgsl_typecheck(&p->ast, &p->src, &p->arena,
                                  &p->diag, &p->types, &p->res, &p->cev,
                                  &p->tc);
    p->ok = lex_ok && parse_ok && res_ok && ce_ok && tc_ok;
    if (!p->ok) {
        for (size_t i = 0; i < p->diag.count; i++) {
            fprintf(stderr, "DIAG: %s\n", p->diag.items[i].message);
        }
    }
}

static void done(P *p) {
    wgsl_typecheck_destroy(&p->tc);
    wgsl_consteval_destroy(&p->cev);
    wgsl_resolver_destroy(&p->res);
    wgsl_types_destroy(&p->types);
    wgsl_diag_destroy(&p->diag);
    wgsl_source_destroy(&p->src);
    wgsl_arena_destroy(&p->arena);
}

static int has_error_with(const WGSLDiagBag *b, const char *needle) {
    for (size_t i = 0; i < b->count; i++) {
        if (strstr(b->items[i].message, needle)) return 1;
    }
    return 0;
}

/* Find the symbol with the given name across resolver's all_decls. */
static WGSLSymbol *sym_for(const P *p, const char *want) {
    size_t wl = strlen(want);
    for (size_t i = 0; i < p->res.all_decl_count; i++) {
        WGSLSymbol *s = p->res.all_decls[i];
        if (s->name_len == wl && memcmp(s->name, want, wl) == 0) {
            return s;
        }
    }
    return NULL;
}

/* Find the Nth match of an EXPR_IDENT by source text. */
static const WGSLNode *find_ident(
    const WGSLNode *n, const WGSLSource *src, const char *want, int *skip)
{
    if (!n) return NULL;
    if (n->kind == WGSL_NODE_EXPR_IDENT) {
        uint32_t off = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t len = (uint32_t)(n->payload[0] >> 32);
        if (len == strlen(want) &&
            memcmp(src->bytes + off, want, len) == 0)
        {
            if (*skip == 0) return n;
            (*skip) -= 1;
        }
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        const WGSLNode *h = find_ident(n->children[i], src, want, skip);
        if (h) return h;
    }
    return NULL;
}

static const WGSLNode *find_kind(const WGSLNode *n, WGSLNodeKind kind) {
    if (!n) return NULL;
    if (n->kind == kind) return n;
    for (uint32_t i = 0; i < n->child_count; i++) {
        const WGSLNode *h = find_kind(n->children[i], kind);
        if (h) return h;
    }
    return NULL;
}

int main(void) {
    wgsl_utf8_init();


#include "test_check_cases/basics.inc"
#include "test_check_cases/variables.inc"
#include "test_check_cases/overloads_types.inc"
#include "test_check_cases/declarations.inc"
#include "test_check_cases/expressions.inc"
#include "test_check_cases/statements_attrs_entry.inc"
#include "test_check_cases/builtins_grammar.inc"
    fprintf(stderr, "%s  test_check  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
