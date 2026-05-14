/**
 * Phase 8 Iter A — validator: cycle detection family.
 *
 * Covers:
 *   - call-graph: self-recursion, mutual recursion → "recursion" diag
 *   - acyclic call graph (caller→callee→leaf) → ok
 *   - struct value cycle: self-field, mutual fields, array-of-struct
 *     wrapping → "struct cycle" diag
 *   - acyclic struct nesting → ok
 *   - alias cycle: A=B / B=A → "alias cycle" diag
 *   - acyclic alias chain → ok
 *   - const init cycle: A = A + 1 → "const cycle" diag
 *   - override init cycle → "override cycle" diag
 */
#include "internal/lexer.h"
#include "internal/parser.h"
#include "internal/resolver.h"
#include "internal/consteval.h"
#include "internal/check.h"
#include "internal/validate.h"
#include "internal/types.h"
#include "internal/utf8.h"
#include "internal/ast.h"

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
    WGSLValidator      val;
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
    int val_ok   = wgsl_validate(&p->ast, &p->src, &p->arena,
                                 &p->diag, &p->types, &p->res, &p->cev,
                                 &p->tc, &p->val);
    p->ok = lex_ok && parse_ok && res_ok && ce_ok && tc_ok && val_ok;
}

static void done(P *p) {
    wgsl_validator_destroy(&p->val);
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

int main(void) {
    wgsl_utf8_init();


#include "test_validate_cases/cycles_attrs.inc"
#include "test_validate_cases/attrs_uniformity_calls.inc"
#include "test_validate_cases/uniformity_lattice.inc"
#include "test_validate_cases/behavior_alias_layout.inc"
#include "test_validate_cases/entry_address_tags.inc"
    fprintf(stderr, "%s  test_validate  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
