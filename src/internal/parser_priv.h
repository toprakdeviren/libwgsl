/**
 * @file parser_priv.h — shared parser state and helpers.
 */
#ifndef WGSL_INTERNAL_PARSER_PRIV_H
#define WGSL_INTERNAL_PARSER_PRIV_H

#include "internal/parser.h"
#include "internal/token.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef WGSL_PARSE_MAX_DEPTH
#define WGSL_PARSE_MAX_DEPTH 128
#endif

typedef struct {
    const WGSLToken *tokens;
    size_t           count;
    size_t           pos;
    const WGSLSource *src;
    WGSLArena        *arena;
    WGSLDiagBag      *diag;
    int               had_error;
    int               has_seen_decl;
    int               depth;
} P;

typedef struct {
    WGSLNode **items;
    size_t     count;
    size_t     capacity;
} NL;

typedef enum {
    BIN_GROUP_NONE      = 0,
    BIN_GROUP_MUL       = 1u << 0,
    BIN_GROUP_ADD       = 1u << 1,
    BIN_GROUP_SHIFT     = 1u << 2,
    BIN_GROUP_REL       = 1u << 3,
    BIN_GROUP_BIT_AND   = 1u << 4,
    BIN_GROUP_BIT_XOR   = 1u << 5,
    BIN_GROUP_BIT_OR    = 1u << 6,
    BIN_GROUP_LOGICAL   = 1u << 7,
} BinaryOpGroup;


void nl_init(NL *nl);
void nl_destroy(NL *nl);
int nl_push(NL *nl, WGSLNode *n);
int is_trivia_kind(WGSLTokenKind k);
void skip_trivia(P *p);
const WGSLToken * peek_n(P *p, size_t n);
const WGSLToken * peek(P *p);
const WGSLToken * advance(P *p);
int at(P *p, WGSLTokenKind k);
int match(P *p, WGSLTokenKind k);
void parse_error(P *p, uint32_t off, uint32_t len, const char *fmt, ...);
const WGSLToken * expect(P *p, WGSLTokenKind k, const char *what);
void recover_to_sync(P *p);
WGSLNode * make_node(P *p, WGSLNodeKind kind, WGSLSpan span);
int finalize_children(P *p, WGSLNode *parent, NL *list);
BinaryOpGroup binary_group(WGSLTokenKind k);
uint32_t can_precede_groups(BinaryOpGroup g);
int requires_parens_ordered(WGSLTokenKind prev, WGSLTokenKind next);
int prec_of(WGSLTokenKind k);
WGSLNode * parse_binary_rhs(P *p, WGSLNode *lhs, int min_prec);
WGSLNode * parse_expression(P *p);
WGSLNode * parse_unary_impl(P *p); WGSLNode *parse_unary(P *p);
WGSLNode * parse_template_arg_list(P *p, WGSLNode *ident);
WGSLNode * parse_postfix(P *p, WGSLNode *base);
WGSLNode * parse_primary(P *p);
WGSLNode * parse_type_specifier(P *p);
WGSLNode * parse_attribute(P *p);
int parse_attributes(P *p, NL *out);
int attach_statement_attrs(P *p, WGSLNode *node, const NL *attrs);
int stmt_kind_accepts_attrs(WGSLNodeKind k);
WGSLNode * parse_statement(P *p); WGSLNode *parse_compound_statement(P *p);
void parse_discard_attributes(P *p);
WGSLNode * parse_attributed_compound_statement(P *p);
const WGSLToken * expect_attributed_lbrace(P *p, const char *label, NL *attrs);
uint32_t finish_stmt_or_inline(P *p, uint32_t fallback_end, int as_statement);
WGSLNode * parse_let_decl(P *p, int as_statement);
WGSLNode * parse_var_decl(P *p, int as_statement, int allow_template, NL *attrs_or_null);
WGSLNode * parse_const_decl(P *p, int as_statement);
WGSLNode * parse_const_assert_statement(P *p);
WGSLNode * parse_return_statement(P *p);
WGSLNode * parse_default_statement(P *p, int as_statement);
WGSLNode * parse_if_statement(P *p);
WGSLNode * parse_while_statement(P *p);
WGSLNode * parse_for_statement(P *p);
WGSLNode * parse_loop_statement(P *p);
WGSLNode * parse_continuing_statement(P *p);
WGSLNode * parse_break_statement(P *p);
WGSLNode * parse_continue_statement(P *p);
WGSLNode * parse_discard_statement(P *p);
WGSLNode * parse_case_clause(P *p);
WGSLNode * parse_switch_statement(P *p);
WGSLNode * parse_statement_impl(P *p); WGSLNode *parse_statement(P *p);
WGSLNode * parse_param(P *p);
WGSLNode * parse_function_decl(P *p, NL *fn_attrs);
WGSLNode * make_ident_from(P *p, const WGSLToken *t);
WGSLNode * parse_struct_member(P *p);
WGSLNode * parse_struct_decl(P *p);
WGSLNode * parse_alias_decl(P *p);
WGSLNode * parse_override_decl(P *p, NL *attrs);
WGSLNode * parse_ident_list_directive(P *p, WGSLTokenKind kw_kind, const char *kw_name, WGSLNodeKind node_kind);
WGSLNode * parse_diagnostic_directive(P *p);
int parse_global_decl(P *p, NL *out);
int wgsl_parse(const WGSLLexResult *tokens, const WGSLSource *source, WGSLArena *arena, WGSLDiagBag *diag, WGSLAst *out);

#endif /* WGSL_INTERNAL_PARSER_PRIV_H */
