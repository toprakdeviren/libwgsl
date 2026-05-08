/**
 * @file ast_dump.c — AST → indented S-expression printer.
 */
#include "internal/ast_dump.h"

#include <stdint.h>
#include <string.h>

static const char *node_kind_short(WGSLNodeKind k) {
    switch (k) {
    case WGSL_NODE_INVALID:                return "invalid";
    case WGSL_NODE_TRANSLATION_UNIT:       return "translation_unit";
    case WGSL_NODE_DIR_ENABLE:             return "dir_enable";
    case WGSL_NODE_DIR_REQUIRES:           return "dir_requires";
    case WGSL_NODE_DIR_DIAGNOSTIC:         return "dir_diagnostic";
    case WGSL_NODE_DECL_STRUCT:            return "decl_struct";
    case WGSL_NODE_DECL_STRUCT_MEMBER:     return "struct_member";
    case WGSL_NODE_DECL_FUNCTION:          return "fn";
    case WGSL_NODE_DECL_PARAM:             return "param";
    case WGSL_NODE_DECL_ALIAS:             return "alias";
    case WGSL_NODE_DECL_CONST:             return "const";
    case WGSL_NODE_DECL_OVERRIDE:          return "override";
    case WGSL_NODE_DECL_VAR:               return "var";
    case WGSL_NODE_DECL_LET:               return "let";
    case WGSL_NODE_DECL_CONST_ASSERT:      return "const_assert";
    case WGSL_NODE_STMT_COMPOUND:          return "compound";
    case WGSL_NODE_STMT_ASSIGN:            return "assign";
    case WGSL_NODE_STMT_COMPOUND_ASSIGN:   return "compound_assign";
    case WGSL_NODE_STMT_PHONY_ASSIGN:      return "phony_assign";
    case WGSL_NODE_STMT_INCREMENT:         return "increment";
    case WGSL_NODE_STMT_DECREMENT:         return "decrement";
    case WGSL_NODE_STMT_IF:                return "if";
    case WGSL_NODE_STMT_ELSE_IF:           return "else_if";
    case WGSL_NODE_STMT_ELSE:              return "else";
    case WGSL_NODE_STMT_SWITCH:            return "switch";
    case WGSL_NODE_STMT_CASE_CLAUSE:       return "case";
    case WGSL_NODE_STMT_LOOP:              return "loop";
    case WGSL_NODE_STMT_FOR:               return "for";
    case WGSL_NODE_STMT_WHILE:             return "while";
    case WGSL_NODE_STMT_BREAK:             return "break";
    case WGSL_NODE_STMT_BREAK_IF:          return "break_if";
    case WGSL_NODE_STMT_CONTINUE:          return "continue";
    case WGSL_NODE_STMT_CONTINUING:        return "continuing";
    case WGSL_NODE_STMT_RETURN:            return "return";
    case WGSL_NODE_STMT_DISCARD:           return "discard";
    case WGSL_NODE_STMT_FN_CALL:           return "expr_stmt";
    case WGSL_NODE_EXPR_LITERAL_BOOL:      return "bool";
    case WGSL_NODE_EXPR_LITERAL_INT:       return "int";
    case WGSL_NODE_EXPR_LITERAL_FLOAT:     return "float";
    case WGSL_NODE_EXPR_IDENT:             return "ident";
    case WGSL_NODE_EXPR_TEMPLATED_IDENT:   return "templ";
    case WGSL_NODE_EXPR_PAREN:             return "paren";
    case WGSL_NODE_EXPR_BINARY:            return "binary";
    case WGSL_NODE_EXPR_UNARY:             return "unary";
    case WGSL_NODE_EXPR_CALL:              return "call";
    case WGSL_NODE_EXPR_INDEX:             return "index";
    case WGSL_NODE_EXPR_MEMBER:            return "member";
    case WGSL_NODE_EXPR_ADDR_OF:           return "addr_of";
    case WGSL_NODE_EXPR_INDIRECTION:       return "indirection";
    case WGSL_NODE_ATTRIBUTE:              return "attr";
    case WGSL_NODE_TEMPLATE_ARG_LIST:      return "tmpl_args";
    case WGSL_NODE_KIND_COUNT:             break;
    }
    return "<unknown>";
}

static void indent(FILE *out, int level) {
    for (int i = 0; i < level; i++) fputs("  ", out);
}

/* Print a clipped snippet from the source for an ident-like node. */
static void print_name(FILE *out, const WGSLSource *src, uint32_t off, uint32_t len) {
    if (!src || off + len > src->length) {
        fputs("\"\"", out);
        return;
    }
    fputc('"', out);
    for (uint32_t i = 0; i < len && i < 64; i++) {
        unsigned char c = (unsigned char)src->bytes[off + i];
        if (c == '"' || c == '\\') fputc('\\', out);
        if (c < 0x20) fprintf(out, "\\x%02X", c);
        else fputc((char)c, out);
    }
    if (len > 64) fputs("…", out);
    fputc('"', out);
}

static void dump_node(const WGSLNode *n, const WGSLSource *src, FILE *out, int level) {
    if (!n) {
        indent(out, level);
        fputs("(null)\n", out);
        return;
    }
    indent(out, level);
    fprintf(out, "(%s", node_kind_short((WGSLNodeKind)n->kind));

    /* Per-kind details */
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_DECL_FUNCTION: {
        uint32_t name_off = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t name_len = (uint32_t)(n->payload[0] >> 32);
        uint32_t fn_attrs = (uint32_t)(n->payload[1] & 0xFFFFFFFFu);
        uint32_t params   = (uint32_t)(n->payload[1] >> 32);
        uint32_t ret_attrs= (uint32_t)(n->payload[2] & 0xFFFFFFFFu);
        uint32_t has_ret  = (uint32_t)(n->payload[2] >> 32);
        fputs(" name=", out);
        print_name(out, src, name_off, name_len);
        fprintf(out, " fn_attrs=%u params=%u ret_attrs=%u ret_type=%s",
                fn_attrs, params, ret_attrs, has_ret ? "yes" : "no");
        break;
    }
    case WGSL_NODE_DECL_PARAM:
    case WGSL_NODE_DECL_LET:
    case WGSL_NODE_EXPR_IDENT: {
        uint32_t name_off = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t name_len = (uint32_t)(n->payload[0] >> 32);
        if (name_len > 0) {
            fputs(" ", out);
            print_name(out, src, name_off, name_len);
        }
        break;
    }
    case WGSL_NODE_EXPR_LITERAL_INT:
    case WGSL_NODE_EXPR_LITERAL_FLOAT:
    case WGSL_NODE_EXPR_LITERAL_BOOL: {
        if (src && n->span_offset + n->span_length <= src->length) {
            fputs(" ", out);
            print_name(out, src, n->span_offset, n->span_length);
        }
        break;
    }
    case WGSL_NODE_EXPR_BINARY:
    case WGSL_NODE_EXPR_UNARY: {
        uint32_t op_off = (uint32_t)(n->payload[1] & 0xFFFFFFFFu);
        uint32_t op_len = (uint32_t)(n->payload[1] >> 32);
        if (src && op_len > 0) {
            fputs(" op=", out);
            print_name(out, src, op_off, op_len);
        }
        break;
    }
    default:
        break;
    }

    if (n->child_count == 0) {
        fputs(")\n", out);
        return;
    }
    fputs("\n", out);
    for (uint32_t i = 0; i < n->child_count; i++) {
        dump_node(n->children[i], src, out, level + 1);
    }
    indent(out, level);
    fputs(")\n", out);
}

void wgsl_ast_dump(const WGSLNode *root, const WGSLSource *src, FILE *out) {
    dump_node(root, src, out, 0);
}
