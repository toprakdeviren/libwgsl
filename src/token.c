/**
 * @file token.c — Token kind names + keyword classifier.  See internal/token.h.
 */
#include "internal/token.h"

#include <stddef.h>
#include <string.h>

const char *wgsl_token_kind_name(WGSLTokenKind k) {
    switch (k) {
    case WGSL_TOK_INVALID:                   return "invalid";
    case WGSL_TOK_EOF:                       return "eof";
    case WGSL_TOK_IDENT:                     return "ident";
    case WGSL_TOK_INT_LIT:                   return "int_lit";
    case WGSL_TOK_FLOAT_LIT:                 return "float_lit";

    case WGSL_TOK_KW_ALIAS:                  return "kw_alias";
    case WGSL_TOK_KW_BREAK:                  return "kw_break";
    case WGSL_TOK_KW_CASE:                   return "kw_case";
    case WGSL_TOK_KW_CONST:                  return "kw_const";
    case WGSL_TOK_KW_CONST_ASSERT:           return "kw_const_assert";
    case WGSL_TOK_KW_CONTINUE:               return "kw_continue";
    case WGSL_TOK_KW_CONTINUING:             return "kw_continuing";
    case WGSL_TOK_KW_DEFAULT:                return "kw_default";
    case WGSL_TOK_KW_DIAGNOSTIC:             return "kw_diagnostic";
    case WGSL_TOK_KW_DISCARD:                return "kw_discard";
    case WGSL_TOK_KW_ELSE:                   return "kw_else";
    case WGSL_TOK_KW_ENABLE:                 return "kw_enable";
    case WGSL_TOK_KW_FALSE:                  return "kw_false";
    case WGSL_TOK_KW_FN:                     return "kw_fn";
    case WGSL_TOK_KW_FOR:                    return "kw_for";
    case WGSL_TOK_KW_IF:                     return "kw_if";
    case WGSL_TOK_KW_LET:                    return "kw_let";
    case WGSL_TOK_KW_LOOP:                   return "kw_loop";
    case WGSL_TOK_KW_OVERRIDE:               return "kw_override";
    case WGSL_TOK_KW_REQUIRES:               return "kw_requires";
    case WGSL_TOK_KW_RETURN:                 return "kw_return";
    case WGSL_TOK_KW_STRUCT:                 return "kw_struct";
    case WGSL_TOK_KW_SWITCH:                 return "kw_switch";
    case WGSL_TOK_KW_TRUE:                   return "kw_true";
    case WGSL_TOK_KW_VAR:                    return "kw_var";
    case WGSL_TOK_KW_WHILE:                  return "kw_while";

    case WGSL_TOK_LPAREN:                    return "(";
    case WGSL_TOK_RPAREN:                    return ")";
    case WGSL_TOK_LBRACE:                    return "{";
    case WGSL_TOK_RBRACE:                    return "}";
    case WGSL_TOK_LBRACKET:                  return "[";
    case WGSL_TOK_RBRACKET:                  return "]";
    case WGSL_TOK_COMMA:                     return ",";
    case WGSL_TOK_SEMICOLON:                 return ";";
    case WGSL_TOK_COLON:                     return ":";
    case WGSL_TOK_DOT:                       return ".";
    case WGSL_TOK_AT:                        return "@";
    case WGSL_TOK_UNDERSCORE:                return "_";

    case WGSL_TOK_AMP:                       return "&";
    case WGSL_TOK_AMP_AMP:                   return "&&";
    case WGSL_TOK_AMP_EQUAL:                 return "&=";
    case WGSL_TOK_PIPE:                      return "|";
    case WGSL_TOK_PIPE_PIPE:                 return "||";
    case WGSL_TOK_PIPE_EQUAL:                return "|=";
    case WGSL_TOK_CARET:                     return "^";
    case WGSL_TOK_CARET_EQUAL:               return "^=";
    case WGSL_TOK_TILDE:                     return "~";
    case WGSL_TOK_BANG:                      return "!";
    case WGSL_TOK_BANG_EQUAL:                return "!=";
    case WGSL_TOK_PLUS:                      return "+";
    case WGSL_TOK_PLUS_PLUS:                 return "++";
    case WGSL_TOK_PLUS_EQUAL:                return "+=";
    case WGSL_TOK_MINUS:                     return "-";
    case WGSL_TOK_MINUS_MINUS:               return "--";
    case WGSL_TOK_MINUS_EQUAL:               return "-=";
    case WGSL_TOK_ARROW:                     return "->";
    case WGSL_TOK_STAR:                      return "*";
    case WGSL_TOK_STAR_EQUAL:                return "*=";
    case WGSL_TOK_SLASH:                     return "/";
    case WGSL_TOK_SLASH_EQUAL:               return "/=";
    case WGSL_TOK_PERCENT:                   return "%";
    case WGSL_TOK_PERCENT_EQUAL:             return "%=";
    case WGSL_TOK_LESS:                      return "<";
    case WGSL_TOK_LESS_EQUAL:                return "<=";
    case WGSL_TOK_LESS_LESS:                 return "<<";
    case WGSL_TOK_LESS_LESS_EQUAL:           return "<<=";
    case WGSL_TOK_GREATER:                   return ">";
    case WGSL_TOK_GREATER_EQUAL:             return ">=";
    case WGSL_TOK_GREATER_GREATER:           return ">>";
    case WGSL_TOK_GREATER_GREATER_EQUAL:     return ">>=";
    case WGSL_TOK_EQUAL:                     return "=";
    case WGSL_TOK_EQUAL_EQUAL:               return "==";

    case WGSL_TOK_TEMPLATE_START:            return "<:tmpl";
    case WGSL_TOK_TEMPLATE_END:              return ">:tmpl";

    case WGSL_TOK_LINE_COMMENT:              return "line_comment";
    case WGSL_TOK_BLOCK_COMMENT:             return "block_comment";
    case WGSL_TOK_BLANKSPACE:                return "blankspace";

    case WGSL_TOK_KIND_COUNT:
        break;
    }
    return "<unknown>";
}

/* — Keyword table — sorted by length, then lexicographically.  All ASCII,
 * so a byte memcmp is sufficient. */
typedef struct { const char *text; uint8_t length; WGSLTokenKind kind; } KW;

static const KW KW_TABLE[] = {
    /* length 2 */
    { "fn",            2, WGSL_TOK_KW_FN          },
    { "if",            2, WGSL_TOK_KW_IF          },
    /* length 3 */
    { "for",           3, WGSL_TOK_KW_FOR         },
    { "let",           3, WGSL_TOK_KW_LET         },
    { "var",           3, WGSL_TOK_KW_VAR         },
    /* length 4 */
    { "case",          4, WGSL_TOK_KW_CASE        },
    { "else",          4, WGSL_TOK_KW_ELSE        },
    { "loop",          4, WGSL_TOK_KW_LOOP        },
    { "true",          4, WGSL_TOK_KW_TRUE        },
    /* length 5 */
    { "alias",         5, WGSL_TOK_KW_ALIAS       },
    { "break",         5, WGSL_TOK_KW_BREAK       },
    { "const",         5, WGSL_TOK_KW_CONST       },
    { "false",         5, WGSL_TOK_KW_FALSE       },
    { "while",         5, WGSL_TOK_KW_WHILE       },
    /* length 6 */
    { "enable",        6, WGSL_TOK_KW_ENABLE      },
    { "return",        6, WGSL_TOK_KW_RETURN      },
    { "struct",        6, WGSL_TOK_KW_STRUCT      },
    { "switch",        6, WGSL_TOK_KW_SWITCH      },
    /* length 7 */
    { "default",       7, WGSL_TOK_KW_DEFAULT     },
    { "discard",       7, WGSL_TOK_KW_DISCARD     },
    /* length 8 */
    { "continue",      8, WGSL_TOK_KW_CONTINUE    },
    { "override",      8, WGSL_TOK_KW_OVERRIDE    },
    { "requires",      8, WGSL_TOK_KW_REQUIRES    },
    /* length 10 */
    { "continuing",   10, WGSL_TOK_KW_CONTINUING  },
    { "diagnostic",   10, WGSL_TOK_KW_DIAGNOSTIC  },
    /* length 12 */
    { "const_assert", 12, WGSL_TOK_KW_CONST_ASSERT},
};

#define KW_COUNT ((int)(sizeof(KW_TABLE) / sizeof(KW_TABLE[0])))

WGSLTokenKind wgsl_token_classify_ident(const char *bytes, size_t length) {
    /* Lengths range 2..12; quick reject. */
    if (length < 2 || length > 12) return WGSL_TOK_IDENT;
    for (int i = 0; i < KW_COUNT; i++) {
        const KW *k = &KW_TABLE[i];
        if (k->length == (uint8_t)length && memcmp(k->text, bytes, length) == 0) {
            return k->kind;
        }
    }
    return WGSL_TOK_IDENT;
}
