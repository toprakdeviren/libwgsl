/**
 * @file parser.c — WGSL recursive-descent + Pratt parser (umbrella).
 *
 * Implementation lives in:
 *   parser/helpers.c           — P/NL cursor + node construction
 *   parser/expressions_attrs.c — expressions + attributes
 *   parser/statements.c        — statements
 *   parser/decls_entry.c       — global decls + wgsl_parse()
 */
#include "internal/parser_priv.h"
