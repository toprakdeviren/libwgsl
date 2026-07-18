/**
 * @file toml.h — minimal TOML subset parser.
 *
 * Supported grammar (deliberately small — `wgslconfig.toml` doesn't
 * need more):
 *
 *   document   = (line)*
 *   line       = comment | section_header | assignment
 *   comment    = "#" .* '\n'
 *   section    = "[" key "]" '\n'
 *   assignment = key "=" value '\n'
 *   value      = string | array
 *   string     = '"' (escape | non-quote)* '"'
 *   escape     = '\' ('\\' | '"' | 'n' | 't' | 'r' | '0')
 *   array      = "[" ( string ("," string)* ","? )? "]"
 *   key        = [A-Za-z0-9_-]+
 *
 * No inline tables, no integers, no booleans, no datetimes, no
 * multi-line strings, no nested keys.  Add support when a real config
 * needs it; this keeps the parser at ~200 LOC.
 */
#ifndef WGSL_INTERNAL_TOML_H
#define WGSL_INTERNAL_TOML_H

#include <stddef.h>

typedef struct WGSLToml WGSLToml;

/**
 * Parse a UTF-8 TOML buffer.  On error returns `NULL` and writes a
 * single-line description into `err` (truncated to `err_len-1`).
 * Caller must `wgsl_toml_free` the result.
 */
WGSLToml *wgsl_toml_parse(const char *src, size_t len, char *err, size_t err_len);

void wgsl_toml_free(WGSLToml *t);

/**
 * Lookup a string value at `[section]` `key`.  Returns `NULL` when
 * the key is absent or its value is an array.  Pass an empty string
 * for `section` to look up root-level keys.
 */
const char *wgsl_toml_get_string(const WGSLToml *t, const char *section, const char *key);

/**
 * Lookup a string-array value at `[section]` `key`.  Returns `NULL`
 * when the key is absent or its value is a scalar; on success writes
 * the element count to `*out_count`.  The returned pointer is owned
 * by the `WGSLToml` and remains valid until `wgsl_toml_free`.
 */
const char *const *wgsl_toml_get_string_array(
    const WGSLToml *t, const char *section, const char *key, int *out_count);

#endif /* WGSL_INTERNAL_TOML_H */
