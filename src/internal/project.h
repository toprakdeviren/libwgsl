/**
 * @file project.h — `wgslconfig.toml` loader.
 *
 * A `WGSLProject` is an in-memory snapshot of the project config:
 * which preamble files exist, which file-path globs they inject into.
 * It does NOT hold open file handles, does NOT read shader content,
 * does NOT cache compiled results.  The public API delegates file
 * IO to the embedder (so WASM consumers don't drag emscripten FS in
 * and native CLIs can hook up their own caching).
 *
 * Config schema (subset):
 *
 *   [preamble]
 *   file             = "00_shared.wgsl"            # single (legacy)
 *   files            = ["a.wgsl", "b.wgsl"]        # ordered list
 *   auto_inject_into = ["doublestar-slash glob (any depth, any *.wgsl)"]               # path globs
 *
 * `file` and `files` accumulate (file goes first when both present).
 * Path globs are evaluated against the path passed to
 * `wgsl_project_match`; the embedder is responsible for normalising
 * the path relative to the project root.
 */
#ifndef WGSL_INTERNAL_PROJECT_H
#define WGSL_INTERNAL_PROJECT_H

#include <stddef.h>

typedef struct WGSLProject WGSLProject;

/**
 * Parse a TOML config buffer.  On error returns `NULL` and writes a
 * single-line message into `err` (truncated to `err_len-1`).
 */
WGSLProject *wgsl_project_open_from_string(
    const char *toml_text, size_t toml_len,
    char *err, size_t err_len);

void wgsl_project_close(WGSLProject *p);

/**
 * Return the preamble files this project wants prepended to
 * `file_path`.  `file_path` is a forward-slash path relative to the
 * project root; if it matches any of `auto_inject_into`, we return
 * the full ordered preamble list.  Otherwise `*out_count = 0` and
 * the return value is `NULL`.
 *
 * Returned strings are owned by the project until `wgsl_project_close`.
 */
const char *const *wgsl_project_match(
    const WGSLProject *p, const char *file_path, int *out_count);

/**
 * Direct preamble-file accessors — for callers that want every
 * preamble regardless of inject globs (e.g. a CLI dumping the
 * effective project state).
 */
const char *const *wgsl_project_preamble_files(
    const WGSLProject *p, int *out_count);

#endif /* WGSL_INTERNAL_PROJECT_H */
