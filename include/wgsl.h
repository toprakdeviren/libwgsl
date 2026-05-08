/**
 * @file wgsl.h
 * @brief WGSL frontend — lex · parse · resolve · validate.  Consumes
 *        WGSL source and produces LSP-grade diagnostics, semantic
 *        tokens, and (when valid) a module summary.
 * @version 0.2.0
 *
 * "A frontend earns its keep when the compiler that comes after it can
 *  trust everything it gets."
 *
 * DESIGN PRINCIPLES
 *
 *   - One-shot API first (`wgsl_check` does parse + resolve + validate)
 *   - Always returns non-NULL — failure is on the result, not the call
 *   - Owned-output strings (no caller buffers, no length guessing)
 *   - Structured diagnostics with LSP-compatible source ranges
 *   - Loud rejects over silent fallbacks (every spec violation surfaces
 *     as a diagnostic; never silently accepted)
 *   - No hidden global state (each WGSLResult is self-contained;
 *     the library has one idempotent process-init for Unicode tables)
 *
 * QUICK START
 *
 *   #include <wgsl.h>
 *
 *   WGSLResult *r = wgsl_check(src);
 *   if (wgsl_ok(r)) {
 *       use(wgsl_module_json(r));   // entry points, bindings, layout
 *   } else {
 *       fprintf(stderr, "%s\n", wgsl_error(r));
 *   }
 *   wgsl_free(r);
 *
 * SECTIONS
 *
 *   1.  Process lifecycle    — one-time Unicode-table init
 *   2.  Check                — run the full frontend on source
 *   3.  Result               — inspect ok / module summary / spec-pin
 *   4.  Diagnostics          — structured errors with LSP positions
 *   5.  Semantic tokens      — token stream for syntax highlighters
 *   6.  Hover & navigation   — type-at-offset, definition-at-offset
 *   7.  Module summary       — `wgsl_module_json` shape (entry pts, …)
 *   8.  Project              — `wgslconfig.toml` preamble injection
 *
 * SPEC TARGET
 *
 *   W3C WGSL Recommendation, Candidate Recommendation Draft of
 *   2026-05-07.  Use `wgsl_spec_pin()` to confirm at runtime.
 */
#ifndef WGSL_H
#define WGSL_H

#define WGSL_VERSION_MAJOR 0
#define WGSL_VERSION_MINOR 2
#define WGSL_VERSION_PATCH 0
#define WGSL_VERSION_STRING "0.2.0"

/** @brief W3C WGSL spec version this library targets, e.g. "CRD-2026-05-07". */
#define WGSL_SPEC_PIN "CRD-2026-05-07"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Opaque result handle.  Allocated by `wgsl_check`,
 *         released with `wgsl_free`. */
typedef struct WGSLResult WGSLResult;

/* ┌──────────────────────────────────────────────────────────────────────────┐
 * │  1. PROCESS LIFECYCLE — one-time Unicode-table init                    │
 * └──────────────────────────────────────────────────────────────────────────┘
 *
 *  The library carries its own Unicode 17.0.0 tables (XID, NFC, blankspace).
 *  `wgsl_init()` loads them; `wgsl_check` calls it implicitly on first use,
 *  so explicit init is optional.  Both are thread-safe and idempotent.
 *  `wgsl_shutdown()` is provided for symmetry with hosting frameworks that
 *  want to leak-check at exit; it is not required for correctness.
 */

void wgsl_init(void);
void wgsl_shutdown(void);

/** @brief Returns the W3C WGSL spec pin string (compile-time constant). */
const char *wgsl_spec_pin(void);

/** @brief Returns the underlying Unicode Standard version, e.g. "17.0.0". */
const char *wgsl_unicode_version(void);

/* ┌──────────────────────────────────────────────────────────────────────────┐
 * │  2. CHECK — run the full frontend on source                            │
 * └──────────────────────────────────────────────────────────────────────────┘
 *
 *  `wgsl_check` runs the full pipeline: lex → parse → resolve → const-eval
 *  → type-check → validate.  It always returns a non-NULL `WGSLResult` that
 *  the caller owns.  Inspect `wgsl_ok` for the verdict and walk diagnostics
 *  (section 4) for details.
 *
 *  WGSL has no `#include`, no module imports; one source file is one module.
 *  There is therefore no `_file` / `_bundle` variant — callers slurp the
 *  source themselves and pass the bytes.
 */

/**
 * @brief Run the full WGSL frontend on a source string.
 * @param src  NUL-terminated WGSL source.  May be NULL (treated as empty).
 * @return     Non-NULL result.  Use `wgsl_ok()` to check the verdict.
 */
WGSLResult *wgsl_check(const char *src);

/**
 * @brief Run the WGSL frontend on a non-NUL-terminated byte range.
 * @param src    Pointer to source bytes.  May be NULL when `src_len == 0`.
 * @param src_len  Length in bytes.  Source need not be NUL-terminated.
 * @return       Non-NULL result.
 */
WGSLResult *wgsl_check_n(const char *src, size_t src_len);

/* ┌──────────────────────────────────────────────────────────────────────────┐
 * │  3. RESULT — inspect ok / module summary / spec pin                    │
 * └──────────────────────────────────────────────────────────────────────────┘
 *
 *  Strings returned by these accessors are owned by the WGSLResult and live
 *  until `wgsl_free()`.  Every accessor handles a NULL result gracefully —
 *  empty strings, zero counts, no crashes.
 */

/**
 * @brief Did the module pass the full check (lex + parse + resolve + validate)?
 * @return 1 on success, 0 if any error-severity diagnostic was raised.
 *         A 0 return guarantees `wgsl_diagnostic_count(r) >= 1` with at
 *         least one `WGSL_DIAG_ERROR` entry.
 */
int wgsl_ok(const WGSLResult *r);

/**
 * @brief First diagnostic message, formatted for human display.
 *
 * Convenience for `if (!wgsl_ok(r)) fprintf(stderr, "%s\n", wgsl_error(r));`.
 * Use the structured diagnostic API (section 4) for full access.
 *
 * @return  NUL-terminated string.  Empty when the module is valid.
 */
const char *wgsl_error(const WGSLResult *r);

/**
 * @brief Module summary as JSON: entry points, resource bindings,
 *        @group/@binding tuples, vertex/fragment/compute attributes,
 *        struct layouts, override IDs.
 *
 *        Empty string when the check failed.  The schema is documented
 *        in `docs/MODULE_JSON.md`.
 *
 * @return  NUL-terminated JSON.  Empty string on error.
 */
const char *wgsl_module_json(const WGSLResult *r);

/** @brief Byte length of the module JSON (excluding NUL). */
size_t wgsl_module_json_len(const WGSLResult *r);

/**
 * @brief Free all memory owned by the result.  Safe to call with NULL.
 */
void wgsl_free(WGSLResult *r);

/* ┌──────────────────────────────────────────────────────────────────────────┐
 * │  4. DIAGNOSTICS — structured errors with LSP positions                 │
 * └──────────────────────────────────────────────────────────────────────────┘
 *
 *  Every spec violation surfaces as a `WGSLDiagnostic` with a 1-based
 *  line/column range.  Editors can map these onto squiggly underlines;
 *  CLIs can pretty-print them.  Messages are owned by the WGSLResult.
 *
 *  Severity model matches the W3C WGSL spec §2.3: error / warning /
 *  info / hint.  The `rule` field carries the filterable rule name for
 *  diagnostics that the spec gates on `@diagnostic` (e.g.
 *  "derivative_uniformity").  For unconditional diagnostics, `rule` is
 *  the empty string.
 */

/** @brief Severity level.  Stable for serialization; do not reorder. */
typedef enum {
    WGSL_DIAG_ERROR   = 1,
    WGSL_DIAG_WARNING = 2,
    WGSL_DIAG_INFO    = 3,
    WGSL_DIAG_HINT    = 4,
} WGSLDiagSeverity;

/** @brief A single diagnostic.  Offsets are 1-based; end column is
 *         exclusive (matches LSP semantics, mirrors `metal/`). */
typedef struct {
    WGSLDiagSeverity severity;
    uint32_t         line;        /**< 1-based start line.            */
    uint32_t         column;      /**< 1-based start column.          */
    uint32_t         end_line;    /**< 1-based end line, inclusive.   */
    uint32_t         end_column;  /**< 1-based end column, exclusive. */
    const char      *message;     /**< Owned by WGSLResult.           */
    const char      *rule;        /**< Filterable rule name, or "".   */
} WGSLDiagnostic;

int wgsl_diagnostic_count(const WGSLResult *r);
const WGSLDiagnostic *wgsl_diagnostic(const WGSLResult *r, int index);

/* ┌──────────────────────────────────────────────────────────────────────────┐
 * │  5. SEMANTIC TOKENS — token stream for syntax highlighters             │
 * └──────────────────────────────────────────────────────────────────────────┘
 *
 *  `wgsl_lex` runs only the tokenizer; `wgsl_semantic_tokens` additionally
 *  walks the resolved AST to upgrade identifier tokens with their
 *  declarative role (struct, function, parameter, var, const, override,
 *  field, type-alias).  Both emit comment and whitespace tokens so clients
 *  can round-trip the full stream.
 *
 *  Output ownership: caller frees with `wgsl_lex_free`.  When parsing
 *  fails, `wgsl_semantic_tokens` falls back to `wgsl_lex`-only output —
 *  parse errors don't make editor highlighting fall over.
 */

/** @brief Lexical / semantic kind.  Stable for serialization. */
typedef enum {
    WGSL_LEX_UNKNOWN = 0,
    WGSL_LEX_IDENTIFIER,        /**< User-defined name, unclassified.       */
    WGSL_LEX_KEYWORD,           /**< `fn`, `let`, `var`, `if`, ...          */
    WGSL_LEX_TYPE,              /**< Predeclared type (`vec4f`, `u32`).     */
    WGSL_LEX_BUILTIN_FUNC,      /**< Predeclared function (`textureSample`).*/
    WGSL_LEX_BUILTIN_VALUE,     /**< Predeclared value (`true`, `false`).   */
    WGSL_LEX_NUMBER,            /**< Integer or float literal.              */
    WGSL_LEX_OPERATOR,          /**< Arithmetic / logical / bitwise.        */
    WGSL_LEX_PUNCT,             /**< `( ) { } [ ] , ; : ->`.                */
    WGSL_LEX_TEMPLATE_DELIM,    /**< `<` or `>` recognized as template arg. */
    WGSL_LEX_ATTRIBUTE,         /**< `@` introducer + attribute name.       */
    WGSL_LEX_DIRECTIVE,         /**< `enable`, `requires`, `diagnostic`.    */
    WGSL_LEX_COMMENT,           /**< Line `//…` or block `/ * … * /`.       */
    WGSL_LEX_WHITESPACE,        /**< Inter-token blankspace.                */
    /* — Resolver-enriched kinds (`wgsl_semantic_tokens` only) ───────────── */
    WGSL_LEX_STRUCT_NAME,       /**< Struct declaration or use.             */
    WGSL_LEX_FUNCTION_NAME,     /**< User-defined function decl or call.    */
    WGSL_LEX_PARAMETER,         /**< Function parameter.                    */
    WGSL_LEX_FIELD,             /**< Struct member name.                    */
    WGSL_LEX_VAR,               /**< Module/function-scope `var`.           */
    WGSL_LEX_LET,               /**< Function-scope `let`.                  */
    WGSL_LEX_CONST,             /**< `const`.                               */
    WGSL_LEX_OVERRIDE,          /**< `override`.                            */
    WGSL_LEX_TYPE_ALIAS,        /**< `alias`-defined type name.             */
} WGSLLexTokenKind;

/** @brief A single token.  `offset` and `length` are byte units. */
typedef struct {
    WGSLLexTokenKind kind;
    uint32_t         offset;   /**< Byte offset into source (0-based). */
    uint32_t         length;   /**< Byte length.                       */
    uint32_t         line;     /**< 1-based line number.               */
    uint32_t         column;   /**< 1-based column number.             */
} WGSLLexToken;

int wgsl_lex(const char *src, WGSLLexToken **out_tokens, int *out_count);
int wgsl_semantic_tokens(const char *src, WGSLLexToken **out_tokens, int *out_count);
void wgsl_lex_free(WGSLLexToken *tokens);

/* ┌──────────────────────────────────────────────────────────────────────────┐
 * │  6. HOVER & NAVIGATION — type-at-offset, definition-at-offset          │
 * └──────────────────────────────────────────────────────────────────────────┘
 *
 *  Both queries operate on a `WGSLResult` returned by `wgsl_check`, so the
 *  LSP server can answer N requests from one parse.  Offsets are 0-based
 *  byte offsets into the source string passed to `wgsl_check`.
 *
 *  `wgsl_hover_at` returns the resolved type and a short signature line
 *  for the symbol under the offset.  `wgsl_definition_at` returns the
 *  byte range of the declaration the symbol resolves to (in the same
 *  source — there's no cross-file resolution in WGSL v1).
 */

/** @brief Hover info at a source offset.  Strings are owned by the result. */
typedef struct {
    int          present;        /**< 0 if no symbol under offset.        */
    const char  *type_repr;      /**< Resolved type, e.g. "vec3<f32>".    */
    const char  *signature;      /**< For functions: signature line.      */
    const char  *doc;            /**< Spec-derived doc (builtins only).   */
    uint32_t     symbol_offset;  /**< Byte offset of the matched symbol.  */
    uint32_t     symbol_length;  /**< Byte length of the matched symbol.  */
} WGSLHover;

WGSLHover wgsl_hover_at(const WGSLResult *r, uint32_t offset);

/**
 * @brief Pointer-out variant of `wgsl_hover_at` — fills `*out` instead
 *        of returning the struct by value.  Functionally identical;
 *        provided so that bindings without ergonomic by-value-struct
 *        FFI (wasm32, Python ctypes, JNI, …) can read the result
 *        through plain memory loads.  Safe to call with `out == NULL`.
 */
void wgsl_hover_at_into(const WGSLResult *r, uint32_t offset, WGSLHover *out);

/** @brief Definition location.  Empty range when no def is found. */
typedef struct {
    int      present;     /**< 0 if no definition resolves.         */
    uint32_t offset;      /**< Byte offset of the declaration.      */
    uint32_t length;      /**< Byte length of the declaration name. */
} WGSLDefinition;

WGSLDefinition wgsl_definition_at(const WGSLResult *r, uint32_t offset);

/** @brief Pointer-out variant of `wgsl_definition_at`.  See
 *         `wgsl_hover_at_into` for rationale. */
void wgsl_definition_at_into(
    const WGSLResult *r, uint32_t offset, WGSLDefinition *out);

/* ─────────────────────────────────────────────────────────────────
 * 8.  Project — `wgslconfig.toml` preamble injection
 * ───────────────────────────────────────────────────────────────── */

/**
 * @brief Opaque project handle.  In-memory snapshot of a parsed
 *        `wgslconfig.toml`; holds preamble file paths and inject
 *        globs.  Does NOT open files itself — the embedder reads
 *        each preamble and feeds the bytes to
 *        `wgsl_check_with_preamble`.
 */
typedef struct WGSLProject WGSLProject;

/**
 * @brief Parse a `wgslconfig.toml` buffer.  Returns NULL on parse
 *        error and writes a single-line description into `err`.
 *
 * Schema (subset; expand as needed):
 *
 *     [preamble]
 *     file             = "00_shared.wgsl"             # optional, single
 *     files            = ["a.wgsl", "b.wgsl"]         # optional, ordered
 *     auto_inject_into = ["doublestar-slash glob (any depth, any *.wgsl)"]                # required for matches
 *
 * `file` and `files` accumulate (file first when both present);
 * `auto_inject_into` decides which target paths receive the
 * preamble.
 */
WGSLProject *wgsl_project_open_from_string(
    const char *toml_text, size_t toml_len,
    char *err, size_t err_len);

void wgsl_project_close(WGSLProject *p);

/**
 * @brief Return the preamble files this project wants prepended to
 *        `file_path`.  When the path doesn't match any inject glob,
 *        sets `*out_count = 0` and returns NULL.  Returned strings
 *        are owned by the project until `wgsl_project_close`.
 *
 * `file_path` should be forward-slash, relative to the project
 * root.  The matcher honours `*` (within a directory) and `**`
 * (across directories); no `?` / `[…]` ranges yet.
 */
const char *const *wgsl_project_match(
    const WGSLProject *p, const char *file_path, int *out_count);

/**
 * @brief Run the full pipeline on `preamble + src` (concatenated in
 *        that order).  Diagnostic line numbers are reported in the
 *        synthesised concatenation; the caller subtracts the
 *        preamble line count to map back to the original source.
 *
 * Convenience over `wgsl_check_n` — saves a heap copy when the
 * embedder already has separate preamble and source buffers.
 */
WGSLResult *wgsl_check_with_preamble(
    const char *preamble, size_t preamble_len,
    const char *src,      size_t src_len);

#ifndef WGSL_NO_FS
/**
 * @brief Open a `wgslconfig.toml` from disk.  Reads
 *        `<root_dir>/wgslconfig.toml`, parses it, and fills the
 *        project handle.  Returns NULL on read or parse error and
 *        writes a single-line description into `err`.
 *
 * Compiled out when the library is built with `-DWGSL_NO_FS`
 * (e.g. the WASM build, where the embedder does its own file IO).
 */
WGSLProject *wgsl_project_open(const char *root_dir, char *err, size_t err_len);

/**
 * @brief Filesystem-aware check: opens `file_path`, looks up the
 *        project's preamble for it, reads each preamble file from
 *        `<root_dir>/<preamble_path>`, concatenates, and runs the
 *        full pipeline.  Returns NULL only on a hard IO error.
 */
WGSLResult *wgsl_check_in_project(
    const WGSLProject *p, const char *root_dir, const char *file_path);
#endif

#ifdef __cplusplus
}
#endif

#endif /* WGSL_H */
