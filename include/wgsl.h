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
 * API SURFACE
 *
 *   Process lifecycle, ABI layout, checking, sessions, result access,
 *   diagnostics, token streams, hover/navigation, language-service
 *   queries, project preambles, optimization, MSL emission, ML-kernel
 *   analysis, golden-reference checks, and interpreter/oracle entry points.
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

/**
 * Interpreter JSON truncation / simulation caps (public so hosts and
 * bindings agree with the C side — also reported by `wgsl_abi_layout`).
 *
 *  - `WGSL_INTERP_STEP_CAP`          max SIMT scheduler steps before
 *                                    `truncated:true` (shared all-lane budget)
 *  - `WGSL_INTERP_BUFFER_VALUES_CAP` max elements dumped per buffer in
 *                                    the JSON `buffers[].values` array
 *  - `WGSL_INTERP_MAX_LANES`         max workgroup lanes simulated
 *                                    (∏ workgroup_size is clamped here)
 */
#define WGSL_INTERP_STEP_CAP          2000000u
#define WGSL_INTERP_BUFFER_VALUES_CAP 4096u
#define WGSL_INTERP_MAX_LANES         256u

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Shared-library export (Makefile uses -fvisibility=hidden). */
#if defined(__GNUC__) || defined(__clang__)
#  define WGSL_API __attribute__((visibility("default")))
#else
#  define WGSL_API
#endif


/** @brief Opaque result handle.  Allocated by `wgsl_check`,
 *         released with `wgsl_free`. */
typedef struct WGSLResult WGSLResult;

/* Process lifecycle.
 *
 *  The library carries its own Unicode 17.0.0 XID tables; UTF-8 decode and
 *  Pattern_White_Space are implemented locally to match WGSL's exact rules.
 *  `wgsl_init()` loads them; `wgsl_check` calls it implicitly on first use,
 *  so explicit init is optional.  Both are thread-safe and idempotent.
 *  `wgsl_shutdown()` is provided for symmetry with hosting frameworks that
 *  want to leak-check at exit; it is not required for correctness.
 */

WGSL_API void wgsl_init(void);
WGSL_API void wgsl_shutdown(void);

/** @brief Returns the W3C WGSL spec pin string (compile-time constant). */
WGSL_API const char *wgsl_spec_pin(void);

/** @brief Returns the underlying Unicode Standard version, e.g. "17.0.0". */
WGSL_API const char *wgsl_unicode_version(void);

/* ABI layout.
 *
 *  Foreign bindings (wasm32 JS, Python ctypes, Rust, JNI) must not hardcode
 *  public-struct field offsets — they differ between LP64 and ILP32.  Call
 *  `wgsl_abi_layout()` once after loading the library; every field below is a
 *  `uint32_t` so the layout table itself is pointer-size-agnostic and can be
 *  read with plain 4-byte loads starting at the returned pointer.
 *
 *  `abi_version` bumps when fields are added/reordered (bindings should
 *  refuse to run against an unexpected version).  Current version: 1.
 */

/** @brief Bumped when `WGSLAbiLayout` fields change. */
#define WGSL_ABI_LAYOUT_VERSION 1u

typedef struct WGSLAbiLayout {
    uint32_t abi_version;               /**< Always `WGSL_ABI_LAYOUT_VERSION`. */
    uint32_t ptr_size;                  /**< `sizeof(void *)` (4 or 8).        */

    uint32_t diagnostic_size;
    uint32_t diagnostic_severity;
    uint32_t diagnostic_line;
    uint32_t diagnostic_column;
    uint32_t diagnostic_end_line;
    uint32_t diagnostic_end_column;
    uint32_t diagnostic_message;
    uint32_t diagnostic_rule;

    uint32_t lex_token_size;
    uint32_t lex_token_kind;
    uint32_t lex_token_offset;
    uint32_t lex_token_length;
    uint32_t lex_token_line;
    uint32_t lex_token_column;

    uint32_t hover_size;
    uint32_t hover_present;
    uint32_t hover_type_repr;
    uint32_t hover_signature;
    uint32_t hover_doc;
    uint32_t hover_symbol_offset;
    uint32_t hover_symbol_length;

    uint32_t definition_size;
    uint32_t definition_present;
    uint32_t definition_offset;
    uint32_t definition_length;

    /* Interpreter caps (same values as the WGSL_INTERP_* macros). */
    uint32_t interp_step_cap;
    uint32_t interp_buffer_values_cap;
    uint32_t interp_max_lanes;
} WGSLAbiLayout;

/**
 * @brief Process-wide ABI layout table (static storage; never free).
 *        Safe to call before `wgsl_init`.
 */
WGSL_API const WGSLAbiLayout *wgsl_abi_layout(void);

/* Checking.
 *
 *  `wgsl_check` runs the full pipeline: lex → parse → resolve → const-eval
 *  → type-check → validate.  It returns a caller-owned `WGSLResult`; NULL is
 *  reserved for catastrophic allocation failure before an error result can be
 *  constructed.  Inspect `wgsl_ok` for the verdict and walk diagnostics
 *  (section 4) for details.
 *
 *  WGSL has no `#include`, no module imports; one source file is one module.
 *  There is therefore no `_file` / `_bundle` variant — callers slurp the
 *  source themselves and pass the bytes.
 *
 *  For repeated checks of an evolving buffer (editors / studio), prefer
 *  the session API (section 2b) — it content-addresses the full source
 *  and reports which top-level decls changed between checks.
 */

/**
 * @brief Run the full WGSL frontend on a source string.
 * @param src  NUL-terminated WGSL source.  May be NULL (treated as empty).
 * @return     Result handle, or NULL only on catastrophic allocation failure.
 *             Use `wgsl_ok()` to check the verdict.
 */
WGSL_API WGSLResult *wgsl_check(const char *src);

/**
 * @brief Run the WGSL frontend on a non-NUL-terminated byte range.
 * @param src    Pointer to source bytes.  May be NULL when `src_len == 0`.
 * @param src_len  Length in bytes.  Source need not be NUL-terminated.
 * @return       Result handle, or NULL only on catastrophic allocation failure.
 */
WGSL_API WGSLResult *wgsl_check_n(const char *src, size_t src_len);

/**
 * @brief Canonical pretty-print of WGSL source (parse → print).
 *        Layout is normalized (4-space indent, consistent spacing);
 *        identifiers/literals keep their original spelling.  Line and
 *        block comments are reattached in source order (leading before
 *        decls/stmts, same-line trailing when possible).
 *        Returns a malloc'd NUL-terminated string; free with
 *        `wgsl_free_string` (or `free`).  Empty string / NULL on failure.
 */
WGSL_API char *wgsl_format(const char *src);
WGSL_API char *wgsl_format_n(const char *src, size_t src_len);

/**
 * @brief Canonical pretty-print for the top-level unit(s) touched by a byte
 *        selection.  The returned text replaces `[*edit_start, *edit_end)`.
 *        When the selection is empty, the containing top-level unit is chosen.
 *        If no complete unit is touched, the selected bytes are copied back.
 *
 *        `edit_start` / `edit_end` are optional.  Free the returned string with
 *        `wgsl_free_string` (or `free`).
 */
WGSL_API char *wgsl_format_range(const char *src,
                                 uint32_t start_offset, uint32_t end_offset,
                                 uint32_t *edit_start, uint32_t *edit_end);
WGSL_API char *wgsl_format_range_n(const char *src, size_t src_len,
                                   uint32_t start_offset, uint32_t end_offset,
                                   uint32_t *edit_start, uint32_t *edit_end);

/**
 * @brief Pretty-print from an existing `WGSLResult` (reuses its AST).
 *        Returns NULL if `r` has no AST.  Free with `wgsl_free_string`.
 */
WGSL_API char *wgsl_format_result(const WGSLResult *r);

/* Sessions.
 *
 *  A session owns the last successful pipeline result for an evolving
 *  source buffer (editor document, studio shader).  Repeated checks of
 *  **byte-identical** source return the cached `WGSLResult` without
 *  re-running lex/parse/resolve/check/validate.
 *
 *  Between checks the session also fingerprints every top-level
 *  declaration (content hash of its source span) and a **module
 *  interface** hash (directives + non-function decls + function
 *  signatures excluding bodies).  Hosts use this to:
 *    - highlight which decls the edit touched
 *    - know when a body-only edit left the interface stable
 *      (`iface_changed == 0`) — the gate for per-function body
 *      re-typecheck (unchanged, diagnostic-free function bodies are
 *      skipped; hover lazily types a skipped body on demand).
 *    - inspect `typecheck_ms` / `skipped_nodes` to quantify the
 *      typecheck work avoided by the partial pass.
 *
 *  The result pointer is owned by the session: it is freed on the next
 *  `wgsl_session_check*` that misses the cache, or on
 *  `wgsl_session_destroy`.  Do **not** call `wgsl_free` on it.
 */

typedef struct WGSLSession WGSLSession;

/** @brief Snapshot of the last `wgsl_session_check*` decision. */
typedef struct {
    int      cache_hit;       /**< 1 = full pipeline skipped (byte-identical). */
    int      iface_changed;   /**< 1 = module interface hash differs. */
    int      decls_total;     /**< Top-level decls fingerprinted this check. */
    int      decls_changed;   /**< Decls whose content hash changed or are new. */
    int      decls_removed;   /**< Decls present last check, gone now. */
    int      fn_bodies_typed;   /**< Function bodies walked this check. */
    int      fn_bodies_skipped; /**< Function bodies skipped (partial TC). */
    int      skipped_nodes;     /**< AST nodes inside skipped function bodies. */
    uint32_t typecheck_ms;      /**< Wall time spent in the typecheck stage. */
    uint64_t src_hash;        /**< FNV-1a of full source bytes. */
    uint64_t iface_hash;      /**< FNV-1a of module interface material. */
} WGSLSessionInfo;

/** @brief Kind of a fingerprinted top-level unit. */
typedef enum {
    WGSL_SESSION_DECL_FUNCTION = 1,
    WGSL_SESSION_DECL_STRUCT,
    WGSL_SESSION_DECL_VAR,
    WGSL_SESSION_DECL_CONST,
    WGSL_SESSION_DECL_OVERRIDE,
    WGSL_SESSION_DECL_ALIAS,
    WGSL_SESSION_DECL_OTHER,
} WGSLSessionDeclKind;

WGSL_API WGSLSession *wgsl_session_new(void);
WGSL_API void         wgsl_session_destroy(WGSLSession *s);

/**
 * @brief Check `src` under the session.  Returns the session-owned
 *        result (never free with `wgsl_free`).  NULL only on
 *        catastrophic allocation failure.
 */
WGSL_API const WGSLResult *wgsl_session_check(WGSLSession *s, const char *src);
WGSL_API const WGSLResult *wgsl_session_check_n(WGSLSession *s, const char *src, size_t src_len);

/** @brief Info about the most recent session check.  Never NULL for a live session. */
WGSL_API const WGSLSessionInfo *wgsl_session_info(const WGSLSession *s);

/**
 * @brief Enumerate top-level decls that **changed** on the last check
 *        (content hash differs, or newly appeared).  Returns 1 and fills
 *        outputs when `index` is in range; 0 at end / on error.
 *        `name` points into the session's current source (valid until
 *        the next check that is not a cache hit).
 */
WGSL_API int wgsl_session_changed_decl(const WGSLSession *s, int index,
                              const char **name, uint32_t *name_len,
                              WGSLSessionDeclKind *kind);

/* Result access.
 *
 *  Strings returned by these accessors are owned by the WGSLResult and live
 *  until `wgsl_free()`.  Every accessor handles a NULL result gracefully —
 *  empty strings, zero counts, no crashes.
 */

/**
 * @brief Did the module pass the full check (lex + parse + resolve + validate)?
 * @return 1 on success, 0 if any error-severity diagnostic was raised.
 *         For a non-NULL result, a 0 return guarantees
 *         `wgsl_diagnostic_count(r) >= 1` with at least one
 *         `WGSL_DIAG_ERROR` entry.
 */
WGSL_API int wgsl_ok(const WGSLResult *r);

/**
 * @brief First diagnostic message, formatted for human display.
 *
 * Convenience for `if (!wgsl_ok(r)) fprintf(stderr, "%s\n", wgsl_error(r));`.
 * Use the structured diagnostic API (section 4) for full access.
 *
 * @return  NUL-terminated string.  Empty when the module is valid.
 */
WGSL_API const char *wgsl_error(const WGSLResult *r);

/**
 * @brief Module summary as JSON for WebGPU pipeline creation.
 *
 *        Per entry point: the bind-group layout it needs (`bindings` —
 *        the (group,binding) resources it transitively touches),
 *        used `features`, compute `workgroup_bytes`, synthesized
 *        `bind_group_layouts`, and IO layout (`inputs`/`outputs` with
 *        @location/@builtin/@interpolate).  Compute `workgroup_size`
 *        dimensions are JSON numbers when const-known, or source strings
 *        for pipeline-time override expressions.  Top-level `pipelines[]`
 *        cross-entry synthesis (compute + inferred render pairs with
 *        unioned visibility, `vertex_attributes`, `color_targets`).
 *        Per resource: address space / access (incl. texture & sampler
 *        `handle` sub-records) and AS-correct buffer `size`/`layout`.
 *        Plus `structs`, `overrides`, and mirrored
 *        `enable`/`requires`/`diagnostics` directives.
 *
 *        Empty string when the check failed.  Full schema:
 *        `docs/libwgsl.md`.
 *
 * @return  NUL-terminated JSON.  Empty string on error.
 */
WGSL_API const char *wgsl_module_json(const WGSLResult *r);

/** @brief Byte length of the module JSON (excluding NUL). */
WGSL_API size_t wgsl_module_json_len(const WGSLResult *r);

/**
 * @brief Free all memory owned by the result.  Safe to call with NULL.
 */
WGSL_API void wgsl_free(WGSLResult *r);

/* Diagnostics.
 *
 *  Every spec violation surfaces as a `WGSLDiagnostic` with a 1-based
 *  line/column range.  Columns are in UTF-16 code units (LSP's default
 *  position encoding), so editor squiggles stay aligned on lines that
 *  contain non-ASCII text.  Messages are owned by the WGSLResult.
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

WGSL_API int wgsl_diagnostic_count(const WGSLResult *r);
WGSL_API const WGSLDiagnostic *wgsl_diagnostic(const WGSLResult *r, int index);

/* Semantic tokens.
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
    /* Resolver-enriched kinds (`wgsl_semantic_tokens` only). */
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

/** @brief A single token.  `offset` and `length` are byte units;
 *         `column` is UTF-16 code units (LSP default), matching
 *         `WGSLDiagnostic.column`. */
typedef struct {
    WGSLLexTokenKind kind;
    uint32_t         offset;   /**< Byte offset into source (0-based). */
    uint32_t         length;   /**< Byte length.                       */
    uint32_t         line;     /**< 1-based line number.               */
    uint32_t         column;   /**< 1-based UTF-16 column (LSP).       */
} WGSLLexToken;

WGSL_API int wgsl_lex(const char *src, WGSLLexToken **out_tokens, int *out_count);
WGSL_API int wgsl_semantic_tokens(const char *src, WGSLLexToken **out_tokens, int *out_count);
WGSL_API void wgsl_lex_free(WGSLLexToken *tokens);

/* Hover and navigation.
 *
 *  Both queries operate on a `WGSLResult` returned by `wgsl_check`, so the
 *  LSP server can answer N requests from one parse.  Offsets are 0-based
 *  byte offsets into the source string passed to `wgsl_check`.
 *
 *  `wgsl_hover_at` returns the resolved type and the matched source span
 *  for the symbol under the offset.  The `signature` / `doc` string fields
 *  are ABI-reserved for richer hover text and are currently the empty
 *  string.  `wgsl_definition_at` returns the byte range of the declaration
 *  the symbol resolves to (in the same source — there's no cross-file
 *  resolution in WGSL v1).
 */

/** @brief Hover info at a source offset.  Strings are owned by the result. */
typedef struct {
    int          present;        /**< 0 if no symbol under offset.        */
    const char  *type_repr;      /**< Resolved type, e.g. "vec3<f32>".    */
    const char  *signature;      /**< Reserved; currently "".             */
    const char  *doc;            /**< Reserved; currently "".             */
    uint32_t     symbol_offset;  /**< Byte offset of the matched symbol.  */
    uint32_t     symbol_length;  /**< Byte length of the matched symbol.  */
} WGSLHover;

WGSL_API WGSLHover wgsl_hover_at(const WGSLResult *r, uint32_t offset);

/**
 * @brief Pointer-out variant of `wgsl_hover_at` — fills `*out` instead
 *        of returning the struct by value.  Functionally identical;
 *        provided so that bindings without ergonomic by-value-struct
 *        FFI (wasm32, Python ctypes, JNI, …) can read the result
 *        through plain memory loads.  Safe to call with `out == NULL`.
 */
WGSL_API void wgsl_hover_at_into(const WGSLResult *r, uint32_t offset, WGSLHover *out);

/** @brief Definition location.  Empty range when no def is found. */
typedef struct {
    int      present;     /**< 0 if no definition resolves.         */
    uint32_t offset;      /**< Byte offset of the declaration.      */
    uint32_t length;      /**< Byte length of the declaration name. */
} WGSLDefinition;

WGSL_API WGSLDefinition wgsl_definition_at(const WGSLResult *r, uint32_t offset);

/** @brief Pointer-out variant of `wgsl_definition_at`.  See
 *         `wgsl_hover_at_into` for rationale. */
WGSL_API void wgsl_definition_at_into(
    const WGSLResult *r, uint32_t offset, WGSLDefinition *out);

/* Language-service queries.
 *
 *  Primitives an LSP / editor shell needs beyond hover and goto-def.
 *  All operate on a `WGSLResult` from `wgsl_check` (byte offsets into that
 *  source).  Array outputs are malloc'd; free with the matching `*_free`.
 *  A stdio JSON-RPC shell is shipped as `make lsp` → `.build/wgsl_lsp`.
 */

/** @brief Outline / document-symbol kind (LSP SymbolKind subset). */
typedef enum {
    WGSL_OUTLINE_FUNCTION  = 1,
    WGSL_OUTLINE_STRUCT    = 2,
    WGSL_OUTLINE_VARIABLE  = 3,  /**< module `var` / function `var`/`let`. */
    WGSL_OUTLINE_CONSTANT  = 4,  /**< `const`. */
    WGSL_OUTLINE_OVERRIDE  = 5,
    WGSL_OUTLINE_TYPE_ALIAS = 6,
    WGSL_OUTLINE_FIELD     = 7,
    WGSL_OUTLINE_PARAMETER = 8,
} WGSLOutlineKind;

/** @brief One document-symbol row.  Name/range are byte spans into source. */
typedef struct {
    WGSLOutlineKind kind;
    uint32_t        name_offset;
    uint32_t        name_length;
    uint32_t        range_offset;   /**< Full declaration span. */
    uint32_t        range_length;
    int32_t         parent;         /**< Index into the same array, or -1. */
} WGSLOutlineSymbol;

/**
 * @brief Document symbols (outline): module decls + struct fields +
 *        function params.  Returns 1 and sets `*out`/`*out_count`
 *        (0 count → `*out` may be NULL).  Free with `wgsl_outline_free`.
 */
WGSL_API int wgsl_document_symbols(const WGSLResult *r,
                          WGSLOutlineSymbol **out, int *out_count);
WGSL_API void wgsl_outline_free(WGSLOutlineSymbol *syms);

/** @brief One reference to a symbol (use site or definition name). */
typedef struct {
    uint32_t offset;
    uint32_t length;
    int      is_definition;   /**< 1 for the declaration name span. */
} WGSLReference;

/**
 * @brief All references to the user symbol under `offset` (definition
 *        name + every resolved use).  Predeclared symbols → 0 refs.
 *        Free with `wgsl_references_free`.
 */
WGSL_API int wgsl_references_at(const WGSLResult *r, uint32_t offset,
                       WGSLReference **out, int *out_count);
WGSL_API void wgsl_references_free(WGSLReference *refs);

/**
 * @brief 1 if the symbol at `offset` is a user decl that can be renamed;
 *        fills the declaration-name span.  Hosts apply renames by
 *        rewriting every span from `wgsl_references_at`.
 */
WGSL_API int wgsl_prepare_rename(const WGSLResult *r, uint32_t offset,
                        uint32_t *name_offset, uint32_t *name_length);

/** @brief Foldable range (byte offsets; end exclusive). */
typedef struct {
    uint32_t start_offset;
    uint32_t end_offset;
} WGSLFoldingRange;

/**
 * @brief Folding ranges for functions, structs, compounds, and control
 *        structures.  Free with `wgsl_folding_free`.
 */
WGSL_API int wgsl_folding_ranges(const WGSLResult *r,
                        WGSLFoldingRange **out, int *out_count);
WGSL_API void wgsl_folding_free(WGSLFoldingRange *ranges);

/** @brief Completion item kind. */
typedef enum {
    WGSL_COMPLETION_KEYWORD  = 1,
    WGSL_COMPLETION_TYPE     = 2,
    WGSL_COMPLETION_FUNCTION = 3,
    WGSL_COMPLETION_VARIABLE = 4,
    WGSL_COMPLETION_FIELD    = 5,
    WGSL_COMPLETION_VALUE    = 6,
} WGSLCompletionKind;

/**
 * @brief One completion.  `label` and `detail` are heap-owned strings
 *        (possibly empty `detail`); free the array with
 *        `wgsl_completions_free(items, count)`.
 */
typedef struct {
    WGSLCompletionKind kind;
    char              *label;
    char              *detail;   /**< Type / signature hint; never NULL. */
} WGSLCompletionItem;

/**
 * @brief Completions at `offset`: keywords, predeclared types/fns,
 *        module symbols, and locals of the enclosing function (params
 *        and `let`/`var`/`const` declared before the cursor).  When the
 *        cursor sits on/after an identifier prefix, results are filtered
 *        to that prefix (case-sensitive).  Free with
 *        `wgsl_completions_free`.
 */
WGSL_API int wgsl_completions_at(const WGSLResult *r, uint32_t offset,
                        WGSLCompletionItem **out, int *out_count);
WGSL_API void wgsl_completions_free(WGSLCompletionItem *items, int count);

/**
 * @brief Signature help at a call site.  `label` is arena-owned (lives
 *        until `wgsl_free`); empty strings when `present == 0`.
 */
typedef struct {
    int          present;
    const char  *label;              /**< e.g. "add(a: i32, b: i32) -> i32". */
    uint32_t     active_parameter;   /**< 0-based index of the arg under cursor. */
    uint32_t     parameter_count;
} WGSLSignatureHelp;

WGSL_API WGSLSignatureHelp wgsl_signature_help_at(const WGSLResult *r, uint32_t offset);
WGSL_API void wgsl_signature_help_at_into(
    const WGSLResult *r, uint32_t offset, WGSLSignatureHelp *out);

/**
 * @brief One text edit (byte offsets into the checked source; end exclusive).
 *        `new_text` is heap-owned.
 */
typedef struct {
    uint32_t start_offset;
    uint32_t end_offset;
    char    *new_text;
} WGSLTextEdit;

/**
 * @brief One code action / quickfix.  Strings and edits are heap-owned;
 *        free the array with `wgsl_code_actions_free(actions, count)`.
 */
typedef struct {
    char         *title;       /**< Human label, e.g. "Add 'enable f16;'". */
    char         *kind;        /**< Stable slug, e.g. "quickfix.enable.f16". */
    WGSLTextEdit *edits;
    int           edit_count;
} WGSLCodeAction;

/**
 * @brief Suggested fixes derived from diagnostics / IO shape on `r`.
 *
 * Emits:
 *   - **enable-extension** quickfixes for messages matching
 *     `requires 'enable NAME;'` (f16, subgroups, …)
 *   - **@location(N)** inserts on vertex/fragment entry-point params
 *     and non-struct returns that lack `@location` / `@builtin`
 *   - **@align** power-of-two rewrites on struct members
 *   - **quickfix.opt.apply** when `wgsl_optimize_apply` can safely rewrite
 *     the checked source (whole-document edit)
 *
 * Free with `wgsl_code_actions_free`.
 */
WGSL_API int wgsl_code_actions(const WGSLResult *r,
                      WGSLCodeAction **out, int *out_count);
WGSL_API void wgsl_code_actions_free(WGSLCodeAction *actions, int count);

/* Project preamble injection. */

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
WGSL_API WGSLProject *wgsl_project_open_from_string(
    const char *toml_text, size_t toml_len,
    char *err, size_t err_len);

WGSL_API void wgsl_project_close(WGSLProject *p);

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
WGSL_API const char *const *wgsl_project_match(
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
WGSL_API WGSLResult *wgsl_check_with_preamble(
    const char *preamble, size_t preamble_len,
    const char *src,      size_t src_len);

/* Optimization analysis.
 *
 *  Analysis-only passes over a checked module (no IR yet):
 *    dce           — statements after return/discard
 *    unused        — unread locals; uncalled non-entry functions
 *    const_branch  — if (true|false) with a literal condition
 *    cse           — repeated pure expression spellings in a function
 *    loop          — for/loop hints (const trip counts, fuse candidates)
 *
 *  `wgsl_optimize_apply` applies the safe source rewrite pipeline:
 *    unreachable-after-terminal deletion, unused pure local / uncalled
 *    non-entry-function deletion, literal-bool branch simplification, and
 *    repeated pure binary-expression CSE via a generated `let`.
 */

/**
 * @brief JSON optimization report for an existing check result.
 *        Free with `wgsl_free_string`.  Shape:
 *        `{ "passes":[...], "summary":{...},
 *           "findings":[{"pass","kind","message","offset","length",
 *                        "savings_hint"}...] }`.
 */
WGSL_API char *wgsl_optimize_json(const WGSLResult *r);

/** @brief Check `src` then analyze.  Free with `wgsl_free_string`. */
WGSL_API char *wgsl_optimize_json_src(const char *src);

/**
 * @brief Apply safe DCE rewrites (unreachable after return/discard).
 *        Returns a new source string (malloc); free with `wgsl_free_string`.
 *        If nothing applies, returns a copy of `src`.
 */
WGSL_API char *wgsl_optimize_apply(const char *src);

/* Metal Shading Language emission.
 *
 *  Emits MSL text from WGSL source / a checked result.  Type and
 *  attribute maps reverse the tables from the Metal→WGSL sibling
 *  project (no link dependency).
 *
 *  MSL structured-diagnostic contract: best-effort MSL may still be
 *  produced, but unsupported / partial constructs are reported as *typed* codes on
 *  `WGSLMslEmit` — never rely on message text for control flow
 *  (see ARCHITECTURE).  `incomplete != 0` means "do not treat as
 *  fully valid MSL."
 */

/** @brief Machine-readable MSL emit diagnostic codes. */
typedef enum {
    WGSL_MSL_DIAG_NONE = 0,
    WGSL_MSL_DIAG_UNSUPPORTED_STMT = 1,   /* for/switch/… not emitted */
    WGSL_MSL_DIAG_UNSUPPORTED_EXPR = 2,
    WGSL_MSL_DIAG_UNSUPPORTED_TYPE = 3,   /* texture/… unsupported gap */
    WGSL_MSL_DIAG_PARTIAL_IO       = 4,   /* partial entry-IO lowering */
    WGSL_MSL_DIAG_SHADOWED_REWRITE = 5,   /* stage_in rewrite unsafe */
} WGSLMslDiagCode;

/**
 * @brief One structured MSL emit finding.
 *        `code` is the control-flow signal; `message` is human-only.
 */
typedef struct {
    int      code;       /* WGSLMslDiagCode */
    uint32_t offset;     /* byte offset into source (0 if N/A) */
    uint32_t length;     /* byte length */
    const char *message; /* owned by WGSLMslEmit; do not free */
} WGSLMslDiag;

typedef struct WGSLMslEmit WGSLMslEmit;

/** @brief Check + emit with structured diags. Free with `wgsl_msl_emit_free`. */
WGSL_API WGSLMslEmit *wgsl_msl_emit(const char *src);
WGSL_API WGSLMslEmit *wgsl_msl_emit_n(const char *src, size_t src_len);
/** @brief Emit from an existing pipeline result. */
WGSL_API WGSLMslEmit *wgsl_msl_emit_result(const WGSLResult *r);
WGSL_API void wgsl_msl_emit_free(WGSLMslEmit *e);

/** @brief Best-effort MSL text (empty string if none). Owned by emit. */
WGSL_API const char *wgsl_msl_emit_text(const WGSLMslEmit *e);
/** @brief 1 iff no incomplete/unsupported findings. */
WGSL_API int wgsl_msl_emit_ok(const WGSLMslEmit *e);
/** @brief 1 iff any finding marks the emit incomplete. */
WGSL_API int wgsl_msl_emit_incomplete(const WGSLMslEmit *e);
WGSL_API int wgsl_msl_emit_diag_count(const WGSLMslEmit *e);
WGSL_API const WGSLMslDiag *wgsl_msl_emit_diag(const WGSLMslEmit *e, int index);

/**
 * @brief Check + emit MSL string only (legacy). Free with `wgsl_free_string`.
 *        Does not surface incompleteness — prefer `wgsl_msl_emit_*`.
 *        NULL only on allocation failure.
 */
WGSL_API char *wgsl_to_msl(const char *src);
WGSL_API char *wgsl_to_msl_n(const char *src, size_t src_len);

/** @brief Emit MSL from an existing pipeline result (legacy string only). */
WGSL_API char *wgsl_to_msl_result(const WGSLResult *r);

/* ML-kernel analysis.
 *
 *  Heuristic analysis for compute kernels used in ML workbenches:
 *  binding layout, shape hints, pattern tags (softmax, attention, …),
 *  `shape_flow` inference for fixed storage array-of-vector tensors
 *  (for example attention `seq_len` / `head_dim` / `softmax_axis` and
 *  matmul `m` / `n` / `k` / `broadcast`), plus a versioned autotune
 *  protocol over workgroup size candidates.  Free with `wgsl_free_string`.
 */

WGSL_API char *wgsl_ml_analyze_json(const WGSLResult *r);
WGSL_API char *wgsl_ml_analyze_json_src(const char *src);

/* Golden-reference checks.
 *
 *  Run `wgsl_interp` against expected buffer needles + determinism.
 *  Used by the in-tree golden suite; embedders can pin semantics the
 *  same way.  `expect_ok`: 1 require ok, 0 require fail, -1 ignore.
 *  `needles` is a NULL-terminated list of JSON substrings (may be NULL).
 */

WGSL_API int wgsl_golden_run(const char *label,
                    const char *src, const char *entry,
                    unsigned len,
                    int expect_ok,
                    const char *const *needles);

/* Debug oracle. */

/**
 * @brief Run the deterministic N-lane interpreter and return CI-oriented
 *        correctness JSON.
 *
 * This API is a CI-oriented correctness oracle rather than an interactive
 * step debugger.  It fails closed on shared/storage
 * data races, non-finite numeric anomalies, interpreter warnings, and execution
 * truncation.  The returned JSON includes:
 *
 * `{ "ok":bool, "verdict":"pass"|"fail",
 *    "wedge":"correctness-oracle+explainer",
 *    "issues":[{"kind":str,"severity":"error","line":u?,
 *               "summary":str,"why":str,"ci":"fail"}...],
 *    "summary":{"issue_count":u,"race_count":u,"nan":u,"inf":u,
 *               "truncated":bool,"race_truncated":bool},
 *    "why":str, "source":"wgsl_interp" }`
 *
 * Parameters match `wgsl_interp`; free the result with `wgsl_free_string`.
 */
WGSL_API char *wgsl_debug_oracle(const char *src, const char *entry,
                    unsigned gx, unsigned gy, unsigned gz,
                    unsigned default_len, const char *seeds);

/* Invocation inspector. */

/**
 * @brief Interpret a workgroup containing ONE focus invocation on the CPU
 *        and return a JSON execution trace.
 *
 * Runs the full frontend on `src`, then executes the entry point `entry`
 * as an N-lane SIMT bytecode VM.  `gx,gy,gz` seed the *focus* invocation's
 * `@builtin(global_invocation_id)` (and related builtins).  Every module-
 * scope `var` in storage / uniform / workgroup / private is auto-discovered
 * and zero-seeded; runtime-sized arrays get `default_len` elements.
 *
 * Arithmetic follows runtime IEEE-754 semantics (a float divide by zero
 * yields ±Inf/NaN rather than a shader-creation error) so numerical
 * anomalies can be surfaced where they are produced.
 *
 * `gx,gy,gz` name the *focus* invocation's global id — the one whose trace and
 * per-invocation cost are reported.  Its whole workgroup (N = ∏ @workgroup_size,
 * capped at 256) is simulated as N SIMT lanes so cooperative ops (subgroup/quad),
 * barriers, and shared `var<workgroup>`/`var<storage>` memory are correct.
 *
 * @return A malloc'd, NUL-terminated JSON string (caller frees with
 *         `wgsl_free_string`).  Shape:
 *         `{ "ok":bool, "entry":str, "gid":[u,u,u],
 *            "workgroup_size":[u,u,u], "lanes":u,
 *            "steps":[{"line":u,"op":str,"name":str,"value":str,
 *                      "anomaly":str?,"active":u,"epoch":u}...], // active =
 *                      // lanes in lockstep; epoch = barrier segment;
 *                      // op "barrier" marks an epoch boundary
 *            "buffers":[{"name":str,"type":str,"values":[str...]}...],
 *            "anomalies":{"nan":u,"inf":u},
 *            "cost":{"flops":u,"iops":u,"bytes_loaded":u,"bytes_stored":u,
 *                    "mix":{"fadd":u,"fmul":u,"fdiv":u,"ftrans":u,"iadd":u,
 *                           "imul":u,"idiv":u,"ishift":u,"cmp":u}},  // op mix
 *            "roofline":{"bytes":u,"flop_intensity":f,"op_intensity":f,
 *                        "f16_flops":u,"packed_ops":u,"effective_flops":f,
 *                        "tensor_path":bool,   // f16 2x + packed-dot fast path
 *                        "presets":[{"name":str,"peak_gflops":f,
 *                                    "bandwidth_gbps":f,"ridge":f,
 *                                    "attainable_gflops":f,
 *                                    "bound":"memory"|"compute"}...]},
 *            "memory":{"dram_theoretical_bytes":u,"dram_effective_bytes":u,
 *                      "coalescing":f,          // 1.0 = fully coalesced
 *                      "shared_accesses":u,"bank_conflicts":u},
 *            "cost_band":{"method":"interval"|"interval+sampled", // seedless band
 *                         "flops":{"min":u,"max":u},"iops":{...},"bytes":{...},
 *                         "intensity":{"min":f,"max":f},
 *                         "data_dependent_branches":u,
 *                         "savings_hint":u, // rough optimization priority
 *                         "loops":[{"line":u,"trip_min":u,"trip_max":u,
 *                                   "bounded":bool}...]},
 *            "races":{"count":u,   // shared-cell data races
 *                     "truncated":bool?, // true if cell table hit internal cap
 *                     "samples":[{"line":u,
 *                                 "kind":"write-write"|"read-after-write"
 *                                       |"write-after-read"}...]},
 *            "divergence":{"warp_steps":u,"lane_steps":u,
 *                          "occupancy":f,"tax":f,   // SIMT efficiency & waste
 *                          "by_line":[{"line":u,"steps":u,"avg_active":f,
 *                                      "min_active":u,"max_active":u}...]},
 *            "return":str?, "steps_executed":u, "truncated":bool,
 *            "warnings":[{"line":u,"message":str}...], "error":str? }`.
 *         `ok` is false when `warnings` is non-empty (constructs not modelled —
 *         cooperative ops inside helper functions, unimplemented builtins) or
 *         the run truncated.  Cost (flops/iops/bytes/mix/f16/packed),
 *         anomalies, and the step trace are the focus lane's alone (peer-lane
 *         cost is rolled back); buffers reflect whole-workgroup shared memory.
 *
 * Truncation / simulation caps (also in `wgsl_abi_layout()` / macros above):
 *   - scheduler steps > `WGSL_INTERP_STEP_CAP` → `truncated:true`, `ok:false`
 *   - buffer dump capped at `WGSL_INTERP_BUFFER_VALUES_CAP` elements
 *   - workgroup lanes clamped to `WGSL_INTERP_MAX_LANES` (warning emitted)
 *
 *         Never NULL except on allocation failure.
 */
WGSL_API char *wgsl_interp(const char *src, const char *entry,
                  unsigned gx, unsigned gy, unsigned gz,
                  unsigned default_len, const char *seeds);

/** @brief Release a string returned by `wgsl_interp`. */
WGSL_API void wgsl_free_string(char *s);

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
WGSL_API WGSLProject *wgsl_project_open(const char *root_dir, char *err, size_t err_len);

/**
 * @brief Filesystem-aware check: opens `file_path`, looks up the
 *        project's preamble for it, reads each preamble file from
 *        `<root_dir>/<preamble_path>`, concatenates, and runs the
 *        full pipeline.  Returns NULL only on a hard IO error.
 */
WGSL_API WGSLResult *wgsl_check_in_project(
    const WGSLProject *p, const char *root_dir, const char *file_path);
#endif

#ifdef __cplusplus
}
#endif

#endif /* WGSL_H */
