/**
 * @file validate_priv.h — internal-only sharing between validate*.c files.
 *
 * The validator is split across:
 *   validate/validate.c   — driver + cycle DFS infra + cycle passes
 *   validate/attrs.c      — §12 attribute conformance + builtin name table
 *   validate/io.c         — §13 entry-point IO shape
 *   validate/behavior.c   — §9.7 behavior-set analysis
 *   validate/access.c     — §14.5 op classification + §14.3 AS-stage gating +
 *                           §9.4.11 discard-stage reachability
 *   validate/layout.c     — §14.4 layouts + §13.3.2 resource bindings
 *   validate/uniformity.c — §15.2 uniformity analysis
 *
 * Functions / types declared here are private to the validator but
 * shared across the split files.  None of these are visible to consumers
 * of the validator — the public surface is `internal/validate.h`.
 */
#ifndef WGSL_INTERNAL_VALIDATE_PRIV_H
#define WGSL_INTERNAL_VALIDATE_PRIV_H

#include "internal/validate.h"

/* ── Shader-stage bitset ───────────────────────────────────────────── */

#define STAGE_VX (1u << 0)
#define STAGE_FR (1u << 1)
#define STAGE_CO (1u << 2)

/* ── Generic cycle DFS infra (validate.c) ──────────────────────────── */

typedef enum { COLOR_WHITE = 0, COLOR_GRAY, COLOR_BLACK } Color;

/* `enumerate_edges`: for the given `from` symbol, invoke `out_cb` once
 * per outgoing edge with the target symbol and a representative AST
 * node (used for diagnostic spans).  The DFS engine is opaque to the
 * concrete edge shape. */
typedef void (*EdgeCb)(
    WGSLValidator *v,
    const WGSLSymbol *to, const WGSLNode *at, void *ctx);

typedef void (*EdgeEnum)(
    WGSLValidator *v,
    const WGSLSymbol *from, EdgeCb cb, void *ctx);

int  wgsl_val_run_cycle_check(
    WGSLValidator *v,
    int (*pick)(const WGSLSymbol *),
    EdgeEnum edges, const char *label);

/* Call-graph edge enumerator: shared with access.c (analyze_static_access)
 * and uniformity.c (compute_fn_uniformity_reqs). */
void wgsl_val_enum_call_edges(
    WGSLValidator *v, const WGSLSymbol *from, EdgeCb cb, void *ctx);

/* ── Diagnostics (validate.c) ──────────────────────────────────────── */

void wgsl_val_error(WGSLValidator *v, const WGSLNode *at, const char *fmt, ...);
void wgsl_val_filterable(
    WGSLValidator *v, const WGSLNode *at,
    WGSLDiagSeverity severity, const char *rule,
    const char *fmt, ...);

/* ── Symbol helpers (validate.c) ───────────────────────────────────── */

int         wgsl_val_sym_index(const WGSLValidator *v, const WGSLSymbol *s);
WGSLSymbol *wgsl_val_find_symbol_for_ast(
    const WGSLValidator *v, const WGSLNode *ast);

/* ── Per-decl × stage access bitsets (access.c) ────────────────────── */

typedef struct {
    uint32_t *read_table;
    uint32_t *write_table;
} AccessTables;

/* ── §12 attribute helpers (attrs.c) ───────────────────────────────── */

/* Find an `@<name>` attribute inside `node->children[start..start+count)`.
 * Returns NULL if not found. */
WGSLNode *wgsl_val_find_attr(
    const WGSLValidator *v, const WGSLNode *node,
    uint32_t start, uint32_t count, const char *name);

/* Pull the (single) integer argument out of an attribute via const-eval.
 * Returns 0 on missing / non-int / non-const (caller can't distinguish). */
long long wgsl_val_attr_int_arg(
    const WGSLValidator *v, const WGSLNode *attr);

/* Like `wgsl_val_attr_int_arg`, but distinguishes "ok" from "missing /
 * non-int / non-const".  Returns 1 on success (writes to *out), 0
 * otherwise.  Consteval-bag noise is rolled back on failure. */
int wgsl_val_attr_int_arg_typed(
    WGSLValidator *v, const WGSLNode *attr, long long *out);

/* ── §13 builtin-name table (attrs.c) ──────────────────────────────── */

typedef enum {
    BT_NONE = 0,
    BT_U32,
    BT_I32,
    BT_F32,
    BT_BOOL,
    BT_VEC3_U32,
    BT_VEC4_F32,
    BT_ARRAY_F32_1_8,
} BuiltinTypeHint;

typedef struct {
    const char     *name;
    uint32_t        ext;        /* WGSLExtension bit or 0 (enable) */
    uint32_t        feat;       /* WGSLLanguageFeature bit or 0 (requires) */
    uint8_t         stage_mask; /* STAGE_VX | STAGE_FR | STAGE_CO */
    uint8_t         dir_in;     /* legal as entry-point input  */
    uint8_t         dir_out;    /* legal as entry-point output */
    BuiltinTypeHint type;
} BuiltinEntry;

const BuiltinEntry *wgsl_val_find_builtin_entry(
    const char *nm, uint32_t nl);

/* ── Per-pass driver hooks ─────────────────────────────────────────── */

/* Each split file owns one (or two) of these.  The validate.c driver
 * calls them in sequence after running the cycle passes. */
void          validate_attributes      (WGSLValidator *v, WGSLNode *tu); /* attrs.c    */
void          validate_behavior        (WGSLValidator *v, WGSLNode *tu); /* behavior.c */
void          validate_entry_points    (WGSLValidator *v, WGSLNode *tu); /* io.c       */
AccessTables  analyze_static_access    (WGSLValidator *v);               /* access.c   */
void          validate_address_spaces  (WGSLValidator *v, WGSLNode *tu,
                                        const uint32_t *read_table,
                                        const uint32_t *write_table);    /* access.c   */
void          validate_alias_analysis  (WGSLValidator *v);               /* access.c   */
void          validate_discard_stages  (WGSLValidator *v);               /* access.c   */
void          validate_layouts         (WGSLValidator *v, WGSLNode *tu); /* layout.c   */
void          validate_resource_bindings(WGSLValidator *v, WGSLNode *tu);/* layout.c   */
void          validate_uniformity      (WGSLValidator *v, WGSLNode *tu); /* uniformity.c */

#endif /* WGSL_INTERNAL_VALIDATE_PRIV_H */
