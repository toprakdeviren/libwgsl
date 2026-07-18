/**
 * @file queries_project.c — public API module (was wgsl .inc fragment) (queries_project.inc).
 * Public API implementation; see include/wgsl.h.
 */
#include "wgsl.h"
#include "internal/result.h"
#include "internal/api_priv.h"
#include "internal/check.h"
#include "internal/resolver.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal/project.h"

/* Hover / goto-def. */

/* Find the smallest EXPR_IDENT span that contains `offset`.  Returns
 * NULL when no ident covers the offset. */
const WGSLNode *wgsl_api_find_ident_at(const WGSLNode *n, uint32_t offset, int depth) {
    if (!n) return NULL;
    /* Bound recursion: a pathologically deep AST (e.g. a long `a+a+…`
     * chain) reached through the public wgsl_hover_at / wgsl_definition_at
     * would otherwise overflow the stack. */
    if (depth > WGSL_MAX_AST_DEPTH) return NULL;
    if (offset < n->span_offset ||
        offset >= n->span_offset + n->span_length)
    {
        return NULL;
    }
    /* Recurse into children first — the deepest hit is the smallest. */
    for (uint32_t i = 0; i < n->child_count; i++) {
        const WGSLNode *hit = wgsl_api_find_ident_at(n->children[i], offset, depth + 1);
        if (hit) return hit;
    }
    if (n->kind == WGSL_NODE_EXPR_IDENT) return n;
    return NULL;
}

static int node_is_bound_decl(WGSLNodeKind k) {
    switch (k) {
    case WGSL_NODE_DECL_FUNCTION:
    case WGSL_NODE_DECL_PARAM:
    case WGSL_NODE_DECL_VAR:
    case WGSL_NODE_DECL_LET:
    case WGSL_NODE_DECL_CONST:
    case WGSL_NODE_DECL_OVERRIDE:
    case WGSL_NODE_DECL_STRUCT:
    case WGSL_NODE_DECL_ALIAS:
        return 1;
    default:
        return 0;
    }
}

static const WGSLNode *find_decl_node_at(
    const WGSLNode *n, uint32_t offset, int depth)
{
    if (!n) return NULL;
    if (depth > WGSL_MAX_AST_DEPTH) return NULL;
    if (offset < n->span_offset ||
        offset >= n->span_offset + n->span_length)
    {
        return NULL;
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        const WGSLNode *hit = find_decl_node_at(n->children[i], offset, depth + 1);
        if (hit) return hit;
    }
    if (!node_is_bound_decl((WGSLNodeKind)n->kind)) return NULL;
    uint32_t no = wgsl_node_name_span(n).offset;
    uint32_t nl = wgsl_node_name_span(n).length;
    if (nl == 0) return NULL;
    return (offset >= no && offset < no + nl) ? n : NULL;
}

/* Find a user-declared symbol whose decl-name span contains `offset`.
 * Decl-name spans live in `payload[0] = (off | len << 32)` for every
 * kind the resolver binds. */
WGSLSymbol *wgsl_api_find_decl_at(
    const WGSLResolver *res, const WGSLNode *root, uint32_t offset)
{
    if (!res || !root) return NULL;
    const WGSLNode *decl = find_decl_node_at(root, offset, 0);
    return wgsl_resolver_symbol_for_decl(res, decl);
}

/* Format a TypeInfo into an arena-owned NUL-terminated string. */
static const char *fmt_type(WGSLArena *arena, WGSLTypeInfo *t) {
    if (!t) return "";
    size_t need = wgsl_type_format(t, NULL, 0);
    char *buf = (char *)wgsl_arena_alloc(arena, need + 1);
    if (!buf) return "";
    wgsl_type_format(t, buf, need + 1);
    return buf;
}

/* Empty `WGSLHover` value with non-NULL string fields so callers can
 * dereference unconditionally. */
static WGSLHover empty_hover(void) {
    WGSLHover h = {0};
    h.type_repr = "";
    h.signature = "";
    h.doc       = "";
    return h;
}

/* Partial typecheck: if a local symbol was left untyped because its function
 * body was skipped, lazily type that body so hover still works. */
static void hover_ensure_body_typed(const WGSLResult *r, WGSLSymbol *sym) {
    if (!r || !sym || sym->type) return;
    if (sym->kind != WGSL_SYM_LET && sym->kind != WGSL_SYM_VAR &&
        sym->kind != WGSL_SYM_CONST && sym->kind != WGSL_SYM_PARAM)
        return;
    if (!sym->ast || !r->ast.root) return;
    uint32_t off = sym->ast->span_offset;
    WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *fn = tu->children[i];
        if (!fn || fn->kind != WGSL_NODE_DECL_FUNCTION) continue;
        uint32_t so = fn->span_offset, sl = fn->span_length;
        if (off < so || off >= so + sl) continue;
        /* Result is const at the public API; the side-table is a cache. */
        (void)wgsl_typecheck_ensure_function_body(
            (WGSLTypeChecker *)&r->tc, fn);
        return;
    }
}

WGSLHover wgsl_hover_at(const WGSLResult *r, uint32_t offset) {
    if (!r || !r->ast.root) return empty_hover();

    /* First try the use-site path: an EXPR_IDENT whose `payload[2]`
     * already carries the resolver-bound symbol. */
    WGSLSymbol *sym = NULL;
    uint32_t span_off = 0, span_len = 0;
    const WGSLNode *id = wgsl_api_find_ident_at(r->ast.root, offset, 0);
    if (id) {
        sym = wgsl_node_resolved_symbol(id);
        span_off = id->span_offset;
        span_len = id->span_length;
    }
    /* Fall back to the decl-name path (the `x` in `let x: i32 = …`,
     * the `Foo` in `struct Foo { … }`, etc.). */
    if (!sym) {
        sym = wgsl_api_find_decl_at(&r->res, r->ast.root, offset);
        if (sym && sym->ast) {
            span_off = wgsl_node_name_span(sym->ast).offset;
            span_len = wgsl_node_name_span(sym->ast).length;
        }
    }
    if (!sym) return empty_hover();

    hover_ensure_body_typed(r, sym);

    WGSLHover h     = empty_hover();
    h.present       = 1;
    h.symbol_offset = span_off;
    h.symbol_length = span_len;
    if (sym->type) {
        h.type_repr = fmt_type((WGSLArena *)&r->arena, sym->type);
    }
    return h;
}

WGSLDefinition wgsl_definition_at(const WGSLResult *r, uint32_t offset) {
    WGSLDefinition d = {0};
    if (!r || !r->ast.root) return d;

    WGSLSymbol *sym = NULL;
    const WGSLNode *id = wgsl_api_find_ident_at(r->ast.root, offset, 0);
    if (id) sym = wgsl_node_resolved_symbol(id);
    if (!sym) sym = wgsl_api_find_decl_at(&r->res, r->ast.root, offset);
    if (!sym || !sym->ast) return d;     /* predeclared symbols → absent */

    WGSLNode *decl = sym->ast;
    uint32_t no = wgsl_node_name_span(decl).offset;
    uint32_t nl = wgsl_node_name_span(decl).length;
    if (nl == 0) return d;
    d.present = 1;
    d.offset  = no;
    d.length  = nl;
    return d;
}

/* Pointer-out wrappers (wasm FFI ergonomics). */

void wgsl_hover_at_into(
    const WGSLResult *r, uint32_t offset, WGSLHover *out)
{
    if (!out) return;
    *out = wgsl_hover_at(r, offset);
}

void wgsl_definition_at_into(
    const WGSLResult *r, uint32_t offset, WGSLDefinition *out)
{
    if (!out) return;
    *out = wgsl_definition_at(r, offset);
}

/* Project / preamble injection (FS-aware variants gated). */

#ifndef WGSL_NO_FS

/* Read the entire file at `path` into a freshly-malloc'd buffer.
 * On success returns the buffer (caller free); writes the byte count
 * to `*out_len`.  On failure returns NULL. */
static char *read_file_all(const char *path, size_t *out_len) {
    if (out_len) *out_len = 0;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) { free(buf); return NULL; }
    buf[n] = '\0';
    if (out_len) *out_len = (size_t)n;
    return buf;
}

static char *path_join(const char *a, const char *b) {
    size_t la = a ? strlen(a) : 0;
    size_t lb = b ? strlen(b) : 0;
    int    sep = (la > 0 && a[la - 1] != '/') ? 1 : 0;
    if (la > SIZE_MAX - (size_t)sep ||
        la + (size_t)sep > SIZE_MAX - lb ||
        la + (size_t)sep + lb == SIZE_MAX)
    {
        return NULL;
    }
    char  *out = (char *)malloc(la + (size_t)sep + lb + 1);
    if (!out) return NULL;
    if (la > 0) memcpy(out, a, la);
    if (sep)    out[la] = '/';
    if (lb > 0) memcpy(out + la + (size_t)sep, b, lb);
    out[la + (size_t)sep + lb] = '\0';
    return out;
}

WGSLProject *wgsl_project_open(const char *root_dir, char *err, size_t err_len) {
    if (err && err_len > 0) err[0] = '\0';
    if (!root_dir) {
        if (err && err_len > 0) snprintf(err, err_len, "root_dir is NULL");
        return NULL;
    }
    char *toml_path = path_join(root_dir, "wgslconfig.toml");
    if (!toml_path) {
        if (err && err_len > 0) snprintf(err, err_len, "out of memory");
        return NULL;
    }
    size_t toml_len = 0;
    char *toml_text = read_file_all(toml_path, &toml_len);
    free(toml_path);
    if (!toml_text) {
        if (err && err_len > 0) snprintf(err, err_len, "could not read wgslconfig.toml");
        return NULL;
    }
    WGSLProject *p = wgsl_project_open_from_string(toml_text, toml_len, err, err_len);
    free(toml_text);
    return p;
}

WGSLResult *wgsl_check_in_project(
    const WGSLProject *p, const char *root_dir, const char *file_path)
{
    /* Read source. */
    char *src_full = path_join(root_dir, file_path);
    if (!src_full) return NULL;
    size_t src_len = 0;
    char *src = read_file_all(src_full, &src_len);
    free(src_full);
    if (!src) return NULL;

    /* Resolve preamble list for this file path. */
    int preamble_count = 0;
    const char *const *preamble_files =
        p ? wgsl_project_match(p, file_path, &preamble_count) : NULL;

    /* Concatenate all preamble files (with trailing newlines). */
    char  *preamble = NULL;
    size_t plen     = 0;
    for (int i = 0; i < preamble_count; i++) {
        char  *full = path_join(root_dir, preamble_files[i]);
        if (!full) { free(preamble); free(src); return NULL; }
        size_t one_len = 0;
        char  *one = read_file_all(full, &one_len);
        free(full);
        if (!one) { free(preamble); free(src); return NULL; }
        int sep = (one_len > 0 && one[one_len - 1] != '\n') ? 1 : 0;
        if (plen > SIZE_MAX - one_len ||
            plen + one_len > SIZE_MAX - (size_t)sep)
        {
            free(one); free(preamble); free(src); return NULL;
        }
        char *grown = (char *)realloc(preamble, plen + one_len + (size_t)sep);
        if (!grown) { free(one); free(preamble); free(src); return NULL; }
        preamble = grown;
        memcpy(preamble + plen, one, one_len);
        plen += one_len;
        if (sep) preamble[plen++] = '\n';
        free(one);
    }

    WGSLResult *r = wgsl_check_with_preamble(preamble, plen, src, src_len);
    free(preamble);
    free(src);
    return r;
}

#endif /* WGSL_NO_FS */
