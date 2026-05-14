/* ── Hover / goto-def — Iter B ────────────────────────────────────── */

/* Find the smallest EXPR_IDENT span that contains `offset`.  Returns
 * NULL when no ident covers the offset. */
static const WGSLNode *find_ident_at(const WGSLNode *n, uint32_t offset) {
    if (!n) return NULL;
    if (offset < n->span_offset ||
        offset >= n->span_offset + n->span_length)
    {
        return NULL;
    }
    /* Recurse into children first — the deepest hit is the smallest. */
    for (uint32_t i = 0; i < n->child_count; i++) {
        const WGSLNode *hit = find_ident_at(n->children[i], offset);
        if (hit) return hit;
    }
    if (n->kind == WGSL_NODE_EXPR_IDENT) return n;
    return NULL;
}

/* Find a user-declared symbol whose decl-name span contains `offset`.
 * Decl-name spans live in `payload[0] = (off | len << 32)` for every
 * kind the resolver binds. */
static WGSLSymbol *find_decl_at(const WGSLResolver *res, uint32_t offset) {
    if (!res) return NULL;
    for (size_t i = 0; i < res->all_decl_count; i++) {
        WGSLSymbol *s = res->all_decls[i];
        if (!s || !s->ast) continue;
        uint32_t no = (uint32_t)(s->ast->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(s->ast->payload[0] >> 32);
        if (nl == 0) continue;
        if (offset >= no && offset < no + nl) return s;
    }
    return NULL;
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

WGSLHover wgsl_hover_at(const WGSLResult *r, uint32_t offset) {
    if (!r || !r->ast.root) return empty_hover();

    /* First try the use-site path: an EXPR_IDENT whose `payload[2]`
     * already carries the resolver-bound symbol. */
    WGSLSymbol *sym = NULL;
    uint32_t span_off = 0, span_len = 0;
    const WGSLNode *id = find_ident_at(r->ast.root, offset);
    if (id) {
        sym = wgsl_node_resolved_symbol(id);
        span_off = id->span_offset;
        span_len = id->span_length;
    }
    /* Fall back to the decl-name path (the `x` in `let x: i32 = …`,
     * the `Foo` in `struct Foo { … }`, etc.). */
    if (!sym) {
        sym = find_decl_at(&r->res, offset);
        if (sym && sym->ast) {
            span_off = (uint32_t)(sym->ast->payload[0] & 0xFFFFFFFFu);
            span_len = (uint32_t)(sym->ast->payload[0] >> 32);
        }
    }
    if (!sym) return empty_hover();

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
    const WGSLNode *id = find_ident_at(r->ast.root, offset);
    if (id) sym = wgsl_node_resolved_symbol(id);
    if (!sym) sym = find_decl_at(&r->res, offset);
    if (!sym || !sym->ast) return d;     /* predeclared symbols → absent */

    WGSLNode *decl = sym->ast;
    uint32_t no = (uint32_t)(decl->payload[0] & 0xFFFFFFFFu);
    uint32_t nl = (uint32_t)(decl->payload[0] >> 32);
    if (nl == 0) return d;
    d.present = 1;
    d.offset  = no;
    d.length  = nl;
    return d;
}

/* ── Pointer-out wrappers (wasm FFI ergonomics) ───────────────────── */

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

/* ── Project / preamble injection (FS-aware variants gated) ───────── */

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
