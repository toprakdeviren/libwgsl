typedef struct {
    char  *buf;
    size_t len;
    size_t cap;
    int    oom;
} JsonBuf;

static void jb_grow(JsonBuf *j, size_t need) {
    if (j->oom) return;
    if (need > SIZE_MAX - j->len || j->len + need == SIZE_MAX) {
        j->oom = 1;
        return;
    }
    size_t want = j->len + need + 1;
    if (want <= j->cap) return;
    size_t cap = j->cap ? j->cap : 256;
    while (cap < want) {
        if (cap > SIZE_MAX / 2) {
            cap = want;
            break;
        }
        cap *= 2;
    }
    char *g = (char *)realloc(j->buf, cap);
    if (!g) { j->oom = 1; return; }
    j->buf = g;
    j->cap = cap;
}

static void jb_putc(JsonBuf *j, char c) {
    jb_grow(j, 1);
    if (j->oom) return;
    j->buf[j->len++] = c;
}

static void jb_putn(JsonBuf *j, const char *s, size_t n) {
    jb_grow(j, n);
    if (j->oom) return;
    memcpy(j->buf + j->len, s, n);
    j->len += n;
}

static void jb_puts(JsonBuf *j, const char *s) {
    jb_putn(j, s, strlen(s));
}

static void jb_put_int(JsonBuf *j, long long v) {
    char tmp[32];
    int n = snprintf(tmp, sizeof tmp, "%lld", v);
    if (n > 0) jb_putn(j, tmp, (size_t)n);
}

/* JSON-escape a byte range. */
static void jb_put_jstr(JsonBuf *j, const char *s, size_t n) {
    jb_putc(j, '"');
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"':  jb_putn(j, "\\\"", 2); break;
        case '\\': jb_putn(j, "\\\\", 2); break;
        case '\b': jb_putn(j, "\\b", 2);  break;
        case '\f': jb_putn(j, "\\f", 2);  break;
        case '\n': jb_putn(j, "\\n", 2);  break;
        case '\r': jb_putn(j, "\\r", 2);  break;
        case '\t': jb_putn(j, "\\t", 2);  break;
        default:
            if (c < 0x20) {
                char tmp[8];
                int k = snprintf(tmp, sizeof tmp, "\\u%04x", c);
                if (k > 0) jb_putn(j, tmp, (size_t)k);
            } else {
                jb_putc(j, (char)c);
            }
        }
    }
    jb_putc(j, '"');
}

/* Convenience: emit a JSON string from a (offset, length) source span. */
static void jb_put_span(JsonBuf *j, const WGSLSource *src,
                        uint32_t off, uint32_t len)
{
    if (!src || !src->bytes ||
        len > UINT32_MAX - off ||
        (size_t)off + (size_t)len > src->length)
    {
        jb_puts(j, "\"\"");
        return;
    }
    jb_put_jstr(j, src->bytes + off, len);
}

/* Format a TypeInfo into the JSON buffer as a quoted string. */
static void jb_put_type(JsonBuf *j, WGSLTypeInfo *t) {
    if (!t) { jb_puts(j, "\"\""); return; }
    char tmp[128];
    size_t n = wgsl_type_format(t, tmp, sizeof tmp);
    if (n >= sizeof tmp) n = sizeof tmp - 1;
    jb_put_jstr(j, tmp, n);
}

/* Resolve an attribute argument (an EXPR_*) to an integer.  Returns 0
 * if the arg isn't a const-eval'd integer; uses the result's existing
 * `WGSLConstEvaluator` (no second pass). */
static long long wgsl_attr_int_arg(WGSLResult *r, WGSLNode *arg) {
    if (!arg) return 0;
    /* Cast away const so we can reuse `wgsl_consteval_expr`; the
     * call only writes `*out`, not the AST. */
    WGSLValue v = {0};
    if (!wgsl_consteval_expr(&r->cev, arg, &v)) return 0;
    if (v.kind != WGSL_VAL_INT) return 0;
    return (long long)v.u.i;
}

/* Look up an attribute on `[parent->children + first .. + first+count)`
 * by name.  Returns the ATTRIBUTE node or NULL. */
static WGSLNode *json_find_attr(
    WGSLResult *r, const WGSLNode *parent,
    uint32_t first, uint32_t count, const char *want)
{
    size_t wl = strlen(want);
    for (uint32_t i = 0; i < count && first + i < parent->child_count; i++) {
        WGSLNode *a = parent->children[first + i];
        if (!a || a->kind != WGSL_NODE_ATTRIBUTE) continue;
        if (a->child_count == 0) continue;
        WGSLNode *nm = a->children[0];
        if (!nm) continue;
        uint32_t off = (uint32_t)(nm->payload[0] & 0xFFFFFFFFu);
        uint32_t len = (uint32_t)(nm->payload[0] >> 32);
        if (len == wl &&
            memcmp(r->source.bytes + off, want, wl) == 0)
        {
            return a;
        }
    }
    return NULL;
}

static const char *json_stage_for(WGSLResult *r, WGSLNode *fn,
                                  uint32_t fn_attr_count)
{
    if (json_find_attr(r, fn, 0, fn_attr_count, "vertex"))   return "vertex";
    if (json_find_attr(r, fn, 0, fn_attr_count, "fragment")) return "fragment";
    if (json_find_attr(r, fn, 0, fn_attr_count, "compute"))  return "compute";
    return NULL;
}

static void emit_entry_points(WGSLResult *r, JsonBuf *j) {
    jb_puts(j, "\"entry_points\":[");
    int first = 1;
    if (!r->ast.root) { jb_puts(j, "]"); return; }
    WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n || n->kind != WGSL_NODE_DECL_FUNCTION) continue;

        uint32_t FA = (uint32_t)(n->payload[1] & 0xFFFFFFFFu);
        const char *stage = json_stage_for(r, n, FA);
        if (!stage) continue;

        if (!first) jb_putc(j, ',');
        first = 0;
        jb_puts(j, "{\"name\":");
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        jb_put_span(j, &r->source, no, nl);
        jb_puts(j, ",\"stage\":\"");
        jb_puts(j, stage);
        jb_putc(j, '"');

        if (strcmp(stage, "compute") == 0) {
            WGSLNode *wgs = json_find_attr(r, n, 0, FA, "workgroup_size");
            if (wgs) {
                jb_puts(j, ",\"workgroup_size\":[");
                /* wgs->children: [name_ident, arg1, arg2?, arg3?] */
                long long dims[3] = { 1, 1, 1 };
                for (uint32_t k = 1; k < wgs->child_count && k <= 3; k++) {
                    dims[k - 1] = wgsl_attr_int_arg(r, wgs->children[k]);
                }
                jb_put_int(j, dims[0]); jb_putc(j, ',');
                jb_put_int(j, dims[1]); jb_putc(j, ',');
                jb_put_int(j, dims[2]);
                jb_putc(j, ']');
            }
        }
        jb_putc(j, '}');
    }
    jb_putc(j, ']');
}

static void emit_resources(WGSLResult *r, JsonBuf *j) {
    jb_puts(j, ",\"resources\":[");
    int first = 1;
    if (!r->ast.root) { jb_putc(j, ']'); return; }
    WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n || n->kind != WGSL_NODE_DECL_VAR) continue;

        uint32_t A = (uint32_t)(n->payload[1] & 0xFFFFFFFFu);
        WGSLNode *gp = json_find_attr(r, n, 0, A, "group");
        WGSLNode *bd = json_find_attr(r, n, 0, A, "binding");
        if (!gp || !bd) continue;       /* not a bound resource          */

        long long group   = (gp->child_count >= 2)
            ? wgsl_attr_int_arg(r, gp->children[1]) : 0;
        long long binding = (bd->child_count >= 2)
            ? wgsl_attr_int_arg(r, bd->children[1]) : 0;

        /* Address space + access mode from the template list. */
        uint32_t T = (uint32_t)(n->payload[1] >> 32);
        const char *as_name = "private";
        const char *acc_name = NULL;
        if (T >= 1 && A < n->child_count) {
            WGSLNode *as_node = n->children[A];
            if (as_node && as_node->kind == WGSL_NODE_EXPR_IDENT) {
                uint32_t off = (uint32_t)(as_node->payload[0] & 0xFFFFFFFFu);
                uint32_t len = (uint32_t)(as_node->payload[0] >> 32);
                if      (len == 7 && memcmp(r->source.bytes + off, "storage", 7) == 0) as_name = "storage";
                else if (len == 7 && memcmp(r->source.bytes + off, "uniform", 7) == 0) as_name = "uniform";
                else if (len == 9 && memcmp(r->source.bytes + off, "workgroup", 9) == 0) as_name = "workgroup";
                else if (len == 7 && memcmp(r->source.bytes + off, "private", 7) == 0) as_name = "private";
            }
        }
        if (T >= 2 && A + 1 < n->child_count) {
            WGSLNode *ac_node = n->children[A + 1];
            if (ac_node && ac_node->kind == WGSL_NODE_EXPR_IDENT) {
                uint32_t off = (uint32_t)(ac_node->payload[0] & 0xFFFFFFFFu);
                uint32_t len = (uint32_t)(ac_node->payload[0] >> 32);
                if      (len == 4 && memcmp(r->source.bytes + off, "read", 4) == 0) acc_name = "read";
                else if (len == 5 && memcmp(r->source.bytes + off, "write", 5) == 0) acc_name = "write";
                else if (len == 10 && memcmp(r->source.bytes + off, "read_write", 10) == 0) acc_name = "read_write";
            }
        }
        /* Spec defaults: storage → read, uniform → read. */
        if (!acc_name) acc_name = "read";

        WGSLSymbol *sym = NULL;
        for (size_t k = 0; k < r->res.all_decl_count; k++) {
            if (r->res.all_decls[k]->ast == n) {
                sym = r->res.all_decls[k]; break;
            }
        }

        if (!first) jb_putc(j, ',');
        first = 0;
        jb_puts(j, "{\"name\":");
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        jb_put_span(j, &r->source, no, nl);
        jb_puts(j, ",\"group\":");   jb_put_int(j, group);
        jb_puts(j, ",\"binding\":"); jb_put_int(j, binding);
        jb_puts(j, ",\"address_space\":\""); jb_puts(j, as_name); jb_putc(j, '"');
        jb_puts(j, ",\"access\":\"");        jb_puts(j, acc_name); jb_putc(j, '"');
        jb_puts(j, ",\"type\":");
        jb_put_type(j, sym ? sym->type : NULL);
        jb_putc(j, '}');
    }
    jb_putc(j, ']');
}

static void emit_structs(WGSLResult *r, JsonBuf *j) {
    jb_puts(j, ",\"structs\":[");
    int first = 1;
    if (!r->ast.root) { jb_putc(j, ']'); return; }
    WGSLNode *tu = r->ast.root;
    /* Layout is computed per spec §14.4.2 using the default address
     * space (WGSL_AS_NONE, i.e. AlignOf without uniform-AS bump).  A
     * struct used as `var<uniform>` would re-walk with WGSL_AS_UNIFORM;
     * the validator already enforces that case.  For module-JSON the
     * default layout is the most useful baseline for engine bindings. */
    WGSLLayoutCtx lcx = { &r->tc, &r->cev, &r->source };
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n || n->kind != WGSL_NODE_DECL_STRUCT) continue;

        if (!first) jb_putc(j, ',');
        first = 0;
        jb_puts(j, "{\"name\":");
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        jb_put_span(j, &r->source, no, nl);

        /* Resolve the struct's TypeInfo (member 0 of all_decls is the
         * struct symbol with `sym->type` set during typecheck).  Walk
         * once to fill offsets[]. */
        const WGSLTypeInfo *st = NULL;
        for (size_t s = 0; s < r->res.all_decl_count; s++) {
            if (r->res.all_decls[s]->ast == n) {
                st = r->res.all_decls[s]->type;
                break;
            }
        }
        uint32_t offsets[64], salign = 0, ssize = 0;
        int have_layout = st &&
            wgsl_layout_struct(&lcx, st, WGSL_AS_NONE, offsets, 64, &salign, &ssize);
        if (have_layout) {
            jb_puts(j, ",\"size\":");      jb_put_int(j, ssize);
            jb_puts(j, ",\"align\":");     jb_put_int(j, salign);
        }

        jb_puts(j, ",\"members\":[");
        int mfirst = 1;
        uint32_t midx = 0;
        for (uint32_t k = 0; k < n->child_count; k++) {
            WGSLNode *m = n->children[k];
            if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            if (!mfirst) jb_putc(j, ',');
            mfirst = 0;
            jb_puts(j, "{\"name\":");
            uint32_t mno = (uint32_t)(m->payload[0] & 0xFFFFFFFFu);
            uint32_t mnl = (uint32_t)(m->payload[0] >> 32);
            jb_put_span(j, &r->source, mno, mnl);
            jb_puts(j, ",\"type\":");
            uint32_t MA = (uint32_t)(m->payload[1] & 0xFFFFFFFFu);
            WGSLTypeInfo *mt = NULL;
            if (MA < m->child_count) {
                /* The TC's `resolve_type_spec` records the resolved
                 * type on the side-table for every type-spec node it
                 * visits, so we can pull it back here. */
                mt = wgsl_typecheck_type_of(&r->tc, m->children[MA]);
            }
            jb_put_type(j, mt);
            if (have_layout && midx < 64) {
                jb_puts(j, ",\"offset\":"); jb_put_int(j, offsets[midx]);
                jb_puts(j, ",\"size\":");   jb_put_int(j, wgsl_layout_size_of(&lcx, mt, WGSL_AS_NONE));
                jb_puts(j, ",\"align\":");  jb_put_int(j, wgsl_layout_align_of(&lcx, mt, WGSL_AS_NONE));
            }
            jb_putc(j, '}');
            midx += 1;
        }
        jb_puts(j, "]}");
    }
    jb_putc(j, ']');
}

static void emit_overrides(WGSLResult *r, JsonBuf *j) {
    jb_puts(j, ",\"overrides\":[");
    int first = 1;
    if (!r->ast.root) { jb_putc(j, ']'); return; }
    WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n || n->kind != WGSL_NODE_DECL_OVERRIDE) continue;

        uint32_t A = (uint32_t)(n->payload[1] & 0xFFFFFFFFu);
        WGSLNode *id = json_find_attr(r, n, 0, A, "id");

        WGSLSymbol *sym = NULL;
        for (size_t k = 0; k < r->res.all_decl_count; k++) {
            if (r->res.all_decls[k]->ast == n) {
                sym = r->res.all_decls[k]; break;
            }
        }
        if (!first) jb_putc(j, ',');
        first = 0;
        jb_puts(j, "{\"name\":");
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        jb_put_span(j, &r->source, no, nl);
        jb_puts(j, ",\"type\":");
        jb_put_type(j, sym ? sym->type : NULL);
        jb_puts(j, ",\"id\":");
        if (id && id->child_count >= 2) {
            jb_put_int(j, wgsl_attr_int_arg(r, id->children[1]));
        } else {
            jb_puts(j, "null");
        }
        jb_putc(j, '}');
    }
    jb_putc(j, ']');
}

static void build_module_json(WGSLResult *r) {
    JsonBuf j = {0};
    jb_putc(&j, '{');
    jb_puts(&j, "\"spec_pin\":\""); jb_puts(&j, WGSL_SPEC_PIN); jb_putc(&j, '"');
    jb_putc(&j, ',');
    emit_entry_points(r, &j);
    emit_resources   (r, &j);
    emit_structs     (r, &j);
    emit_overrides   (r, &j);
    jb_putc(&j, '}');

    if (j.oom || !j.buf) {
        free(j.buf);
        r->json_cache     = "";
        r->json_cache_len = 0;
        r->json_built     = 1;
        return;
    }
    /* Copy into the arena so the buffer survives `wgsl_free` semantics
     * and lives alongside the rest of the result's owned strings. */
    char *out = (char *)wgsl_arena_alloc(&r->arena, j.len + 1);
    if (out) {
        memcpy(out, j.buf, j.len);
        out[j.len] = '\0';
        r->json_cache     = out;
        r->json_cache_len = j.len;
    } else {
        r->json_cache     = "";
        r->json_cache_len = 0;
    }
    free(j.buf);
    r->json_built = 1;
}

const char *wgsl_module_json(const WGSLResult *r) {
    if (!r) return "";
    /* Only emit JSON when the module passed the full check.  Per the
     * header contract: empty string when the check failed. */
    if (!r->pipeline_ok) return "";
    /* Cast away const for lazy initialisation — the cache is logically
     * derived state, and `wgsl_module_json` is documented as
     * idempotent. */
    WGSLResult *m = (WGSLResult *)r;
    if (!m->json_built) build_module_json(m);
    return m->json_cache ? m->json_cache : "";
}

size_t wgsl_module_json_len(const WGSLResult *r) {
    if (!r) return 0;
    if (!r->pipeline_ok) return 0;
    WGSLResult *m = (WGSLResult *)r;
    if (!m->json_built) build_module_json(m);
    return m->json_cache_len;
}

