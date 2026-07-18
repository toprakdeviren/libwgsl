static char *m_strdup(const char *s) {
    size_t n = s ? strlen(s) : 0;
    char *out = (char *)malloc(n + 1);
    if (!out) return NULL;
    if (n) memcpy(out, s, n);
    out[n] = 0;
    return out;
}

static void m_diag(M *m, int code, uint32_t off, uint32_t len,
                   const char *msg)
{
    if (!m) return;
    m->incomplete = 1;
    if (m->ndiags == m->dcap) {
        int nc = m->dcap ? m->dcap * 2 : 8;
        WGSLMslDiag *g = (WGSLMslDiag *)realloc(
            m->diags, (size_t)nc * sizeof *g);
        if (!g) return;
        m->diags = g;
        m->dcap = nc;
    }
    WGSLMslDiag *d = &m->diags[m->ndiags++];
    d->code = code;
    d->offset = off;
    d->length = len;
    d->message = m_strdup(msg ? msg : "");
}

static void m_grow(M *m, size_t n) {
    if (m->oom) return;
    if (m->len + n + 1 <= m->cap) return;
    size_t nc = m->cap ? m->cap : 512;
    while (nc < m->len + n + 1) {
        if (nc > SIZE_MAX / 2) { m->oom = 1; return; }
        nc *= 2;
    }
    char *g = (char *)realloc(m->buf, nc);
    if (!g) { m->oom = 1; return; }
    m->buf = g; m->cap = nc;
}
static void m_putn(M *m, const char *s, size_t n) {
    if (!s || !n || m->oom) return;
    m_grow(m, n);
    if (m->oom) return;
    memcpy(m->buf + m->len, s, n);
    m->len += n;
    m->buf[m->len] = 0;
    m->at_line_start = (s[n - 1] == '\n');
}
static void m_puts(M *m, const char *s) { if (s) m_putn(m, s, strlen(s)); }
static void m_putc(M *m, char c) { m_putn(m, &c, 1); }
static void m_nl(M *m) { m_putc(m, '\n'); m->at_line_start = 1; }
static void m_indent(M *m) {
    if (!m->at_line_start) return;
    for (int i = 0; i < m->indent; i++) m_puts(m, "    ");
    m->at_line_start = 0;
}
static void m_span(M *m, uint32_t o, uint32_t l) {
    if (!m->src || (size_t)o + (size_t)l > m->src_len) return;
    m_indent(m);
    m_putn(m, m->src + o, l);
}
static int m_name_eq(M *m, uint32_t o, uint32_t l, const char *s) {
    size_t n = strlen(s);
    return l == n && m->src && (size_t)o + n <= m->src_len &&
           memcmp(m->src + o, s, n) == 0;
}
static int m_ident_eq(M *m, const WGSLNode *n, const char *s) {
    if (!n || n->kind != WGSL_NODE_EXPR_IDENT) return 0;
    uint32_t o = wgsl_node_name_span(n).offset;
    uint32_t l = wgsl_node_name_span(n).length;
    return m_name_eq(m, o, l, s);
}

static const WGSLNode *templated_arg_at(const WGSLNode *n, uint32_t want) {
    if (!n || n->kind != WGSL_NODE_EXPR_TEMPLATED_IDENT) return NULL;
    uint32_t ai = 0;
    for (uint32_t i = 1; i < n->child_count; i++) {
        const WGSLNode *c = n->children[i];
        if (!c) continue;
        if (c->kind == WGSL_NODE_TEMPLATE_ARG_LIST) {
            for (uint32_t k = 0; k < c->child_count; k++) {
                if (ai == want) return c->children[k];
                ai++;
            }
        } else {
            if (ai == want) return c;
            ai++;
        }
    }
    return NULL;
}

static const char *storage_texture_base(M *m, uint32_t o, uint32_t l) {
    if (m_name_eq(m, o, l, "texture_storage_1d")) return "texture1d";
    if (m_name_eq(m, o, l, "texture_storage_2d")) return "texture2d";
    if (m_name_eq(m, o, l, "texture_storage_2d_array")) return "texture2d_array";
    if (m_name_eq(m, o, l, "texture_storage_3d")) return "texture3d";
    return NULL;
}

static int span_contains(const char *s, uint32_t n, const char *needle) {
    size_t nl = strlen(needle);
    if (!s || nl == 0 || nl > n) return 0;
    for (uint32_t i = 0; i + nl <= n; i++)
        if (memcmp(s + i, needle, nl) == 0) return 1;
    return 0;
}

static const char *storage_texture_elem(M *m, const WGSLNode *fmt) {
    if (!fmt || fmt->kind != WGSL_NODE_EXPR_IDENT || !m->src) return NULL;
    uint32_t o = wgsl_node_name_span(fmt).offset;
    uint32_t l = wgsl_node_name_span(fmt).length;
    if ((size_t)o + l > m->src_len) return NULL;
    const char *s = m->src + o;
    if (span_contains(s, l, "uint")) return "uint";
    if (span_contains(s, l, "sint")) return "int";
    if (span_contains(s, l, "16float")) return "half";
    return "float";
}

static const char *storage_texture_access(M *m, const WGSLNode *acc) {
    if (m_ident_eq(m, acc, "read")) return "read";
    if (m_ident_eq(m, acc, "write")) return "write";
    if (m_ident_eq(m, acc, "read_write")) return "read_write";
    return NULL;
}
static void m_ident(M *m, const WGSLNode *n) {
    if (!n) return;
    uint32_t o = wgsl_node_name_span(n).offset;
    uint32_t l = wgsl_node_name_span(n).length;
    if (n->kind == WGSL_NODE_EXPR_IDENT ||
        n->kind == WGSL_NODE_DECL_FUNCTION ||
        n->kind == WGSL_NODE_DECL_PARAM ||
        n->kind == WGSL_NODE_DECL_STRUCT ||
        n->kind == WGSL_NODE_DECL_STRUCT_MEMBER ||
        n->kind == WGSL_NODE_DECL_CONST ||
        n->kind == WGSL_NODE_DECL_LET ||
        n->kind == WGSL_NODE_DECL_VAR ||
        n->kind == WGSL_NODE_DECL_ALIAS)
        m_span(m, o, l);
    else
        m_span(m, n->span_offset, n->span_length);
}

/* ── WGSL type name (from source text) → MSL ─────────────────────── */

static const char *map_simple_type(const char *w, size_t n) {
    struct { const char *w; const char *m; } t[] = {
        { "void", "void" }, { "bool", "bool" },
        { "f32", "float" }, { "f16", "half" },
        { "i32", "int" }, { "u32", "uint" },
        { "vec2f", "float2" }, { "vec3f", "float3" }, { "vec4f", "float4" },
        { "vec2i", "int2" }, { "vec3i", "int3" }, { "vec4i", "int4" },
        { "vec2u", "uint2" }, { "vec3u", "uint3" }, { "vec4u", "uint4" },
        { "vec2h", "half2" }, { "vec3h", "half3" }, { "vec4h", "half4" },
        { "mat2x2f", "float2x2" }, { "mat2x3f", "float2x3" }, { "mat2x4f", "float2x4" },
        { "mat3x2f", "float3x2" }, { "mat3x3f", "float3x3" }, { "mat3x4f", "float3x4" },
        { "mat4x2f", "float4x2" }, { "mat4x3f", "float4x3" }, { "mat4x4f", "float4x4" },
        { "mat2x2h", "half2x2" }, { "mat3x3h", "half3x3" }, { "mat4x4h", "half4x4" },
        { "vec2", "float2" }, { "vec3", "float3" }, { "vec4", "float4" }, /* bare → f32 */
    };
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) {
        if (n == strlen(t[i].w) && memcmp(w, t[i].w, n) == 0) return t[i].m;
    }
    return NULL;
}

static const char *map_builtin(const char *w, size_t n) {
    struct { const char *w; const char *m; } t[] = {
        { "position", "position" },
        { "vertex_index", "vertex_id" },
        { "instance_index", "instance_id" },
        { "front_facing", "front_facing" },
        { "frag_depth", "depth(any)" },
        { "sample_index", "sample_id" },
        { "sample_mask", "sample_mask" },
        { "local_invocation_id", "thread_position_in_threadgroup" },
        { "local_invocation_index", "thread_index_in_threadgroup" },
        { "global_invocation_id", "thread_position_in_grid" },
        { "workgroup_id", "threadgroup_position_in_grid" },
        { "num_workgroups", "threadgroups_per_grid" },
        { "subgroup_invocation_id", "thread_index_in_simdgroup" },
        { "subgroup_size", "threads_per_simdgroup" },
    };
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) {
        if (n == strlen(t[i].w) && memcmp(w, t[i].w, n) == 0) return t[i].m;
    }
    return NULL;
}

/* Map common WGSL builtin / constructor call names to MSL. */
static const char *map_callee(const char *w, size_t n) {
    const char *s = map_simple_type(w, n);
    if (s) return s;
    struct { const char *w; const char *m; } t[] = {
        { "vec2", "float2" }, { "vec3", "float3" }, { "vec4", "float4" },
        { "mat2x2", "float2x2" }, { "mat3x3", "float3x3" }, { "mat4x4", "float4x4" },
        { "bitcast", "as_type" }, /* special-cased in emit */
        /* most math names match: sin cos tan abs min max clamp mix step … */
    };
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) {
        if (n == strlen(t[i].w) && memcmp(w, t[i].w, n) == 0) return t[i].m;
    }
    return NULL; /* keep original spelling */
}

static void emit_type_spec(M *m, const WGSLNode *n, int depth);
static void emit_expr(M *m, const WGSLNode *n, int depth);
static void emit_stmt(M *m, const WGSLNode *n, int depth);
static void emit_node(M *m, const WGSLNode *n, int depth);
static void emit_compound(M *m, const WGSLNode *n, int depth);
static void emit_for(M *m, const WGSLNode *n, int depth);
static void emit_loop(M *m, const WGSLNode *n, int depth);
static void emit_switch(M *m, const WGSLNode *n, int depth);
static void detect_shadowing(M *m, const WGSLNode *n,
                             const WGSLSymbol **syms, int nsym,
                             uint8_t *shadowed);
static WGSLTypeInfo *node_type(M *m, const WGSLNode *n);
static void emit_type_info(M *m, const WGSLTypeInfo *t);

