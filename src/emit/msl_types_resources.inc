static void emit_type_spec(M *m, const WGSLNode *n, int depth) {
    if (!n) { m_puts(m, "void"); return; }
    if (depth > 64) { m_puts(m, "/*deep*/"); return; }

    if (n->kind == WGSL_NODE_EXPR_IDENT) {
        uint32_t o = wgsl_node_name_span(n).offset;
        uint32_t l = wgsl_node_name_span(n).length;
        if (m->src && (size_t)o + l <= m->src_len) {
            const char *mapped = map_simple_type(m->src + o, l);
            if (mapped) { m_puts(m, mapped); return; }
        }
        m_ident(m, n);
        return;
    }
    if (n->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT && n->child_count > 0) {
        const WGSLNode *base = n->children[0];
        uint32_t o = 0, l = 0;
        if (base && base->kind == WGSL_NODE_EXPR_IDENT) {
            o = wgsl_node_name_span(base).offset;
            l = wgsl_node_name_span(base).length;
        }
        /* array<T,N> → metal::array or T[N] — use T name[N] style via array<> */
        if (m_name_eq(m, o, l, "array")) {
            m_puts(m, "array<");
            const WGSLNode *elem = templated_arg_at(n, 0);
            const WGSLNode *len_arg = templated_arg_at(n, 1);
            if (elem) emit_type_spec(m, elem, depth + 1);
            else m_puts(m, "uint");
            if (len_arg) {
                m_puts(m, ", ");
                if (len_arg->kind == WGSL_NODE_EXPR_LITERAL_INT)
                    m_span(m, len_arg->span_offset, len_arg->span_length);
                else
                    emit_expr(m, len_arg, depth + 1);
            } else {
                m_diag(m, WGSL_MSL_DIAG_UNSUPPORTED_TYPE, o, l,
                       "runtime-sized array has no whole-value MSL array form");
            }
            m_putc(m, '>');
            return;
        }
        if (m_name_eq(m, o, l, "atomic")) {
            /* atomic<u32> → atomic_uint, atomic<i32> → atomic_int */
            const WGSLNode *el = NULL;
            for (uint32_t i = 1; i < n->child_count && !el; i++) {
                if (n->children[i] &&
                    n->children[i]->kind == WGSL_NODE_TEMPLATE_ARG_LIST &&
                    n->children[i]->child_count > 0)
                    el = n->children[i]->children[0];
                else if (n->children[i])
                    el = n->children[i];
            }
            if (el && m_ident_eq(m, el, "u32")) { m_puts(m, "atomic_uint"); return; }
            if (el && m_ident_eq(m, el, "i32")) { m_puts(m, "atomic_int"); return; }
            m_puts(m, "atomic_uint");
            return;
        }
        if (m_name_eq(m, o, l, "vec2") || m_name_eq(m, o, l, "vec3") ||
            m_name_eq(m, o, l, "vec4")) {
            char pref = m->src[o + 3]; /* '2','3','4' */
            const char *el = "float";
            for (uint32_t i = 1; i < n->child_count; i++) {
                const WGSLNode *c = n->children[i];
                if (!c) continue;
                const WGSLNode *a = c;
                if (c->kind == WGSL_NODE_TEMPLATE_ARG_LIST && c->child_count)
                    a = c->children[0];
                if (m_ident_eq(m, a, "f32")) el = "float";
                else if (m_ident_eq(m, a, "f16")) el = "half";
                else if (m_ident_eq(m, a, "i32")) el = "int";
                else if (m_ident_eq(m, a, "u32")) el = "uint";
                else if (m_ident_eq(m, a, "bool")) el = "bool";
            }
            if (strcmp(el, "float") == 0) {
                m_putc(m, 'f'); m_puts(m, "loat"); m_putc(m, pref);
            } else if (strcmp(el, "half") == 0) {
                m_puts(m, "half"); m_putc(m, pref);
            } else if (strcmp(el, "int") == 0) {
                m_puts(m, "int"); m_putc(m, pref);
            } else if (strcmp(el, "uint") == 0) {
                m_puts(m, "uint"); m_putc(m, pref);
            } else {
                m_puts(m, "bool"); m_putc(m, pref);
            }
            return;
        }
        if (l >= 3 && m->src && m_name_eq(m, o, 3, "mat")) {
            /* matCxR<f32> → floatCxR */
            m_puts(m, "float");
            if (l >= 5) m_putn(m, m->src + o + 3, l - 3); /* e.g. 4x4 */
            return;
        }
        if (m_name_eq(m, o, l, "ptr")) {
            /* ptr<AS, T, AM> → T AS*  (simplified) */
            const WGSLNode *as = NULL, *ty = NULL;
            int ai = 0;
            for (uint32_t i = 1; i < n->child_count; i++) {
                const WGSLNode *c = n->children[i];
                if (!c) continue;
                if (c->kind == WGSL_NODE_TEMPLATE_ARG_LIST) {
                    for (uint32_t k = 0; k < c->child_count; k++) {
                        if (ai == 0) as = c->children[k];
                        else if (ai == 1) ty = c->children[k];
                        ai++;
                    }
                } else {
                    if (ai == 0) as = c;
                    else if (ai == 1) ty = c;
                    ai++;
                }
            }
            const char *space = "device";
            if (as && m_ident_eq(m, as, "function")) space = "thread";
            else if (as && m_ident_eq(m, as, "private")) space = "thread";
            else if (as && m_ident_eq(m, as, "workgroup")) space = "threadgroup";
            else if (as && m_ident_eq(m, as, "uniform")) space = "constant";
            else if (as && m_ident_eq(m, as, "storage")) space = "device";
            if (ty) emit_type_spec(m, ty, depth + 1);
            else m_puts(m, "void");
            m_putc(m, ' ');
            m_puts(m, space);
            m_putc(m, '*');
            return;
        }
        /* texture_* / sampler — §B3b data-plane maps. */
        if (l >= 7 && m->src && memcmp(m->src + o, "texture", 7) == 0) {
            /* Map common WGSL texture type names → MSL.  Element type from
             * template arg (f32→float, f16→half, …); default float. */
            const char *msl_base = NULL;
            if (m_name_eq(m, o, l, "texture_1d")) msl_base = "texture1d";
            else if (m_name_eq(m, o, l, "texture_2d")) msl_base = "texture2d";
            else if (m_name_eq(m, o, l, "texture_2d_array")) msl_base = "texture2d_array";
            else if (m_name_eq(m, o, l, "texture_3d")) msl_base = "texture3d";
            else if (m_name_eq(m, o, l, "texture_cube")) msl_base = "texturecube";
            else if (m_name_eq(m, o, l, "texture_cube_array")) msl_base = "texturecube_array";
            else if (m_name_eq(m, o, l, "texture_multisampled_2d"))
                msl_base = "texture2d_ms";
            else if (m_name_eq(m, o, l, "texture_depth_2d")) msl_base = "depth2d";
            else if (m_name_eq(m, o, l, "texture_depth_2d_array"))
                msl_base = "depth2d_array";
            else if (m_name_eq(m, o, l, "texture_depth_cube")) msl_base = "depthcube";
            else if (m_name_eq(m, o, l, "texture_depth_cube_array"))
                msl_base = "depthcube_array";
            else if (m_name_eq(m, o, l, "texture_depth_multisampled_2d"))
                msl_base = "depth2d_ms";
            else if (l >= 15 && memcmp(m->src + o, "texture_storage", 15) == 0) {
                const char *base_name = storage_texture_base(m, o, l);
                const WGSLNode *fmt = templated_arg_at(n, 0);
                const WGSLNode *acc = templated_arg_at(n, 1);
                const char *elem = storage_texture_elem(m, fmt);
                const char *access = storage_texture_access(m, acc);
                if (base_name && elem && access) {
                    m_puts(m, base_name);
                    m_putc(m, '<');
                    m_puts(m, elem);
                    m_puts(m, ", access::");
                    m_puts(m, access);
                    m_putc(m, '>');
                } else {
                    m_puts(m, "texture2d<float, access::write>");
                    m_diag(m, WGSL_MSL_DIAG_UNSUPPORTED_TYPE, o, l,
                           "unknown texture_storage* format/access for MSL emit");
                }
                return;
            }
            if (!msl_base) {
                m_puts(m, "texture2d<float>");
                m_diag(m, WGSL_MSL_DIAG_UNSUPPORTED_TYPE, o, l,
                       "unknown texture_* type; lowered as texture2d<float>");
                return;
            }
            const char *el = "float";
            for (uint32_t i = 1; i < n->child_count; i++) {
                const WGSLNode *c = n->children[i];
                if (!c) continue;
                const WGSLNode *a = c;
                if (c->kind == WGSL_NODE_TEMPLATE_ARG_LIST && c->child_count)
                    a = c->children[0];
                if (m_ident_eq(m, a, "f32")) el = "float";
                else if (m_ident_eq(m, a, "f16")) el = "half";
                else if (m_ident_eq(m, a, "i32")) el = "int";
                else if (m_ident_eq(m, a, "u32")) el = "uint";
            }
            /* depth* types take no element template in MSL. */
            if (msl_base[0] == 'd') {
                m_puts(m, msl_base);
            } else {
                m_puts(m, msl_base);
                m_putc(m, '<');
                m_puts(m, el);
                m_putc(m, '>');
            }
            return;
        }
        if (m_name_eq(m, o, l, "sampler") || m_name_eq(m, o, l, "sampler_comparison")) {
            m_puts(m, "sampler");
            return;
        }
        /* fallback: emit base + template args mapped */
        if (base) m_ident(m, base);
        m_putc(m, '<');
        int first = 1;
        for (uint32_t i = 1; i < n->child_count; i++) {
            const WGSLNode *c = n->children[i];
            if (!c) continue;
            if (c->kind == WGSL_NODE_TEMPLATE_ARG_LIST) {
                for (uint32_t k = 0; k < c->child_count; k++) {
                    if (!first) m_puts(m, ", ");
                    first = 0;
                    emit_type_spec(m, c->children[k], depth + 1);
                }
            } else {
                if (!first) m_puts(m, ", ");
                first = 0;
                emit_type_spec(m, c, depth + 1);
            }
        }
        m_putc(m, '>');
        return;
    }
    /* fallback original */
    m_span(m, n->span_offset, n->span_length);
}

/* Collect @attr info for a parameter / member / return. */
typedef struct {
    int has_builtin;
    char builtin[48];
    int has_location;
    long long location;
    int has_group, has_binding;
    long long group, binding;
    int stage; /* 1 vertex 2 fragment 3 compute */
    int workgroup[3];
    int has_workgroup;
} AttrInfo;

static long long attr_int_arg(M *m, const WGSLNode *attr) {
    if (!attr || attr->child_count < 2) return 0;
    const WGSLNode *a = attr->children[1];
    if (!a || a->kind != WGSL_NODE_EXPR_LITERAL_INT) return 0;
    uint32_t o = a->span_offset, l = a->span_length;
    if (!m->src || (size_t)o + l > m->src_len) return 0;
    long long v = 0;
    for (uint32_t i = 0; i < l; i++) {
        char c = m->src[o + i];
        if (c >= '0' && c <= '9') v = v * 10 + (c - '0');
        else break;
    }
    return v;
}

static void collect_attrs(M *m, const WGSLNode *const *kids, uint32_t first,
                          uint32_t count, AttrInfo *ai)
{
    memset(ai, 0, sizeof *ai);
    ai->workgroup[0] = ai->workgroup[1] = ai->workgroup[2] = 1;
    if (!kids) return;
    for (uint32_t i = 0; i < count; i++) {
        const WGSLNode *a = kids[first + i];
        if (!a || a->kind != WGSL_NODE_ATTRIBUTE || a->child_count == 0) continue;
        const WGSLNode *nm = a->children[0];
        if (!nm || nm->kind != WGSL_NODE_EXPR_IDENT) continue;
        uint32_t o = wgsl_node_name_span(nm).offset;
        uint32_t l = wgsl_node_name_span(nm).length;
        if (m_name_eq(m, o, l, "vertex")) ai->stage = 1;
        else if (m_name_eq(m, o, l, "fragment")) ai->stage = 2;
        else if (m_name_eq(m, o, l, "compute")) ai->stage = 3;
        else if (m_name_eq(m, o, l, "location")) {
            ai->has_location = 1;
            ai->location = attr_int_arg(m, a);
        } else if (m_name_eq(m, o, l, "group")) {
            ai->has_group = 1;
            ai->group = attr_int_arg(m, a);
        } else if (m_name_eq(m, o, l, "binding")) {
            ai->has_binding = 1;
            ai->binding = attr_int_arg(m, a);
        } else if (m_name_eq(m, o, l, "builtin") && a->child_count >= 2) {
            const WGSLNode *bv = a->children[1];
            if (bv && bv->kind == WGSL_NODE_EXPR_IDENT) {
                uint32_t bo = wgsl_node_name_span(bv).offset;
                uint32_t bl = wgsl_node_name_span(bv).length;
                const char *mapped = NULL;
                if (m->src && (size_t)bo + bl <= m->src_len)
                    mapped = map_builtin(m->src + bo, bl);
                ai->has_builtin = 1;
                if (mapped)
                    snprintf(ai->builtin, sizeof ai->builtin, "%s", mapped);
                else if (m->src && bl < sizeof ai->builtin)
                    snprintf(ai->builtin, sizeof ai->builtin, "%.*s",
                             (int)bl, m->src + bo);
            }
        } else if (m_name_eq(m, o, l, "workgroup_size")) {
            ai->has_workgroup = 1;
            for (uint32_t k = 1; k < a->child_count && k <= 3; k++) {
                const WGSLNode *c = a->children[k];
                if (c && c->kind == WGSL_NODE_EXPR_LITERAL_INT) {
                    long long v = 0;
                    uint32_t co = c->span_offset, cl = c->span_length;
                    if (m->src && (size_t)co + cl <= m->src_len) {
                        for (uint32_t j = 0; j < cl; j++) {
                            char ch = m->src[co + j];
                            if (ch >= '0' && ch <= '9') v = v * 10 + (ch - '0');
                            else break;
                        }
                    }
                    ai->workgroup[k - 1] = (int)(v > 0 ? v : 1);
                }
            }
        }
    }
}

/* loc_role: 0=inter-stage user, 1=vertex attribute, 2=fragment color out
 * res_attr: 0=none/legacy-buffer-flag, 1=buffer, 2=texture, 3=sampler,
 *           4=threadgroup.  for_buffer (old API) maps to res_attr=1. */
static void emit_msl_attrs_trail(M *m, const AttrInfo *ai, int for_buffer,
                                 int loc_role)
{
    int res_attr = for_buffer ? 1 : 0;
    if (ai->has_builtin) {
        m_puts(m, " [[");
        m_puts(m, ai->builtin);
        m_puts(m, "]]");
    } else if (ai->has_location) {
        char t[40];
        if (loc_role == 2)
            snprintf(t, sizeof t, " [[color(%lld)]]", ai->location);
        else if (loc_role == 1)
            snprintf(t, sizeof t, " [[attribute(%lld)]]", ai->location);
        else
            snprintf(t, sizeof t, " [[user(locn%lld)]]", ai->location);
        m_puts(m, t);
    } else if (ai->has_binding && res_attr > 0) {
        char t[56];
        if (res_attr == 2)
            snprintf(t, sizeof t, " [[texture(%lld)]]", ai->binding);
        else if (res_attr == 3)
            snprintf(t, sizeof t, " [[sampler(%lld)]]", ai->binding);
        else if (res_attr == 4)
            snprintf(t, sizeof t, " [[threadgroup(%lld)]]", ai->binding);
        else
            snprintf(t, sizeof t, " [[buffer(%lld)]]", ai->binding);
        m_puts(m, t);
    }
}

/* Same as emit_msl_attrs_trail but with explicit resource attribute class. */
static void emit_msl_res_attr(M *m, long long binding, int res_attr)
{
    if (res_attr <= 0) return;
    char t[56];
    if (res_attr == 2)
        snprintf(t, sizeof t, " [[texture(%lld)]]", binding);
    else if (res_attr == 3)
        snprintf(t, sizeof t, " [[sampler(%lld)]]", binding);
    else if (res_attr == 4)
        snprintf(t, sizeof t, " [[threadgroup(%lld)]]", binding);
    else
        snprintf(t, sizeof t, " [[buffer(%lld)]]", binding);
    m_puts(m, t);
}

/* ── §B3b resource data-plane ──────────────────────────────────────── */

static int type_spec_is_texture(M *m, const WGSLNode *ts) {
    if (!ts || !m->src) return 0;
    const WGSLNode *id = ts;
    if (ts->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT && ts->child_count)
        id = ts->children[0];
    if (!id || id->kind != WGSL_NODE_EXPR_IDENT) return 0;
    uint32_t o = wgsl_node_name_span(id).offset;
    uint32_t l = wgsl_node_name_span(id).length;
    return l >= 7 && (size_t)o + l <= m->src_len &&
           memcmp(m->src + o, "texture", 7) == 0;
}

static int type_spec_is_sampler(M *m, const WGSLNode *ts) {
    if (!ts || !m->src) return 0;
    const WGSLNode *id = ts;
    if (ts->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT && ts->child_count)
        id = ts->children[0];
    if (!id || id->kind != WGSL_NODE_EXPR_IDENT) return 0;
    uint32_t o = wgsl_node_name_span(id).offset;
    uint32_t l = wgsl_node_name_span(id).length;
    return m_name_eq(m, o, l, "sampler") ||
           m_name_eq(m, o, l, "sampler_comparison");
}

/* array<T> / array<T,N> → 1 and *out_elem = T. */
static int type_spec_array_elem(M *m, const WGSLNode *ts, const WGSLNode **out_elem) {
    if (!ts || !out_elem) return 0;
    *out_elem = NULL;
    if (ts->kind != WGSL_NODE_EXPR_TEMPLATED_IDENT || !ts->child_count) return 0;
    if (!m_ident_eq(m, ts->children[0], "array")) return 0;
    for (uint32_t i = 1; i < ts->child_count; i++) {
        const WGSLNode *c = ts->children[i];
        if (!c) continue;
        if (c->kind == WGSL_NODE_TEMPLATE_ARG_LIST && c->child_count) {
            *out_elem = c->children[0];
            return 1;
        }
        if (c->kind != WGSL_NODE_EXPR_LITERAL_INT) {
            *out_elem = c;
            return 1;
        }
    }
    return 0;
}

static int classify_module_var(M *m, const WGSLNode *n, MslRes *out) {
    if (!n || n->kind != WGSL_NODE_DECL_VAR) return 0;
    memset(out, 0, sizeof *out);
    out->decl = n;
    out->name_off = wgsl_node_name_span(n).offset;
    out->name_len = wgsl_node_name_span(n).length;
    size_t nn = out->name_len < sizeof out->name - 1 ? out->name_len : sizeof out->name - 1;
    if (m->src && (size_t)out->name_off + nn <= m->src_len) {
        memcpy(out->name, m->src + out->name_off, nn);
        out->name[nn] = 0;
    } else {
        snprintf(out->name, sizeof out->name, "res");
    }

    uint32_t attrs = wgsl_varlike_attr_count(n);
    uint32_t tpls  = wgsl_varlike_tpl_count(n);
    int has_type   = wgsl_varlike_has_type(n);
    AttrInfo ai;
    collect_attrs(m, (const WGSLNode *const *)n->children, 0, attrs, &ai);
    out->has_binding = ai.has_binding;
    out->binding = ai.binding;

    uint32_t ci = attrs;
    const char *aspace = NULL;
    int read_only = 0;
    if (tpls > 0 && ci < n->child_count) {
        const WGSLNode *t0 = n->children[ci];
        if (t0 && m_ident_eq(m, t0, "storage")) aspace = "storage";
        else if (t0 && m_ident_eq(m, t0, "uniform")) aspace = "uniform";
        else if (t0 && m_ident_eq(m, t0, "workgroup")) aspace = "workgroup";
        else if (t0 && m_ident_eq(m, t0, "private")) aspace = "private";
        else if (t0 && m_ident_eq(m, t0, "function")) aspace = "function";
        /* access mode for storage */
        if (tpls >= 2 && ci + 1 < n->child_count) {
            const WGSLNode *am = n->children[ci + 1];
            if (am && m_ident_eq(m, am, "read")) read_only = 1;
        }
        ci += tpls;
    }
    out->read_only = read_only;
    out->type_spec = (has_type && ci < n->child_count) ? n->children[ci] : NULL;

    if (type_spec_is_texture(m, out->type_spec)) {
        out->kind = MSL_RES_TEXTURE;
        return 1;
    }
    if (type_spec_is_sampler(m, out->type_spec)) {
        out->kind = MSL_RES_SAMPLER;
        return 1;
    }
    if (aspace && strcmp(aspace, "storage") == 0) {
        out->kind = MSL_RES_BUFFER;
        return 1;
    }
    if (aspace && strcmp(aspace, "uniform") == 0) {
        out->kind = MSL_RES_CONSTANT;
        return 1;
    }
    if (aspace && strcmp(aspace, "workgroup") == 0) {
        out->kind = MSL_RES_THREADGROUP;
        return 1;
    }
    if (aspace && strcmp(aspace, "private") == 0) {
        out->kind = MSL_RES_PRIVATE;
        return 1;
    }
    /* Handle-space without template (texture/sampler already handled). */
    return 0;
}

static void collect_module_resources(M *m, const WGSLNode *tu) {
    m->nresources = 0;
    if (!tu || tu->kind != WGSL_NODE_TRANSLATION_UNIT) return;
    for (uint32_t i = 0; i < tu->child_count && m->nresources < MSL_RES_MAX; i++) {
        const WGSLNode *c = tu->children[i];
        if (!c || c->kind != WGSL_NODE_DECL_VAR) continue;
        MslRes r;
        if (!classify_module_var(m, c, &r)) continue;
        m->resources[m->nresources++] = r;
    }
    if (m->nresources == MSL_RES_MAX) {
        m_diag(m, WGSL_MSL_DIAG_PARTIAL_IO, 0, 0,
               "resource data-plane cap (64) hit; extra module vars not "
               "injected as entry params");
    }
}

/* Emit pointee type for buffer-like params: peel array → element. */
static void emit_res_pointee(M *m, const MslRes *r, int depth) {
    const WGSLNode *elem = NULL;
    if (r->type_spec && type_spec_array_elem(m, r->type_spec, &elem) && elem) {
        emit_type_spec(m, elem, depth + 1);
        return;
    }
    if (r->type_spec)
        emit_type_spec(m, r->type_spec, depth + 1);
    else
        m_puts(m, "uint");
}

static int res_attr_class(MslResKind k) {
    switch (k) {
    case MSL_RES_BUFFER:
    case MSL_RES_CONSTANT:    return 1;
    case MSL_RES_TEXTURE:     return 2;
    case MSL_RES_SAMPLER:     return 3;
    case MSL_RES_THREADGROUP: return 4;
    default:                  return 0;
    }
}

/* Emit one resource as an entry-parameter clause (no leading comma). */
static void emit_resource_param(M *m, const MslRes *r, int depth) {
    long long bind = r->has_binding ? r->binding : 0;
    switch (r->kind) {
    case MSL_RES_BUFFER:
        if (r->read_only) m_puts(m, "const device ");
        else m_puts(m, "device ");
        emit_res_pointee(m, r, depth);
        m_puts(m, " *");
        m_puts(m, r->name);
        emit_msl_res_attr(m, bind, 1);
        break;
    case MSL_RES_CONSTANT:
        m_puts(m, "constant ");
        emit_res_pointee(m, r, depth);
        m_puts(m, " *");
        m_puts(m, r->name);
        emit_msl_res_attr(m, bind, 1);
        break;
    case MSL_RES_THREADGROUP:
        m_puts(m, "threadgroup ");
        emit_res_pointee(m, r, depth);
        m_puts(m, " *");
        m_puts(m, r->name);
        emit_msl_res_attr(m, bind, 4);
        break;
    case MSL_RES_TEXTURE:
        if (r->type_spec) emit_type_spec(m, r->type_spec, depth + 1);
        else m_puts(m, "texture2d<float>");
        m_putc(m, ' ');
        m_puts(m, r->name);
        emit_msl_res_attr(m, bind, 2);
        break;
    case MSL_RES_SAMPLER:
        m_puts(m, "sampler ");
        m_puts(m, r->name);
        emit_msl_res_attr(m, bind, 3);
        break;
    case MSL_RES_PRIVATE:
        /* Injected as body local, not a param. */
        break;
    }
    (void)res_attr_class;
}

/* thread locals for var<private> at the start of an entry body. */
static void emit_private_preamble(M *m, int depth) {
    for (int i = 0; i < m->nresources; i++) {
        const MslRes *r = &m->resources[i];
        if (r->kind != MSL_RES_PRIVATE) continue;
        m_indent(m);
        m_puts(m, "thread ");
        if (r->type_spec) emit_type_spec(m, r->type_spec, depth + 1);
        else m_puts(m, "float");
        m_putc(m, ' ');
        m_puts(m, r->name);
        m_puts(m, ";\n");
    }
}

static int module_var_is_resource(M *m, const WGSLNode *n) {
    for (int i = 0; i < m->nresources; i++)
        if (m->resources[i].decl == n) return 1;
    return 0;
}

static int texture_type_is_arrayed(const WGSLTypeInfo *t) {
    if (!t || t->kind != WGSL_TYPE_TEXTURE) return 0;
    return t->width == WGSL_TEX_DIM_2D_ARRAY ||
           t->width == WGSL_TEX_DIM_CUBE_ARRAY ||
           t->width == WGSL_TEX_DIM_DEPTH_2D_ARRAY ||
           t->width == WGSL_TEX_DIM_DEPTH_CUBE_ARRAY ||
           t->width == WGSL_TEX_DIM_STORAGE_2D_ARRAY;
}

static int expr_texture_is_arrayed(M *m, const WGSLNode *tex) {
    return texture_type_is_arrayed(node_type(m, tex));
}

static void emit_remaining_args(M *m, const WGSLNode *n, uint32_t first, int depth) {
    for (uint32_t i = first; i < n->child_count; i++) {
        m_puts(m, ", ");
        emit_expr(m, n->children[i], depth + 1);
    }
}

static int emit_texture_sample_call(
    M *m, const WGSLNode *n, const char *name, uint32_t len, int depth)
{
    if (!n || n->child_count < 4) return 0;
    int is_sample =
        (len == 13 && memcmp(name, "textureSample", 13) == 0);
    int is_level =
        (len == 18 && memcmp(name, "textureSampleLevel", 18) == 0);
    int is_bias =
        (len == 17 && memcmp(name, "textureSampleBias", 17) == 0);
    int is_grad =
        (len == 17 && memcmp(name, "textureSampleGrad", 17) == 0);
    int is_compare =
        (len == 20 && memcmp(name, "textureSampleCompare", 20) == 0);
    int is_compare_level =
        (len == 25 && memcmp(name, "textureSampleCompareLevel", 25) == 0);
    int is_base_clamp =
        (len == 28 && memcmp(name, "textureSampleBaseClampToEdge", 28) == 0);
    if (!is_sample && !is_level && !is_bias && !is_grad &&
        !is_compare && !is_compare_level && !is_base_clamp)
    {
        return 0;
    }

    int arrayed = expr_texture_is_arrayed(m, n->children[1]);
    emit_expr(m, n->children[1], depth + 1);
    m_puts(m, (is_compare || is_compare_level) ? ".sample_compare(" : ".sample(");
    emit_expr(m, n->children[2], depth + 1); /* sampler */
    m_puts(m, ", ");
    emit_expr(m, n->children[3], depth + 1); /* coords */
    uint32_t next = 4;
    if (arrayed && next < n->child_count) {
        m_puts(m, ", ");
        emit_expr(m, n->children[next++], depth + 1);
    }

    if ((is_compare || is_compare_level) && next < n->child_count) {
        m_puts(m, ", ");
        emit_expr(m, n->children[next++], depth + 1);
        emit_remaining_args(m, n, next, depth);
    } else if ((is_level || is_bias) && next < n->child_count) {
        m_puts(m, is_level ? ", level(" : ", bias(");
        emit_expr(m, n->children[next++], depth + 1);
        m_putc(m, ')');
        emit_remaining_args(m, n, next, depth);
    } else if (is_grad && next + 1 < n->child_count) {
        m_puts(m, ", gradient2d(");
        emit_expr(m, n->children[next++], depth + 1);
        m_puts(m, ", ");
        emit_expr(m, n->children[next++], depth + 1);
        m_putc(m, ')');
        emit_remaining_args(m, n, next, depth);
    } else {
        emit_remaining_args(m, n, next, depth);
    }
    m_putc(m, ')');
    return 1;
}

static int emit_texture_store_call(M *m, const WGSLNode *n, int depth) {
    if (!n || n->child_count < 4) return 0;
    int arrayed = expr_texture_is_arrayed(m, n->children[1]);
    uint32_t value_idx = arrayed ? 4 : 3;
    if (value_idx >= n->child_count) return 0;
    emit_expr(m, n->children[1], depth + 1);
    m_puts(m, ".write(");
    emit_expr(m, n->children[value_idx], depth + 1);
    m_puts(m, ", ");
    emit_expr(m, n->children[2], depth + 1);
    if (arrayed && n->child_count > 3) {
        m_puts(m, ", ");
        emit_expr(m, n->children[3], depth + 1);
    }
    m_putc(m, ')');
    return 1;
}
