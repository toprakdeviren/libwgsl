/**
 * @file pipelines.c — pipelines[] synthesis (compute + render candidates)
 */
#include "internal/module_json_priv.h"

/* pipelines[].
 *
 * Cross-entry synthesis: a render pipeline's bind-group layout needs
 * the *union* of vertex+fragment visibility, which no single entry's
 * bind_group_layouts can express.  We also emit the attribute / color-
 * target lists that are fully determined by WGSL IO (buffer packing and
 * texture formats remain host-defined).
 *
 * Pairing model (documented in libwgsl.md §5):
 *   - one compute pipeline per @compute entry
 *   - render: Cartesian product of vertex × fragment (or vertex-only
 *     when the module has no fragment entry), capped at 16 candidates.
 *     Beyond the cap we emit nothing for render — hosts combine
 *     entry_points[] themselves. */

enum {
    REFLECT_STAGE_VERTEX   = 1,
    REFLECT_STAGE_FRAGMENT = 2,
    REFLECT_STAGE_COMPUTE  = 4
};

static unsigned reflect_stage_bit(const char *stage) {
    if (!stage) return REFLECT_STAGE_COMPUTE;
    if (strcmp(stage, "vertex") == 0)   return REFLECT_STAGE_VERTEX;
    if (strcmp(stage, "fragment") == 0) return REFLECT_STAGE_FRAGMENT;
    return REFLECT_STAGE_COMPUTE;
}

typedef struct {
    long long   group;
    long long   binding;
    WGSLSymbol *sym;
    unsigned    stages;   /* REFLECT_STAGE_* bitset */
} MergedBindRef;

static int merged_bindref_cmp(const void *a, const void *b) {
    const MergedBindRef *x = (const MergedBindRef *)a;
    const MergedBindRef *y = (const MergedBindRef *)b;
    if (x->group   != y->group)   return x->group   < y->group   ? -1 : 1;
    if (x->binding != y->binding) return x->binding < y->binding ? -1 : 1;
    return 0;
}

static void merged_add(MergedBindRef **arr, uint32_t *n, uint32_t *cap,
                       long long group, long long binding,
                       WGSLSymbol *sym, unsigned stage_bit)
{
    for (uint32_t i = 0; i < *n; i++) {
        if ((*arr)[i].group == group && (*arr)[i].binding == binding) {
            (*arr)[i].stages |= stage_bit;
            if (!(*arr)[i].sym && sym) (*arr)[i].sym = sym;
            return;
        }
    }
    if (*n + 1 > *cap) {
        uint32_t nc = *cap ? *cap * 2 : 8;
        MergedBindRef *g = (MergedBindRef *)realloc(*arr, (size_t)nc * sizeof *g);
        if (!g) return;
        *arr = g;
        *cap = nc;
    }
    (*arr)[*n].group   = group;
    (*arr)[*n].binding = binding;
    (*arr)[*n].sym     = sym;
    (*arr)[*n].stages  = stage_bit;
    (*n)++;
}

static void emit_visibility_array(JsonBuf *j, unsigned stages) {
    jb_puts(j, ",\"visibility\":[");
    int first = 1;
    if (stages & REFLECT_STAGE_VERTEX) {
        jb_puts(j, "\"vertex\""); first = 0;
    }
    if (stages & REFLECT_STAGE_FRAGMENT) {
        if (!first) jb_putc(j, ',');
        jb_puts(j, "\"fragment\""); first = 0;
    }
    if (stages & REFLECT_STAGE_COMPUTE) {
        if (!first) jb_putc(j, ',');
        jb_puts(j, "\"compute\"");
    }
    jb_putc(j, ']');
}

static void emit_merged_bind_group_layouts(WGSLResult *r, JsonBuf *j,
                                           const MergedBindRef *refs,
                                           uint32_t nref)
{
    jb_puts(j, ",\"bind_group_layouts\":[");
    if (!refs || nref == 0) { jb_putc(j, ']'); return; }

    uint32_t i = 0;
    int first_group = 1;
    while (i < nref) {
        long long group = refs[i].group;
        if (!first_group) jb_putc(j, ',');
        first_group = 0;
        jb_puts(j, "{\"group\":");
        jb_put_int(j, group);
        jb_puts(j, ",\"entries\":[");
        int first_entry = 1;
        while (i < nref && refs[i].group == group) {
            if (!first_entry) jb_putc(j, ',');
            first_entry = 0;
            jb_puts(j, "{\"binding\":");
            jb_put_int(j, refs[i].binding);
            emit_visibility_array(j, refs[i].stages);
            emit_layout_resource_fields(r, j, refs[i].sym);
            jb_putc(j, '}');
            i++;
        }
        jb_puts(j, "]}");
    }
    jb_putc(j, ']');
}

/* Map a WGSL IO type to a GPUVertexFormat name when unambiguous.
 * Returns NULL for types with no single WebGPU format (e.g. f16x3). */
static const char *reflect_vertex_format(const WGSLTypeInfo *t) {
    if (!t) return NULL;
    switch (t->kind) {
    case WGSL_TYPE_F32: return "float32";
    case WGSL_TYPE_I32: return "sint32";
    case WGSL_TYPE_U32: return "uint32";
    case WGSL_TYPE_F16: return NULL; /* no scalar float16 GPUVertexFormat */
    case WGSL_TYPE_VEC: {
        const WGSLTypeInfo *el = (const WGSLTypeInfo *)t->ref;
        uint8_t w = t->width;
        if (!el) return NULL;
        if (el->kind == WGSL_TYPE_F32) {
            if (w == 2) return "float32x2";
            if (w == 3) return "float32x3";
            if (w == 4) return "float32x4";
        } else if (el->kind == WGSL_TYPE_I32) {
            if (w == 2) return "sint32x2";
            if (w == 3) return "sint32x3";
            if (w == 4) return "sint32x4";
        } else if (el->kind == WGSL_TYPE_U32) {
            if (w == 2) return "uint32x2";
            if (w == 3) return "uint32x3";
            if (w == 4) return "uint32x4";
        } else if (el->kind == WGSL_TYPE_F16) {
            if (w == 2) return "float16x2";
            if (w == 4) return "float16x4";
            /* no float16x3 in WebGPU */
        }
        return NULL;
    }
    default:
        return NULL;
    }
}

/* Emit @location-bearing IO slots as pipeline vertex attributes or
 * fragment color targets.  Struct-typed IO fans out like entry IO. */
static void emit_pipeline_io_slots(WGSLResult *r, JsonBuf *j, int *first,
                                   WGSLTypeInfo *type,
                                   const WGSLNode *ap, uint32_t afirst,
                                   uint32_t acount, int as_vertex_attr)
{
    if (type && type->kind == WGSL_TYPE_STRUCT && type->ref) {
        WGSLNode *sd = (WGSLNode *)type->ref;
        for (uint32_t k = 0; k < sd->child_count; k++) {
            WGSLNode *m = sd->children[k];
            if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            uint32_t MA = wgsl_member_attr_count(m);
            WGSLTypeInfo *mt = (MA < m->child_count)
                ? wgsl_typecheck_type_of(&r->tc, m->children[MA]) : NULL;
            emit_pipeline_io_slots(r, j, first, mt, m, 0, MA, as_vertex_attr);
        }
        return;
    }
    WGSLNode *loc = json_find_attr(r, ap, afirst, acount, "location");
    if (!loc || loc->child_count < 2) return; /* builtins / unlocated */
    long long location = wgsl_attr_int_arg(r, loc->children[1]);
    if (!*first) jb_putc(j, ',');
    *first = 0;
    jb_putc(j, '{');
    if (as_vertex_attr) {
        jb_puts(j, "\"shaderLocation\":");
        jb_put_int(j, location);
        const char *fmt = reflect_vertex_format(type);
        if (fmt) {
            jb_puts(j, ",\"format\":\"");
            jb_puts(j, fmt);
            jb_putc(j, '"');
        }
        jb_puts(j, ",\"type\":");
        jb_put_type(j, type);
    } else {
        jb_puts(j, "\"location\":");
        jb_put_int(j, location);
        jb_puts(j, ",\"type\":");
        jb_put_type(j, type);
    }
    jb_putc(j, '}');
}

static void emit_vertex_attributes(WGSLResult *r, JsonBuf *j, const WGSLNode *fn) {
    uint32_t FA = wgsl_fn_attr_count(fn);
    uint32_t P  = wgsl_fn_param_count(fn);
    jb_puts(j, ",\"vertex_attributes\":[");
    int first = 1;
    for (uint32_t k = 0; k < P; k++) {
        uint32_t ci = FA + k;
        if (ci >= fn->child_count) break;
        WGSLNode *p = fn->children[ci];
        if (!p || p->kind != WGSL_NODE_DECL_PARAM) continue;
        uint32_t PA = wgsl_param_attr_count(p);
        WGSLTypeInfo *pt = (PA < p->child_count)
            ? wgsl_typecheck_type_of(&r->tc, p->children[PA]) : NULL;
        emit_pipeline_io_slots(r, j, &first, pt, p, 0, PA, 1);
    }
    jb_putc(j, ']');
}

static void emit_color_targets(WGSLResult *r, JsonBuf *j, const WGSLNode *fn) {
    uint32_t FA = wgsl_fn_attr_count(fn);
    uint32_t P  = wgsl_fn_param_count(fn);
    uint32_t RA = wgsl_fn_ret_attr_count(fn);
    uint32_t HR = wgsl_fn_has_return_type(fn);
    jb_puts(j, ",\"color_targets\":[");
    int first = 1;
    if (HR) {
        uint32_t rti = FA + P + RA;
        WGSLTypeInfo *rt = (rti < fn->child_count)
            ? wgsl_typecheck_type_of(&r->tc, fn->children[rti]) : NULL;
        emit_pipeline_io_slots(r, j, &first, rt, fn, FA + P, RA, 0);
    }
    jb_putc(j, ']');
}

typedef struct {
    const WGSLNode *fn;
    const char     *stage;
    uint32_t        name_off;
    uint32_t        name_len;
    ReflectBindRef *refs;
    uint32_t        nref;
    WGSLNode       *wgs_args[3];
    int             has_wgs;
    uint64_t        workgroup_bytes;
} PipelineEntry;

static void pipeline_entry_free(PipelineEntry *e) {
    if (!e) return;
    free(e->refs);
    e->refs = NULL;
    e->nref = 0;
}

static int pipeline_entry_init(WGSLResult *r, PipelineEntry *e, const WGSLNode *fn,
                               const char *stage)
{
    memset(e, 0, sizeof *e);
    e->fn = fn;
    e->stage = stage;
    e->name_off = wgsl_node_name_span(fn).offset;
    e->name_len = wgsl_node_name_span(fn).length;

    EntryUsage eu = {0};
    eu.r = r;
    eu_collect(&eu, fn);
    reflect_fill_bind_refs(r, &eu, &e->refs, &e->nref);

    if (stage && strcmp(stage, "compute") == 0) {
        uint32_t FA = wgsl_fn_attr_count(fn);
        WGSLNode *wgs = json_find_attr(r, fn, 0, FA, "workgroup_size");
        if (wgs) {
            e->has_wgs = 1;
            for (uint32_t k = 1; k < wgs->child_count && k <= 3; k++)
                e->wgs_args[k - 1] = wgs->children[k];
        }
        e->workgroup_bytes = reflect_workgroup_bytes(r, &eu);
    }
    free(eu.fns);
    free(eu.vars);
    return 1;
}

static void emit_merged_from_entries(WGSLResult *r, JsonBuf *j,
                                     const PipelineEntry *a,
                                     const PipelineEntry *b)
{
    MergedBindRef *merged = NULL;
    uint32_t n = 0, cap = 0;
    if (a) {
        unsigned sb = reflect_stage_bit(a->stage);
        for (uint32_t i = 0; i < a->nref; i++)
            merged_add(&merged, &n, &cap, a->refs[i].group, a->refs[i].binding,
                       a->refs[i].sym, sb);
    }
    if (b) {
        unsigned sb = reflect_stage_bit(b->stage);
        for (uint32_t i = 0; i < b->nref; i++)
            merged_add(&merged, &n, &cap, b->refs[i].group, b->refs[i].binding,
                       b->refs[i].sym, sb);
    }
    if (merged && n > 1) qsort(merged, n, sizeof *merged, merged_bindref_cmp);
    emit_merged_bind_group_layouts(r, j, merged, n);
    free(merged);
}

static void emit_compute_pipeline(WGSLResult *r, JsonBuf *j, const PipelineEntry *e) {
    jb_puts(j, "{\"kind\":\"compute\",\"entry_point\":");
    jb_put_span(j, &r->source, e->name_off, e->name_len);
    if (e->has_wgs) {
        jb_puts(j, ",\"workgroup_size\":[");
        jb_put_wgs_dim(r, j, e->wgs_args[0]); jb_putc(j, ',');
        jb_put_wgs_dim(r, j, e->wgs_args[1]); jb_putc(j, ',');
        jb_put_wgs_dim(r, j, e->wgs_args[2]);
        jb_putc(j, ']');
    }
    jb_puts(j, ",\"workgroup_bytes\":");
    jb_put_int(j, (long long)e->workgroup_bytes);
    /* Pipeline-level visibility is an array even for single-stage. */
    {
        MergedBindRef *merged = NULL;
        uint32_t n = 0, cap = 0;
        unsigned sb = REFLECT_STAGE_COMPUTE;
        for (uint32_t i = 0; i < e->nref; i++)
            merged_add(&merged, &n, &cap, e->refs[i].group, e->refs[i].binding,
                       e->refs[i].sym, sb);
        if (merged && n > 1) qsort(merged, n, sizeof *merged, merged_bindref_cmp);
        emit_merged_bind_group_layouts(r, j, merged, n);
        free(merged);
    }
    jb_putc(j, '}');
}

static void emit_render_pipeline(WGSLResult *r, JsonBuf *j,
                                 const PipelineEntry *vs,
                                 const PipelineEntry *fs,
                                 int multi_candidate)
{
    jb_puts(j, "{\"kind\":\"render\",\"vertex\":");
    jb_put_span(j, &r->source, vs->name_off, vs->name_len);
    if (fs) {
        jb_puts(j, ",\"fragment\":");
        jb_put_span(j, &r->source, fs->name_off, fs->name_len);
    }
    if (multi_candidate)
        jb_puts(j, ",\"inferred\":true");
    emit_merged_from_entries(r, j, vs, fs);
    emit_vertex_attributes(r, j, vs->fn);
    if (fs) emit_color_targets(r, j, fs->fn);
    else    jb_puts(j, ",\"color_targets\":[]");
    jb_putc(j, '}');
}

/* Internal reflection cap: not a WGSL limit.  When V×F exceeds this,
 * render pipelines are omitted and the caller must emit analysis_truncated. */
#define REFLECT_PIPELINE_RENDER_CAP 16

int emit_pipelines(WGSLResult *r, JsonBuf *j) {
    int truncated = 0;
    jb_puts(j, ",\"pipelines\":[");
    if (!r->ast.root) { jb_putc(j, ']'); return 0; }

    PipelineEntry *entries = NULL;
    uint32_t nent = 0, ecap = 0;
    uint32_t *vi = NULL, nvi = 0, vcap = 0;
    uint32_t *fi = NULL, nfi = 0, fcap = 0;
    uint32_t *ci = NULL, nci = 0, ccap = 0;

    WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n || n->kind != WGSL_NODE_DECL_FUNCTION) continue;
        uint32_t FA = wgsl_fn_attr_count(n);
        const char *stage = json_stage_for(r, n, FA);
        if (!stage) continue;

        if (nent + 1 > ecap) {
            uint32_t nc = ecap ? ecap * 2 : 4;
            PipelineEntry *g = (PipelineEntry *)realloc(entries, (size_t)nc * sizeof *g);
            if (!g) goto done;
            entries = g;
            ecap = nc;
        }
        if (!pipeline_entry_init(r, &entries[nent], n, stage)) continue;
        uint32_t idx = nent++;

        uint32_t **list = NULL, *ln = NULL, *lc = NULL;
        if (strcmp(stage, "vertex") == 0)        { list = &vi; ln = &nvi; lc = &vcap; }
        else if (strcmp(stage, "fragment") == 0) { list = &fi; ln = &nfi; lc = &fcap; }
        else                                      { list = &ci; ln = &nci; lc = &ccap; }
        if (*ln + 1 > *lc) {
            uint32_t nc = *lc ? *lc * 2 : 4;
            uint32_t *g = (uint32_t *)realloc(*list, (size_t)nc * sizeof *g);
            if (!g) goto done;
            *list = g;
            *lc = nc;
        }
        (*list)[(*ln)++] = idx;
    }

    int first = 1;
    for (uint32_t i = 0; i < nci; i++) {
        if (!first) jb_putc(j, ',');
        first = 0;
        emit_compute_pipeline(r, j, &entries[ci[i]]);
    }

    /* Render pairing: V×F product (F=0 → vertex-only). Cap is an internal
     * reflection budget; over-cap → omit render entries + truncated. */
    {
        uint32_t fslots = nfi ? nfi : 1; /* one null-fragment slot when F=0 */
        uint64_t product = (uint64_t)nvi * (uint64_t)fslots;
        if (nvi > 0 && product > REFLECT_PIPELINE_RENDER_CAP) {
            truncated = 1;
        } else if (nvi > 0 && product <= REFLECT_PIPELINE_RENDER_CAP) {
            int multi = (product > 1) ? 1 : 0;
            for (uint32_t v = 0; v < nvi; v++) {
                if (nfi == 0) {
                    if (!first) jb_putc(j, ',');
                    first = 0;
                    emit_render_pipeline(r, j, &entries[vi[v]], NULL, multi);
                } else {
                    for (uint32_t f = 0; f < nfi; f++) {
                        if (!first) jb_putc(j, ',');
                        first = 0;
                        emit_render_pipeline(r, j, &entries[vi[v]],
                                             &entries[fi[f]], multi);
                    }
                }
            }
        }
    }

done:
    for (uint32_t i = 0; i < nent; i++) pipeline_entry_free(&entries[i]);
    free(entries);
    free(vi);
    free(fi);
    free(ci);
    jb_putc(j, ']');
    return truncated;
}

/* Structured `handle` sub-record for a texture resource, giving
 * the WebGPU GPUBindGroupLayoutEntry.texture / .storageTexture fields
 * (view dimension, sample type, format, access, multisampled) that the
 * flat "type" string can't be pattern-matched for reliably. */
