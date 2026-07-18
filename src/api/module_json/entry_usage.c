/**
 * @file entry_usage.c — per-entry resource usage analysis + bind-group layouts
 */
#include "internal/module_json_priv.h"

/* Entry-scope usage analysis.
 *
 * WebGPU derives a *per-entry-point* bind-group layout, so reflection
 * has to answer "which (group,binding) resources does THIS entry touch"
 * — directly or through any function it transitively calls.  The single
 * global `resources[]` cannot.  We traverse the entry's call graph as a
 * worklist (WGSL forbids recursion — §10.3.1 — so it is a DAG) and
 * collect every module-scope resource var referenced by name in the
 * reachable code.  The traversal is a BFS over functions (bounded by the
 * function count) with only AST-structural recursion inside each body
 * (bounded by WGSL_MAX_AST_DEPTH) — so no deep call chain can overflow
 * the stack.  The same reachable set feeds per-entry feature flags and
 * the workgroup shared-memory byte total. */

/* Returns 1 if `fn` was newly enqueued, 0 if already present or on OOM. */
static int eu_add_fn(EntryUsage *eu, const WGSLNode *fn) {
    for (uint32_t i = 0; i < eu->fn_count; i++)
        if (eu->fns[i] == fn) return 0;
    if (eu->fn_count == eu->fn_cap) {
        uint32_t nc = eu->fn_cap ? eu->fn_cap * 2 : 8;
        const WGSLNode **g = (const WGSLNode **)realloc(eu->fns, (size_t)nc * sizeof *g);
        if (!g) { eu->oom = 1; return 0; }
        eu->fns = g; eu->fn_cap = nc;
    }
    eu->fns[eu->fn_count++] = fn;
    return 1;
}

static void eu_add_var(EntryUsage *eu, WGSLSymbol *s) {
    for (uint32_t i = 0; i < eu->var_count; i++)
        if (eu->vars[i] == s) return;
    if (eu->var_count == eu->var_cap) {
        uint32_t nc = eu->var_cap ? eu->var_cap * 2 : 8;
        WGSLSymbol **g = (WGSLSymbol **)realloc(eu->vars, (size_t)nc * sizeof *g);
        if (!g) { eu->oom = 1; return; }
        eu->vars = g; eu->var_cap = nc;
    }
    eu->vars[eu->var_count++] = s;
}

/* Classify a call by callee name to note subgroup / quad cooperative
 * use — the pipeline-relevant capability an entry may need enabled. */
static void eu_note_builtin(EntryUsage *eu, const char *nm, uint32_t len) {
    if ((len >= 8 && memcmp(nm, "subgroup", 8) == 0) ||
        (len >= 4 && memcmp(nm, "quad", 4) == 0))
        eu->uses |= REFLECT_USE_SUBGROUPS;
}

static int reflect_type_has_f16(const WGSLTypeInfo *t, int depth) {
    if (!t || depth > 16) return 0;
    if (t->kind == WGSL_TYPE_F16) return 1;
    switch (t->kind) {
    case WGSL_TYPE_VEC: case WGSL_TYPE_MAT:
    case WGSL_TYPE_ATOMIC: case WGSL_TYPE_ARRAY:
        return reflect_type_has_f16((const WGSLTypeInfo *)t->ref, depth + 1);
    default:
        /* Struct members aren't walked here; an actual f16 field use
         * surfaces through its member-access expression's type. */
        return 0;
    }
}

/* Walk one function body (AST-structural recursion only); enqueue any
 * user function it calls; record referenced resource vars and features. */
static void eu_walk_body(EntryUsage *eu, const WGSLNode *n, int depth) {
    if (!n || eu->oom) return;
    if (depth > WGSL_MAX_AST_DEPTH) { eu->oom = 1; return; }

    switch (n->kind) {
    case WGSL_NODE_EXPR_IDENT: {
        WGSLSymbol *s = wgsl_node_resolved_symbol(n);
        if (s && s->kind == WGSL_SYM_VAR &&
            (s->as == WGSL_AS_STORAGE || s->as == WGSL_AS_UNIFORM ||
             s->as == WGSL_AS_HANDLE  || s->as == WGSL_AS_WORKGROUP))
            eu_add_var(eu, s);
        break;
    }
    case WGSL_NODE_EXPR_CALL: {
        const WGSLNode *callee = n->child_count ? n->children[0] : NULL;
        if (callee && callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
            callee->child_count)
            callee = callee->children[0];
        if (callee && callee->kind == WGSL_NODE_EXPR_IDENT) {
            WGSLSymbol *fs = wgsl_node_resolved_symbol(callee);
            if (fs && fs->kind == WGSL_SYM_FUNCTION && fs->ast) {
                eu_add_fn(eu, fs->ast);   /* enqueue; walked by eu_collect */
            } else if (!fs || fs->kind == WGSL_SYM_PREDECLARED_FN) {
                /* A builtin call — classify by name.  Gate on "is a
                 * predeclared fn (or unresolved builtin)" so a value
                 * constructor for a user struct/alias whose name happens
                 * to start with "subgroup"/"quad" doesn't false-trigger
                 * the subgroups feature (it resolves to WGSL_SYM_STRUCT /
                 * _TYPE_ALIAS, not a function). */
                uint32_t off = wgsl_node_name_span(callee).offset;
                uint32_t len = wgsl_node_name_span(callee).length;
                if (eu->r->source.bytes &&
                    (size_t)off + len <= eu->r->source.length)
                    eu_note_builtin(eu, eu->r->source.bytes + off, len);
            }
        }
        break;
    }
    default: break;
    }

    if (!(eu->uses & REFLECT_USE_F16) &&
        n->kind >= WGSL_NODE_EXPR_LITERAL_BOOL &&
        n->kind <= WGSL_NODE_EXPR_INDIRECTION) {
        WGSLTypeInfo *t = wgsl_typecheck_type_of(&eu->r->tc, n);
        if (reflect_type_has_f16(t, 0)) eu->uses |= REFLECT_USE_F16;
    }

    for (uint32_t i = 0; i < n->child_count; i++)
        eu_walk_body(eu, n->children[i], depth + 1);
}

/* Compute the reachable-function closure + referenced resources for an
 * entry point.  Caller owns eu.fns / eu.vars and must free() them. */
void eu_collect(EntryUsage *eu, const WGSLNode *entry_fn) {
    if (!entry_fn) return;
    eu_add_fn(eu, entry_fn);
    for (uint32_t idx = 0; idx < eu->fn_count && !eu->oom; idx++)
        eu_walk_body(eu, eu->fns[idx], 0);
}

static int reflect_bindref_cmp(const void *a, const void *b) {
    const ReflectBindRef *x = (const ReflectBindRef *)a;
    const ReflectBindRef *y = (const ReflectBindRef *)b;
    if (x->group   != y->group)   return x->group   < y->group   ? -1 : 1;
    if (x->binding != y->binding) return x->binding < y->binding ? -1 : 1;
    return 0;
}

static uint64_t reflect_round_up(uint64_t x, uint64_t a) {
    if (a == 0) return x;
    uint64_t r = x % a;
    return r ? x + (a - r) : x;
}

/* WebGPU workgroup-storage size for a compute entry: pack the reachable
 * `var<workgroup>` in declaration order, each rounded up to its own
 * alignment, and round the total to the max alignment (WebGPU
 * §"workgroup storage size" — the device workgroupStorageSize limit). */
uint64_t reflect_workgroup_bytes(WGSLResult *r, EntryUsage *eu) {
    WGSLSymbol **wg = NULL;
    uint32_t nwg = 0;
    for (uint32_t i = 0; i < eu->var_count; i++) {
        WGSLSymbol *s = eu->vars[i];
        if (s->as != WGSL_AS_WORKGROUP || !s->type) continue;
        WGSLSymbol **g = (WGSLSymbol **)realloc(wg, (size_t)(nwg + 1) * sizeof *g);
        if (!g) { free(wg); return 0; }
        wg = g; wg[nwg++] = s;
    }
    if (!wg) return 0;
    /* deterministic packing order: source declaration order */
    for (uint32_t a = 1; a < nwg; a++)
        for (uint32_t b = a; b > 0 &&
                 wg[b - 1]->ast && wg[b]->ast &&
                 wg[b - 1]->ast->span_offset > wg[b]->ast->span_offset; b--) {
            WGSLSymbol *t = wg[b - 1]; wg[b - 1] = wg[b]; wg[b] = t;
        }
    WGSLLayoutCtx lcx = { &r->tc, &r->cev, &r->source, 0 };
    uint64_t off = 0, maxa = 1;
    for (uint32_t i = 0; i < nwg; i++) {
        uint32_t al = wgsl_layout_align_of(&lcx, wg[i]->type, WGSL_AS_WORKGROUP);
        uint32_t sz = wgsl_layout_size_of (&lcx, wg[i]->type, WGSL_AS_WORKGROUP);
        if (al == 0) al = 1;
        if (al > maxa) maxa = al;
        off = reflect_round_up(off, al) + sz;
    }
    free(wg);
    return reflect_round_up(off, maxa);
}

/* WebGPU visibility is entry-stage local for synthesized
 * bind-group layout.  Keep this as lower-case stage names for host
 * readability. */
static const char *reflect_stage_visibility(const char *stage) {
    if (!stage) return "compute";
    if (strcmp(stage, "vertex") == 0)   return "vertex";
    if (strcmp(stage, "fragment") == 0) return "fragment";
    return "compute";
}

/* Texture kind helpers for bind-group-layout synthesis from checker-resolved
 * texture types. */
static const char *reflect_texture_view_dimension(const WGSLTextureDim d,
                                                int *multisampled,
                                                int *is_storage,
                                                int *is_depth,
                                                int *is_external)
{
    const char *view = "2d";
    *multisampled = 0;
    *is_storage = 0;
    *is_depth = 0;
    *is_external = 0;
    switch (d) {
    case WGSL_TEX_DIM_1D:            view = "1d";         break;
    case WGSL_TEX_DIM_2D:            view = "2d";         break;
    case WGSL_TEX_DIM_2D_ARRAY:      view = "2d-array";   break;
    case WGSL_TEX_DIM_3D:            view = "3d";         break;
    case WGSL_TEX_DIM_CUBE:          view = "cube";       break;
    case WGSL_TEX_DIM_CUBE_ARRAY:    view = "cube-array"; break;
    case WGSL_TEX_DIM_MULTISAMPLED_2D:
        *multisampled = 1;
        view = "2d";
        break;
    case WGSL_TEX_DIM_EXTERNAL:
        *is_external = 1;
        view = "2d";
        break;
    case WGSL_TEX_DIM_DEPTH_2D:
        *is_depth = 1;
        view = "2d";
        break;
    case WGSL_TEX_DIM_DEPTH_2D_ARRAY:
        *is_depth = 1;
        view = "2d-array";
        break;
    case WGSL_TEX_DIM_DEPTH_CUBE:
        *is_depth = 1;
        view = "cube";
        break;
    case WGSL_TEX_DIM_DEPTH_CUBE_ARRAY:
        *is_depth = 1;
        view = "cube-array";
        break;
    case WGSL_TEX_DIM_DEPTH_MULTISAMPLED_2D:
        *is_depth = 1;
        *multisampled = 1;
        view = "2d";
        break;
    case WGSL_TEX_DIM_STORAGE_1D:
        *is_storage = 1;
        view = "1d";
        break;
    case WGSL_TEX_DIM_STORAGE_2D:
        *is_storage = 1;
        view = "2d";
        break;
    case WGSL_TEX_DIM_STORAGE_2D_ARRAY:
        *is_storage = 1;
        view = "2d-array";
        break;
    case WGSL_TEX_DIM_STORAGE_3D:
        *is_storage = 1;
        view = "3d";
        break;
    default:
        break;
    }
    return view;
}

static const char *reflect_texture_sample_type(const WGSLTypeInfo *t, int is_depth) {
    if (is_depth) return "depth";
    const WGSLTypeInfo *el = (const WGSLTypeInfo *)t->ref;
    if (el && el->kind == WGSL_TYPE_U32) return "uint";
    if (el && el->kind == WGSL_TYPE_I32) return "sint";
    return "float";
}

static const char *reflect_sampler_type(const WGSLTypeInfo *t) {
    return t && t->width ? "comparison" : "filtering";
}

static const char *reflect_storage_buffer_type(const WGSLSymbol *sym) {
    if (!sym) return "storage";
    if (sym->as == WGSL_AS_UNIFORM) return "uniform";
    if (sym->am == WGSL_ACCESS_READ) return "read-only-storage";
    return "storage";
}

/* Resource-kind body of a GPUBindGroupLayoutEntry (everything after
 * binding + visibility).  Shared by per-entry and pipeline synthesis. */
void emit_layout_resource_fields(WGSLResult *r, JsonBuf *j,
                                        WGSLSymbol *sym)
{
    WGSLTypeInfo *rt = sym ? sym->type : NULL;
    WGSLAddressSpace as = sym ? (WGSLAddressSpace)sym->as : WGSL_AS_NONE;
    if (!rt) return;

    if (rt->kind == WGSL_TYPE_SAMPLER) {
        jb_puts(j, ",\"sampler\":{\"type\":\"");
        jb_puts(j, reflect_sampler_type(rt));
        jb_putc(j, '"');
        jb_putc(j, '}');
    } else if (rt->kind == WGSL_TYPE_TEXTURE) {
        int ms = 0, is_storage = 0, is_depth = 0, is_external = 0;
        const char *vd = reflect_texture_view_dimension((WGSLTextureDim)rt->width,
            &ms, &is_storage, &is_depth, &is_external);
        if (is_external) {
            jb_puts(j, ",\"externalTexture\":{}");
        } else if (is_storage) {
            /* WebGPU storageTexture.access: write-only | read-only | read-write.
             * WGSL stores the access on type->rows as WGSLAccessMode. */
            const char *acc = "write-only";
            switch ((WGSLAccessMode)rt->rows) {
            case WGSL_ACCESS_READ:       acc = "read-only";  break;
            case WGSL_ACCESS_READ_WRITE: acc = "read-write"; break;
            case WGSL_ACCESS_WRITE:
            case WGSL_ACCESS_NONE:
            default:                     acc = "write-only"; break;
            }
            jb_puts(j, ",\"storageTexture\":{\"access\":\"");
            jb_puts(j, acc);
            jb_puts(j, "\",\"format\":\"");
            jb_puts(j, wgsl_texel_format_name((WGSLTexelFormat)rt->array_len));
            jb_puts(j, "\",\"viewDimension\":\"");
            jb_puts(j, vd);
            jb_putc(j, '"');
            jb_putc(j, '}');
        } else {
            jb_puts(j, ",\"texture\":{\"sampleType\":\"");
            jb_puts(j, reflect_texture_sample_type(rt, is_depth));
            jb_puts(j, "\",\"viewDimension\":\"");
            jb_puts(j, vd);
            jb_putc(j, '"');
            if (ms) jb_puts(j, ",\"multisampled\":true");
            jb_putc(j, '}');
        }
    } else if (as == WGSL_AS_STORAGE || as == WGSL_AS_UNIFORM) {
        WGSLLayoutCtx lcx = { &r->tc, &r->cev, &r->source, 0 };
        uint32_t size = wgsl_layout_size_of(&lcx, rt, as);
        jb_puts(j, ",\"buffer\":{\"type\":\"");
        jb_puts(j, reflect_storage_buffer_type(sym));
        jb_puts(j, "\",\"hasDynamicOffset\":false,\"minBindingSize\":");
        jb_put_int(j, (long long)size);
        jb_putc(j, '}');
    } else {
        /* Should not arise for a real (group,binding) resource; keep the
         * entry well-formed so hosts can still enumerate the slot. */
        jb_puts(j, ",\"unsupported\":{}");
    }
}

/* Per-entry visibility is a single stage string (the entry's stage). */
static void emit_entry_layout_entry(WGSLResult *r, JsonBuf *j,
                                   const ReflectBindRef *ref,
                                   const char *stage)
{
    const char *vis = reflect_stage_visibility(stage);
    jb_puts(j, "{\"binding\":");
    jb_put_int(j, ref ? ref->binding : 0);
    jb_puts(j, ",\"visibility\":\"");
    jb_puts(j, vis);
    jb_putc(j, '"');
    emit_layout_resource_fields(r, j, ref ? ref->sym : NULL);
    jb_putc(j, '}');
}

static void emit_bind_group_layouts_from_refs(WGSLResult *r, JsonBuf *j,
                                              const ReflectBindRef *refs,
                                              uint32_t nref, const char *stage,
                                              int leading_comma)
{
    if (leading_comma) jb_puts(j, ",\"bind_group_layouts\":[");
    else               jb_puts(j, "\"bind_group_layouts\":[");
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
            emit_entry_layout_entry(r, j, refs + i, stage);
            i++;
        }
        jb_puts(j, "]}");
    }
    jb_putc(j, ']');
}

static void emit_entry_bind_group_layouts(WGSLResult *r, JsonBuf *j,
                                         const ReflectBindRef *refs,
                                         uint32_t nref, const char *stage)
{
    emit_bind_group_layouts_from_refs(r, j, refs, nref, stage, 1);
}

/* Build the sorted (group,binding,sym) list from an EntryUsage walk. */
void reflect_fill_bind_refs(WGSLResult *r, EntryUsage *eu,
                                   ReflectBindRef **out_refs, uint32_t *out_n)
{
    *out_refs = NULL;
    *out_n = 0;
    if (!eu || !eu->var_count) return;
    ReflectBindRef *refs = (ReflectBindRef *)malloc(
        (size_t)eu->var_count * sizeof *refs);
    if (!refs) return;
    uint32_t nref = 0;
    for (uint32_t i = 0; i < eu->var_count; i++) {
        WGSLSymbol *s = eu->vars[i];
        if (!s || !s->ast) continue;
        uint32_t A = wgsl_varlike_attr_count(s->ast);
        WGSLNode *gp = json_find_attr(r, s->ast, 0, A, "group");
        WGSLNode *bd = json_find_attr(r, s->ast, 0, A, "binding");
        if (!gp || !bd) continue;   /* workgroup/private var: not a binding */
        long long group = (gp->child_count >= 2) ? wgsl_attr_int_arg(r, gp->children[1]) : 0;
        long long binding = (bd->child_count >= 2) ? wgsl_attr_int_arg(r, bd->children[1]) : 0;
        int duplicate = 0;
        for (uint32_t k = 0; k < nref; k++) {
            if (refs[k].group == group && refs[k].binding == binding) {
                duplicate = 1;
                break;
            }
        }
        if (duplicate) continue;
        refs[nref].group   = group;
        refs[nref].binding = binding;
        refs[nref].sym     = s;
        nref++;
    }
    if (nref > 1) qsort(refs, nref, sizeof *refs, reflect_bindref_cmp);
    *out_refs = refs;
    *out_n = nref;
}

/* Emit the per-entry usage-derived fields: bindings, features,
 * bind_group_layouts, and workgroup_bytes.  Computes the reachable closure
 * once. */
void emit_entry_usage(WGSLResult *r, JsonBuf *j, const WGSLNode *fn,
                             const char *stage) {
    EntryUsage eu = {0};
    eu.r = r;
    eu_collect(&eu, fn);

    ReflectBindRef *refs = NULL;
    uint32_t nref = 0;
    reflect_fill_bind_refs(r, &eu, &refs, &nref);

    jb_puts(j, ",\"bindings\":[");
    for (uint32_t i = 0; i < nref; i++) {
        if (i) jb_putc(j, ',');
        jb_puts(j, "{\"group\":");   jb_put_int(j, refs[i].group);
        jb_puts(j, ",\"binding\":"); jb_put_int(j, refs[i].binding);
        jb_putc(j, '}');
    }
    jb_putc(j, ']');

    /* Capabilities this entry actually uses (map to the WebGPU
     * device features a pipeline must enable). */
    jb_puts(j, ",\"features\":[");
    int ff = 0;
    if (eu.uses & REFLECT_USE_SUBGROUPS) { jb_puts(j, "\"subgroups\""); ff = 1; }
    if (eu.uses & REFLECT_USE_F16)       { if (ff) jb_putc(j, ','); jb_puts(j, "\"f16\""); }
    jb_putc(j, ']');

    emit_entry_bind_group_layouts(r, j, refs, nref, stage);

    /* Total workgroup shared memory (compute only). */
    if (stage && strcmp(stage, "compute") == 0) {
        jb_puts(j, ",\"workgroup_bytes\":");
        jb_put_int(j, (long long)reflect_workgroup_bytes(r, &eu));
    }

    free(refs);
    free(eu.fns);
    free(eu.vars);
}
