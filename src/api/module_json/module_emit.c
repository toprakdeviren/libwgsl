/**
 * @file module_emit.c — resources/structs/overrides/directives + public module JSON API
 */
#include "internal/module_json_priv.h"

static void emit_texture_handle(JsonBuf *j, const WGSLTypeInfo *t) {
    WGSLTextureDim dim = (WGSLTextureDim)t->width;
    const char *view = "2d";
    int storage = 0, depth = 0, ms = 0, external = 0;
    switch (dim) {
    case WGSL_TEX_DIM_1D:                     view = "1d";         break;
    case WGSL_TEX_DIM_2D:                     view = "2d";         break;
    case WGSL_TEX_DIM_2D_ARRAY:               view = "2d-array";   break;
    case WGSL_TEX_DIM_3D:                     view = "3d";         break;
    case WGSL_TEX_DIM_CUBE:                   view = "cube";       break;
    case WGSL_TEX_DIM_CUBE_ARRAY:             view = "cube-array"; break;
    case WGSL_TEX_DIM_MULTISAMPLED_2D:        view = "2d"; ms = 1; break;
    case WGSL_TEX_DIM_EXTERNAL:               view = "2d"; external = 1; break;
    case WGSL_TEX_DIM_DEPTH_2D:               view = "2d";       depth = 1; break;
    case WGSL_TEX_DIM_DEPTH_2D_ARRAY:         view = "2d-array"; depth = 1; break;
    case WGSL_TEX_DIM_DEPTH_CUBE:             view = "cube";     depth = 1; break;
    case WGSL_TEX_DIM_DEPTH_CUBE_ARRAY:       view = "cube-array"; depth = 1; break;
    case WGSL_TEX_DIM_DEPTH_MULTISAMPLED_2D:  view = "2d"; depth = 1; ms = 1; break;
    case WGSL_TEX_DIM_STORAGE_1D:             view = "1d";       storage = 1; break;
    case WGSL_TEX_DIM_STORAGE_2D:             view = "2d";       storage = 1; break;
    case WGSL_TEX_DIM_STORAGE_2D_ARRAY:       view = "2d-array"; storage = 1; break;
    case WGSL_TEX_DIM_STORAGE_3D:             view = "3d";       storage = 1; break;
    }
    jb_puts(j, ",\"handle\":{\"kind\":\"");
    jb_puts(j, external ? "external_texture" :
               storage  ? "storage_texture"  :
               depth    ? "depth_texture"    : "texture");
    jb_puts(j, "\",\"view_dimension\":\""); jb_puts(j, view); jb_putc(j, '"');
    if (ms) jb_puts(j, ",\"multisampled\":true");
    if (storage) {
        jb_puts(j, ",\"format\":\"");
        jb_puts(j, wgsl_texel_format_name((WGSLTexelFormat)t->array_len));
        jb_puts(j, "\",\"access\":\"");
        jb_puts(j, wgsl_access_mode_name((WGSLAccessMode)t->rows));
        jb_putc(j, '"');
    } else if (depth) {
        jb_puts(j, ",\"sample_type\":\"depth\"");
    } else if (!external) {
        const WGSLTypeInfo *el = (const WGSLTypeInfo *)t->ref;
        const char *st = "float";
        if (el && el->kind == WGSL_TYPE_U32)      st = "uint";
        else if (el && el->kind == WGSL_TYPE_I32) st = "sint";
        jb_puts(j, ",\"sample_type\":\""); jb_puts(j, st); jb_putc(j, '"');
    }
    jb_putc(j, '}');
}

static void emit_sampler_handle(JsonBuf *j, const WGSLTypeInfo *t) {
    jb_puts(j, ",\"handle\":{\"kind\":\"");
    jb_puts(j, t->width ? "comparison_sampler" : "sampler");
    jb_puts(j, "\"}");
}

/* Per-buffer-binding byte size (minBindingSize) + alignment in
 * the binding's own address space, and — for struct-typed bindings —
 * AS-correct member offsets.  Uniform bindings round struct/array
 * alignment up to 16 (§14.4.5), so these differ from the AS_NONE
 * baseline emitted in structs[]; an embedder must use these for
 * uniform buffers or it computes wrong offsets. */
static void emit_binding_size(WGSLResult *r, JsonBuf *j,
                              WGSLSymbol *sym, WGSLAddressSpace as) {
    WGSLTypeInfo *t = sym ? sym->type : NULL;
    if (!t) return;
    WGSLLayoutCtx lcx = { &r->tc, &r->cev, &r->source, 0 };
    jb_puts(j, ",\"size\":");  jb_put_int(j, wgsl_layout_size_of(&lcx, t, as));
    jb_puts(j, ",\"align\":"); jb_put_int(j, wgsl_layout_required_align_of(&lcx, t, as));

    if (t->kind != WGSL_TYPE_STRUCT || !t->ref) return;
    WGSLNode *sn = (WGSLNode *)t->ref;
    uint32_t nmem = 0;
    for (uint32_t k = 0; k < sn->child_count; k++)
        if (sn->children[k] && sn->children[k]->kind == WGSL_NODE_DECL_STRUCT_MEMBER) nmem++;
    uint32_t *offs = (uint32_t *)malloc((nmem ? nmem : 1) * sizeof *offs);
    uint32_t a2 = 0, s2 = 0;
    if (offs && wgsl_layout_struct(&lcx, t, as, offs, nmem, &a2, &s2)) {
        jb_puts(j, ",\"layout\":[");
        int mf = 1; uint32_t mi = 0;
        for (uint32_t k = 0; k < sn->child_count; k++) {
            WGSLNode *m = sn->children[k];
            if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            if (!mf) jb_putc(j, ',');
            mf = 0;
            jb_puts(j, "{\"name\":");
            uint32_t mno = wgsl_node_name_span(m).offset;
            uint32_t mnl = wgsl_node_name_span(m).length;
            jb_put_span(j, &r->source, mno, mnl);
            if (mi < nmem) { jb_puts(j, ",\"offset\":"); jb_put_int(j, offs[mi]); }
            jb_putc(j, '}');
            mi++;
        }
        jb_putc(j, ']');
    }
    free(offs);
}

void emit_resources(WGSLResult *r, JsonBuf *j) {
    jb_puts(j, ",\"resources\":[");
    int first = 1;
    if (!r->ast.root) { jb_putc(j, ']'); return; }
    WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n || n->kind != WGSL_NODE_DECL_VAR) continue;

        uint32_t A = wgsl_varlike_attr_count(n);
        WGSLNode *gp = json_find_attr(r, n, 0, A, "group");
        WGSLNode *bd = json_find_attr(r, n, 0, A, "binding");
        if (!gp || !bd) continue;       /* not a bound resource          */

        long long group   = (gp->child_count >= 2)
            ? wgsl_attr_int_arg(r, gp->children[1]) : 0;
        long long binding = (bd->child_count >= 2)
            ? wgsl_attr_int_arg(r, bd->children[1]) : 0;

        WGSLSymbol *sym = wgsl_resolver_symbol_for_decl(&r->res, n);

        /* Address space + access mode come from the checker's resolved
         * symbol (sym->as/am), not a template re-parse: textures /
         * samplers carry no template AS but the checker infers
         * WGSL_AS_HANDLE for them (§14.3), so the old template-only path
         * mislabelled every handle resource as "private". */
        WGSLTypeInfo *rt   = sym ? sym->type : NULL;
        WGSLAddressSpace as = sym ? (WGSLAddressSpace)sym->as : WGSL_AS_NONE;
        const char *as_name = (as != WGSL_AS_NONE)
            ? wgsl_address_space_name(as) : "private";
        const char *acc_name = "read";
        if (sym) {
            const char *a = wgsl_access_mode_name((WGSLAccessMode)sym->am);
            if (a[0]) acc_name = a;
        }

        if (!first) jb_putc(j, ',');
        first = 0;
        jb_puts(j, "{\"name\":");
        uint32_t no = wgsl_node_name_span(n).offset;
        uint32_t nl = wgsl_node_name_span(n).length;
        jb_put_span(j, &r->source, no, nl);
        jb_puts(j, ",\"group\":");   jb_put_int(j, group);
        jb_puts(j, ",\"binding\":"); jb_put_int(j, binding);
        jb_puts(j, ",\"address_space\":\""); jb_puts(j, as_name); jb_putc(j, '"');
        jb_puts(j, ",\"access\":\"");        jb_puts(j, acc_name); jb_putc(j, '"');
        jb_puts(j, ",\"type\":");
        jb_put_resource_type(r, j, rt);

        if (rt && rt->kind == WGSL_TYPE_TEXTURE)      emit_texture_handle(j, rt);
        else if (rt && rt->kind == WGSL_TYPE_SAMPLER) emit_sampler_handle(j, rt);
        else if (as == WGSL_AS_STORAGE || as == WGSL_AS_UNIFORM)
            emit_binding_size(r, j, sym, as);

        jb_putc(j, '}');
    }
    jb_putc(j, ']');
}

void emit_structs(WGSLResult *r, JsonBuf *j) {
    jb_puts(j, ",\"structs\":[");
    int first = 1;
    if (!r->ast.root) { jb_putc(j, ']'); return; }
    WGSLNode *tu = r->ast.root;
    /* Layout is computed per spec §14.4.2 using the default address
     * space (WGSL_AS_NONE, i.e. AlignOf without uniform-AS bump).  A
     * struct used as `var<uniform>` would re-walk with WGSL_AS_UNIFORM;
     * the validator already enforces that case.  For module-JSON the
     * default layout is the most useful baseline for engine bindings. */
    WGSLLayoutCtx lcx = { &r->tc, &r->cev, &r->source, 0 };
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n || n->kind != WGSL_NODE_DECL_STRUCT) continue;

        if (!first) jb_putc(j, ',');
        first = 0;
        jb_puts(j, "{\"name\":");
        uint32_t no = wgsl_node_name_span(n).offset;
        uint32_t nl = wgsl_node_name_span(n).length;
        jb_put_span(j, &r->source, no, nl);

        /* Resolve the struct's TypeInfo (member 0 of all_decls is the
         * struct symbol with `sym->type` set during typecheck).  Walk
         * once to fill offsets[]. */
        WGSLSymbol *ssym = wgsl_resolver_symbol_for_decl(&r->res, n);
        const WGSLTypeInfo *st = ssym ? ssym->type : NULL;
        uint32_t nmem = 0;
        for (uint32_t k = 0; k < n->child_count; k++)
            if (n->children[k] &&
                n->children[k]->kind == WGSL_NODE_DECL_STRUCT_MEMBER) nmem++;
        uint32_t *offsets = (uint32_t *)malloc((nmem ? nmem : 1) * sizeof *offsets);
        uint32_t salign = 0, ssize = 0;
        int have_layout = st && offsets &&
            wgsl_layout_struct(&lcx, st, WGSL_AS_NONE, offsets, nmem, &salign, &ssize);
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
            uint32_t mno = wgsl_node_name_span(m).offset;
            uint32_t mnl = wgsl_node_name_span(m).length;
            jb_put_span(j, &r->source, mno, mnl);
            jb_puts(j, ",\"type\":");
            uint32_t MA = wgsl_member_attr_count(m);
            WGSLTypeInfo *mt = NULL;
            if (MA < m->child_count) {
                /* The TC's `resolve_type_spec` records the resolved
                 * type on the side-table for every type-spec node it
                 * visits, so we can pull it back here. */
                mt = wgsl_typecheck_type_of(&r->tc, m->children[MA]);
            }
            jb_put_type(j, mt);
            if (have_layout && midx < nmem) {
                jb_puts(j, ",\"offset\":"); jb_put_int(j, offsets[midx]);
                jb_puts(j, ",\"size\":");   jb_put_int(j, wgsl_layout_size_of(&lcx, mt, WGSL_AS_NONE));
                jb_puts(j, ",\"align\":");  jb_put_int(j, wgsl_layout_align_of(&lcx, mt, WGSL_AS_NONE));
            }
            jb_putc(j, '}');
            midx += 1;
        }
        jb_puts(j, "]}");
        free(offsets);
    }
    jb_putc(j, ']');
}

void emit_overrides(WGSLResult *r, JsonBuf *j) {
    jb_puts(j, ",\"overrides\":[");
    int first = 1;
    if (!r->ast.root) { jb_putc(j, ']'); return; }
    WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n || n->kind != WGSL_NODE_DECL_OVERRIDE) continue;

        uint32_t A = wgsl_varlike_attr_count(n);
        WGSLNode *id = json_find_attr(r, n, 0, A, "id");

        WGSLSymbol *sym = wgsl_resolver_symbol_for_decl(&r->res, n);
        if (!first) jb_putc(j, ',');
        first = 0;
        jb_puts(j, "{\"name\":");
        uint32_t no = wgsl_node_name_span(n).offset;
        uint32_t nl = wgsl_node_name_span(n).length;
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

/* Raw (unquoted) ident source bytes are JSON-safe here: no quotes,
 * backslashes, or controls. */
static void jb_put_ident_bytes(WGSLResult *r, JsonBuf *j, const WGSLNode *id) {
    if (!id || id->kind != WGSL_NODE_EXPR_IDENT) return;
    uint32_t o = wgsl_node_name_span(id).offset;
    uint32_t l = wgsl_node_name_span(id).length;
    if (r->source.bytes && (size_t)o + l <= r->source.length)
        jb_putn(j, r->source.bytes + o, l);
}

/* Mirror the module's `enable` / `requires` / `diagnostic`
 * directives verbatim from source, so an editor can round-trip its
 * settings.  Faithful to what was written (not a bit→name reconstruction
 * that could drift from the parser). */
void emit_directives(WGSLResult *r, JsonBuf *j) {
    WGSLNode *tu = r->ast.root;

    jb_puts(j, ",\"enable\":[");
    int first = 1;
    for (uint32_t i = 0; tu && i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n || n->kind != WGSL_NODE_DIR_ENABLE) continue;
        for (uint32_t k = 0; k < n->child_count; k++) {
            if (!first) jb_putc(j, ',');
            first = 0;
            jb_put_ident_name(r, j, n->children[k]);
        }
    }
    jb_putc(j, ']');

    jb_puts(j, ",\"requires\":[");
    first = 1;
    for (uint32_t i = 0; tu && i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n || n->kind != WGSL_NODE_DIR_REQUIRES) continue;
        for (uint32_t k = 0; k < n->child_count; k++) {
            if (!first) jb_putc(j, ',');
            first = 0;
            jb_put_ident_name(r, j, n->children[k]);
        }
    }
    jb_putc(j, ']');

    jb_puts(j, ",\"diagnostics\":[");
    first = 1;
    for (uint32_t i = 0; tu && i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n || n->kind != WGSL_NODE_DIR_DIAGNOSTIC || n->child_count < 2) continue;
        if (!first) jb_putc(j, ',');
        first = 0;
        jb_puts(j, "{\"severity\":");
        jb_put_ident_name(r, j, n->children[0]);
        jb_puts(j, ",\"rule\":\"");
        jb_put_ident_bytes(r, j, n->children[1]);
        if (wgsl_diag_rule_parts(n) == 2 && n->child_count >= 3) {
            jb_putc(j, '.');
            jb_put_ident_bytes(r, j, n->children[2]);
        }
        jb_puts(j, "\"}");
    }
    jb_putc(j, ']');
}

void build_module_json(WGSLResult *r) {
    JsonBuf j = {0};
    jb_putc(&j, '{');
    jb_puts(&j, "\"spec_pin\":\""); jb_puts(&j, WGSL_SPEC_PIN); jb_putc(&j, '"');
    jb_putc(&j, ',');
    emit_entry_points(r, &j);
    /* Pipelines may omit over-cap V×F pairs; surface honesty flag. */
    int pipelines_truncated = emit_pipelines(r, &j);
    emit_resources   (r, &j);
    emit_structs     (r, &j);
    emit_overrides   (r, &j);
    emit_directives  (r, &j);
    if (pipelines_truncated)
        jb_puts(&j, ",\"analysis_truncated\":true");
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
