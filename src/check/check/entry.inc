/* ── Entry point ───────────────────────────────────────────────────── */

static void collect_directives(WGSLTypeChecker *tc, WGSLNode *tu) {
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n) continue;
        if (n->kind == WGSL_NODE_DIR_ENABLE) {
            for (uint32_t j = 0; j < n->child_count; j++) {
                WGSLNode *arg = n->children[j];
                if (!arg || arg->kind != WGSL_NODE_EXPR_IDENT) continue;
                /* §4.1.1: enable-extension names are NOT identifiers and
                 * therefore are not resolved against the symbol table.
                 * Read the name directly from source text. */
                uint32_t off = (uint32_t)(arg->payload[0] & 0xFFFFFFFFu);
                uint32_t nl  = (uint32_t)(arg->payload[0] >> 32);
                const char *nm = tc->src->bytes + off;
                uint32_t bit = 0;
                if (nl == 3 && memcmp(nm, "f16", 3) == 0)
                    bit = WGSL_EXT_F16;
                else if (nl == 14 && memcmp(nm, "clip_distances", 14) == 0)
                    bit = WGSL_EXT_CLIP_DISTANCES;
                else if (nl == 20 && memcmp(nm, "dual_source_blending", 20) == 0)
                    bit = WGSL_EXT_DUAL_SOURCE_BLENDING;
                else if (nl == 9 && memcmp(nm, "subgroups", 9) == 0)
                    bit = WGSL_EXT_SUBGROUPS;
                else if (nl == 15 && memcmp(nm, "primitive_index", 15) == 0)
                    bit = WGSL_EXT_PRIMITIVE_INDEX;
                else if (nl == 34 && memcmp(nm, "chromium_experimental_primitive_id", 34) == 0)
                    bit = WGSL_EXT_PRIMITIVE_INDEX;

                if (bit) {
                    /* §4.1.1 example: "A redundant enable directive is
                     * ok." — do not emit a duplicate-extension diagnostic. */
                    tc->enabled_extensions |= bit;
                } else {
                    wgsl_tc_error(tc, arg, "unknown extension '%.*s'", (int)nl, nm);
                }
            }
        } else if (n->kind == WGSL_NODE_DIR_REQUIRES) {
            /* §4.1.2 + §3.8.5: language extension names are NOT
             * identifiers and don't resolve.  Read names directly
             * from source text.  All 11 spec-listed names are
             * recognized; anything else is rejected. */
            static const struct {
                const char *name;
                uint32_t    len;
                uint32_t    bit;
            } kLangFeatures[] = {
                { "readonly_and_readwrite_storage_textures", 39, WGSL_FEAT_RO_RW_STORAGE_TEXTURES },
                { "packed_4x8_integer_dot_product",          30, WGSL_FEAT_PACKED_4X8_INT_DOT },
                { "unrestricted_pointer_parameters",         31, WGSL_FEAT_UNRESTRICTED_PTR_PARAMS },
                { "pointer_composite_access",                24, WGSL_FEAT_POINTER_COMPOSITE_ACCESS },
                { "uniform_buffer_standard_layout",          30, WGSL_FEAT_UNIFORM_BUF_STD_LAYOUT },
                { "subgroup_id",                             11, WGSL_FEAT_SUBGROUP_ID },
                { "subgroup_uniformity",                     19, WGSL_FEAT_SUBGROUP_UNIFORMITY },
                { "texture_and_sampler_let",                 23, WGSL_FEAT_TEXTURE_AND_SAMPLER_LET },
                { "texture_formats_tier1",                   21, WGSL_FEAT_TEXTURE_FORMATS_TIER1 },
                { "linear_indexing",                         15, WGSL_FEAT_LINEAR_INDEXING },
                { "immediate_data",                          14, WGSL_FEAT_IMMEDIATE_DATA },
            };
            for (uint32_t j = 0; j < n->child_count; j++) {
                WGSLNode *arg = n->children[j];
                if (!arg || arg->kind != WGSL_NODE_EXPR_IDENT) continue;
                uint32_t off = (uint32_t)(arg->payload[0] & 0xFFFFFFFFu);
                uint32_t nl  = (uint32_t)(arg->payload[0] >> 32);
                const char *nm = tc->src->bytes + off;
                uint32_t bit = 0;
                for (size_t k = 0; k < sizeof kLangFeatures / sizeof kLangFeatures[0]; k++) {
                    if (kLangFeatures[k].len == nl &&
                        memcmp(kLangFeatures[k].name, nm, nl) == 0) {
                        bit = kLangFeatures[k].bit;
                        break;
                    }
                }
                if (bit) {
                    /* Redundant `requires` is harmless (mirrors §4.1.1
                     * note for `enable`). */
                    tc->required_features |= bit;
                } else {
                    wgsl_tc_error(tc, arg,
                        "unknown language feature '%.*s'", (int)nl, nm);
                }
            }
        } else if (n->kind == WGSL_NODE_DIR_DIAGNOSTIC) {
            /* children: [severity_ident, rule_ident, optional sub_ident]
             * payload[0]: rule_parts (1 or 2). */
            int sev_val = -1;  /* -1 = invalid; 0 = off; 1..4 = WGSLDiagSeverity */
            if (n->child_count >= 1) {
                WGSLNode *sev = n->children[0];
                if (sev && sev->kind == WGSL_NODE_EXPR_IDENT) {
                    uint32_t off = (uint32_t)(sev->payload[0] & 0xFFFFFFFFu);
                    uint32_t len = (uint32_t)(sev->payload[0] >> 32);
                    const char *nm = tc->src->bytes + off;
                    if (len == 3 && memcmp(nm, "off", 3) == 0)     sev_val = 0;
                    if (len == 4 && memcmp(nm, "info", 4) == 0)    sev_val = WGSL_DIAG_INFO;
                    if (len == 7 && memcmp(nm, "warning", 7) == 0) sev_val = WGSL_DIAG_WARNING;
                    if (len == 5 && memcmp(nm, "error", 5) == 0)   sev_val = WGSL_DIAG_ERROR;
                    if (sev_val < 0) {
                        wgsl_tc_error(tc, sev, "invalid diagnostic severity '%.*s' (expected 'off', 'info', 'warning', or 'error')", (int)len, nm);
                    }
                }
            }
            /* §3.8.3 / §2.3.2: single-token rule name must be one of the
             * pre-defined names; unknown multi-token names ("vendor.foo")
             * are permitted (spec: "may itself trigger a diagnostic" —
             * we silently accept). */
            uint32_t rule_parts = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
            const char *rule_nm = NULL;
            uint32_t rule_len = 0;
            if (rule_parts == 1 && n->child_count >= 2) {
                WGSLNode *rule = n->children[1];
                if (rule && rule->kind == WGSL_NODE_EXPR_IDENT) {
                    uint32_t off = (uint32_t)(rule->payload[0] & 0xFFFFFFFFu);
                    uint32_t len = (uint32_t)(rule->payload[0] >> 32);
                    const char *nm = tc->src->bytes + off;
                    int valid =
                        (len == 21 && memcmp(nm, "derivative_uniformity", 21) == 0) ||
                        (len == 19 && memcmp(nm, "subgroup_uniformity", 19) == 0);
                    if (!valid) {
                        wgsl_diag_emit_at(tc->diag, tc->src, WGSL_DIAG_WARNING,
                            rule->span_offset, rule->span_length, NULL,
                            "unknown diagnostic rule '%.*s' (expected "
                            "'derivative_uniformity' or "
                            "'subgroup_uniformity')",
                            (int)len, nm);
                    } else {
                        rule_nm = nm;
                        rule_len = len;
                    }
                }
            }
            /* §4.2: register the filter on the diagnostic bag so downstream
             * filterable diagnostics honour it. */
            if (sev_val >= 0 && rule_nm) {
                wgsl_diag_set_filter(tc->diag, rule_nm, rule_len, sev_val);
            }
        }
    }
}

int wgsl_typecheck(
    WGSLAst             *ast,
    const WGSLSource    *src,
    WGSLArena           *arena,
    WGSLDiagBag         *diag,
    WGSLTypeStore       *types,
    WGSLResolver        *res,
    WGSLConstEvaluator  *cev,
    WGSLTypeChecker     *out)
{
    if (!ast || !src || !arena || !diag || !types || !res || !out) return 0;

    memset(out, 0, sizeof *out);
    out->arena = arena;
    out->diag  = diag;
    out->types = types;
    out->src   = src;
    out->res   = res;
    out->cev   = cev;
    out->supported_features = WGSL_FEAT_ALL_SUPPORTED;

    if (!ast->root) return 0;
    WGSLNode *tu = ast->root;

    collect_directives(out, tu);

    /* Pre-pass — type all module-scope structs and aliases first so
     * downstream type-specs that reference them resolve.  Source order
     * is good enough for v1; cycle detection lives in Phase 8. */
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n) continue;
        if (n->kind == WGSL_NODE_DECL_STRUCT)  wgsl_tc_type_decl_struct(out, n);
        if (n->kind == WGSL_NODE_DECL_ALIAS)   wgsl_tc_type_decl_alias(out, n);
    }

    /* Main pass — every other module-scope decl. */
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n) continue;
        if (n->kind == WGSL_NODE_DECL_STRUCT) continue;
        if (n->kind == WGSL_NODE_DECL_ALIAS)  continue;
        wgsl_tc_type_module_decl(out, n);
    }

    return out->had_error ? 0 : 1;
}
