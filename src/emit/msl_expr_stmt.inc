static void emit_expr(M *m, const WGSLNode *n, int depth) {
    if (!n) return;
    if (depth > 96) { m_puts(m, "/*deep*/"); return; }
    m_indent(m);
    switch (n->kind) {
    case WGSL_NODE_EXPR_IDENT: {
        /* §B1: stage_in param uses rewrite via resolved symbol. */
        if (m->rw_n > 0) {
            WGSLSymbol *sym = wgsl_node_resolved_symbol(n);
            if (sym) {
                for (int i = 0; i < m->rw_n; i++) {
                    if (m->rw_sym[i] == sym) {
                        m_puts(m, m->rw_to[i]);
                        return;
                    }
                }
            }
        }
        uint32_t o = wgsl_node_name_span(n).offset;
        uint32_t l = wgsl_node_name_span(n).length;
        if (m->src && (size_t)o + l <= m->src_len) {
            const char *mapped = map_simple_type(m->src + o, l);
            /* Only map type-ish idents when used as bare type names in
             * constructors are handled in CALL; leave user idents alone.
             * true/false stay. */
            if (m_name_eq(m, o, l, "true") || m_name_eq(m, o, l, "false")) {
                m_span(m, o, l);
                return;
            }
            (void)mapped;
        }
        m_ident(m, n);
        break;
    }
    case WGSL_NODE_EXPR_LITERAL_BOOL:
        m_puts(m, wgsl_lit_bool(n) ? "true" : "false");
        break;
    case WGSL_NODE_EXPR_LITERAL_INT:
    case WGSL_NODE_EXPR_LITERAL_FLOAT: {
        /* Strip WGSL suffixes i/u/f/h for MSL (Metal uses typed casts). */
        uint32_t o = n->span_offset, l = n->span_length;
        if (m->src && (size_t)o + l <= m->src_len && l > 0) {
            size_t end = l;
            char last = m->src[o + end - 1];
            if (last == 'i' || last == 'u' || last == 'f' || last == 'h')
                end--;
            m_putn(m, m->src + o, end);
        }
        break;
    }
    case WGSL_NODE_EXPR_PAREN:
        m_putc(m, '(');
        if (n->child_count > 0) emit_expr(m, n->children[0], depth + 1);
        m_putc(m, ')');
        break;
    case WGSL_NODE_EXPR_UNARY: {
        WGSLTokenKind op = (WGSLTokenKind)wgsl_node_op_kind_u32(n);
        const char *ops = wgsl_token_kind_name(op);
        /* Skip diagnostic-style names like "<eof>"; keep real ops incl. "!". */
        if (ops && !(ops[0] == '<' && ops[1] != '\0' && ops[1] != '<' &&
                     ops[1] != '='))
            m_puts(m, ops);
        if (n->child_count > 0) emit_expr(m, n->children[0], depth + 1);
        break;
    }
    case WGSL_NODE_EXPR_BINARY: {
        WGSLTokenKind op = (WGSLTokenKind)wgsl_node_op_kind_u32(n);
        const char *ops = wgsl_token_kind_name(op);
        if (n->child_count > 0) emit_expr(m, n->children[0], depth + 1);
        m_putc(m, ' ');
        /* Emit <, <=, <<, >, >=, >> — do not drop comparison/shift ops.
         * Only suppress names that look like <token-kind> diagnostics. */
        if (ops) {
            int diag_name = (ops[0] == '<' && ops[1] != '\0' &&
                             ops[1] != '<' && ops[1] != '=' &&
                             ops[strlen(ops) - 1] == '>');
            if (!diag_name) m_puts(m, ops);
        }
        m_putc(m, ' ');
        if (n->child_count > 1) emit_expr(m, n->children[1], depth + 1);
        break;
    }
    case WGSL_NODE_EXPR_CALL: {
        if (n->child_count == 0) break;
        const WGSLNode *cal = n->children[0];
        /* bitcast<T>(x) → as_type<T>(x) */
        if (cal && cal->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
            cal->child_count > 0 && m_ident_eq(m, cal->children[0], "bitcast")) {
            m_puts(m, "as_type<");
            /* type args */
            for (uint32_t i = 1; i < cal->child_count; i++) {
                if (cal->children[i] &&
                    cal->children[i]->kind == WGSL_NODE_TEMPLATE_ARG_LIST &&
                    cal->children[i]->child_count > 0)
                    emit_type_spec(m, cal->children[i]->children[0], depth + 1);
                else if (cal->children[i])
                    emit_type_spec(m, cal->children[i], depth + 1);
            }
            m_puts(m, ">(");
            for (uint32_t i = 1; i < n->child_count; i++) {
                if (i > 1) m_puts(m, ", ");
                emit_expr(m, n->children[i], depth + 1);
            }
            m_putc(m, ')');
            break;
        }
        if (cal && cal->kind == WGSL_NODE_EXPR_IDENT) {
            uint32_t o = wgsl_node_name_span(cal).offset;
            uint32_t l = wgsl_node_name_span(cal).length;
            /* §B3b: textureSample* family → Metal texture methods. */
            if (m->src && (size_t)o + l <= m->src_len &&
                emit_texture_sample_call(m, n, m->src + o, l, depth))
            {
                break;
            }
            /* textureStore(t, coords[, array], value) → t.write(value, coords[, array]) */
            if (m_name_eq(m, o, l, "textureStore") && emit_texture_store_call(m, n, depth)) {
                break;
            }
            /* textureLoad(t, coords, level) → t.read(uintN(coords), level) */
            if (m_name_eq(m, o, l, "textureLoad") && n->child_count >= 3) {
                emit_expr(m, n->children[1], depth + 1);
                m_puts(m, ".read(");
                for (uint32_t i = 2; i < n->child_count; i++) {
                    if (i > 2) m_puts(m, ", ");
                    emit_expr(m, n->children[i], depth + 1);
                }
                m_putc(m, ')');
                break;
            }
            /* Barriers are full expressions (no extra ()). */
            if (m_name_eq(m, o, l, "workgroupBarrier")) {
                m_puts(m, "threadgroup_barrier(mem_flags::mem_threadgroup)");
                break;
            }
            if (m_name_eq(m, o, l, "storageBarrier")) {
                m_puts(m, "threadgroup_barrier(mem_flags::mem_device)");
                break;
            }
            if (m_name_eq(m, o, l, "textureBarrier")) {
                m_puts(m, "threadgroup_barrier(mem_flags::mem_texture)");
                break;
            }
            const char *mapped = NULL;
            if (m->src && (size_t)o + l <= m->src_len)
                mapped = map_callee(m->src + o, l);
            if (mapped) m_puts(m, mapped);
            else m_ident(m, cal);
        } else if (cal && cal->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT) {
            emit_type_spec(m, cal, depth + 1); /* vec3<f32>(...) → float3(...) */
        } else {
            emit_expr(m, cal, depth + 1);
        }
        m_putc(m, '(');
        for (uint32_t i = 1; i < n->child_count; i++) {
            if (i > 1) m_puts(m, ", ");
            emit_expr(m, n->children[i], depth + 1);
        }
        m_putc(m, ')');
        break;
    }
    case WGSL_NODE_EXPR_INDEX:
        if (n->child_count > 0) emit_expr(m, n->children[0], depth + 1);
        m_putc(m, '[');
        if (n->child_count > 1) emit_expr(m, n->children[1], depth + 1);
        m_putc(m, ']');
        break;
    case WGSL_NODE_EXPR_MEMBER:
        if (n->child_count > 0) emit_expr(m, n->children[0], depth + 1);
        m_putc(m, '.');
        if (n->child_count > 1) emit_expr(m, n->children[1], depth + 1);
        else {
            uint32_t o = wgsl_node_name_span(n).offset;
            uint32_t l = wgsl_node_name_span(n).length;
            if (l) m_span(m, o, l);
        }
        break;
    case WGSL_NODE_EXPR_ADDR_OF:
        m_putc(m, '&');
        if (n->child_count > 0) emit_expr(m, n->children[0], depth + 1);
        break;
    case WGSL_NODE_EXPR_INDIRECTION:
        m_putc(m, '*');
        if (n->child_count > 0) emit_expr(m, n->children[0], depth + 1);
        break;
    case WGSL_NODE_EXPR_TEMPLATED_IDENT:
        emit_type_spec(m, n, depth);
        break;
    default:
        m_span(m, n->span_offset, n->span_length);
        break;
    }
}

static void emit_compound(M *m, const WGSLNode *n, int depth) {
    m_indent(m);
    m_puts(m, "{\n");
    m->indent++;
    for (uint32_t i = 0; i < n->child_count; i++) {
        const WGSLNode *c = n->children[i];
        if (!c || c->kind == WGSL_NODE_ATTRIBUTE) continue;
        emit_stmt(m, c, depth + 1);
    }
    m->indent--;
    m_indent(m);
    m_puts(m, "}");
}

static void emit_type_info(M *m, const WGSLTypeInfo *t) {
    if (!t) { m_puts(m, "float"); return; }
    switch ((WGSLTypeKind)t->kind) {
    case WGSL_TYPE_VOID: m_puts(m, "void"); break;
    case WGSL_TYPE_BOOL: m_puts(m, "bool"); break;
    case WGSL_TYPE_I32: case WGSL_TYPE_ABSTRACT_INT: m_puts(m, "int"); break;
    case WGSL_TYPE_U32: m_puts(m, "uint"); break;
    case WGSL_TYPE_F32: case WGSL_TYPE_ABSTRACT_FLOAT: m_puts(m, "float"); break;
    case WGSL_TYPE_F16: m_puts(m, "half"); break;
    case WGSL_TYPE_VEC: {
        const char *el = "float";
        if (t->ref) {
            WGSLTypeKind ek = (WGSLTypeKind)((const WGSLTypeInfo *)t->ref)->kind;
            if (ek == WGSL_TYPE_I32) el = "int";
            else if (ek == WGSL_TYPE_U32) el = "uint";
            else if (ek == WGSL_TYPE_F16) el = "half";
            else if (ek == WGSL_TYPE_BOOL) el = "bool";
        }
        if (strcmp(el, "float") == 0) m_puts(m, "float");
        else m_puts(m, el);
        m_putc(m, (char)('0' + (t->width ? t->width : 3)));
        break;
    }
    case WGSL_TYPE_MAT: {
        m_puts(m, "float");
        m_putc(m, (char)('0' + (t->width ? t->width : 4)));
        m_putc(m, 'x');
        m_putc(m, (char)('0' + (t->rows ? t->rows : 4)));
        break;
    }
    case WGSL_TYPE_ATOMIC:
        m_puts(m, "atomic_uint");
        break;
    default:
        m_puts(m, "float");
        break;
    }
}

static void emit_var_like(M *m, const WGSLNode *n, const char *kw_msl, int depth) {
    /* let/const → const T x = …;  var → T x = …; */
    m_indent(m);
    uint32_t attrs = 0, tpls = 0;
    int has_type = 0, has_init = 0;
    if (n->kind == WGSL_NODE_DECL_VAR) {
        attrs = wgsl_varlike_attr_count(n);
        tpls = wgsl_varlike_tpl_count(n);
        has_type = wgsl_varlike_has_type(n);
        has_init = wgsl_varlike_has_init(n);
    } else if (n->kind == WGSL_NODE_DECL_LET || n->kind == WGSL_NODE_DECL_CONST) {
        has_type = wgsl_letlike_has_type(n);
        has_init = 1;
    } else if (n->kind == WGSL_NODE_DECL_OVERRIDE) {
        attrs = wgsl_varlike_attr_count(n);
        has_type = wgsl_varlike_has_type(n);
        has_init = wgsl_varlike_has_init(n);
        m_puts(m, "/*override→constant*/ ");
    }

    AttrInfo ai;
    if (attrs)
        collect_attrs(m, (const WGSLNode *const *)n->children, 0, attrs, &ai);

    /* address space from var<template> */
    const char *aspace = NULL;
    uint32_t ci = attrs;
    if (n->kind == WGSL_NODE_DECL_VAR && tpls > 0 && ci < n->child_count) {
        const WGSLNode *t0 = n->children[ci];
        if (t0 && m_ident_eq(m, t0, "storage")) aspace = "device";
        else if (t0 && m_ident_eq(m, t0, "uniform")) aspace = "constant";
        else if (t0 && m_ident_eq(m, t0, "workgroup")) aspace = "threadgroup";
        else if (t0 && m_ident_eq(m, t0, "private")) aspace = "thread";
        else if (t0 && m_ident_eq(m, t0, "function")) aspace = "thread";
        ci += tpls;
    }

    if (kw_msl && kw_msl[0]) {
        m_puts(m, kw_msl);
        m_putc(m, ' ');
    }
    if (aspace) {
        m_puts(m, aspace);
        m_putc(m, ' ');
    }

    /* type: declared spec, else inferred from symbol type store */
    if (has_type && ci < n->child_count) {
        emit_type_spec(m, n->children[ci], depth + 1);
        m_putc(m, ' ');
        ci++;
    } else {
        WGSLTypeInfo *ti = NULL;
        if (m->res) {
            WGSLSymbol *sym = wgsl_resolver_symbol_for_decl(&m->res->res, n);
            if (sym) ti = sym->type;
        }
        emit_type_info(m, ti);
        m_putc(m, ' ');
    }

    uint32_t no = wgsl_node_name_span(n).offset;
    uint32_t nl = wgsl_node_name_span(n).length;
    m_span(m, no, nl);

    if (n->kind == WGSL_NODE_DECL_VAR && aspace &&
        (strcmp(aspace, "device") == 0 || strcmp(aspace, "constant") == 0))
        emit_msl_attrs_trail(m, &ai, 1, 0);

    if (has_init && ci < n->child_count) {
        m_puts(m, " = ");
        emit_expr(m, n->children[ci], depth + 1);
    }
    m_puts(m, ";\n");
}

/* §B3a: for-init / for-update as a single clause (no trailing ';'). */
static void emit_for_clause(M *m, const WGSLNode *n, int depth, int is_update)
{
    if (!n) return;
    if (n->kind == WGSL_NODE_DECL_LET || n->kind == WGSL_NODE_DECL_VAR ||
        n->kind == WGSL_NODE_DECL_CONST)
    {
        if (is_update) {
            m_diag(m, WGSL_MSL_DIAG_UNSUPPORTED_STMT,
                   n->span_offset, n->span_length,
                   "for update must not be a declaration");
            m_puts(m, "/*bad for-update decl*/");
            return;
        }
        /* Inline decl: `T name = init` (Metal for-init). */
        int has_type = 0, has_init = 0;
        uint32_t attrs = 0, tpls = 0;
        if (n->kind == WGSL_NODE_DECL_VAR) {
            attrs = wgsl_varlike_attr_count(n);
            tpls = wgsl_varlike_tpl_count(n);
            has_type = wgsl_varlike_has_type(n);
            has_init = wgsl_varlike_has_init(n);
        } else {
            has_type = wgsl_letlike_has_type(n);
            has_init = 1;
        }
        uint32_t ci = attrs + tpls;
        if (has_type && ci < n->child_count) {
            emit_type_spec(m, n->children[ci], depth + 1);
            ci++;
        } else {
            /* Infer from checker when possible. */
            WGSLTypeInfo *t = node_type(m, n);
            if (t) emit_type_info(m, t);
            else m_puts(m, "int");
        }
        m_putc(m, ' ');
        m_span(m, wgsl_node_name_span(n).offset, wgsl_node_name_span(n).length);
        if (has_init && ci < n->child_count) {
            m_puts(m, " = ");
            emit_expr(m, n->children[ci], depth + 1);
        }
        return;
    }
    if (n->kind == WGSL_NODE_STMT_ASSIGN) {
        if (n->child_count > 0) emit_expr(m, n->children[0], depth + 1);
        m_puts(m, " = ");
        if (n->child_count > 1) emit_expr(m, n->children[1], depth + 1);
        return;
    }
    if (n->kind == WGSL_NODE_STMT_COMPOUND_ASSIGN) {
        WGSLTokenKind op = (WGSLTokenKind)wgsl_node_op_kind_u32(n);
        const char *ops = wgsl_token_kind_name(op);
        if (n->child_count > 0) emit_expr(m, n->children[0], depth + 1);
        m_putc(m, ' ');
        if (ops) m_puts(m, ops);
        m_putc(m, ' ');
        if (n->child_count > 1) emit_expr(m, n->children[1], depth + 1);
        return;
    }
    if (n->kind == WGSL_NODE_STMT_INCREMENT) {
        if (n->child_count > 0) emit_expr(m, n->children[0], depth + 1);
        m_puts(m, "++");
        return;
    }
    if (n->kind == WGSL_NODE_STMT_DECREMENT) {
        if (n->child_count > 0) emit_expr(m, n->children[0], depth + 1);
        m_puts(m, "--");
        return;
    }
    /* Expression statement as for-update (e.g. call) — rare. */
    if (n->kind == WGSL_NODE_STMT_FN_CALL && n->child_count > 0) {
        emit_expr(m, n->children[0], depth + 1);
        return;
    }
    m_diag(m, WGSL_MSL_DIAG_UNSUPPORTED_STMT,
           n->span_offset, n->span_length,
           "unsupported for-header clause in MSL emit");
    m_puts(m, "/*for-clause*/");
}

static void emit_for(M *m, const WGSLNode *n, int depth) {
    uint32_t flags = wgsl_for_flags(n);
    uint32_t idx = 0;
    const WGSLNode *init = NULL, *cond = NULL, *update = NULL, *body = NULL;
    if ((flags & 1u) && idx < n->child_count) init = n->children[idx++];
    if ((flags & 2u) && idx < n->child_count) cond = n->children[idx++];
    if ((flags & 4u) && idx < n->child_count) update = n->children[idx++];
    if (idx < n->child_count) body = n->children[idx];

    /* Nested breakable inside desugared loop body. */
    if (m->loop_in_body) m->loop_break_nest++;

    m_indent(m);
    m_puts(m, "for (");
    if (init) emit_for_clause(m, init, depth, 0);
    m_puts(m, "; ");
    if (cond) emit_expr(m, cond, depth + 1);
    m_puts(m, "; ");
    if (update) emit_for_clause(m, update, depth, 1);
    m_puts(m, ") ");
    if (body && body->kind == WGSL_NODE_STMT_COMPOUND)
        emit_compound(m, body, depth + 1);
    else if (body) {
        m_puts(m, "{\n");
        m->indent++;
        emit_stmt(m, body, depth + 1);
        m->indent--;
        m_indent(m);
        m_puts(m, "}");
    } else {
        m_puts(m, "{}");
    }
    m_nl(m);

    if (m->loop_in_body) m->loop_break_nest--;
}

/* WGSL loop { body… continuing { c… } }:
 *   continue in body must still run continuing; break must skip it.
 * Lowering (no goto — MSL forbids):
 *   for (;;) {
 *     bool __wgsl_brk_N = false;
 *     do { body  // continue→break; break→{brk=true;break}
 *     } while (0);
 *     if (__wgsl_brk_N) break;
 *     continuing…
 *   }
 * Without continuing: plain for (;;) { body }.
 */
static void emit_loop(M *m, const WGSLNode *n, int depth) {
    const WGSLNode *cont = NULL;
    for (uint32_t i = 0; i < n->child_count; i++) {
        if (n->children[i] &&
            n->children[i]->kind == WGSL_NODE_STMT_CONTINUING)
        {
            cont = n->children[i];
            break;
        }
    }

    /* Nested loop inside a desugared loop body is a nested breakable. */
    int nested_in_desugar = m->loop_in_body;
    if (nested_in_desugar) m->loop_break_nest++;

    m_indent(m);
    m_puts(m, "for (;;) {\n");
    m->indent++;

    if (cont) {
        int id = m->loop_cf_depth;
        char t[72];
        m_indent(m);
        snprintf(t, sizeof t, "bool __wgsl_brk_%d = false;\n", id);
        m_puts(m, t);
        m_indent(m);
        m_puts(m, "do {\n");
        m->indent++;
        m->loop_cf_depth++;
        int prev_in_body = m->loop_in_body;
        int prev_nest = m->loop_break_nest;
        m->loop_in_body = 1;
        m->loop_break_nest = 0; /* this loop is the nearest break target */
        for (uint32_t i = 0; i < n->child_count; i++) {
            const WGSLNode *c = n->children[i];
            if (!c || c->kind == WGSL_NODE_STMT_CONTINUING) continue;
            emit_stmt(m, c, depth + 1);
        }
        m->loop_in_body = prev_in_body;
        m->loop_break_nest = prev_nest;
        m->loop_cf_depth--;
        m->indent--;
        m_indent(m);
        m_puts(m, "} while (0);\n");
        m_indent(m);
        snprintf(t, sizeof t, "if (__wgsl_brk_%d) break;\n", id);
        m_puts(m, t);
        /* continuing block (break-if → if(c) break already handled) */
        for (uint32_t i = 0; i < cont->child_count; i++)
            emit_stmt(m, cont->children[i], depth + 1);
    } else {
        for (uint32_t i = 0; i < n->child_count; i++) {
            const WGSLNode *c = n->children[i];
            if (!c) continue;
            emit_stmt(m, c, depth + 1);
        }
    }

    m->indent--;
    m_indent(m);
    m_puts(m, "}\n");

    if (nested_in_desugar) m->loop_break_nest--;
}

/* WGSL switch: no fallthrough; multi-selector case; default alone or in list.
 * Metal switch needs explicit break after each clause body. */
static void emit_switch(M *m, const WGSLNode *n, int depth) {
    if (m->loop_in_body) m->loop_break_nest++;

    m_indent(m);
    m_puts(m, "switch (");
    if (n->child_count > 0) emit_expr(m, n->children[0], depth + 1);
    m_puts(m, ") {\n");
    m->indent++;

    for (uint32_t i = 1; i < n->child_count; i++) {
        const WGSLNode *cl = n->children[i];
        if (!cl || cl->kind != WGSL_NODE_STMT_CASE_CLAUSE) continue;

        uint32_t sel_n = wgsl_case_selector_count(cl);
        int def_alone = wgsl_case_is_default_alone(cl);
        int def_in_list = wgsl_case_has_default_in_list(cl);

        if (def_alone) {
            m_indent(m);
            m_puts(m, "default: {\n");
            m->indent++;
            /* children[0] is body when default-alone */
            if (cl->child_count > 0)
                emit_stmt(m, cl->children[0], depth + 1);
            m_indent(m);
            m_puts(m, "break;\n");
            m->indent--;
            m_indent(m);
            m_puts(m, "}\n");
            continue;
        }

        for (uint32_t s = 0; s < sel_n && s < cl->child_count; s++) {
            m_indent(m);
            m_puts(m, "case ");
            emit_expr(m, cl->children[s], depth + 1);
            m_puts(m, ":\n");
        }
        if (def_in_list) {
            m_indent(m);
            m_puts(m, "default:\n");
        }
        m_indent(m);
        m_puts(m, "{\n");
        m->indent++;
        /* body is after selectors */
        if (sel_n < cl->child_count)
            emit_stmt(m, cl->children[sel_n], depth + 1);
        m_indent(m);
        m_puts(m, "break;\n"); /* WGSL has no fallthrough */
        m->indent--;
        m_indent(m);
        m_puts(m, "}\n");
    }

    m->indent--;
    m_indent(m);
    m_puts(m, "}\n");

    if (m->loop_in_body) m->loop_break_nest--;
}

static void emit_stmt(M *m, const WGSLNode *n, int depth) {
    if (!n) return;
    switch (n->kind) {
    case WGSL_NODE_STMT_COMPOUND:
        emit_compound(m, n, depth);
        m_nl(m);
        break;
    case WGSL_NODE_DECL_LET:
        emit_var_like(m, n, "const", depth); /* Metal: const T x */
        break;
    case WGSL_NODE_DECL_CONST:
        emit_var_like(m, n, "constant", depth); /* module const → constant */
        break;
    case WGSL_NODE_DECL_VAR:
        emit_var_like(m, n, NULL, depth);
        break;
    case WGSL_NODE_STMT_RETURN:
        m_indent(m);
        m_puts(m, "return");
        if (n->child_count > 0 && n->children[0]) {
            m_putc(m, ' ');
            emit_expr(m, n->children[0], depth + 1);
        }
        m_puts(m, ";\n");
        break;
    case WGSL_NODE_STMT_DISCARD:
        m_indent(m);
        m_puts(m, "discard_fragment();\n");
        break;
    case WGSL_NODE_STMT_BREAK:
        m_indent(m);
        /* Flag-rewrite only when this break's nearest enclosing
         * loop-or-switch is the desugared outer loop (nest==0). */
        if (m->loop_in_body && m->loop_cf_depth > 0 &&
            m->loop_break_nest == 0)
        {
            /* Exit do-wrapper + outer for (skip continuing). */
            char t[48];
            snprintf(t, sizeof t, "{ __wgsl_brk_%d = true; break; }\n",
                     m->loop_cf_depth - 1);
            m_puts(m, t);
        } else {
            m_puts(m, "break;\n");
        }
        break;
    case WGSL_NODE_STMT_CONTINUE:
        m_indent(m);
        /* continue always targets nearest loop (never switch).  Nested
         * for/while/loop use native continue; only body of desugared
         * loop-with-continuing rewrites continue → break out of do. */
        if (m->loop_in_body && m->loop_cf_depth > 0 &&
            m->loop_break_nest == 0)
        {
            /* Exit do-wrapper only → fall into continuing. */
            m_puts(m, "break;\n");
        } else {
            m_puts(m, "continue;\n");
        }
        break;
    case WGSL_NODE_STMT_ASSIGN:
        m_indent(m);
        if (n->child_count > 0) emit_expr(m, n->children[0], depth + 1);
        m_puts(m, " = ");
        if (n->child_count > 1) emit_expr(m, n->children[1], depth + 1);
        m_puts(m, ";\n");
        break;
    case WGSL_NODE_STMT_COMPOUND_ASSIGN: {
        m_indent(m);
        WGSLTokenKind op = (WGSLTokenKind)wgsl_node_op_kind_u32(n);
        const char *ops = wgsl_token_kind_name(op);
        if (n->child_count > 0) emit_expr(m, n->children[0], depth + 1);
        m_putc(m, ' ');
        if (ops) m_puts(m, ops);
        m_putc(m, ' ');
        if (n->child_count > 1) emit_expr(m, n->children[1], depth + 1);
        m_puts(m, ";\n");
        break;
    }
    case WGSL_NODE_STMT_PHONY_ASSIGN:
        m_indent(m);
        m_puts(m, "(void)(");
        if (n->child_count > 0) emit_expr(m, n->children[0], depth + 1);
        m_puts(m, ");\n");
        break;
    case WGSL_NODE_STMT_INCREMENT:
        m_indent(m);
        if (n->child_count > 0) emit_expr(m, n->children[0], depth + 1);
        m_puts(m, "++;\n");
        break;
    case WGSL_NODE_STMT_DECREMENT:
        m_indent(m);
        if (n->child_count > 0) emit_expr(m, n->children[0], depth + 1);
        m_puts(m, "--;\n");
        break;
    case WGSL_NODE_STMT_IF: {
        m_indent(m);
        m_puts(m, "if (");
        if (n->child_count > 0) emit_expr(m, n->children[0], depth + 1);
        m_puts(m, ") ");
        int i = 1;
        if (i < (int)n->child_count && n->children[i] &&
            n->children[i]->kind == WGSL_NODE_STMT_COMPOUND) {
            emit_compound(m, n->children[i], depth + 1);
            i++;
        }
        while (i < (int)n->child_count) {
            const WGSLNode *c = n->children[i++];
            if (!c) continue;
            if (c->kind == WGSL_NODE_STMT_ELSE_IF) {
                m_puts(m, " else if (");
                if (c->child_count > 0) emit_expr(m, c->children[0], depth + 1);
                m_puts(m, ") ");
                if (c->child_count > 1) emit_compound(m, c->children[1], depth + 1);
            } else if (c->kind == WGSL_NODE_STMT_ELSE) {
                m_puts(m, " else ");
                if (c->child_count > 0) emit_compound(m, c->children[0], depth + 1);
            } else if (c->kind == WGSL_NODE_STMT_COMPOUND) {
                m_puts(m, " else ");
                emit_compound(m, c, depth + 1);
            }
        }
        m_nl(m);
        break;
    }
    case WGSL_NODE_STMT_WHILE:
        if (m->loop_in_body) m->loop_break_nest++;
        m_indent(m);
        m_puts(m, "while (");
        if (n->child_count > 0) emit_expr(m, n->children[0], depth + 1);
        m_puts(m, ") ");
        if (n->child_count > 1) emit_compound(m, n->children[1], depth + 1);
        m_nl(m);
        if (m->loop_in_body) m->loop_break_nest--;
        break;
    case WGSL_NODE_STMT_LOOP:
        emit_loop(m, n, depth);
        break;
    case WGSL_NODE_STMT_FOR:
        emit_for(m, n, depth);
        break;
    case WGSL_NODE_STMT_SWITCH:
        emit_switch(m, n, depth);
        break;
    case WGSL_NODE_STMT_BREAK_IF:
        /* break if cond; → if (cond) break;  (WGSL: only in continuing) */
        m_indent(m);
        m_puts(m, "if (");
        if (n->child_count > 0) emit_expr(m, n->children[0], depth + 1);
        m_puts(m, ") break;\n");
        break;
    case WGSL_NODE_STMT_CONTINUING:
        /* Standalone continuing only appears as loop child; if reached here,
         * emit its body (used by emit_loop). */
        for (uint32_t i = 0; i < n->child_count; i++)
            emit_stmt(m, n->children[i], depth);
        break;
    case WGSL_NODE_STMT_FN_CALL:
        m_indent(m);
        if (n->child_count > 0) emit_expr(m, n->children[0], depth + 1);
        m_puts(m, ";\n");
        break;
    case WGSL_NODE_DECL_CONST_ASSERT:
        m_indent(m);
        m_puts(m, "/* const_assert ");
        if (n->child_count > 0) emit_expr(m, n->children[0], depth + 1);
        m_puts(m, "; */\n");
        break;
    default:
        m_indent(m);
        m_puts(m, "/*stmt*/ ");
        m_span(m, n->span_offset, n->span_length);
        m_nl(m);
        m_diag(m, WGSL_MSL_DIAG_UNSUPPORTED_STMT,
               n->span_offset, n->span_length,
               "statement kind not emitted as MSL");
        break;
    }
}
