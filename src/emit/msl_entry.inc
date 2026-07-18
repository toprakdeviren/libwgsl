static void emit_struct(M *m, const WGSLNode *n, int depth) {
    m_indent(m);
    m_puts(m, "struct ");
    uint32_t no = wgsl_node_name_span(n).offset;
    uint32_t nl = wgsl_node_name_span(n).length;
    m_span(m, no, nl);
    m_puts(m, " {\n");
    m->indent++;
    for (uint32_t i = 0; i < n->child_count; i++) {
        const WGSLNode *mem = n->children[i];
        if (!mem || mem->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
        uint32_t MA = wgsl_member_attr_count(mem);
        AttrInfo ai;
        collect_attrs(m, (const WGSLNode *const *)mem->children, 0, MA, &ai);
        m_indent(m);
        if (MA < mem->child_count)
            emit_type_spec(m, mem->children[MA], depth + 1);
        m_putc(m, ' ');
        uint32_t mno = wgsl_node_name_span(mem).offset;
        uint32_t mnl = wgsl_node_name_span(mem).length;
        m_span(m, mno, mnl);
        emit_msl_attrs_trail(m, &ai, 0, 0);
        m_puts(m, ";\n");
    }
    m->indent--;
    m_indent(m);
    m_puts(m, "};\n");
}

/* ── §B1: non-mutating entry-IO plan (emit-only; AST untouched) ─────
 *
 * Metal requires user-defined inter-stage IO in a [[stage_in]] struct;
 * fragment color outs use [[color(N)]], vertex attrs use [[attribute(N)]]
 * (MVP: free params, not packed stage_in — intentional).
 * V↔F location matching is out of scope (pipeline validation).
 */

typedef struct {
    AttrInfo ai;
    const WGSLNode *type_spec;
    uint32_t name_off, name_len;
    WGSLSymbol *sym;          /* param symbol for body rewrite; NULL if none */
    int is_stage_in;          /* 1 → member of synthetic stage_in */
    int loc_role;             /* 0 user, 1 attribute, 2 color */
    char field[64];
} MslIoSlot;

typedef struct {
    int stage; /* 1 vx 2 fs 3 cs */
    MslIoSlot free_params[MSL_SLOT_MAX];
    int n_free;
    MslIoSlot stage_in[MSL_SLOT_MAX];
    int n_stage_in;
    /* Return: either named/type-spec struct kept as-is, or scalar+trail attr. */
    const WGSLNode *ret_type_spec;
    AttrInfo ret_ai;
    int ret_loc_role;
    int has_ret;
    char stage_in_ty[80];
    char fn_name[64];
    const WGSLNode *fn;
    const WGSLNode *body;
} MslEntryPlan;

static void slot_set_name(M *m, MslIoSlot *s, uint32_t off, uint32_t len) {
    s->name_off = off;
    s->name_len = len;
    size_t n = len < sizeof s->field - 1 ? len : sizeof s->field - 1;
    if (m->src && (size_t)off + n <= m->src_len) {
        memcpy(s->field, m->src + off, n);
        s->field[n] = 0;
    } else {
        snprintf(s->field, sizeof s->field, "p");
    }
}

static int plan_push_free(MslEntryPlan *ep, const MslIoSlot *s) {
    if (ep->n_free >= MSL_SLOT_MAX) return 0;
    ep->free_params[ep->n_free++] = *s;
    return 1;
}
static int plan_push_in(MslEntryPlan *ep, const MslIoSlot *s) {
    if (ep->n_stage_in >= MSL_SLOT_MAX) return 0;
    ep->stage_in[ep->n_stage_in++] = *s;
    return 1;
}

/* Classify one IO surface (param or flattened member) into free vs stage_in. */
static void plan_classify_input(MslEntryPlan *ep, MslIoSlot *s) {
    s->is_stage_in = 0;
    s->loc_role = 0;
    if (s->ai.has_builtin) {
        s->is_stage_in = 0; /* builtins never enter stage_in */
        plan_push_free(ep, s);
        return;
    }
    if (s->ai.has_location) {
        if (ep->stage == 1) {
            /* Vertex inputs: free [[attribute(N)]] (MVP shape). */
            s->loc_role = 1;
            s->is_stage_in = 0;
            plan_push_free(ep, s);
        } else if (ep->stage == 2) {
            /* Fragment user varyings → stage_in [[user(locnN)]]. */
            s->loc_role = 0;
            s->is_stage_in = 1;
            plan_push_in(ep, s);
        } else {
            plan_push_free(ep, s);
        }
        return;
    }
    /* Untagged / buffer-ish: free (compute resources are module-scope). */
    plan_push_free(ep, s);
}

static WGSLSymbol *param_symbol(M *m, const WGSLNode *p) {
    if (!m->res) return NULL;
    return wgsl_resolver_symbol_for_decl(&m->res->res, p);
}

static WGSLTypeInfo *node_type(M *m, const WGSLNode *n) {
    if (!m->res || !n) return NULL;
    return wgsl_typecheck_type_of(&m->res->tc, n);
}

static void plan_add_param(M *m, MslEntryPlan *ep, const WGSLNode *p) {
    if (!p || p->kind != WGSL_NODE_DECL_PARAM) return;
    uint32_t PA = wgsl_param_attr_count(p);
    const WGSLNode *ts = (PA < p->child_count) ? p->children[PA] : NULL;
    AttrInfo pai;
    collect_attrs(m, (const WGSLNode *const *)p->children, 0, PA, &pai);

    WGSLTypeInfo *pt = node_type(m, ts);
    if (pt && pt->kind == WGSL_TYPE_STRUCT && pt->ref) {
        /* Flatten struct-typed entry param members into free/stage_in slots. */
        const WGSLNode *st = (const WGSLNode *)pt->ref;
        m_diag(m, WGSL_MSL_DIAG_PARTIAL_IO,
               wgsl_node_name_span(p).offset, wgsl_node_name_span(p).length,
               "struct-typed entry param: members flattened for signature; "
               "nested field uses in the body are not rewritten (MVP)");
        for (uint32_t i = 0; i < st->child_count; i++) {
            const WGSLNode *mem = st->children[i];
            if (!mem || mem->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            uint32_t MA = wgsl_member_attr_count(mem);
            MslIoSlot s;
            memset(&s, 0, sizeof s);
            collect_attrs(m, (const WGSLNode *const *)mem->children, 0, MA, &s.ai);
            s.type_spec = (MA < mem->child_count) ? mem->children[MA] : NULL;
            slot_set_name(m, &s, wgsl_node_name_span(mem).offset,
                          wgsl_node_name_span(mem).length);
            s.sym = NULL; /* member path rewrite not in MVP */
            plan_classify_input(ep, &s);
        }
        return;
    }

    MslIoSlot s;
    memset(&s, 0, sizeof s);
    s.ai = pai;
    s.type_spec = ts;
    slot_set_name(m, &s, wgsl_node_name_span(p).offset,
                  wgsl_node_name_span(p).length);
    s.sym = param_symbol(m, p);
    plan_classify_input(ep, &s);
}

static int build_entry_plan(M *m, const WGSLNode *fn, MslEntryPlan *ep) {
    memset(ep, 0, sizeof *ep);
    ep->fn = fn;
    uint32_t FA = wgsl_fn_attr_count(fn);
    uint32_t P  = wgsl_fn_param_count(fn);
    uint32_t RA = wgsl_fn_ret_attr_count(fn);
    uint32_t HR = wgsl_fn_has_return_type(fn);

    AttrInfo fai;
    collect_attrs(m, (const WGSLNode *const *)fn->children, 0, FA, &fai);
    ep->stage = fai.stage;
    if (ep->stage != 1 && ep->stage != 2 && ep->stage != 3) return 0;

    uint32_t no = wgsl_node_name_span(fn).offset;
    uint32_t nl = wgsl_node_name_span(fn).length;
    size_t nn = nl < sizeof ep->fn_name - 1 ? nl : sizeof ep->fn_name - 1;
    if (m->src && (size_t)no + nn <= m->src_len) {
        memcpy(ep->fn_name, m->src + no, nn);
        ep->fn_name[nn] = 0;
    } else {
        snprintf(ep->fn_name, sizeof ep->fn_name, "fn");
    }
    snprintf(ep->stage_in_ty, sizeof ep->stage_in_ty, "__wgsl_in_%s", ep->fn_name);

    for (uint32_t k = 0; k < P; k++) {
        uint32_t ci = FA + k;
        if (ci >= fn->child_count) break;
        plan_add_param(m, ep, fn->children[ci]);
    }

    ep->has_ret = (int)HR;
    if (HR) {
        uint32_t rti = FA + P + RA;
        ep->ret_type_spec =
            (rti < fn->child_count) ? fn->children[rti] : NULL;
        collect_attrs(m, (const WGSLNode *const *)fn->children, FA + P, RA,
                      &ep->ret_ai);
        /* Fragment @location return → [[color(N)]]; vertex position stays
         * builtin; other locations on return → user (rare for scalar). */
        if (ep->stage == 2 && ep->ret_ai.has_location && !ep->ret_ai.has_builtin)
            ep->ret_loc_role = 2;
        else if (ep->stage == 1 && ep->ret_ai.has_location &&
                 !ep->ret_ai.has_builtin)
            ep->ret_loc_role = 0; /* VS scalar user out — unusual */
        else
            ep->ret_loc_role = 0;
    }

    for (uint32_t i = 0; i < fn->child_count; i++) {
        if (fn->children[i] &&
            fn->children[i]->kind == WGSL_NODE_STMT_COMPOUND) {
            ep->body = fn->children[i];
            break;
        }
    }
    return 1;
}

static void emit_slot_type_name(M *m, const MslIoSlot *s, int depth) {
    if (s->type_spec)
        emit_type_spec(m, s->type_spec, depth);
    else
        m_puts(m, "float");
}

static void emit_entry_from_plan(M *m, MslEntryPlan *ep, int depth) {
    uint32_t FA = wgsl_fn_attr_count(ep->fn);
    AttrInfo fai;
    collect_attrs(m, (const WGSLNode *const *)ep->fn->children, 0, FA, &fai);

    if (fai.has_workgroup) {
        char t[80];
        snprintf(t, sizeof t,
                 "// workgroup_size(%d, %d, %d) — host sets MTLSize\n",
                 fai.workgroup[0], fai.workgroup[1], fai.workgroup[2]);
        m_puts(m, t);
    }

    /* Prelude: synthetic stage_in struct for user varyings. */
    if (ep->n_stage_in > 0) {
        m_puts(m, "// §B1: synthetic stage_in (user @location inputs); "
                  "AST not mutated\n");
        m_puts(m, "struct ");
        m_puts(m, ep->stage_in_ty);
        m_puts(m, " {\n");
        for (int i = 0; i < ep->n_stage_in; i++) {
            MslIoSlot *s = &ep->stage_in[i];
            m_puts(m, "    ");
            emit_slot_type_name(m, s, depth + 1);
            m_putc(m, ' ');
            m_puts(m, s->field);
            emit_msl_attrs_trail(m, &s->ai, 0, s->loc_role);
            m_puts(m, ";\n");
        }
        m_puts(m, "};\n");
    }

    if (ep->stage == 1 && ep->n_free > 0) {
        /* Note free [[attribute]] MVP so it is not mistaken for final design. */
        int any_attr = 0;
        for (int i = 0; i < ep->n_free; i++)
            if (ep->free_params[i].loc_role == 1) any_attr = 1;
        if (any_attr)
            m_puts(m, "// §B1 MVP: vertex @location inputs as free "
                      "[[attribute(N)]] params (not packed stage_in)\n");
    }

    m_indent(m);
    if (ep->stage == 1) m_puts(m, "vertex ");
    else if (ep->stage == 2) m_puts(m, "fragment ");
    else if (ep->stage == 3) m_puts(m, "kernel ");

    if (ep->has_ret && ep->ret_type_spec)
        emit_type_spec(m, ep->ret_type_spec, depth + 1);
    else if (ep->has_ret)
        m_puts(m, "void");
    else
        m_puts(m, "void");
    m_putc(m, ' ');
    m_puts(m, ep->fn_name);
    m_putc(m, '(');

    int first = 1;
    for (int i = 0; i < ep->n_free; i++) {
        MslIoSlot *s = &ep->free_params[i];
        if (!first) m_puts(m, ", ");
        first = 0;
        emit_slot_type_name(m, s, depth + 1);
        m_putc(m, ' ');
        m_puts(m, s->field);
        emit_msl_attrs_trail(m, &s->ai, 1, s->loc_role);
    }
    if (ep->n_stage_in > 0) {
        if (!first) m_puts(m, ", ");
        first = 0;
        m_puts(m, ep->stage_in_ty);
        m_puts(m, " in [[stage_in]]");
    }
    /* §B3b: inject module-scope resources as entry params. */
    for (int i = 0; i < m->nresources; i++) {
        const MslRes *r = &m->resources[i];
        if (r->kind == MSL_RES_PRIVATE) continue; /* body local */
        if (!first) m_puts(m, ", ");
        first = 0;
        emit_resource_param(m, r, depth + 1);
    }
    m_putc(m, ')');

    /* Return attributes: Apple Metal rejects `fn(...) [[position]]` /
     * `[[color(N)]]` trailing form ("cannot be applied to types").
     * Convention that compiles:
     *   - vertex float4  → implicit [[position]]
     *   - fragment float4 → implicit [[color(0)]]
     *   - multi-slot / non-default → named return struct with member attrs
     *     (user structs already emitted with member [[…]]).
     * Scalar @location(N) for N!=0 is a known MVP gap (needs out-struct
     * + return rewrite); fixtures use position / color(0). */
    (void)ep->ret_loc_role;
    (void)ep->ret_ai;

    m_putc(m, ' ');

    /* Body rewrite table for stage_in params — skip shadowed names (§B2). */
    m->rw_n = 0;
    {
        const WGSLSymbol *syms[MSL_SLOT_MAX];
        uint8_t shadowed[MSL_SLOT_MAX];
        int nsym = 0;
        memset(shadowed, 0, sizeof shadowed);
        for (int i = 0; i < ep->n_stage_in && nsym < MSL_SLOT_MAX; i++) {
            if (ep->stage_in[i].sym)
                syms[nsym++] = ep->stage_in[i].sym;
        }
        if (ep->body && nsym > 0)
            detect_shadowing(m, ep->body, syms, nsym, shadowed);
        int si = 0;
        for (int i = 0; i < ep->n_stage_in && m->rw_n < MSL_RW_MAX; i++) {
            MslIoSlot *s = &ep->stage_in[i];
            if (!s->sym) {
                m_diag(m, WGSL_MSL_DIAG_PARTIAL_IO,
                       s->name_off, s->name_len,
                       "stage_in field has no resolver symbol; body uses "
                       "of this name are not rewritten");
                continue;
            }
            int sh = 0;
            for (int k = 0; k < nsym; k++) {
                if (syms[k] == s->sym) { sh = shadowed[k]; break; }
            }
            (void)si;
            if (sh) continue; /* diag already emitted; do not mis-rewrite */
            m->rw_sym[m->rw_n] = s->sym;
            snprintf(m->rw_to[m->rw_n], sizeof m->rw_to[0], "in.%s", s->field);
            m->rw_n++;
        }
    }

    /* B1 MVP gaps → partial IO diagnostics (loud, not silent wrong MSL). */
    if (ep->stage == 2 && ep->has_ret && ep->ret_ai.has_location &&
        !ep->ret_ai.has_builtin && ep->ret_ai.location != 0)
    {
        m_diag(m, WGSL_MSL_DIAG_PARTIAL_IO, 0, 0,
               "fragment @location(N) return with N!=0 needs out-struct "
               "(MVP emits bare float4 as color0 only)");
    }

    if (ep->body) {
        /* Open compound ourselves so §B3b private locals land first. */
        m_indent(m);
        m_puts(m, "{\n");
        m->indent++;
        emit_private_preamble(m, depth + 1);
        for (uint32_t i = 0; i < ep->body->child_count; i++) {
            const WGSLNode *c = ep->body->children[i];
            if (!c || c->kind == WGSL_NODE_ATTRIBUTE) continue;
            emit_stmt(m, c, depth + 1);
        }
        m->indent--;
        m_indent(m);
        m_puts(m, "}");
    } else {
        m_puts(m, "{}\n");
    }

    m->rw_n = 0;
    m_nl(m);
}

static void emit_fn(M *m, const WGSLNode *n, int depth) {
    uint32_t FA = wgsl_fn_attr_count(n);
    AttrInfo fai;
    collect_attrs(m, (const WGSLNode *const *)n->children, 0, FA, &fai);

    /* Graphics + compute entries: plan-driven IO (§B1). */
    if (fai.stage == 1 || fai.stage == 2 || fai.stage == 3) {
        MslEntryPlan ep;
        if (build_entry_plan(m, n, &ep)) {
            emit_entry_from_plan(m, &ep, depth);
            return;
        }
    }

    /* Non-entry user functions: prior span-style emit. */
    uint32_t P  = wgsl_fn_param_count(n);
    uint32_t RA = wgsl_fn_ret_attr_count(n);
    uint32_t HR = wgsl_fn_has_return_type(n);

    m_indent(m);
    if (HR) {
        uint32_t rti = FA + P + RA;
        if (rti < n->child_count)
            emit_type_spec(m, n->children[rti], depth + 1);
        else
            m_puts(m, "void");
    } else {
        m_puts(m, "void");
    }
    m_putc(m, ' ');
    uint32_t no = wgsl_node_name_span(n).offset;
    uint32_t nl = wgsl_node_name_span(n).length;
    m_span(m, no, nl);
    m_putc(m, '(');
    for (uint32_t k = 0; k < P; k++) {
        if (k) m_puts(m, ", ");
        uint32_t ci = FA + k;
        if (ci >= n->child_count) break;
        const WGSLNode *p = n->children[ci];
        if (!p || p->kind != WGSL_NODE_DECL_PARAM) continue;
        uint32_t PA = wgsl_param_attr_count(p);
        if (PA < p->child_count)
            emit_type_spec(m, p->children[PA], depth + 1);
        m_putc(m, ' ');
        uint32_t pno = wgsl_node_name_span(p).offset;
        uint32_t pnl = wgsl_node_name_span(p).length;
        m_span(m, pno, pnl);
    }
    m_putc(m, ')');
    m_putc(m, ' ');
    for (uint32_t i = 0; i < n->child_count; i++) {
        if (n->children[i] && n->children[i]->kind == WGSL_NODE_STMT_COMPOUND) {
            emit_compound(m, n->children[i], depth + 1);
            break;
        }
    }
    m_nl(m);
}

