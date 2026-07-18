/**
 * @file compile.c — structured-control-flow → bytecode compiler + program cache
 */
#include "internal/interp_priv.h"

/* bytecode ISA types: interp_priv.h. */

/* compiler: structured statement AST -> linear bytecode. */

typedef struct { uint32_t brk, cont; } LoopCtx;   /* break / continue label */

typedef struct {
    Prog     *prog;
    uint32_t *label_pc;  uint32_t nlabels, caplabels;   /* label id → pc     */
    LoopCtx  *loops;     uint32_t nloops, caploops;      /* enclosing loops   */
    SwTable **sws;       uint32_t nsws,  capsws;         /* owned switch tables */
    int       ok;
} Comp;

static uint32_t comp_new_label(Comp *c) {
    if (c->nlabels == c->caplabels) {
        uint32_t nc = c->caplabels ? c->caplabels * 2 : 32;
        uint32_t *g = (uint32_t *)realloc(c->label_pc, nc * sizeof *g);
        if (!g) { c->ok = 0; return 0; }
        c->label_pc = g; c->caplabels = nc;
    }
    c->label_pc[c->nlabels] = LABEL_UNSET;
    return c->nlabels++;
}
static void comp_place(Comp *c, uint32_t label) {
    if (label < c->nlabels) c->label_pc[label] = c->prog->ncode;
}
static void comp_emit(Comp *c, OpKind op, uint32_t target, WGSLNode *node, SwTable *sw) {
    Prog *p = c->prog;
    if (p->ncode == p->capcode) {
        uint32_t nc = p->capcode ? p->capcode * 2 : 64;
        Instr *g = (Instr *)realloc(p->code, nc * sizeof *g);
        if (!g) { c->ok = 0; return; }
        p->code = g; p->capcode = nc;
    }
    Instr *in = &p->code[p->ncode++];
    in->op = (uint16_t)op; in->target = target; in->node = node; in->sw = sw;
}
static void comp_push_loop(Comp *c, uint32_t brk, uint32_t cont) {
    if (c->nloops == c->caploops) {
        uint32_t nc = c->caploops ? c->caploops * 2 : 16;
        LoopCtx *g = (LoopCtx *)realloc(c->loops, nc * sizeof *g);
        if (!g) { c->ok = 0; return; }
        c->loops = g; c->caploops = nc;
    }
    c->loops[c->nloops].brk = brk; c->loops[c->nloops].cont = cont; c->nloops++;
}
static void comp_pop_loop(Comp *c) { if (c->nloops) c->nloops--; }
static uint32_t comp_brk(Comp *c)  { return c->nloops ? c->loops[c->nloops - 1].brk  : LABEL_UNSET; }
static uint32_t comp_cont(Comp *c) { return c->nloops ? c->loops[c->nloops - 1].cont : LABEL_UNSET; }

static SwTable *comp_new_switch(Comp *c, uint32_t ncases) {
    SwTable *sw = (SwTable *)calloc(1, sizeof *sw);
    if (!sw) { c->ok = 0; return NULL; }
    if (ncases) {
        sw->cases = (SwCase *)calloc(ncases, sizeof *sw->cases);
        if (!sw->cases) { free(sw); c->ok = 0; return NULL; }
    }
    if (c->nsws == c->capsws) {
        uint32_t nc = c->capsws ? c->capsws * 2 : 8;
        SwTable **g = (SwTable **)realloc(c->sws, nc * sizeof *g);
        if (!g) { free(sw->cases); free(sw); c->ok = 0; return NULL; }
        c->sws = g; c->capsws = nc;
    }
    c->sws[c->nsws++] = sw;
    return sw;
}

static void comp_stmt(Comp *c, WGSLNode *n);

static void comp_block(Comp *c, WGSLNode *block) {
    if (!block) return;
    for (uint32_t i = 0; i < block->child_count && c->ok; i++)
        comp_stmt(c, block->children[i]);
}

static void comp_if(Comp *c, WGSLNode *n) {
    uint32_t Lend = comp_new_label(c);
    uint32_t Lnext = comp_new_label(c);
    comp_emit(c, OP_BR_IFNOT, Lnext, n->child_count ? n->children[0] : NULL, NULL);
    comp_stmt(c, n->child_count > 1 ? n->children[1] : NULL);
    comp_emit(c, OP_BR, Lend, NULL, NULL);
    comp_place(c, Lnext);
    for (uint32_t i = 2; i < n->child_count && c->ok; i++) {
        WGSLNode *clause = n->children[i];
        if (!clause) continue;
        if (clause->kind == WGSL_NODE_STMT_ELSE_IF) {
            uint32_t Ln = comp_new_label(c);
            comp_emit(c, OP_BR_IFNOT, Ln,
                      clause->child_count ? clause->children[0] : NULL, NULL);
            comp_stmt(c, clause->child_count >= 2 ? clause->children[1] : NULL);
            comp_emit(c, OP_BR, Lend, NULL, NULL);
            comp_place(c, Ln);
        } else if (clause->kind == WGSL_NODE_STMT_ELSE) {
            comp_stmt(c, clause->child_count
                          ? clause->children[clause->child_count - 1] : NULL);
        }
    }
    comp_place(c, Lend);
}

static void comp_for(Comp *c, WGSLNode *n) {
    uint32_t flags = wgsl_for_flags(n);
    uint32_t idx = 0;
    WGSLNode *init = (flags & 1u) && idx < n->child_count ? n->children[idx++] : NULL;
    WGSLNode *cond = (flags & 2u) && idx < n->child_count ? n->children[idx++] : NULL;
    WGSLNode *cont = (flags & 4u) && idx < n->child_count ? n->children[idx++] : NULL;
    WGSLNode *body = idx < n->child_count ? n->children[idx] : NULL;
    if (init) comp_stmt(c, init);
    uint32_t Lhead = comp_new_label(c), Lcont = comp_new_label(c), Lend = comp_new_label(c);
    comp_place(c, Lhead);
    if (cond) comp_emit(c, OP_BR_IFNOT, Lend, cond, NULL);
    comp_push_loop(c, Lend, Lcont);
    comp_stmt(c, body);
    comp_pop_loop(c);
    comp_place(c, Lcont);
    if (cont) comp_stmt(c, cont);
    comp_emit(c, OP_BR, Lhead, NULL, NULL);
    comp_place(c, Lend);
}

static void comp_while(Comp *c, WGSLNode *n) {
    WGSLNode *cond = n->child_count > 0 ? n->children[0] : NULL;
    WGSLNode *body = n->child_count > 1 ? n->children[1] : NULL;
    uint32_t Lhead = comp_new_label(c), Lend = comp_new_label(c);
    comp_place(c, Lhead);
    if (cond) comp_emit(c, OP_BR_IFNOT, Lend, cond, NULL);
    comp_push_loop(c, Lend, Lhead);   /* `continue` re-tests the condition */
    comp_stmt(c, body);
    comp_pop_loop(c);
    comp_emit(c, OP_BR, Lhead, NULL, NULL);
    comp_place(c, Lend);
}

static void comp_loop(Comp *c, WGSLNode *n) {
    /* `loop { body… continuing { … break if e; } }` — the STMT_CONTINUING block
     * (if present) is the last child. */
    uint32_t nbody = n->child_count;
    WGSLNode *cont = NULL;
    if (nbody > 0 && n->children[nbody - 1] &&
        n->children[nbody - 1]->kind == WGSL_NODE_STMT_CONTINUING) {
        cont = n->children[nbody - 1];
        nbody -= 1;
    }
    uint32_t Lhead = comp_new_label(c), Lcont = comp_new_label(c), Lend = comp_new_label(c);
    comp_place(c, Lhead);
    comp_push_loop(c, Lend, Lcont);
    for (uint32_t i = 0; i < nbody && c->ok; i++) comp_stmt(c, n->children[i]);
    comp_place(c, Lcont);
    if (cont) comp_block(c, cont);    /* break-if inside resolves to Lend    */
    comp_emit(c, OP_BR, Lhead, NULL, NULL);
    comp_pop_loop(c);
    comp_place(c, Lend);
}

static void comp_switch(Comp *c, WGSLNode *n) {
    if (n->child_count == 0) return;
    uint32_t nclauses = n->child_count - 1;   /* child 0 is the selector */
    SwTable *sw = comp_new_switch(c, nclauses);
    if (!sw) return;
    uint32_t Lend = comp_new_label(c);
    sw->exit = Lend;
    /* Reserve one label per clause body; fill the jump table with them. */
    uint32_t base_label = c->nlabels;
    for (uint32_t k = 0; k < nclauses; k++) {
        uint32_t Lk = comp_new_label(c);
        sw->cases[k].clause = n->children[1 + k];
        sw->cases[k].target = Lk;
    }
    sw->ncases = nclauses;
    comp_emit(c, OP_SWITCH, Lend, n->children[0], sw);
    /* `break` inside a case exits the switch; `continue` binds to the
     * enclosing loop (inherit the current continue target). */
    comp_push_loop(c, Lend, comp_cont(c));
    for (uint32_t k = 0; k < nclauses && c->ok; k++) {
        WGSLNode *clause = n->children[1 + k];
        comp_place(c, base_label + k);
        if (clause && clause->child_count > 0)
            comp_stmt(c, clause->children[clause->child_count - 1]);
        comp_emit(c, OP_BR, Lend, NULL, NULL);
    }
    comp_pop_loop(c);
    comp_place(c, Lend);
}

static void comp_stmt(Comp *c, WGSLNode *n) {
    if (!n || !c->ok) return;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_STMT_COMPOUND: comp_block(c, n); break;

    case WGSL_NODE_DECL_VAR:
    case WGSL_NODE_DECL_LET:
    case WGSL_NODE_STMT_ASSIGN:
    case WGSL_NODE_STMT_COMPOUND_ASSIGN:
    case WGSL_NODE_STMT_INCREMENT:
    case WGSL_NODE_STMT_DECREMENT:
    case WGSL_NODE_STMT_FN_CALL:
    case WGSL_NODE_STMT_PHONY_ASSIGN:
        comp_emit(c, OP_STMT, 0, n, NULL);
        break;

    case WGSL_NODE_STMT_IF:     comp_if(c, n);     break;
    case WGSL_NODE_STMT_FOR:    comp_for(c, n);    break;
    case WGSL_NODE_STMT_WHILE:  comp_while(c, n);  break;
    case WGSL_NODE_STMT_LOOP:   comp_loop(c, n);   break;
    case WGSL_NODE_STMT_SWITCH: comp_switch(c, n); break;

    case WGSL_NODE_STMT_BREAK:    comp_emit(c, OP_BR, comp_brk(c),  NULL, NULL); break;
    case WGSL_NODE_STMT_CONTINUE: comp_emit(c, OP_BR, comp_cont(c), NULL, NULL); break;
    case WGSL_NODE_STMT_BREAK_IF: comp_emit(c, OP_BR_IF, comp_brk(c),
                                            n->child_count ? n->children[0] : NULL, NULL); break;

    case WGSL_NODE_STMT_RETURN: comp_emit(c, OP_RET, 0, n, NULL); break;

    /* Fn-scope `const` is pre-bound by the const-eval pass; phony/discard/
     * const-assert have no runtime effect here. */
    default: break;
    }
}

/* Resolve all label references to instruction indices, then append a HALT. */
static int comp_finish(Comp *c) {
    comp_emit(c, OP_HALT, 0, NULL, NULL);
    if (!c->ok) return 0;
    for (uint32_t i = 0; i < c->prog->ncode; i++) {
        Instr *in = &c->prog->code[i];
        if (in->op == OP_BR || in->op == OP_BR_IFNOT || in->op == OP_BR_IF) {
            uint32_t pc = (in->target < c->nlabels) ? c->label_pc[in->target] : LABEL_UNSET;
            in->target = (pc == LABEL_UNSET) ? c->prog->ncode - 1 : pc;   /* stray → HALT */
        } else if (in->op == OP_SWITCH && in->sw) {
            SwTable *sw = in->sw;
            uint32_t ex = (sw->exit < c->nlabels) ? c->label_pc[sw->exit] : LABEL_UNSET;
            sw->exit = (ex == LABEL_UNSET) ? c->prog->ncode - 1 : ex;
            for (uint32_t k = 0; k < sw->ncases; k++) {
                uint32_t t = (sw->cases[k].target < c->nlabels)
                             ? c->label_pc[sw->cases[k].target] : LABEL_UNSET;
                sw->cases[k].target = (t == LABEL_UNSET) ? sw->exit : t;
            }
        }
    }
    return 1;
}

/* compiled-program cache (one Prog per function body, stable pointers). */

typedef struct { WGSLNode *body; Prog *prog; SwTable **sws; uint32_t nsws; } ProgEntry;
typedef struct { ProgEntry *v; uint32_t n, cap; } ProgCache;

const Prog *get_prog(Interp *ip, WGSLNode *body) {
    ProgCache *pc = (ProgCache *)ip->progs;
    if (!pc) {
        pc = (ProgCache *)calloc(1, sizeof *pc);
        ip->progs = pc;
        if (!pc) return NULL;
    }
    for (uint32_t i = 0; i < pc->n; i++)
        if (pc->v[i].body == body) return pc->v[i].prog;

    Prog *prog = (Prog *)calloc(1, sizeof *prog);
    if (!prog) return NULL;
    Comp c; memset(&c, 0, sizeof c);
    c.prog = prog; c.ok = 1;
    comp_block(&c, body);
    if (!comp_finish(&c)) {
        free(prog->code); free(c.label_pc); free(c.loops);
        for (uint32_t i = 0; i < c.nsws; i++) { free(c.sws[i]->cases); free(c.sws[i]); }
        free(c.sws); free(prog);
        return NULL;
    }
    free(c.label_pc); free(c.loops);
    if (pc->n == pc->cap) {
        uint32_t nc = pc->cap ? pc->cap * 2 : 8;
        ProgEntry *g = (ProgEntry *)realloc(pc->v, nc * sizeof *g);
        if (!g) { free(prog->code); for (uint32_t i=0;i<c.nsws;i++){free(c.sws[i]->cases);free(c.sws[i]);} free(c.sws); free(prog); return NULL; }
        pc->v = g; pc->cap = nc;
    }
    pc->v[pc->n].body = body; pc->v[pc->n].prog = prog;
    pc->v[pc->n].sws = c.sws; pc->v[pc->n].nsws = c.nsws;
    pc->n++;
    return prog;
}

void progs_free(Interp *ip) {
    ProgCache *pc = (ProgCache *)ip->progs;
    if (!pc) return;
    for (uint32_t i = 0; i < pc->n; i++) {
        free(pc->v[i].prog->code);
        for (uint32_t k = 0; k < pc->v[i].nsws; k++) { free(pc->v[i].sws[k]->cases); free(pc->v[i].sws[k]); }
        free(pc->v[i].sws);
        free(pc->v[i].prog);
    }
    free(pc->v); free(pc);
    ip->progs = NULL;
}
