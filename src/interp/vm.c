/**
 * @file vm.c — SIMT warp scheduler, memory/race analysis, user-call hooks
 */
#include "internal/interp_priv.h"


/* warp scheduler (min-PC lockstep with per-lane program counters).
 *
 * All active lanes run one shared instruction stream, each with its own pc.
 * Every step advances the group of lanes sitting at the *minimum* pc together;
 * a lane that branches ahead simply waits (higher pc) until the lanes behind
 * catch up to it.  For WGSL's structured (reducible) control flow this
 * reconverges divergent lanes at the immediate post-dominator automatically,
 * and makes any post-branch merge point — a barrier included — a real join:
 * no lane can slip past it while another is still behind. */

/* Resolve a switch selector to its target pc for the current lane. */
uint32_t vm_switch_target(Interp *ip, WGSLNode *sw_node, const SwTable *sw) {
    WGSLValue sel;
    if (!sw || !eval(ip, sw_node, &sel)) return sw ? sw->exit : 0;
    uint32_t deflt = sw->exit;
    for (uint32_t k = 0; k < sw->ncases; k++) {
        WGSLNode *clause = sw->cases[k].clause;
        if (!clause || clause->kind != WGSL_NODE_STMT_CASE_CLAUSE) continue;
        uint32_t sel_count = wgsl_case_selector_count(clause);
        if (wgsl_case_is_default_alone(clause) ||
            wgsl_case_has_default_in_list(clause))
            deflt = sw->cases[k].target;
        for (uint32_t j = 0; j < sel_count && j < clause->child_count; j++) {
            WGSLValue lab, eq;
            if (eval(ip, clause->children[j], &lab) &&
                wgsl_consteval_binop(ip->cev, clause, WGSL_TOK_EQUAL_EQUAL,
                                     &sel, &lab, &eq) && is_true(&eq))
                return sw->cases[k].target;
        }
    }
    return deflt;
}

/* Bind the evaluator to lane `laneid` and publish the active group so a
 * cooperative op reached during this lane's evaluation sees its peers.
 * Re-published per lane because a nested user-call run overwrites `act`. */
void vm_enter(Interp *ip, const uint32_t *act, uint32_t nact, uint32_t laneid) {
    ip->act = act; ip->nact = nact;
    set_lane(ip, laneid);
}

/* Per-invocation cost: the reported `cost` is the *focus* lane's alone — the
 * other lanes exist only to make cooperative ops / shared memory correct.  So
 * peer lanes run with their cost-counter deltas rolled back.  (The focus lane's
 * own cross-lane gathers can still bill a little extra, which is fine.) */
void cost_snapshot(const WGSLConstEvaluator *cev, uint64_t s[4]) {
    s[0] = cev->cost.flop_count; s[1] = cev->cost.iop_count;
    s[2] = cev->cost.bytes_loaded; s[3] = cev->cost.bytes_stored;
}
void cost_rollback(WGSLConstEvaluator *cev, const uint64_t s[4]) {
    cev->cost.flop_count = s[0]; cev->cost.iop_count = s[1];
    cev->cost.bytes_loaded = s[2]; cev->cost.bytes_stored = s[3];
}

/* Fold one warp-step's active-lane count into the per-line heatmap. */
void div_line(Interp *ip, uint32_t line, uint32_t active) {
    if (line == 0) return;
    if (line >= ip->linediv_cap) {
        uint32_t nc = ip->linediv_cap ? ip->linediv_cap : 64;
        while (line >= nc) { if (nc > 0x7fffffffu) return; nc *= 2; }
        struct LineDiv *g = (struct LineDiv *)realloc(ip->linediv, (size_t)nc * sizeof *g);
        if (!g) return;
        for (uint32_t i = ip->linediv_cap; i < nc; i++) {
            g[i].steps = 0; g[i].max_active = 0; g[i].active_sum = 0;
            g[i].min_active = 0xffffffffu;
        }
        ip->linediv = g; ip->linediv_cap = nc;
    }
    struct LineDiv *d = &ip->linediv[line];
    d->steps++; d->active_sum += active;
    if (active < d->min_active) d->min_active = active;
    if (active > d->max_active) d->max_active = active;
    if (line > ip->max_line) ip->max_line = line;
}

/* warp memory analysis. */

#define INTERP_CACHE_LINE 128u   /* DRAM coalescing granularity (bytes)   */
#define INTERP_BANKS       32u   /* shared-memory banks                   */

/* Root buffer symbol of an access chain (walk the base to its ident). */
const WGSLSymbol *access_root(WGSLNode *n) {
    while (n) {
        if (n->kind == WGSL_NODE_EXPR_IDENT) return wgsl_node_resolved_symbol(n);
        if (n->kind == WGSL_NODE_EXPR_INDEX || n->kind == WGSL_NODE_EXPR_MEMBER ||
            n->kind == WGSL_NODE_EXPR_PAREN)
            n = n->child_count ? n->children[0] : NULL;
        else return NULL;
    }
    return NULL;
}

/* Default race-cell table cap (internal analysis buffer, not a WGSL limit). */
#ifndef WGSL_RACE_CELLS_CAP_DEFAULT
#define WGSL_RACE_CELLS_CAP_DEFAULT (1u << 16)
#endif

/* Overridable for tests; 0 → use WGSL_RACE_CELLS_CAP_DEFAULT. */
uint32_t wgsl_test_race_cells_cap = 0;

static uint32_t race_cells_cap_of(const Interp *ip) {
    if (ip->race_cells_cap) return ip->race_cells_cap;
    if (wgsl_test_race_cells_cap) return wgsl_test_race_cells_cap;
    return WGSL_RACE_CELLS_CAP_DEFAULT;
}

static uint64_t race_key_hash(const WGSLSymbol *root, uint32_t offset) {
    uint64_t x = (uint64_t)(uintptr_t)root;
    x ^= (uint64_t)offset * 0x9e3779b97f4a7c15ull;
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27; x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x ? x : 1;
}

static int race_cell_matches(const struct RCell *c,
                             const WGSLSymbol *root, uint32_t offset) {
    return c && c->root == root && c->offset == offset;
}

static void race_htab_insert(uint32_t *tab, uint32_t cap,
                             const struct RCell *cells, uint32_t idx) {
    const struct RCell *c = &cells[idx];
    uint32_t mask = cap - 1u;
    uint32_t h = (uint32_t)race_key_hash(c->root, c->offset) & mask;
    while (tab[h]) h = (h + 1u) & mask;
    tab[h] = idx + 1u;
}

static int race_htab_reserve(Interp *ip, uint32_t need) {
    if (!ip) return 0;
    uint32_t want = 512;
    while (want < need * 2u) {
        if (want > 0x80000000u) return 0;
        want <<= 1;
    }
    if (ip->cell_htab_cap >= want) return 1;
    uint32_t *tab = (uint32_t *)calloc((size_t)want, sizeof *tab);
    if (!tab) return 0;
    for (uint32_t i = 0; i < ip->ncells; i++)
        race_htab_insert(tab, want, ip->cells, i);
    free(ip->cell_htab);
    ip->cell_htab = tab;
    ip->cell_htab_cap = want;
    return 1;
}

static struct RCell *race_cell_lookup(Interp *ip,
                                      const WGSLSymbol *root, uint32_t offset) {
    if (!ip) return NULL;
    if (ip->cell_htab && ip->cell_htab_cap) {
        uint32_t mask = ip->cell_htab_cap - 1u;
        uint32_t h = (uint32_t)race_key_hash(root, offset) & mask;
        for (;;) {
            uint32_t slot = ip->cell_htab[h];
            if (!slot) return NULL;
            struct RCell *c = &ip->cells[slot - 1u];
            if (race_cell_matches(c, root, offset)) return c;
            h = (h + 1u) & mask;
        }
    }
    for (uint32_t i = 0; i < ip->ncells; i++)
        if (race_cell_matches(&ip->cells[i], root, offset)) return &ip->cells[i];
    return NULL;
}

/* Data-race check: within the current barrier epoch, a shared cell touched
 * by two different lanes with at least one write is a race.  Cell = (root sym,
 * byte offset); lane = cev->simt.cur_lane. */
void race_check(Interp *ip, const WGSLSymbol *root, uint32_t offset,
                       int is_write, uint32_t line) {
    if (!root) return;
    int32_t lane = (int32_t)ip->cev->simt.cur_lane;
    struct RCell *c = race_cell_lookup(ip, root, offset);
    if (!c) {
        if (ip->ncells == ip->cells_cap) {
            uint32_t lim = race_cells_cap_of(ip);
            if (ip->cells_cap >= lim) {
                /* Never silent: mark analysis truncated. */
                ip->race_cells_truncated = 1;
                return;
            }
            uint32_t nc = ip->cells_cap ? ip->cells_cap * 2 : 256;
            if (nc > lim) nc = lim;
            if (nc <= ip->cells_cap) {
                ip->race_cells_truncated = 1;
                return;
            }
            struct RCell *g = (struct RCell *)realloc(ip->cells, (size_t)nc * sizeof *g);
            if (!g) {
                ip->race_cells_truncated = 1;
                return;
            }
            ip->cells = g; ip->cells_cap = nc;
        }
        int htab_ok = race_htab_reserve(ip, ip->ncells + 1u);
        if (!htab_ok) {
            free(ip->cell_htab);
            ip->cell_htab = NULL;
            ip->cell_htab_cap = 0;
        }
        uint32_t cidx = ip->ncells;
        c = &ip->cells[ip->ncells++];
        c->root = root; c->offset = offset; c->epoch = ip->epoch;
        c->writer = -1; c->reader = -1; c->read_other = 0;
        if (htab_ok && ip->cell_htab && ip->cell_htab_cap)
            race_htab_insert(ip->cell_htab, ip->cell_htab_cap, ip->cells, cidx);
    }
    if (c->epoch != ip->epoch) {   /* a barrier separated prior accesses → no race */
        c->epoch = ip->epoch; c->writer = -1; c->reader = -1; c->read_other = 0;
    }
    int kind = -1;
    if (is_write) {
        if (c->writer >= 0 && c->writer != lane)      kind = 0;   /* WAW */
        /* WAR: any lane other than the writer read this cell (track >1 reader,
         * else a self-read would mask an earlier peer read). */
        else if ((c->reader >= 0 && c->reader != lane) || c->read_other) kind = 2;
        c->writer = lane;
    } else {
        if (c->writer >= 0 && c->writer != lane)      kind = 1;   /* RAW */
        if (c->reader < 0) c->reader = lane;
        else if (c->reader != lane) c->read_other = 1;
    }
    if (kind >= 0) {
        ip->race_total++;
        if (ip->nraces < sizeof ip->races / sizeof ip->races[0]) {
            ip->races[ip->nraces].line = line;
            ip->races[ip->nraces].kind = (uint8_t)kind;
            ip->nraces++;
        }
    }
}

/* consteval mem_access hook: buffer one lane's direct buf[idx] access for this
 * warp-step (entry warp only; not during a cooperative-op operand re-eval). */
void mem_access_cb(WGSLConstEvaluator *cev, WGSLNode *node, uint32_t index,
                          uint32_t elem_bytes, int is_store, int is_workgroup, void *ud) {
    (void)cev;
    Interp *ip = (Interp *)ud;
    if (ip->call_depth != 0 || ip->in_gather) return;
    /* Race on *loads* here (indexed rvalues via consteval).
     * Shared *stores* race through mem_store only — calling race_check on
     * stores here would double-count with the mem_store chokepoint.  Only
     * DIRECT 1-D (`buf[i]`, base is root ident) so nested/member offsets
     * do not collide distinct cells into false races. */
    if (!is_store && node && node->kind == WGSL_NODE_EXPR_INDEX &&
        node->child_count && node->children[0] &&
        node->children[0]->kind == WGSL_NODE_EXPR_IDENT)
        race_check(ip, access_root(node), index * elem_bytes, 0,
                   line_of(ip->src, node));
    if (ip->macc_n == ip->macc_cap) {
        uint32_t nc = ip->macc_cap ? ip->macc_cap * 2 : 64;
        struct MemAcc *g = (struct MemAcc *)realloc(ip->macc, (size_t)nc * sizeof *g);
        if (!g) return;
        ip->macc = g; ip->macc_cap = nc;
    }
    ip->macc[ip->macc_n].node = node;
    ip->macc[ip->macc_n].offset = index * elem_bytes;   /* contiguous array stride */
    ip->macc[ip->macc_n].elem_bytes = elem_bytes;
    ip->macc[ip->macc_n].is_wg = (uint8_t)(is_workgroup != 0);
    ip->macc_n++;
}

static void mem_process_slow(Interp *ip) {
    uint32_t n = ip->macc_n;
    for (uint32_t i = 0; i < n; i++) {
        WGSLNode *site = ip->macc[i].node;
        if (!site) continue;
        int is_wg = ip->macc[i].is_wg;
        /* first occurrence of this site drives the group; skip later dups */
        int seen = 0;
        for (uint32_t k = 0; k < i; k++) if (ip->macc[k].node == site) { seen = 1; break; }
        if (seen) continue;

        if (!is_wg) {
            /* DRAM coalescing: count distinct cache lines the group touches. */
            uint64_t theoretical = 0, lines = 0;
            for (uint32_t k = i; k < n; k++) {
                if (ip->macc[k].node != site) continue;
                theoretical += ip->macc[k].elem_bytes;
                uint32_t line = ip->macc[k].offset / INTERP_CACHE_LINE;
                int dup = 0;
                for (uint32_t m = i; m < k; m++)
                    if (ip->macc[m].node == site &&
                        ip->macc[m].offset / INTERP_CACHE_LINE == line) { dup = 1; break; }
                if (!dup) lines++;
            }
            ip->dram_theoretical += theoretical;
            ip->dram_effective   += lines * INTERP_CACHE_LINE;
        } else {
            /* Shared-memory bank conflict: max lanes landing on one bank. */
            uint32_t per_bank[INTERP_BANKS] = {0};
            for (uint32_t k = i; k < n; k++)
                if (ip->macc[k].node == site)
                    per_bank[(ip->macc[k].offset / 4u) % INTERP_BANKS]++;
            uint32_t worst = 0;
            for (uint32_t b = 0; b < INTERP_BANKS; b++)
                if (per_bank[b] > worst) worst = per_bank[b];
            ip->shared_accesses++;
            if (worst > 1) ip->bank_conflicts += (worst - 1);   /* extra serialized */
        }
    }
    ip->macc_n = 0;
}

typedef struct {
    WGSLNode *site;
    uint64_t theoretical;
    uint64_t lines;
    uint32_t per_bank[INTERP_BANKS];
    uint8_t  used;
    uint8_t  is_wg;
} MemGroup;

typedef struct {
    WGSLNode *site;
    uint32_t line;
    uint8_t  used;
} MemLine;

static uint32_t mem_hash_cap(uint32_t n) {
    uint32_t cap = 16;
    while (cap < n * 2u && cap < 0x80000000u) cap <<= 1;
    return cap;
}

static uint64_t mem_hash_mix(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27; x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x ? x : 1;
}

static uint64_t mem_site_hash(const WGSLNode *site) {
    return mem_hash_mix((uint64_t)(uintptr_t)site);
}

static uint64_t mem_line_hash(const WGSLNode *site, uint32_t line) {
    uint64_t x = (uint64_t)(uintptr_t)site;
    x ^= (uint64_t)line * 0x9e3779b97f4a7c15ull;
    return mem_hash_mix(x);
}

static MemGroup *mem_group_get(MemGroup *groups, uint32_t cap,
                               WGSLNode *site, uint8_t is_wg) {
    uint32_t mask = cap - 1u;
    uint32_t h = (uint32_t)mem_site_hash(site) & mask;
    for (;;) {
        MemGroup *g = &groups[h];
        if (!g->used) {
            g->used = 1;
            g->site = site;
            g->is_wg = is_wg;
            return g;
        }
        if (g->site == site) return g;
        h = (h + 1u) & mask;
    }
}

static int mem_line_insert(MemLine *lines, uint32_t cap,
                           WGSLNode *site, uint32_t line) {
    uint32_t mask = cap - 1u;
    uint32_t h = (uint32_t)mem_line_hash(site, line) & mask;
    for (;;) {
        MemLine *l = &lines[h];
        if (!l->used) {
            l->used = 1;
            l->site = site;
            l->line = line;
            return 1;
        }
        if (l->site == site && l->line == line) return 0;
        h = (h + 1u) & mask;
    }
}

/* At a warp-step's end: group the buffered accesses by site and score each.
 * Storage → DRAM coalescing (distinct 128B lines × 128 vs bytes requested);
 * workgroup → bank conflicts (max lanes colliding on a bank, minus one). */
void mem_process(Interp *ip) {
    uint32_t n = ip->macc_n;
    if (!n) return;

    uint32_t cap = mem_hash_cap(n);
    MemGroup *groups = (MemGroup *)calloc((size_t)cap, sizeof *groups);
    MemLine  *lines  = (MemLine  *)calloc((size_t)cap, sizeof *lines);
    if (!groups || !lines) {
        free(groups);
        free(lines);
        mem_process_slow(ip);
        return;
    }

    for (uint32_t i = 0; i < n; i++) {
        WGSLNode *site = ip->macc[i].node;
        if (!site) continue;
        MemGroup *g = mem_group_get(groups, cap, site, ip->macc[i].is_wg);
        if (g->is_wg) {
            uint32_t bank = (ip->macc[i].offset / 4u) % INTERP_BANKS;
            g->per_bank[bank]++;
        } else {
            g->theoretical += ip->macc[i].elem_bytes;
            uint32_t line = ip->macc[i].offset / INTERP_CACHE_LINE;
            if (mem_line_insert(lines, cap, site, line)) g->lines++;
        }
    }

    for (uint32_t i = 0; i < cap; i++) {
        MemGroup *g = &groups[i];
        if (!g->used) continue;
        if (g->is_wg) {
            uint32_t worst = 0;
            for (uint32_t b = 0; b < INTERP_BANKS; b++)
                if (g->per_bank[b] > worst) worst = g->per_bank[b];
            ip->shared_accesses++;
            if (worst > 1) ip->bank_conflicts += (worst - 1);
        } else {
            ip->dram_theoretical += g->theoretical;
            ip->dram_effective   += g->lines * INTERP_CACHE_LINE;
        }
    }

    free(groups);
    free(lines);
    ip->macc_n = 0;
}

void vm_run(Interp *ip, const Prog *prog,
                   const uint32_t *lane_ids, uint32_t nactive, WGSLValue *rets) {
    if (rets) memset(rets, 0, (size_t)nactive * sizeof *rets);  /* even on early-out */
    if (!prog || nactive == 0) return;
    uint32_t *pc  = (uint32_t *)malloc((size_t)nactive * sizeof *pc);
    uint32_t *grp = (uint32_t *)malloc((size_t)nactive * sizeof *grp);
    uint32_t *act = (uint32_t *)malloc((size_t)nactive * sizeof *act);
    uint8_t  *live = (uint8_t *)malloc((size_t)nactive);
    if (!pc || !grp || !act || !live) { free(pc); free(grp); free(act); free(live); return; }
    for (uint32_t i = 0; i < nactive; i++) { pc[i] = 0; live[i] = 1; }

    for (;;) {
        uint32_t minpc = LABEL_UNSET;
        for (uint32_t i = 0; i < nactive; i++)
            if (live[i] && pc[i] < minpc) minpc = pc[i];
        if (minpc == LABEL_UNSET) break;                 /* all lanes halted */
        if (++ip->steps > STEP_CAP) { ip->truncated = 1; break; }

        uint32_t ns = 0;
        for (uint32_t i = 0; i < nactive; i++)
            if (live[i] && pc[i] == minpc) { grp[ns] = i; act[ns] = lane_ids[i]; ns++; }

        const Instr *in = &prog->code[minpc];

        /* Divergence/occupancy: `ns` of the N lanes are converged here; the rest
         * are elsewhere.  Only the entry warp (call_depth == 0) is measured. */
        ip->cur_active = ns;
        if (ip->call_depth == 0) {
            ip->warp_steps++;
            ip->lane_steps += ns;
            div_line(ip, in->node ? line_of(ip->src, in->node) : 0, ns);
        }

        uint64_t cs[4] = {0, 0, 0, 0};
        switch ((OpKind)in->op) {
        case OP_STMT:
            for (uint32_t k = 0; k < ns; k++) {
                uint32_t i = grp[k], foc = (lane_ids[i] == ip->focus_lane);
                if (!foc) cost_snapshot(ip->cev, cs);
                vm_enter(ip, act, ns, lane_ids[i]);
                exec_leaf(ip, in->node);
                if (!foc) cost_rollback(ip->cev, cs);
                pc[i] = minpc + 1;
            }
            break;
        case OP_BR:
            for (uint32_t k = 0; k < ns; k++) pc[grp[k]] = in->target;
            break;
        case OP_BR_IFNOT:
        case OP_BR_IF:
            for (uint32_t k = 0; k < ns; k++) {
                uint32_t i = grp[k], foc = (lane_ids[i] == ip->focus_lane);
                if (!foc) cost_snapshot(ip->cev, cs);
                vm_enter(ip, act, ns, lane_ids[i]);
                WGSLValue c; int t = eval(ip, in->node, &c) && is_true(&c);
                if (!foc) cost_rollback(ip->cev, cs);
                pc[i] = (in->op == OP_BR_IF) ? (t ? in->target : minpc + 1)
                                             : (t ? minpc + 1 : in->target);
            }
            break;
        case OP_SWITCH:
            for (uint32_t k = 0; k < ns; k++) {
                uint32_t i = grp[k], foc = (lane_ids[i] == ip->focus_lane);
                if (!foc) cost_snapshot(ip->cev, cs);
                vm_enter(ip, act, ns, lane_ids[i]);
                pc[i] = vm_switch_target(ip, in->node, in->sw);
                if (!foc) cost_rollback(ip->cev, cs);
            }
            break;
        case OP_RET:
            for (uint32_t k = 0; k < ns; k++) {
                uint32_t i = grp[k], foc = (lane_ids[i] == ip->focus_lane);
                if (!foc) cost_snapshot(ip->cev, cs);
                vm_enter(ip, act, ns, lane_ids[i]);
                WGSLValue r; memset(&r, 0, sizeof r);
                if (in->node && in->node->child_count && eval(ip, in->node->children[0], &r))
                    record_step(ip, line_of(ip->src, in->node), "return", "return",
                                &r, anomaly_tag(ip, &r));
                if (!foc) cost_rollback(ip->cev, cs);
                if (rets) rets[i] = r;
                live[i] = 0;
            }
            break;
        case OP_HALT:
        default:
            for (uint32_t k = 0; k < ns; k++) live[grp[k]] = 0;
            break;
        }
        /* Score this warp-step's buffered memory accesses (entry warp only). */
        if (ip->call_depth == 0 && ip->macc_n) mem_process(ip);
        /* A barrier just executed → advance the race epoch (happens-before). */
        if (ip->call_depth == 0 && ip->barrier_pending) { ip->epoch++; ip->barrier_pending = 0; }
    }
    free(pc); free(grp); free(act); free(live);
}

/* Run a callee body once for the calling lane (`cev->simt.cur_lane`), binding its
 * parameters from the evaluated arguments.  Used for user-function calls and,
 * with a single lane, kept as the entry into a compiled body. */
int exec_fn(Interp *ip, WGSLNode *fn, WGSLValue *args, uint32_t nargs, WGSLValue *out) {
    uint32_t pi = 0;
    for (uint32_t c = 0; c < fn->child_count && pi < nargs; c++) {
        if (fn->children[c]->kind == WGSL_NODE_DECL_PARAM) {
            WGSLSymbol *ps = decl_symbol(ip->cev, fn->children[c]);
            if (ps) {
                if (!args[pi].type) args[pi].type = ps->type;
                wgsl_consteval_set_value(ip->cev, ps, &args[pi]);
            }
            pi++;
        }
    }
    const Prog *prog = get_prog(ip, fn_body(fn));
    WGSLValue r; memset(&r, 0, sizeof r);
    uint32_t lane = ip->cev->simt.cur_lane;
    /* The nested run publishes (and, at its end, frees) its own active-group
     * array via ip->act.  Save/restore ours so a cooperative op later in the
     * *calling* statement doesn't dereference the freed nested array. */
    const uint32_t *save_act = ip->act; uint32_t save_nact = ip->nact;
    uint32_t save_active = ip->cur_active;
    ip->call_depth++;
    vm_run(ip, prog, &lane, 1, &r);
    ip->call_depth--;
    ip->act = save_act; ip->nact = save_nact; ip->cur_active = save_active;
    ip->cev->simt.cur_lane = lane;   /* nested run left cur_lane on its last lane */
    *out = r;
    return prog != NULL;
}

/* consteval user-function-call hook → run the callee via the interpreter. */
int user_call_cb(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out, void *ud) {
    (void)cev;
    Interp *ip = (Interp *)ud;
    WGSLNode *callee = n->children[0];
    if (callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT && callee->child_count) callee = callee->children[0];
    WGSLSymbol *fs = wgsl_node_resolved_symbol(callee);
    if (!fs || !fs->ast || fs->ast->kind != WGSL_NODE_DECL_FUNCTION) return 0;
    WGSLValue args[16]; uint32_t na = 0;
    for (uint32_t i = 1; i < n->child_count && na < 16; i++) {
        if (!eval(ip, n->children[i], &args[na])) return 0;
        na++;
    }
    return exec_fn(ip, fs->ast, args, na, out);
}
