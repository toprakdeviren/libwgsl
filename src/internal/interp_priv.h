/**
 * @file interp_priv.h — multi-TU shared surface for the interpreter.
 *
 * Monolith split into src/interp/ TUs (helpers, exec_leaf, compile, vm,
 * coop_builtins, run) plus interp_band.c. Shared types and cross-TU helpers
 * live here; band analyzer types/API stay below the SIMT core.
 */
#ifndef WGSL_INTERNAL_INTERP_PRIV_H
#define WGSL_INTERNAL_INTERP_PRIV_H

#include "wgsl.h"

#include "internal/arena.h"
#include "internal/ast.h"
#include "internal/check.h"
#include "internal/consteval.h"
#include "internal/diag.h"
#include "internal/lexer.h"
#include "internal/parser.h"
#include "internal/resolver.h"
#include "internal/source.h"
#include "internal/token.h"
#include "internal/types.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Caps live in the public header so bindings/docs cannot drift. */
#define STEP_CAP ((long)WGSL_INTERP_STEP_CAP)

/* One workgroup is simulated as N SIMT lanes; N is capped here.  Lanes are
 * partitioned into subgroups of this width for subgroup/quad ops (quad = 4). */
#define INTERP_MAX_LANES       WGSL_INTERP_MAX_LANES
#define INTERP_SUBGROUP_WIDTH  32u

#define LABEL_UNSET 0xFFFFFFFFu

/* growable string buffer (JSON assembly). */

typedef struct { char *buf; size_t len, cap; } SB;

/* interpreter state. */

typedef struct {
    WGSLConstEvaluator     *cev;
    const WGSLTypeChecker  *tc;      /* for struct member types on seeding */
    const WGSLSource       *src;
    uint32_t                default_len; /* runtime-array default length   */
    long                    steps;
    int                     truncated;
    SB                      trace;   /* JSON array body of step objects    */
    int                     nsteps;  /* comma bookkeeping for `trace`      */
    long                    nan_hits;
    long                    inf_hits;
    SB                      scratch; /* reused value-formatting buffer     */

    /* N-lane SIMT execution.  `nlanes` is the workgroup lane count; only the
     * `focus_lane`'s statements are recorded into `trace` (all N would swamp
     * it — the per-lane view is a later increment).  `recording` caches
     * whether the lane currently bound in the evaluator is the focus lane.
     * `progs` is the compiled-bytecode cache (opaque `ProgCache *`). */
    uint32_t                nlanes;
    uint32_t                focus_lane;
    int                     recording;
    void                   *progs;

    /* Active lane group at the instruction currently executing — the lanes the
     * min-PC scheduler has converged at this pc.  Cooperative ops (subgroup and
     * quad) reduce/shuffle over exactly this set (intersected with the caller's
     * subgroup).  Lane ids, ascending (== subgroup-invocation order). */
    const uint32_t         *act;
    uint32_t                nact;

    /* User-call nesting depth.  A cooperative op or barrier reached while
     * call_depth > 0 is inside a helper we run one lane at a time, so it can't
     * model the workgroup — flagged (once) instead of silently mis-reducing. */
    int                     call_depth;
    int                     helper_coop_warned;

    /* divergence / occupancy. */
    uint32_t                cur_active;     /* active lanes at the current step  */
    uint64_t                warp_steps;     /* entry-warp scheduler steps        */
    uint64_t                lane_steps;     /* Σ cur_active over those steps      */

    /* warp memory analysis. */
    struct MemAcc { WGSLNode *node; uint32_t offset, elem_bytes; uint8_t is_wg; }
                           *macc;
    uint32_t                macc_n, macc_cap;
    int                     in_gather;
    uint64_t                dram_theoretical, dram_effective;  /* bytes         */
    uint64_t                shared_accesses, bank_conflicts;

    /* differential race linter. */
    uint32_t                epoch;
    int                     barrier_pending;
    struct RCell { const WGSLSymbol *root; uint32_t offset, epoch; int32_t writer, reader; uint8_t read_other; }
                           *cells;
    uint32_t                ncells, cells_cap;
    uint32_t               *cell_htab;     /* open-address index+1 by (root,offset) */
    uint32_t                cell_htab_cap; /* power of two; 0 = disabled/oom       */
    uint32_t                race_cells_cap;       /* default 1<<16; overridable */
    int                     race_cells_truncated; /* hit race_cells_cap        */
    struct Race { uint32_t line; uint8_t kind; }   /* 0=WAW 1=RAW 2=WAR */
                            races[64];
    uint32_t                nraces;
    uint64_t                race_total;
    struct LineDiv {
        uint32_t steps;                     /* times a stmt on this line ran     */
        uint32_t min_active, max_active;
        uint64_t active_sum;
    }                      *linediv;        /* indexed by 1-based source line    */
    uint32_t                linediv_cap;
    uint32_t                max_line;       /* highest line touched              */
} Interp;

/* Test hook: if non-zero, overrides the race-cell tracking cap (default
 * 65536).  Production code leaves this 0.  Not a public API. */
extern uint32_t wgsl_test_race_cells_cap;

/* bytecode ISA. */

typedef enum {
    OP_STMT,       /* run leaf action on `node`, fall through            */
    OP_BR,         /* pc = target                                        */
    OP_BR_IFNOT,   /* if !cond(node): pc = target, else fall through     */
    OP_BR_IF,      /* if cond(node): pc = target, else fall through (break-if) */
    OP_SWITCH,     /* multiway on node's selector via `sw`               */
    OP_RET,        /* eval node's return expr (if any), halt the lane    */
    OP_HALT,       /* end of body — halt the lane                        */
} OpKind;

typedef struct { WGSLNode *clause; uint32_t target; } SwCase;
typedef struct { SwCase *cases; uint32_t ncases; uint32_t exit; } SwTable;

typedef struct {
    uint16_t  op;
    uint32_t  target;     /* branch destination (label id pre-resolve → pc) */
    WGSLNode *node;       /* leaf stmt / cond / selector / return           */
    SwTable  *sw;         /* OP_SWITCH only                                 */
} Instr;

typedef struct {
    Instr   *code;
    uint32_t ncode, capcode;
} Prog;

/* helpers.c. */

int sb_reserve(SB *s, size_t extra);
void sb_putc(SB *s, char c);
void sb_write(SB *s, const char *p, size_t n);
void sb_puts(SB *s, const char *p);
void sb_u(SB *s, unsigned long v);
void sb_dbl(SB *s, const char *fmt, double v);
void sb_json(SB *s, const char *p, size_t n);

uint32_t line_of(const WGSLSource *src, const WGSLNode *n);
void span_text(const WGSLSource *src, const WGSLNode *n, char *buf, size_t cap);
void name_text(const WGSLSource *src, const WGSLNode *n, char *buf, size_t cap);
void fmt_value(SB *s, const WGSLValue *v);
int is_true(const WGSLValue *v);
int swizzle_index_h(char c);
void record_step(Interp *ip, uint32_t line, const char *op,
                 const char *name, const WGSLValue *val, const char *anomaly);
const char *anomaly_tag(Interp *ip, const WGSLValue *v);

/* exec_leaf.c. */

WGSLSymbol *decl_symbol(WGSLConstEvaluator *cev, const WGSLNode *decl);
uint32_t interp_leaf_bytes(const WGSLTypeInfo *t);
int eval(Interp *ip, WGSLNode *e, WGSLValue *out);
/* First-class pointer/lvalue memory chokepoint. */
int ptr_from_lvalue(Interp *ip, WGSLNode *lhs, WGSLValue *out_ptr);
WGSLValue *mem_slot(Interp *ip, const WGSLValue *ptr);
int mem_load(Interp *ip, const WGSLValue *ptr, WGSLValue *out, uint32_t line);
int mem_store(Interp *ip, const WGSLValue *ptr, const WGSLValue *v, uint32_t line);
/* Resolve AST lvalue to a mutable slot for member/swizzle writes outside the
 * compact pointer path.  Also fills *out_ptr for race metadata. */
WGSLValue *lvalue_resolve(Interp *ip, WGSLNode *lhs, WGSLValue *out_ptr);
WGSLSymbol *lvalue_root_symbol(WGSLNode *lhs);
WGSLNode *fn_body(WGSLNode *fn);
void store_lvalue(Interp *ip, WGSLNode *lhs, WGSLValue *v);
WGSLTokenKind base_binop_of_compound(WGSLTokenKind t);
void set_lane(Interp *ip, uint32_t laneid);
void exec_leaf(Interp *ip, WGSLNode *n);
WGSLValue zero_value(WGSLConstEvaluator *cev, const WGSLTypeChecker *tc,
                     WGSLTypeInfo *t, uint32_t default_len);

/* Runtime hooks called from consteval when cev->simt.runtime (ADDR_OF / *p). */
int wgsl_runtime_eval_addr_of(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out);
int wgsl_runtime_eval_indirection(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out);

/* compile.c. */

const Prog *get_prog(Interp *ip, WGSLNode *body);
void progs_free(Interp *ip);

/* vm.c. */

void race_check(Interp *ip, const WGSLSymbol *root, uint32_t offset,
                int is_write, uint32_t line);
void mem_process(Interp *ip);
void mem_access_cb(WGSLConstEvaluator *cev, WGSLNode *node, uint32_t index,
                   uint32_t elem_bytes, int is_store, int is_workgroup, void *ud);
void vm_run(Interp *ip, const Prog *prog,
            const uint32_t *lane_ids, uint32_t nactive, WGSLValue *rets);
int exec_fn(Interp *ip, WGSLNode *fn, WGSLValue *args, uint32_t nargs, WGSLValue *out);
int user_call_cb(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out, void *ud);

/* coop_builtins.c. */

int builtin_call_cb(WGSLConstEvaluator *cev, WGSLNode *n,
                    const char *nm, uint32_t len, WGSLValue *out, void *ud);

/* run.c (public entry also in wgsl.h). */

char *wgsl_interp(const char *src, const char *entry,
                  unsigned gx, unsigned gy, unsigned gz,
                  unsigned default_len, const char *seeds);
void wgsl_free_string(char *s);

/* band analyzer (interp_band.c). */

#define BAND_LOOP_CAP  4096.0   /* assumed trips for an unbounded loop         */
#define BAND_GRID      65536.0  /* assumed grid extent for gid-derived values  */

typedef struct { double lo, hi; int top; } Ival;
typedef struct { double flops, iops, bytes_ld, bytes_st; } CV;
typedef struct { CV min, max; } Band;
typedef struct { const WGSLSymbol *sym; Ival iv; } IBind;
typedef struct { IBind *v; uint32_t n, cap; } IEnv;
typedef struct {
    const WGSLTypeChecker *tc;
    const WGSLSource      *src;
    WGSLConstEvaluator    *cev;    /* decl_symbol + const/override values     */
    struct { uint32_t line; double tmin, tmax; int bounded; } loops[64];
    int   nloops;
    int   ddbranches;              /* data-dependent branches encountered     */
    int   imprecise;              /* hit an unbounded/unanalyzable construct  */
} BandA;

Ival iv_top(void);
Ival iv_pt(double x);
Ival iv_rng(double a, double b);
Ival iv_add(Ival a, Ival b);
Ival iv_sub(Ival a, Ival b);
Ival iv_mul(Ival a, Ival b);
Ival iv_join(Ival a, Ival b);
CV cv0(void);
CV cv_add(CV a, CV b);
CV cv_scale(CV a, double k);
double cv_total(CV a);
Band band_of(CV c);
Band band_add(Band a, Band b);
void env_set(IEnv *e, const WGSLSymbol *s, Ival iv);
Ival env_get(const IEnv *e, const WGSLSymbol *s);
void env_copy(IEnv *dst, const IEnv *src);
void env_join(IEnv *dst, const IEnv *other);
int band_is_float(const BandA *A, WGSLNode *n);
uint32_t band_width(const BandA *A, WGSLNode *n);
uint32_t band_flop_weight(const char *nm, uint32_t len);
WGSLAddressSpace band_root_as(WGSLNode *n);
Ival band_ival(BandA *A, WGSLNode *n, const IEnv *env);
int band_cond(BandA *A, WGSLNode *cond, const IEnv *env);
CV band_count_ops(BandA *A, WGSLNode *n);
CV band_store_cost(BandA *A, WGSLNode *lhs);
Band band_block(BandA *A, WGSLNode *b, IEnv *env);
void band_trip(BandA *A, WGSLNode *cond, WGSLNode *update, const IEnv *env, WGSLSymbol *ivar, Ival i0, double *tmin, double *tmax, int *bounded);
void band_note_loop(BandA *A, uint32_t line, double tmin, double tmax, int bounded);
Band band_stmt(BandA *A, WGSLNode *n, IEnv *env);

#endif
