/**
 * @file consteval.h — WGSL const-expression evaluator.
 *
 * Evaluates expressions that WGSL requires to be known at shader-creation
 * time:
 *
 *   - `const NAME = expr;`       (§7.2)  must const-eval
 *   - `override NAME = expr?;`   (§7.2)  default value is override-expr
 *   - `const_assert <expr>;`     (§10.1) must eval to literal `true`
 *   - template-arg const-expr   (e.g. `array<f32, N>`)
 *   - abstract→concrete materialization (§6.2.1)
 *   - scalar + vector FP builtin-call domains for §15.7.7, including
 *     fold-through for common @const numeric builtins (`abs`, `min`,
 *     `max`, `mix`, `ldexp`, `length`, `normalize`, etc.)
 *
 * §11.3 pure user-function const-evaluation (memoized) folds helpers whose
 * bodies are only `const`/`let` + `return` (and simple `if`) of
 * const-expressions.  WGSL forbids `@const` on user functions; this is the
 * pure-function subset evaluable at shader-creation time.
 *
 * Runtime-only operations are reported as non-constant instead of being
 * folded.  The evaluator runs after the resolver.  It uses every
 * `EXPR_IDENT`'s resolved `WGSLSymbol *` (stashed in `payload[2]`) to
 * look up the bound value of a const symbol.
 *
 * Successful evaluation of a `DECL_CONST` / `DECL_OVERRIDE` populates
 * a side-table keyed by `WGSLSymbol *` so later uses (in template args,
 * other consts, `const_assert`) can fetch the value back.  Untyped
 * `const` values keep their abstract numeric type; materialization is
 * performed only for typed consts here, or later by the consuming context.
 */
#ifndef WGSL_INTERNAL_CONSTEVAL_H
#define WGSL_INTERNAL_CONSTEVAL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "internal/arena.h"
#include "internal/ast.h"
#include "internal/token.h"
#include "internal/diag.h"
#include "internal/resolver.h"
#include "internal/source.h"
#include "internal/types.h"

/** Tagged value union.  All instances are arena-allocated by the
 *  evaluator; callers do not free.  Aggregates point at flat element
 *  arrays in the same arena. */
typedef enum {
    WGSL_VAL_INVALID = 0,
    WGSL_VAL_BOOL,
    WGSL_VAL_INT,        /* AbstractInt / i32 / u32 — interpret with `type` */
    WGSL_VAL_FLOAT,      /* AbstractFloat / f32 / f16 — interpret with `type` */
    WGSL_VAL_VEC,        /* `agg.count` elems = type width                  */
    WGSL_VAL_MAT,        /* `agg.count` elems = cols * rows (column-major)  */
    WGSL_VAL_ARRAY,      /* `agg.count` elems = array length                */
    WGSL_VAL_STRUCT,     /* `agg.count` elems = struct member count         */
    /* First-class pointer/lvalue: address = root + path + baked lane.
     * The compact path stores either the whole root or one 1-D array/vec
     * index.  Struct/member paths fall back to a coarse whole-root race cell. */
    WGSL_VAL_PTR,
} WGSLValKind;

typedef struct WGSLValue {
    uint16_t       kind;          /* WGSLValKind                          */
    uint16_t       flags_;        /* reserved                             */
    uint32_t       pad_;
    WGSLTypeInfo  *type;          /* tagged type (Abstract* or concrete)  */
    union {
        bool     b;
        int64_t  i;
        double   f;
        struct {
            uint32_t          count;
            struct WGSLValue *elems;     /* arena-owned                   */
        } agg;
        /* Pointer / lvalue address.  Copied by value (POD); no
         * arena ownership.  `lane` is baked at `&` (cur_lane); function/
         * private deref asserts it still matches cur_lane. */
        struct {
            const WGSLSymbol *root;       /* originating variable         */
            uint32_t          lane;       /* baked at address-of          */
            int32_t           index;      /* path[0] when has_index       */
            uint8_t           has_index;  /* 0 empty path, 1 = 1-D index  */
            uint8_t           race_coarse;/* 1 → whole-root race cell     */
            uint8_t           pad_ptr[2];
        } ptr;
    } u;
} WGSLValue;

/* Side-table entry: WGSLSymbol → cached evaluated value.
 *
 * The single-invocation const path and the N-lane interpreter share this
 * table.  A *shared* binding (module const/override/function, and any
 * `var<storage/uniform/workgroup>`) holds one value that every lane aliases —
 * this is how a workgroup reduction or a storage buffer is visible across
 * lanes.  A *per-lane* binding (function locals/params and `var<private>`,
 * which the SIMT model gives each invocation its own copy of) holds one value
 * *per lane*, indexed by `cev->simt.cur_lane`.  Classification is by address space
 * (see `wgsl_consteval_sym_is_per_lane`); it only takes effect once the
 * interpreter sets `nlanes >= 1`, so the plain const-eval path (nlanes == 0)
 * keeps every binding shared exactly as before. */
typedef struct {
    const WGSLSymbol *sym;
    int               per_lane;    /* 0 shared → `value`; 1 → `lane_values` */
    WGSLValue        *value;       /* arena-owned (shared binding)          */
    WGSLValue       **lane_values; /* heap array [nlanes] (per-lane binding) */
} WGSLConstBinding;

/* Memo entry for §11.3 pure user-function calls: (fn, args) → result. */
typedef struct WGSLConstFnMemo {
    const WGSLNode *fn;
    uint64_t        hash;
    uint32_t        nargs;
    WGSLValue      *args;       /* arena-owned deep copies              */
    WGSLValue       result;     /* arena-owned on success               */
    int             ok;         /* 0 = cached failure                   */
} WGSLConstFnMemo;

struct WGSLConstEvaluator;

typedef struct WGSLConstStore {
    WGSLConstBinding  *bindings;     /* heap-grown, freed in destroy      */
    size_t             binding_count;
    size_t             binding_capacity;
    size_t            *binding_htab;  /* WGSLSymbol* → bindings index      */
    size_t             binding_htab_cap;

    /* §11.3 pure user-fn call memo + recursion depth (shader-creation). */
    WGSLConstFnMemo   *fn_memo;      /* heap-grown, freed in destroy      */
    size_t             fn_memo_count;
    size_t             fn_memo_cap;
    size_t            *fn_memo_htab; /* (fn,args hash) → fn_memo index    */
    size_t             fn_memo_htab_cap;
    int                fn_call_depth;

    int                had_error;
    /* Typed failure class for the most recent eval_expr / cev_error
     * cascade.  Soft-eval (type checker) branches on this enum, never on
     * diagnostic message text.  Escalates to VALUE_ERROR if any such
     * error is noted during the evaluation. */
    int                last_status;    /* WGSLCevStatus */
    int                eval_depth;     /* current eval_expr recursion depth */
} WGSLConstStore;

typedef struct WGSLInterpEnv {
    /* N-lane SIMT interpreter (0 for the plain const-eval path).  `nlanes` is
     * the workgroup lane count; `cur_lane` selects which lane's per-lane
     * bindings `value_of`/`set_value` read and write.  Shared bindings ignore
     * both.  See WGSLConstBinding. */
    uint32_t           nlanes;
    uint32_t           cur_lane;

    /* Interpreter hook (NULL for plain const-eval): when set, an EXPR_CALL to
     * a user function (WGSL_SYM_FUNCTION) is dispatched here instead of
     * erroring — the single-invocation interpreter runs the callee's body. */
    int              (*user_call)(struct WGSLConstEvaluator *cev,
                                  WGSLNode *call, WGSLValue *out, void *ud);
    void              *user_call_ud;

    /* Interpreter hook for predeclared builtins the const-folder can't handle
     * because they need a mutable memory location (`&ptr`): atomics,
     * arrayLength, workgroupUniformLoad.  Returns >0 handled ok, <0 handled
     * with an emitted error, 0 not handled (fall through). */
    int              (*builtin_call)(struct WGSLConstEvaluator *cev,
                                     WGSLNode *call, const char *name,
                                     uint32_t name_len, WGSLValue *out, void *ud);
    void              *builtin_call_ud;

    /* Interpreter hook fired on a direct indexed access to a storage/uniform or
     * workgroup array (`buf[idx]`), for warp-level memory analysis (coalescing,
     * bank conflicts).  `elem_bytes` is the leaf stride; `is_store` distinguishes
     * write from read; `is_workgroup` selects shared-mem (banked) vs DRAM. */
    void             (*mem_access)(struct WGSLConstEvaluator *cev, WGSLNode *node,
                                   uint32_t index, uint32_t elem_bytes,
                                   int is_store, int is_workgroup, void *ud);
    void              *mem_access_ud;

    /* Runtime IEEE mode (0 for plain const-eval): when the single-invocation
     * interpreter drives evaluation it wants *runtime* arithmetic semantics,
     * not the stricter shader-creation rules.  Under §8.7 a float divide by
     * zero yields ±Inf/NaN and an integer divide by zero yields the dividend
     * — both are well-defined at runtime, whereas in a const-expression they
     * are shader-creation errors.  Gated so the const path is unaffected. */
    int                runtime;
} WGSLInterpEnv;

typedef struct WGSLCostProfile {
    /* Static cost model (interpreter mode).  Accumulated while `runtime` is
     * set: `flop_count` bumps on every scalar float arithmetic op (vector ops
     * fold through per-component, so they count as `width`); `bytes_loaded`
     * bumps when a value flagged WGSL_VAL_FLAG_STORAGE is indexed/member-read
     * down to a scalar/vector leaf.  The single-invocation counts feed an
     * arithmetic-intensity (FLOP/byte) roofline verdict — scale-invariant, so
     * one thread answers "memory-bound vs compute-bound" for the whole grid. */
    uint64_t           flop_count;
    /* Integer/bitwise ALU ops: bumps once per scalar integer binary op
     * (+ - * / % & | ^ << >> and the comparisons), vectors folding through
     * per-component like `flop_count`.  Integer-heavy kernels (token/BPE,
     * indexing, bit-twiddling) do real ALU work that carries ZERO FLOPs, so
     * this is the intensity axis that makes their roofline verdict meaningful. */
    uint64_t           iop_count;
    uint64_t           bytes_loaded;
    uint64_t           bytes_stored;

    /* Per-op instruction-mix breakdown (occurrence counts, vectors fold per
     * component like flop_count).  These refine flop_count/iop_count into
     * buckets so a roofline card can show the mix, not one scalar.  Indexed by
     * WGSLOpBucket. */
    uint64_t           op_bucket[9];

    /* f16 / packed-math awareness.  `flop_f16` is the subset of
     * flop_count done in f16 — GPUs run these at ~2× f32 throughput, so the
     * roofline discounts them.  `packed_ops` counts tensor-path-eligible packed
     * instructions (dot4{I,U}8Packed): one instruction doing several MACs. */
    uint64_t           flop_f16;
    uint64_t           packed_ops;
} WGSLCostProfile;

typedef struct WGSLConstEvaluator {
    WGSLArena         *arena;        /* used for values + bindings table  */
    WGSLDiagBag       *diag;
    WGSLTypeStore     *types;
    const WGSLSource  *src;
    WGSLResolver      *res;          /* for EXPR_IDENT → symbol lookups   */

    WGSLConstStore     store;        /* shader-creation const bindings    */
    WGSLInterpEnv      simt;         /* N-lane interpreter execution env  */
    WGSLCostProfile    cost;         /* roofline/profiler counters        */
} WGSLConstEvaluator;

/* Instruction-mix buckets for op_bucket[].  Float: add/mul/div/transcendental;
 * integer: add-or-logic/mul/div/shift; and comparisons (either domain). */
typedef enum {
    WGSL_OPB_FADD, WGSL_OPB_FMUL, WGSL_OPB_FDIV, WGSL_OPB_FTRANS,
    WGSL_OPB_IADD, WGSL_OPB_IMUL, WGSL_OPB_IDIV, WGSL_OPB_ISHIFT,
    WGSL_OPB_CMP,
    WGSL_OPB_COUNT
} WGSLOpBucket;

/**
 * Typed const-eval outcome for soft-eval / noise filtering.
 *
 * Replaces diagnostic-message substring matching ("not a constant", …).
 * Soft-eval rolls back NOT_CONST / NOT_IMPLEMENTED; VALUE_ERROR surfaces.
 */
typedef enum {
    WGSL_CEV_OK = 0,             /**< last eval succeeded (or not yet failed) */
    WGSL_CEV_NOT_CONST,          /**< runtime / non-const expression (noise)  */
    WGSL_CEV_NOT_IMPLEMENTED,    /**< fold path unavailable here (noise)      */
    WGSL_CEV_VALUE_ERROR         /**< real value error: div0, overflow, …     */
} WGSLCevStatus;

/** True when `st` is soft-eval noise (rollback diags; do not surface). */
static inline int wgsl_cev_status_is_noise(int st) {
    return st == WGSL_CEV_NOT_CONST || st == WGSL_CEV_NOT_IMPLEMENTED;
}

/* WGSLValue.flags_ bit: this value aliases a storage/uniform buffer, so
 * indexing/member-reading it is real DRAM traffic for the cost model. */
#define WGSL_VAL_FLAG_STORAGE   (1u << 0)
/* This value aliases workgroup shared memory (banked, not DRAM). */
#define WGSL_VAL_FLAG_WORKGROUP (1u << 1)

/* Initialization / destruction. */

/**
 * Initialise the evaluator state (no walking).  After this, individual
 * expressions can be evaluated with `wgsl_consteval_expr`, or the full
 * module can be processed with `wgsl_consteval`.
 *
 * `res` may be NULL — in that case, identifier lookups fail and only
 * literal-arithmetic expressions can be evaluated.
 */
void wgsl_consteval_init(
    WGSLConstEvaluator *cev,
    WGSLArena          *arena,
    WGSLDiagBag        *diag,
    WGSLTypeStore      *types,
    const WGSLSource   *src,
    WGSLResolver       *res);

/** Release the bindings table.  Values themselves live in the arena. */
void wgsl_consteval_destroy(WGSLConstEvaluator *cev);

/* Single-expression evaluation. */

/**
 * Evaluate one expression node as a const-expression.
 *
 * On success returns 1, populates `*out` with `kind != INVALID`.
 * On failure returns 0, emits one or more diagnostics on the bag, and
 * leaves `*out` zero-initialized.
 *
 * Side effects: NONE on the AST itself.
 */
int wgsl_consteval_expr(
    WGSLConstEvaluator *cev,
    WGSLNode           *expr,
    WGSLValue          *out);

/**
 * Combine two already-evaluated values under a binary operator `op`, reusing
 * the full binary-op machinery (numeric promotion, vector/matrix component
 * ops, integer shift/bitwise rules, signedness).  `ctx` is used only for
 * diagnostics; the inputs are not mutated.
 *
 * Used by the interpreter to implement compound assignment (`a += b`,
 * `x >>= 1u`) and increment/decrement, which the const-expression grammar
 * itself never produces.
 *
 * On success returns 1 and populates `*out`; on failure returns 0 and emits a
 * diagnostic at `ctx`.
 */
int wgsl_consteval_binop(
    WGSLConstEvaluator *cev,
    WGSLNode           *ctx,
    WGSLTokenKind       op,
    const WGSLValue    *a,
    const WGSLValue    *b,
    WGSLValue          *out);

/* Materialisation (§6.2.1). */

/**
 * Materialise abstract numeric values to concrete types per §6.2.1:
 *   AbstractInt    → i32 (default) or u32 / f32 / f16 if `target` requires it
 *   AbstractFloat  → f32 (default) or f16 if `target` requires it
 *
 * `target` may be NULL — that means "default concretisation": AbstractInt → i32,
 * AbstractFloat → f32.  Already-concrete values are returned unchanged
 * if the conversion-rank table allows; otherwise an error diagnostic is
 * emitted at `at_node`.
 *
 * Returns 1 on success (possibly with `*v` rewritten in place); 0 on
 * failure.
 */
int wgsl_consteval_materialize(
    WGSLConstEvaluator *cev,
    WGSLValue          *v,
    WGSLTypeInfo       *target,        /* NULL = default concrete tower */
    const WGSLNode     *at_node);

/* Module-level pass. */

/**
 * Walk the translation unit, evaluating every `DECL_CONST`,
 * `DECL_OVERRIDE`, `DECL_CONST_ASSERT` (module + fn-scope).  Caches
 * resolved values on the side-table.  Emits diagnostics for non-const
 * expressions, divide-by-zero, overflow at materialise, …
 *
 * @return 1 if no error-severity diagnostics emitted; 0 otherwise.
 */
int wgsl_consteval(
    WGSLAst            *ast,
    const WGSLSource   *src,
    WGSLArena          *arena,
    WGSLDiagBag        *diag,
    WGSLTypeStore      *types,
    WGSLResolver       *res,
    WGSLConstEvaluator *out);

/** Symbol→value binding lookup.  Returns NULL if `sym` has no value.
 *  In N-lane mode a per-lane symbol resolves against `cev->simt.cur_lane`. */
const WGSLValue *wgsl_consteval_value_of(
    const WGSLConstEvaluator *cev, const WGSLSymbol *sym);

/**
 * Classify a symbol as per-lane (each SIMT invocation owns a private copy) or
 * shared (one copy the whole workgroup/grid aliases).  Per-lane: function
 * locals/`let`, parameters, and `var<private>`.  Shared: everything else —
 * `var<storage/uniform/workgroup>`, module `const`/`override`, functions.
 */
int wgsl_consteval_sym_is_per_lane(const WGSLSymbol *sym);

/**
 * Insert-or-OVERWRITE a symbol's bound value (mutable environment).
 *
 * Unlike the const path (which treats a bound symbol as immutable), this
 * re-binds an existing symbol — the single-invocation interpreter uses it
 * for `var`/`let` declarations, assignments, and parameter binding.  The
 * value is copied into the evaluator arena.  Returns 1 on success.
 */
int wgsl_consteval_set_value(
    WGSLConstEvaluator *cev, const WGSLSymbol *sym, const WGSLValue *v);

/* Runtime pointer ops (implemented in interp/exec_leaf.c).  Used by the
 * const-evaluator when `cev->simt.runtime` so nested `*p` / `&x` inside binary
 * ops share one evaluation path with the interpreter.  Const-eval mode
 * (runtime==0) never calls these. */
int wgsl_runtime_eval_addr_of(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out);
int wgsl_runtime_eval_indirection(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out);

#endif /* WGSL_INTERNAL_CONSTEVAL_H */
