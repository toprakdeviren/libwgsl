# DESIGN — wgsl frontend, rev 1

> **Phase 0d deliverable.** The architecture of record, frozen at v1.
> Pairs with [`PLAN.md`](PLAN.md) (phases), [`SPEC-MAP.md`](SPEC-MAP.md)
> (per-§ ownership), [`REF-REVIEW.md`](REF-REVIEW.md) (Tint/Naga
> structural review), and [`include/wgsl.h`](../include/wgsl.h) v1.
>
> Breaking changes to anything in this document require a `DESIGN.md`
> rev bump and a corresponding bump on `wgsl.h`'s
> `WGSL_VERSION_MAJOR`.

## 1. Purpose

`wgsl` is a C-native WGSL frontend with three product surfaces:

1. `libwgsl.a` / `wgsl.h` — a frontend library that parses, resolves,
   type-checks, and validates WGSL source against the W3C WGSL
   Recommendation. Output: diagnostics, semantic tokens, module summary.
2. **LSP integration** — hover, go-to-def, semantic tokens, and
   diagnostics. The library is the engine; the LSP server is a thin
   wrapper.
3. **`wgsl.run`** — a browser sandbox: monaco editor (WGSL mode) +
   WebGPU live preview + share-via-URL. Built from a WASM build of the
   library + a minimal JS shell.

Spec target: **W3C WGSL CRD-2026-05-07** (pinned in
`WGSL_SPEC_PIN`). The working reference is the Turkish translation
in [`docs/wgsl/`](wgsl/), 7 modular files derived from the W3C
snapshot of that date.

## 2. The contract — `wgsl.h` v1

The public API is the canonical contract. It surfaces six sections:

| § | Purpose |
|---|---------|
| 1. Process lifecycle | `wgsl_init` / `wgsl_shutdown` — Unicode-table init |
| 2. Check | `wgsl_check`, `wgsl_check_n` — run the full frontend |
| 3. Result | `wgsl_ok`, `wgsl_error`, `wgsl_module_json`, `wgsl_free` |
| 4. Diagnostics | `WGSLDiagnostic` (LSP-compatible spans), `wgsl_diagnostic*` |
| 5. Semantic tokens | `wgsl_lex`, `wgsl_semantic_tokens`, `WGSLLexTokenKind` |
| 6. Hover & navigation | `wgsl_hover_at`, `wgsl_definition_at` |

The header is the only thing consumers include.  Internal headers live
under `src/internal/` and are not redistributed.

### Design principles (also in `wgsl.h`'s preamble)

- **One-shot first.** `wgsl_check` is the main entry; everything else
  inspects the result.
- **Always return non-NULL.** Failure is reported on the result, not
  via NULL or `errno`.
- **Owned-output strings.** The library allocates; the caller frees
  the result handle.
- **Structured diagnostics with LSP spans.** Same shape as `metal/` —
  1-based line/col + end-line/end-col + severity + message + rule.
- **Loud rejects over silent fallbacks.** Every spec violation is a
  diagnostic; nothing is silently accepted.
- **No hidden global state.** The single exception is the Unicode
  table init in `wgsl_init`, which is idempotent and thread-safe.

## 3. Pipeline

```
WGSL source
  │
  ▼
┌────────┐  ┌────────┐  ┌──────────┐  ┌─────────┐  ┌───────────┐  ┌───────────┐
│ Lexer  │→ │ Parser │→ │ Resolver │→ │ ConstEv │→ │ TypeCheck │→ │ Validator │
│ token  │  │  AST   │  │  scope   │  │ values  │  │ TypeInfo  │  │  verdict  │
└────────┘  └────────┘  └──────────┘  └─────────┘  └───────────┘  └───────────┘
                              ▲             ▲             ▲              │
                              └─────────────┴─────────────┴──────────────┤
                                       decorate the AST                  │
                                                                          ▼
                                                                  ┌──────────────┐
                                                                  │ Module JSON  │
                                                                  │ Diagnostics  │
                                                                  │ SemTokens    │
                                                                  │ Hover / Def  │
                                                                  └──────────────┘
```

There is **no separate IR**. Following Tint's "decorated AST" model
(see `REF-REVIEW.md`), Resolver, ConstEv, TypeCheck, and Validator
attach `sem::*`-style information directly to AST nodes via parallel
arenas. Naga's "AST → IR lower" stage is rejected because we don't
emit code.

The pipeline is **strictly sequential** — each pass reads its
predecessors' output and writes its own annotation arena. There is
**one** lexer↔parser feedback channel, used only for template-list
discovery (§3.9 of the spec).

## 4. Module layout

```
wgsl/
  README.md
  Makefile

  include/
    wgsl.h                Public API (single header)

  src/
    arena.c               Bump arena
    source.c              Source buffer + line index
    span.c                Source-range type
    diag.c                Diagnostic accumulator
    utf8.c                Wraps vendor/unicode (XID, NFC, blankspace)
    token.c               Token type, kind enum, predeclared-name table

    lexer.c               Tokenizer + template-list discovery hook

    parser/
      core.c              Top-level dispatch, translation_unit
      decl.c              global_decl: struct, fn, alias, const, override, var
      stmt.c              Statements + control flow
      expr.c              Pratt expression parser
      helpers.c           Type specifiers, attribute lists, template args
      private.h           Parser-internal declarations

    sema/
      types.c             TypeInfo arena, predeclared types, kinds
      resolver.c          Module/function/block scope, name resolution
      consteval.c         Const-expression evaluator + materialization
      check.c             Type checker, ref/ptr handling
      overloads.gen.c     Generated builtin overload table (from def/wgsl.def)
      private.h

    validate/
      mod.c               Validator entry, dispatch
      recursion.c         No-recursion + alias/const/override-cycle check
      uniformity.c        Directed dependency graph (Tint-style)
      address.c           Address-space inference + conformance
      attrs.c             Per-attribute rule table
      entry.c             Entry-point I/O shape, stage attribute
      layout.c            Memory layout, host-shareable rules
      behavior.c          Statement behavior algebra (§9.7)
      alias.c             Pointer alias analysis (§11.4.1)
      private.h

    lsp/
      tokens.c            Semantic-token enrichment using resolver state
      hover.c             Type / signature lookup at offset
      goto_def.c          Definition-at-offset lookup
      private.h

    wasm_api.c            Emscripten shim, called from JS shell

  def/
    wgsl.def              Tint-style builtin DSL (source of truth)

  tools/
    gen-overloads.c       def/wgsl.def → src/sema/overloads.gen.c

  tests/
    01-arena/             Phase 1 …
    …                     One folder per phase

  vendor/
    unicode/              libunicode.{native,wasm}.a + headers
                          (Decoder, Unicode 17.0.0)

  docs/
    PLAN.md               Phased roadmap
    DESIGN.md             This file
    SPEC-MAP.md           Per-§ pass ownership + risk
    REF-REVIEW.md         Tint + Naga structural review
    BENCH.md              Phase 3.5 numbers (generated at gate)
    MODULE_JSON.md        Schema for `wgsl_module_json` output
    wgsl/                 Turkish W3C spec reference (~6000 lines, 7 files)
```

Internal headers under `src/internal/` carry private declarations
shared across files of the same pass; they are not installed.

## 5. Data shapes

The five core data shapes — locked at this revision:

### `WGSLSource`
The source buffer plus a precomputed line index. Holds the original
bytes; spans point into it. Owned by the result.

### `WGSLToken`
```
kind: WGSLLexTokenKind   (see wgsl.h §5)
offset, length           byte coordinates
line, column             1-based for diagnostics
template_role            none / arg-start / arg-end (filled by lexer
                          during template-list discovery)
```

### `WGSLNode` (AST)
A compact union with a fixed-size header. Sized per Phase 1
benchmark; the `sizeof(WGSLNode)` gate prevents accidental field
growth. Children are `WGSLNode*` arrays in the same arena.

### `WGSLTypeInfo`
Lives in a parallel arena, addressed by interned ID. Predeclared
types (scalars, vectors, matrices, textures, samplers) are interned
once at session init.

### `WGSLDiagnostic` (public — see `wgsl.h`)
LSP-compatible: 1-based line/col, end-line/end-col, severity, message,
rule. `rule` is the empty string for unconditional diagnostics.

## 6. Per-stage design

For each stage: what it consumes, what it produces, what's hard.
Hard cases are deferred to `SPEC-MAP.md`'s *Cross-cutting* section.

### Lexer (Phase 2)

Consumes UTF-8 bytes. Produces a `WGSLToken[]`. Calls into
[`vendor/unicode/`](../vendor/unicode/) for XID classification (§3.7),
NFC normalization (§3.7.1), and blankspace (§3.2).

Special: **template-list discovery** runs interleaved with the lexer,
not as a strict pre-pass. The parser exposes a `peek_template_start()`
hook; the lexer consults it before classifying `<`. (Spec §3.1 Note's
interleaved option, confirmed by Tint's parse-time approach in
`REF-REVIEW.md` §A.)

### Parser (Phase 3)

Recursive descent for declarations and statements; Pratt for
expressions (precedence table from §8.19). Builds the AST
into the arena. Recovers on errors so multiple diagnostics
accumulate per pass.

### Resolver (Phase 5)

Walks declarations; populates a scope tree (predeclared / module /
function / block). Resolves identifier expressions to declarations.
Handles **forward references at module scope** but rejects them in
function bodies (per §5.1).

### Const-evaluator (Phase 6)

Evaluates const-expressions and override-expressions; runs `const_assert`
(§10.1) at this stage. Materializes abstract types into concrete types
at well-defined sites (§6.2.1).

`WGSLValue` (32 B) is the tagged value type: `bool`, `int64_t` (covers
AbstractInt + i32 + u32), `double` (covers AbstractFloat + f32 + f16),
plus reserved slots for vec/mat (filled in v1.x). Bindings are kept on
a side-table keyed by `WGSLSymbol *`; resolver bookkeeping
(`WGSLResolver.all_decls`) lets the evaluator do AST-decl-node →
WGSLSymbol reverse lookup without keeping live scope chains.

**v1 cap**: only the const-eval the spec demands at shader-generation
time. The Phase 6 deliverable supports:

  - scalar literal parse (decimal + hex int / hex float, all suffixes)
  - unary `- ! ~` and the full binary operator set
  - §6.1.2 pair-promotion and §6.2.1 abstract→concrete materialisation
  - module + fn-scope `const NAME = expr;` and `const_assert <expr>;`
  - identifier lookup against the const-binding cache

**v1 deferrals** (revisited in v1.x):

  - vec / mat / array constructor folding (`const C = vec3<f32>(1.0);`).
    Validated against the real shader corpus: every `.wgsl` in
    `examples/shaders/` only uses scalar `const`s, so this gap doesn't
    block the v1 ship. Iter D / v1.x will add the constructor table.
  - member access (`.x` / `.y` / `.rgb`) and component-wise binary ops
    on vec values inside a const-expression.
  - `override` decl default-value folding — values are pipeline-time;
    only the *tier* of the init expression is checked, and Phase 7
    will own the override-expr-tier predicate with full type context.
  - user-defined `const fn` graph evaluation (§11.3) — the Naga
    `constant_evaluator.rs` precedent (194 KB) is the explicit risk in
    [`PLAN.md`](PLAN.md)'s Phase 6 trigger.

### Type checker (Phase 7)

Tags every expression with an effective `WGSLTypeInfo *` and an
`is_ref` bit (§6.4.3 Reference vs Pointer).  Expression types live on
a `WGSLTypeChecker.table` side-table keyed by `WGSLNode *`; symbol
types live on `WGSLSymbol.type` (resolver fills predeclared, the
checker fills user decls).

The full WGSL operator surface is dispatched through two helpers:
`common_componentwise` (scalar × scalar, vec×vec, vec×scalar
broadcast, mat×mat same-shape, mat×scalar broadcast) and
`try_matrix_multiply` (mat×vec, vec×mat, mat×mat dimension-matched).
Comparisons over vec operands produce `vec<width, bool>`.

Builtin overload resolution: the spec defines ~130 builtin functions
across §17.2–17.13.  Iter C ships a **hand-coded** overload table
(`kBuiltinTable` in `src/check.c`) covering ~80 entries — every
builtin the real-shader corpus exercises plus near-neighbours.  Each
entry maps `name → return-rule` (one of `BR_VOID`, `BR_BOOL`,
`BR_U32`, `BR_ARG0`, `BR_ARG0_ELEM`, `BR_TEMPLATE`, `BR_VEC4F`,
`BR_VEC4U`).  `bitcast<T>` is the only template-driven entry today.

**v1.x deferrals**:

  - The Tint-style `def/wgsl.def` DSL + `tools/gen-overloads` codegen
    pipeline (`src/sema/overloads.gen.c`).  REF-REVIEW row C is the
    canonical answer; the hand-coded table is a v1 expedient that
    fits the corpus, not a long-term substitute.  The risk register
    in [`PLAN.md`](PLAN.md) tracks the migration — when the table
    grows past ~150 entries or starts needing precise constraint
    reasoning (e.g. `dot4U8Packed`, `subgroupBallot`), switch.
  - `EXPR_ADDR_OF` / `EXPR_INDIRECTION` + full ref-vs-ptr semantics.
    Phase 8 (validator) wants ref tracking for assignment-LHS gating;
    that will pull this in.
  - Mixed-mode vec constructors (`vec3<f32>(a_vec2, scalar)`); strict
    constructor argument-count validation.
  - Struct field validation in member access — currently permissive
    (member access on an unrecognised type returns the object's type
    silently to keep the corpus passing through `atomicCompareExchange
    Weak`'s synthetic struct return).  Phase 8 owns the strict check
    with full struct field type context.

### Validator (Phase 8)

Single `src/validate.c` for v1 (the eight-file split in `PLAN.md`'s
Module layout is the long-term goal; for the current code volume one
file is fine).  Validator does **not** mutate the AST.

v1 surface — five rule families across three iterations:

  - **Cycle detection** (Iter A) — recursion (§11.4), struct value-
    recursion (§6.2.10), alias cycles (§6.7), const cycles (§7.2),
    override cycles (§7.2).  All five share a DFS-coloring engine
    (`run_cycle_check` + `dfs_visit` + `dfs_edge`) keyed by
    `WGSLResolver.all_decls` indices; only the per-decl-kind edge
    enumerator differs.

  - **Attribute conformance** (Iter B) — the `kAttrTable` (17 entries,
    §12.1–12.15) maps `@name → (min_args, max_args, allowed_contexts)`
    where contexts are bitmasked by AST shape (`AC_FN_DECL`,
    `AC_FN_PARAM`, `AC_FN_RETURN`, `AC_VAR_MODULE`, `AC_OVERRIDE`,
    `AC_STRUCT_MEMBER`, `AC_MODULE`).  `validate_attributes` walks
    every decl-shaped node and checks each `@…` against the table.

  - **Behavior-set analysis** (Iter B, §9.7) — `Behavior` bitmask of
    `{Next, Break, Continue, Return}` propagated up through statements.
    `compound` stops Next at the first non-falling-through stmt; `if/
    else` unions branches (no-else adds Next); `for/while/loop` always
    yield Next plus a propagated Return; `switch` unions cases (no-
    default adds Next) and absorbs Break into Next; `break_if` is
    `Break | Next`.  Body constraint: non-void fn must have Return
    without Next ("does not return on all paths").

  - **Entry-point shape** (Iter C, §13) — at most one stage attribute
    (`@vertex` / `@fragment` / `@compute`) per fn; `@compute` requires
    `@workgroup_size`; `@workgroup_size` is rejected anywhere else.

  - **Address-space conformance** (Iter C, §14.3 subset) — module-scope
    `var<uniform>` and `var<storage>` decls require both `@group` and
    `@binding`.  Other address spaces (`private` / `workgroup` /
    `function`) are unconstrained at v1.

**v1.x deferrals** (revisited as the codebase grows past WebGPU CTS
expectations):

  - **Layout (§14.4)** — host-shareable type alignment and size rules
    for uniform / storage buffers.  Spec is straightforward but
    voluminous; v1.x will fold it into `validate_address_spaces`.
  - **Uniformity analysis (§15.2)** — `PLAN.md`'s risk register
    explicitly budgets this as a multi-week sub-project.  Tint and
    Naga both implement it as a directed dependency graph over
    control-and-value flow.  Required for `derivative_uniformity` and
    `subgroup_uniformity` filterable rules.
  - **Alias analysis (§11.4.1)** — originating-variable propagation
    through the call graph for pointer arguments with `read_write`
    access.
  - **Strict entry-point IO type rules (§13.2 / §13.3)** — vertex must
    have `@builtin(position)`; fragment outputs must have `@location`
    or `@builtin(frag_depth)`; etc.  Currently only stage / workgroup_
    size combinations are validated.

The v1 surface is enough to reject every cycle / attribute placement /
arity / return-completeness / address-space violation the real shader
corpus could plausibly hit; the deferred families do not affect any
section in `examples/shaders/`.

### LSP glue (Phase 9)

`src/wgsl.c` is the entire public C API surface — one file, ~1 K LOC,
keyed off the internal pipeline.  The header (`include/wgsl.h`) is
stable for v1; this implementation file ships the four sections it
documents:

  - **Lifecycle** — `WGSLResult` heap-owns the source bytes plus every
    internal state struct (arena, lex, AST, types, resolver, consteval,
    typecheck, validator) so that hover / goto-def / semantic-tokens
    queries can answer N requests from one `wgsl_check` call.  Always-
    non-NULL contract is honoured even on OOM via a stand-in result.
  - **Diagnostics** — the bag's records are exposed directly; spans are
    1-based line/col with end-column exclusive (mirrors `metal/`).
  - **Semantic tokens** — `wgsl_lex` returns the raw classified stream;
    `wgsl_semantic_tokens` walks the resolved AST to upgrade IDENT
    tokens with their resolver-derived role.  Decl-name spans are
    emitted directly off `payload[0]` of the decl node; EXPR_IDENT use
    sites consume the resolver's `payload[2]` symbol.  EXPR_MEMBER's
    member-name child is tagged `FIELD` (the resolver intentionally
    leaves it unresolved).
  - **Hover / goto-def** — `find_ident_at` walks the AST for the
    smallest EXPR_IDENT containing an offset; falls back to
    `find_decl_at` over `WGSLResolver.all_decls` so a hover over the
    `x` in `let x: i32 = …` answers from the decl side too.
  - **Module summary JSON** — `wgsl_module_json` lazy-builds a string
    in the result's arena describing entry_points / resources /
    structs / overrides per `docs/MODULE_JSON.md`.  `@workgroup_size`
    arguments are resolved via the const-evaluator (handles literal
    integers + module-scope const refs uniformly).

Type-spec → TypeInfo is now also recorded on the typechecker's side-
table (`resolve_type_spec` wrapper) so consumers like the JSON
emitter can pull a member's resolved type out via
`wgsl_typecheck_type_of(spec)` instead of duplicating the resolution
logic.

**v1.x deferrals**:

  - VSCode extension shell + `wgsl-lsp` server binary.  The C side is
    ready; what remains is the LSP-protocol-over-stdio glue layer,
    which is mechanical and language-of-choice: a thin Node or Rust
    wrapper around `wgsl_check` + the query helpers.  Tracked in
    [`PLAN.md`](PLAN.md) as part of Phase 9's "done when".
  - Per-resource layout (host-shareable size + alignment) in the JSON
    summary — gated on the validator's Phase 8 v1.x layout pass.
  - Vertex / fragment IO records with `@location` / `@interpolate`
    pairings — gated on Phase 8 v1.x strict entry-point IO rules.
  - Incremental re-check.  `wgsl_check` re-runs the full pipeline per
    edit; v1 picks correctness + simplicity over latency.  The corpus
    parses + resolves + typechecks + validates fast enough on real
    hardware that incremental is not a v1 blocker (see
    [`docs/BENCH.md`](BENCH.md) — 0.37 ms/Kloc warm).

### WASM build (Phase 10)

`make wasm` runs Emscripten over the same `LIB_SRCS` as the native
build and links the prebuilt `vendor/unicode/libunicode.wasm.a`.  The
output is a single `wgsl_compiler.{js,wasm}` pair sized to ~14 KB JS
+ ~178 KB wasm; `MODULARIZE=1` + `EXPORT_NAME=WGSL` give a factory
function usable from Node, web workers, and the browser
(`ENVIRONMENT=node,web,worker`).

The full public C API surface is exposed via `EXPORTED_FUNCTIONS`
(19 functions + `_malloc` / `_free`).  Runtime helpers `cwrap`,
`ccall`, `UTF8ToString`, `getValue`, `setValue`, and the `HEAPU8` /
`HEAPU32` typed-array views are also exported so callers can read
the small structs (`WGSLDiagnostic`, `WGSLLexToken`, `WGSLHover`,
`WGSLDefinition`) directly.

**Pointer-size note.**  Layout asserts on `WGSLTypeInfo` are now
pointer-size-aware: 24 B / 8-aligned on LP64 (native arm64 / x86_64),
20 B / 4-aligned on ILP32 (wasm32).  Both layouts are intentional —
the size shrink is the natural consequence of `void *ref` being
4 bytes on wasm32 — and both are exercised in CI (the native suite
plus `make wasm-test`).

`make wasm-test` runs a Node-driven smoke: `cwrap`s the public
surface and exercises `wgsl_check` over (a) empty source, (b) a
literal compute entry point with `@workgroup_size(64)`, (c) a parse
error (verifying diagnostic shape via `getValue` over the struct
fields), (d) a resolve error, and (e) `examples/shaders/linear.wgsl`
straight through (the only corpus shader that compiles cleanly
without `_shared.wgsl` prepended).  24 checks; ships with the build.

**v1.x deferrals**:

  - Browser HTML demo and a thin idiomatic JS wrapper that hides the
    `cwrap` / `getValue` dance.  The C side is ready and the build
    flags are already `web`-aware; what remains is wrapping it in a
    higher-level `WGSLClient` class that exposes promises rather
    than raw pointers.  Phase 11 (`wgsl.run` sandbox) will land this.
  - WASM SIMD enablement (`-msimd128`).  The corpus + LSP queries are
    not hot enough to make this worthwhile in v1; revisit when
    profiling the sandbox shows lex / parse as a bottleneck.

## 7. Decisions pinned

These are locked at this revision:

| Decision | Pin | Rationale |
|----------|-----|-----------|
| Spec target | W3C WGSL CRD-2026-05-07 | User intent; `WGSL_SPEC_PIN` |
| Unicode source | Decoder (Unicode 17.0.0), MIT, vendored | `vendor/unicode/README.md` |
| IR | None — decorated AST | REF-REVIEW Tint approach |
| Builtin overloads | Tint-style `.def` DSL + codegen | REF-REVIEW C-row |
| Uniformity model | Directed dependency graph | REF-REVIEW H-row |
| Diagnostic spans | LSP-compatible (mirrors `metal/`) | Editor-uniformity goal |
| Template-list discovery | Lexer↔parser interleaved hook | Spec §3.1 Note + Tint |
| Ref vs Ptr | Per-AST-node `is_ref` bit + auto-load | SPEC-MAP §D |
| Naming | `wgsl_*` prefix, single public header | Mirror `metal/` |
| Lifecycle | One-shot `wgsl_check`; no streaming v1 | Match `metal/` |

## 8. Versioning

- Public API surface is locked at `WGSL_VERSION_STRING = "0.1.0"`.
- Patch (`0.1.x`) — bug fixes, internal refactors, no header changes.
- Minor (`0.x.0`) — additive header changes only (new functions,
  enum entries appended).
- Major (`x.0.0`) — anything that breaks consumers; bumps with this
  document's revision number.

`WGSL_SPEC_PIN` is independent. A new W3C spec revision triggers a
re-pin and a SPEC-MAP rev — but that does **not** force a major bump
unless the API surface itself changes.

## 9. What this document doesn't cover

- **Wire format / serialization** — the module-JSON schema lives in
  `docs/MODULE_JSON.md`, not here.
- **Build matrix** — `make` targets and CI steps live in the
  `Makefile` and `docs/BENCH.md` once Phase 3.5 lands them.
- **Test plan** — phase-aligned tests under `tests/NN-*`; details
  ride with each phase's deliverable.
- **License** — TBD before public ship.
