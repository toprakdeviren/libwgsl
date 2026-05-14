# REF-REVIEW — Tint and Naga structural review

> **Phase 0c deliverable.** What modules and pass split each reference
> implementation uses, and the v1 design call this informs.
> Sourced from upstream `arch.md`, `layering.md`, `uniformity_analysis.md`,
> `intrinsic_definition_files.md` for **Tint**, and a directory survey of
> `naga/src/` (`front/wgsl/{parse,lower}`, `valid/`, `proc/`, `ir/`) for
> **Naga**. No code copied; this is structural inspiration only.

## TL;DR — three big calls

| Decision | Tint approach | Naga approach | **`wgsl` v1** |
|----------|---------------|---------------|---------------|
| **Where types are resolved** | Resolver pass after parse populates `sem::*` info on the same AST | Parse → AST (untyped) → Lower (AST→IR with types) | **Tint-style**: a Resolver pass that decorates the AST, no separate IR. Simpler API; we don't backend-translate. |
| **Builtin overload encoding** | DSL file `wgsl.def`, generated into C++ tables | Hand-coded Rust modules in `proc/overloads/` (~64 KB) | **Tint-style DSL** — a `def/wgsl.def` that codegens C overload tables. ~130 builtins justifies the DSL. |
| **Where uniformity runs** | Validator pass on resolved Program; **directed dependency graph** with Value Nodes + Control Flow Nodes | `valid/analyzer.rs` (65 KB) on the IR | Validator pass on the resolved AST. Graph-based, Tint terminology. |

---

## Tint (Chrome / Google)

### Pipeline

```
Reader → ProgramBuilder → Resolver → Program
                                       ├─→ Inspector (optional)
                                       ├─→ Transforms (optional)
                                       └─→ Writer
```

Quotes from `arch.md`:

- **Reader** — "Readers are responsible for parsing a shader program and populating a `ProgramBuilder`" (SPIR-V and WGSL).
- **ProgramBuilder** — "the interface to construct an immutable `Program`".
- **Resolver** — runs automatically; "generates the `Program`s semantic information by analyzing the `Program`s AST".
- **Program** — "holds an immutable version of the information from the `ProgramBuilder` along with semantic information".
- **Transforms** — "operates by cloning the input `Program` into a new `ProgramBuilder`, applying required changes".

Key data shapes:

- `ast::Node` — "directed acyclic graph of `ast::Node`s which encode the syntactic structure".
- `sem::Node` — "describes the program at a higher/more abstract level than the AST".
- `sem::Type*` pointer-equality — "Semantic types are de-duplicated; you can compare `type::Type*` pointers to check type equivalence".

### Module layering (from `layering.md`)

```
Readers | Writers
  └─→ Val | Inspector | Transform
        └─→ AST
              └─→ Program | Sem
                    └─→ AST Hdrs
                          └─→ Clone Context Hdrs
                                └─→ Constant | Types | Builtin
                                      └─→ Symbols
                                            └─→ Utils IO | Initializer
```

This is the layered dependency tree. Bottom is shared; top is per-language.

### Uniformity analysis (from `uniformity_analysis.md`)

- Builds a **Directed Dependency Graph**.
- Two node kinds:
  - **Value Nodes** — uniformity of an expression or variable.
  - **Control Flow (CF) Nodes** — uniformity of the current execution path.
- An edge `A → B` means *A's uniformity depends on B's*.
- Validation walks: can an operation requiring uniform CF reach a non-uniform source through edges?
- Diagnostic example: *"'workgroupBarrier' must only be called from uniform control flow"* with notes *"control flow depends on possibly non-uniform value"* and *"builtin 'local_id' may be non-uniform"*.

### Builtin overload DSL (from `intrinsic_definition_files.md`)

WGSL builtins live at `src/tint/lang/wgsl/wgsl.def` in a custom DSL:

```
fn F()
fn F() -> R
fn F(f32, i32)
fn F[T: scalar](T)
fn F[T: fiu32](T) -> T
fn F[T, N: num](vec<N, T>)
```

- `[T: constraint]` — implicit template parameter.
- `<T>` — explicit template parameter (user-visible).
- Constraints: `scalar`, `fiu32` (one of f32/i32/u32), `num` (template number).

The `.def` is consumed by a code generator that emits the C++ overload table. This is the leverage point for builtin coverage.

---

## Naga (gfx-rs / Mozilla)

### Pipeline

```
naga::front::wgsl::parse_str(src)
                          │
                          ▼
                   ast::TranslationUnit  ──── parse/{lexer,mod,conv,number,directive}.rs
                          │
                          ▼
                       Lowerer            ──── lower/{mod,construction,conversion,template_list}.rs
                          │
                          ▼
                    ir::Module            ──── ir/{mod,block}.rs
                          │
                          ▼
                Validator::validate       ──── valid/{analyzer,expression,function,interface,type,handles,compose,immediates,mod}.rs
                          │
                          ▼
                   ModuleInfo (typed + uniformity-checked)
                          │
                          ▼
                back::*::Writer (out)     ──── back/{spv,msl,hlsl,glsl,wgsl,...}
```

Key API (from `lib.rs`):

```rust
let module: naga::Module = naga::front::wgsl::parse_str(src)?;
let info = naga::valid::Validator::new(flags, caps).validate(&module)?;
```

### Module layout (`naga/src/`)

```
arena/                 id-arena (handles instead of pointers)
front/wgsl/
  parse/
    lexer.rs            (43 KB)
    mod.rs              (96 KB — recursive descent + Pratt)
    conv.rs             (26 KB — type conversion utilities)
    number.rs           (23 KB — abstract numeric literals)
    directive.rs + dir   enable / requires / diagnostic
    ast.rs              (14 KB — AST node types)
  lower/
    mod.rs              (202 KB — AST → IR lowerer)
    construction.rs     (25 KB — vec/mat/array constructors)
    conversion.rs       (19 KB — abstract → concrete materialization)
    template_list.rs    (7 KB — template-arg consumption iterator)
  index.rs              (8 KB — array-bounds info)
  error.rs              (66 KB — exhaustive WGSL error enum)
  tests.rs              (24 KB)
ir/
  mod.rs                (110 KB — Module/Function/Expression/Statement)
  block.rs              (3 KB)
proc/                   "procedures" = analysis/utility passes
  constant_evaluator.rs (194 KB — the heaviest file in the repo)
  typifier.rs           (36 KB — type inference)
  layouter.rs           (10 KB — memory layout)
  type_methods.rs       (27 KB)
  namer.rs              (16 KB — name uniquification for backends)
  index.rs              (26 KB)
  emitter.rs, terminator.rs, keyword_set.rs
  overloads/            (~64 KB total across 10 files — the WGSL builtin set)
    mod.rs (11)         · regular.rs (18)        · mathfunction.rs (8)
    constructor_set.rs (5) · list.rs (6)         · rule.rs (4)
    scalar_set.rs (5)   · any_overload_set.rs (4) · one_bits_iter.rs (3)
    utils.rs (3)
valid/
  mod.rs                (35 KB — entry; flags, capabilities)
  analyzer.rs           (66 KB — uniformity + control-flow analysis)
  expression.rs         (69 KB — expression-level rules)
  function.rs           (93 KB — fn-level rules: recursion, must_use, etc.)
  interface.rs          (72 KB — entry point I/O matching)
  type.rs               (42 KB — type-level rules)
  handles.rs            (43 KB — handle/arena consistency)
  compose.rs (4)        · immediates.rs (7)
diagnostic_filter.rs    (10 KB — `@diagnostic` filter ranges)
span.rs                 (15 KB)
keywords/               WGSL keyword set
compact/                module compaction post-validation
back/                   six backends (spv, msl, hlsl, glsl, wgsl, dot)
common/
ir/                     internal IR
```

### Template-list discovery — Naga's call

`naga/src/front/wgsl/lower/template_list.rs` (7 KB). Decision: **template lists pass through parse opaquely** as `Vec<Handle<ast::Expression>>`. The lowerer iterates them with a typed `TemplateListIter` — `.ty()`, `.ty_with_span()`, etc. — at the moment a builtin or type-generator is invoked.

This is *opposite* to Tint, which resolves templates as part of parsing. Naga buys flexibility (any expression form passes parse); Tint buys earlier errors.

### Constant evaluator — the elephant

`proc/constant_evaluator.rs` is **194 KB**, the largest file in Naga. It folds const-expressions, const-functions, and abstract-type values, and is responsible for materialization. Anyone implementing const-eval should budget more than the ~1–2 weeks Phase 6 currently allocates if going for full parity.

### Validator split

`valid/` ships **10 files, ~480 KB total**. Note the per-rule-family file split — same shape we already plan for `wgsl/src/validate/` in `PLAN.md`.

| Naga validator file | Our planned file |
|---------------------|-------------------|
| `analyzer.rs` | `validate/uniformity.c` |
| `expression.rs` | (subsumed by Phase 7 type checker) |
| `function.rs` | `validate/recursion.c` + `validate/alias.c` + `validate/behavior.c` |
| `interface.rs` | `validate/entry.c` |
| `type.rs` | `validate/layout.c` (host-shareable rules) |
| `handles.rs` | (no equivalent — we don't have Naga's id-arena handles) |
| `compose.rs`, `immediates.rs` | (subsumed by Phase 7) |

Our split into 8 files (recursion · uniformity · address · attrs · entry · layout · behavior · alias) is finer-grained than Naga's 10 — fine, since some Naga files are 60–90 KB.

---

## Side-by-side — the cross-cutting hard problems from SPEC-MAP

| Hard problem | Tint | Naga | **`wgsl` v1** |
|--------------|------|------|---------------|
| **A. Template-list discovery (§3.9)** | Parser-time disambiguation; resolver fixes types | Parse keeps `Vec<Expression>`; Lower's `template_list.rs` consumes typed | Parser-time (per spec §3.1 Note's interleaved option). We don't have a separate Lower stage to defer to. |
| **B. Abstract-type materialization (§6.2.1)** | Resolver + `Constant` module materialize at type-check sites | `parse/number.rs` tags abstract; `lower/conversion.rs` materializes during AST→IR | Resolver-time materialization. Each `TypeInfo` carries an optional `concrete_target` slot set by the contextual rule. |
| **C. Overload resolution (§6.1.3, §17)** | DSL file `wgsl.def`, codegen → C++ table | Hand-coded `proc/overloads/` (10 files, 64 KB) | **DSL-driven** like Tint. ~130 builtins × N overloads each is too much hand-code. v1 ships `def/wgsl.def` + a `tools/gen-overloads` codegen → `src/sema/overloads.gen.c`. |
| **D. Ref vs Ptr (§6.4.3)** | `sem::*` types carry "is reference" attribute | IR distinguishes `Pointer` types; loads are explicit IR ops | Per-AST-node `is_ref` bit + value-context auto-load (per SPEC-MAP §D). Aligns with Tint. |
| **E. Alias analysis (§11.4.1)** | Validator pass | `valid/function.rs` | `validate/alias.c`. Originating-variable per ptr-typed expression, propagated through call graph. |
| **F. Behavior analysis (§9.7)** | (not surfaced separately in Tint docs) | implicit in `valid/function.rs` | `validate/behavior.c` — own pass; cleaner than embedding in fn validator. |
| **G. Memory layout (§14.4)** | Runtime in `Constant`/`Types` | `proc/layouter.rs` | `validate/layout.c`. |
| **H. Uniformity (§15.2)** | Directed Dependency Graph (Value Nodes + CF Nodes) | `valid/analyzer.rs` (66 KB) | Tint-style **directed dep graph**. Phase 8's 2-week budget stands. |

---

## Concrete implications for Phase 0d (API freeze)

1. **No separate IR.** We adopt Tint's "AST + decorated semantic info"
   model. No AST→IR lower stage. Public API exposes `WGSLAst` and
   `WGSLSem` as cooperating structures.
2. **Single `parse` → `resolve` → `validate` callable surface.** Three
   exported functions, all idempotent; each takes the previous output
   plus session/diagnostics state.
3. **Builtin DSL is part of the source tree.** Add `def/wgsl.def` to
   the project layout (Tint format); `make` invokes a small codegen
   tool; the generated C lands in `src/sema/overloads.gen.c`.
   `def/wgsl.def` is the source of truth — never hand-edit `*.gen.*`.
4. **Validator pre-allocated to 8 files** — already in PLAN; no change.
5. **Const-evaluator is bigger than we budgeted.** Naga is 194 KB on
   const-eval alone. v1 const-eval doesn't have to match that breadth
   (we don't run shaders, only const-fold what the spec demands), but
   Phase 6's 1–2 wk estimate may slip; flag in PLAN risk register.
6. **Template-list discovery** — interleaved with the lexer (per spec
   §3.1 Note). Confirmed by Tint's parse-time approach. Naga's
   defer-to-lower is not available to us because we don't lower.

---

## What this review **didn't** answer

- **Tint directory listing** — `arch.md` mentions modules but I didn't pull
  the actual `src/tint/lang/wgsl/` tree. Phase 1 may want a second pass
  here for parser file-split inspiration.
- **Naga LSP** — Naga has no LSP integration (it's a backend
  translation library). Phase 9 will need to look at e.g. wgsl-analyzer
  separately, not Naga.
- **`wesl-rs`** — there is also a wesl-rs (Rust) WGSL compiler with
  module/import extensions. Out of v1 scope; revisit only if module
  imports become a v2 ask.

---

## Sources

- [Tint repo](https://dawn.googlesource.com/dawn/+/refs/heads/main/src/tint)
- [Tint architecture](https://dawn.googlesource.com/dawn/+/refs/heads/main/docs/tint/arch.md)
- [Tint layering](https://dawn.googlesource.com/dawn/+/refs/heads/main/docs/tint/layering.md)
- [Tint uniformity analysis](https://dawn.googlesource.com/dawn/+/refs/heads/main/docs/tint/uniformity_analysis.md)
- [Tint intrinsic definition files](https://dawn.googlesource.com/dawn/+/refs/heads/main/docs/tint/intrinsic_definition_files.md)
- [Naga in wgpu](https://github.com/gfx-rs/wgpu/tree/trunk/naga)
- [Naga `front/wgsl/`](https://github.com/gfx-rs/wgpu/tree/trunk/naga/src/front/wgsl)
- [Naga `valid/`](https://github.com/gfx-rs/wgpu/tree/trunk/naga/src/valid)
- [Naga `proc/`](https://github.com/gfx-rs/wgpu/tree/trunk/naga/src/proc)
- [Naga `proc/overloads/`](https://github.com/gfx-rs/wgpu/tree/trunk/naga/src/proc/overloads)
