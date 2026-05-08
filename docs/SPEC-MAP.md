# SPEC-MAP — W3C WGSL → wgsl pipeline pass ownership

> **Phase 0b deliverable.** Per-section verdict on which pass / module owns
> each part of the W3C WGSL Recommendation, with risk tags. v1 target =
> **full spec**, so the columns are not "in / out" but **owner + risk**.

## Spec source pinned

- **W3C Candidate Recommendation Draft, 7 May 2026** — pinned by
  date as `WGSL_SPEC_PIN = "CRD-2026-05-07"` in `include/wgsl.h`.
- The working reference is [`docs/wgsl/`](wgsl/) — 7 modular Turkish
  files, ~6000 lines, §1–§18 fully translated, derived from the W3C
  CRD snapshot of that date.
- Single source of truth for v1: this snapshot. A future spec rev triggers
  a fresh `SPEC-MAP-rev2.md`, not in-place edits.

## Vocabulary at a glance

Counted from the spec HTML and grammar productions:

| Bucket | Count | Where |
|--------|------:|-------|
| Top-level sections (§1–§18) | 18 | spec |
| Leaf sections | 236 | spec |
| Grammar productions (`syntax-*`) | 125 | §18 + inline |
| Keywords | 27 | §16.1 |
| Attributes | 17 | §12.1–§12.15 |
| Built-in functions | ~130 | §17.1–§17.13 |
| Predeclared types | finite (scalar × vector × matrix × …) | §6.9 |

**Keywords** (27): `alias` `break` `case` `const` `const_assert` `continue`
`continuing` `default` `diagnostic` `discard` `else` `enable` `false` `fn`
`for` `if` `let` `loop` `override` `requires` `return` `struct` `switch`
`true` `var` `while` (+ token-form: `_` blank ident).

**Attributes** (17): `align` `binding` `blend_src` `builtin` `compute`
`const` `diagnostic` `fragment` `group` `id` `interpolate` `invariant`
`location` `must_use` `size` `vertex` `workgroup_size`.

## Pass ownership legend

| Pass | Symbol |
|------|--------|
| Source / arena / utf8 | **L0** |
| Lexer | **LEX** |
| Parser (template-list discovery + recursive descent + Pratt) | **PAR** |
| Resolver (name resolution + scope) | **RES** |
| Const-evaluator (const-expr + override-expr + materialization) | **CE** |
| Type checker / overload resolver | **TC** |
| Validator (spec-mandated structural rules + uniformity) | **VAL** |
| LSP glue / semantic tokens / hover | **LSP** |
| Out (not v1) | **—** |

Risk: **🟢 low** · **🟡 medium** · **🔴 high**.

---

## §1–§18 ownership table

| § | Title | Owner | Risk | Notes |
|---|-------|-------|:----:|-------|
| **1** | Introduction | — | 🟢 | Non-normative narrative. |
| 1.1 | Overview | — | 🟢 | |
| 1.2 | Syntax Notation | PAR | 🟢 | Reading the BNF only; no code. |
| 1.3 | Mathematical Terms and Notation | CE / TC | 🟢 | floor, ceiling, truncate, roundUp, transpose for builtin semantics. |
| **2** | WGSL Module | L0 / VAL | 🟢 | Lifecycle informs the API shape; module struct owned by L0. |
| 2.1 | Shader Lifecycle | L0 | 🟢 | Inform `wgsl.h` lifecycle: *create* / *create-pipeline* phases. |
| 2.2 | Errors | L0 / VAL | 🟢 | Three error classes: shader-creation / pipeline-creation / dynamic. v1 emits the first two. |
| 2.3 | Diagnostics | L0 / VAL | 🟡 | Severity model + filterable rules + `@diagnostic` filter range nesting. |
| 2.3.1 | Diagnostic Processing | L0 / VAL | 🟢 | "Smallest containing range with same rule" lookup — needs a sorted filter table per rule. |
| 2.3.2 | Filterable Triggering Rules | VAL | 🟡 | `derivative_uniformity`, `subgroup_uniformity` — lifted from uniformity analysis. |
| 2.3.3 | Diagnostic Filtering | PAR / VAL | 🟡 | Range filter attached to compound stmt / fn / if / switch / loop / while / for / continuing. **Affected ranges must perfectly nest** — a parse-time sanity check. |
| 2.4 | Limits | VAL | 🟢 | 9 numeric limits — easy table-driven check. |
| **3** | Textual Structure | LEX / PAR | 🟡 | Section that defines the **parser strategy**: LALR(1) + comment-strip + template-list discovery + context-aware tokenization. |
| 3.1 | Parsing | PAR | 🔴 | Top of the lexer/parser interleaving. v1 picks **interleaved** (the spec's Note suggests `_disambiguate_template` synthetic token). See *Cross-cutting* §A. |
| 3.2 | Blankspace and Line Breaks | LEX | 🟢 | 11 Pattern_White_Space code points + UAX14 LB4s/LB5 line-break definition. Line breaks drive diagnostic line numbers. |
| 3.3 | Comments | LEX | 🟢 | `//` line and `/* … */` **nested** block comments. (Comment strip pre-pass per §3.1.) |
| 3.4 | Tokens | LEX | 🟢 | 6 token classes. |
| 3.5 | Literals | LEX | 🟢 | bool + int + float (decimal/hex). |
| 3.5.1 | Boolean Literals | LEX | 🟢 | |
| 3.5.2 | Numeric Literals | LEX / TC | 🟡 | Suffix `i` / `u` / `f` / `h` selects concrete type; absence ⇒ `AbstractInt` / `AbstractFloat`. **No leading zeros** rule. Hex float requires `p`/`P` exponent. |
| 3.6 | Keywords | LEX | 🟢 | 27 keywords — perfect-hash or trie. |
| 3.7 | Identifiers | LEX | 🟡 | Unicode XID_Start + XID_Continue (UAX31). NFC normalization for comparison. Forbidden double-underscore prefix `__`. |
| 3.7.1 | Identifier Comparison | LEX / RES | 🟡 | Compare in NFC form. UTF-8 decoder is a hard dep. |
| 3.8 | Context-Dependent Names | LEX / PAR | 🟢 | Attribute names, builtin value names, diagnostic rule names, severity, extension names, interpolate type / sampling, swizzle. Recognized in context, not as keywords. |
| **3.9** | **Template Lists** | LEX + PAR | 🔴 | The famous `<` / `>` disambiguation pre-pass. **Section A** in *Cross-cutting*. |
| **4** | Directives | PAR | 🟢 | Module head only. |
| 4.1 | Extensions | PAR / VAL | 🟢 | `enable` (host-toggled) and `requires` (language ext). Validate against a known table. |
| 4.1.1 | Enable Extensions | VAL | 🟢 | Known: `f16`, `dual_source_blending`, `subgroups`, `clip_distances`, `texture_compression_*` (extensible). |
| 4.1.2 | Language Extensions | VAL | 🟢 | Known: `readonly_and_readwrite_storage_textures`, `packed_4x8_integer_dot_product`, `unrestricted_pointer_parameters`, `pointer_composite_access` (extensible). |
| 4.2 | Global Diagnostic Filter | PAR / VAL | 🟢 | Same as range filter, but at module scope. |
| **5** | Declaration and Scope | RES | 🟡 | Module scope = unordered (forward refs OK). Function scope = ordered. Predeclared scope is parent. **No shadowing of predeclared**. |
| 5.1 | Scope | RES | 🟢 | Three scope kinds — predeclared / module / function (compound) — and shadowing rules per kind. |
| 5.2 | Name Resolution | RES | 🟡 | Name resolves to declaration at first matching enclosing scope. Module scope is order-independent. |
| 5.3 | Predeclared Objects | RES | 🟢 | Types + values (`true` / `false` are *literals* not values) + builtin functions + enumerants. v1 fills this from a generated table. |
| **6** | Types | TC | 🟡 | Tight, finite type system; the Tier-1 deliverable for TC. |
| 6.1 | Type Checking | TC | 🟢 | |
| 6.1.1 | Type Rule Tables | TC | 🟢 | Spec encodes operator semantics in tables — ours mirrors. |
| 6.1.2 | Conversion Rank | TC | 🟡 | Integer rank ≤ Float rank ≤ Abstract→Concrete tower. |
| 6.1.3 | Overload Resolution | TC | 🔴 | Used by every operator + every builtin. **Section C** in *Cross-cutting*. |
| 6.2.1 | Abstract Numeric Types | CE / TC | 🔴 | `AbstractInt` ⊂ `AbstractFloat`; **materialize-to-concrete** at well-defined points. **Section B**. |
| 6.2.2–5 | Bool / Integer / Float / Scalar | TC | 🟢 | `bool i32 u32 f32 f16`. f16 gated on `enable f16`. |
| 6.2.6 | Vector Types | TC | 🟢 | `vecN<T>` where N ∈ {2,3,4} and T scalar; `vec2f` etc. predeclared aliases. |
| 6.2.7 | Matrix Types | TC | 🟢 | `matCxR<T>` where C,R ∈ {2,3,4}, T ∈ {f32, f16}. |
| 6.2.8 | Atomic Types | TC | 🟢 | `atomic<T>` where T ∈ {i32, u32}. |
| 6.2.9 | Array Types | TC | 🟡 | Fixed `array<T,N>` and runtime-sized `array<T>`. RTS only as last struct member in storage. |
| 6.2.10 | Structure Types | TC / VAL | 🟡 | Members + `@align`/`@size`/`@location`/`@builtin`/`@interpolate`/`@invariant`/`@blend_src`. Layout pass touches this. |
| 6.2.11 | Composite Types | TC | 🟢 | |
| 6.2.12 | Constructible Types | TC / VAL | 🟢 | Used by `let` / `const` legality. |
| 6.2.13 | Fixed-Footprint Types | TC / VAL | 🟢 | Used by var-in-function legality. |
| 6.3 | Enumeration Types | TC | 🟢 | Distinct from the type system: access mode / address space / texel format names. |
| **6.4** | **Memory Views** | TC | 🔴 | Ref vs Ptr semantics. **Section D**. |
| 6.4.1 | Storable Types | TC | 🟢 | |
| 6.4.2 | Host-shareable Types | TC / VAL | 🟡 | Used by uniform/storage layout. |
| 6.4.3 | Reference and Pointer Types | TC | 🔴 | Ref auto-loads; Ptr explicit. v1 must track ref-vs-value at every expression node. |
| 6.4.4 | Valid and Invalid Memory References | TC / VAL | 🟢 | |
| 6.4.5 | Originating Variable | TC / VAL | 🟡 | Feeds alias analysis (§11.4.1). |
| 6.4.6 | Out-of-Bounds Access | VAL | 🟢 | Robustness model — runtime concern; spec only requires bounds-honoring at runtime. |
| 6.4.7–9 | Use cases / Forming / Comparison | — | 🟢 | Non-normative. |
| 6.5 | Texture and Sampler Types | TC | 🟡 | 7 sampled / 1 multisampled / 1 external / 1 storage / 1 depth / 1 sampler kinds × dimensions. Predeclared. |
| 6.5.1 | Texel Formats | TC | 🟢 | Enum table; ~30 formats. |
| 6.5.2–7 | Texture / Sampler kinds | TC | 🟢 | |
| 6.6 | AllTypes | TC | 🟢 | Used in spec table notation only. |
| 6.7 | Type Aliases | RES / TC | 🟢 | Module-scope alias creates a name in same namespace. |
| 6.8 | Type Specifier Grammar | PAR | 🟢 | `template_elaborated_ident` with optional `template_list`. |
| 6.9 | Predeclared Types and Type-Generators Summary | RES | 🟢 | Table to load into predeclared scope. |
| **7** | Variable and Value Declarations | PAR / RES / TC | 🟡 | Four declaration forms: `const` / `override` / `let` / `var`. |
| 7.1 | Variables vs Values | TC | 🟢 | Defines reference rule. |
| 7.2 | Value Declarations | RES / TC / CE | 🟡 | `const` value (must const-eval), `override` (pipeline-overridable), `let` (function only). |
| 7.3 | `var` Declarations | RES / TC / VAL | 🟡 | Address-space inference; `var<…>` with explicit space. |
| 7.4 | Variable and Value Decl Grammar | PAR | 🟢 | |
| **8** | Expressions | PAR / TC / CE | 🟡 | 17 expression productions; precedence in §8.19. |
| 8.1 | Early Evaluation Expressions | CE | 🟡 | const-expr ⊂ override-expr ⊂ runtime-expr — three tiers. |
| 8.2 | Indeterminate Values | TC | 🟢 | |
| 8.3 | Literal Value Expressions | TC | 🟢 | |
| 8.4 | Parenthesized Expressions | PAR | 🟢 | |
| 8.5.x | Composite Decomposition | PAR / TC | 🟡 | Vector/matrix/array/struct access; **swizzle** is parser-recognized identifier in context. |
| 8.6 | Logical Expressions | PAR / TC | 🟢 | Short-circuit `&&` `\|\|`. Vector `& \|` are bitwise, not short-circuit. |
| 8.7 | Arithmetic Expressions | PAR / TC | 🟡 | Unary `-`; binary `+ - * / %`; matrix-vector / matrix-matrix products. |
| 8.8 | Comparison Expressions | PAR / TC | 🟢 | |
| 8.9 | Bit Expressions | PAR / TC | 🟢 | `& \| ^ << >> ~`. |
| 8.10 | Function Call Expression | PAR / TC | 🔴 | `template_elaborated_ident` call site is the heart of overload resolution. |
| 8.11–14 | Variable / Param / Address-Of / Indirection | PAR / TC | 🟡 | Address-of and indirection are the only operators that move between Ref and Ptr. |
| 8.15–17 | Identifier / Enum / Type Expressions | PAR / RES | 🟢 | |
| 8.18 | Expression Grammar Summary | PAR | 🟢 | |
| 8.19 | Operator Precedence and Associativity | PAR | 🟢 | The Pratt table source. |
| **9** | Statements | PAR / TC / VAL | 🟡 | |
| 9.1 | Compound Statement | PAR / RES | 🟢 | New scope; range for diagnostics filter. |
| 9.2 | Assignment Statement | PAR / TC | 🟡 | Simple / phony (`_ = expr`) / compound (`+= -= …`). LHS must be Ref. |
| 9.3 | Increment / Decrement | PAR / TC | 🟢 | LHS must be Ref to {i32, u32}. |
| 9.4.x | Control Flow (if / switch / loop / for / while / break / break-if / continue / continuing / return / discard) | PAR / VAL | 🟡 | Each has narrow placement rules. `continuing` and `break-if` are the loop-only forms. |
| 9.5 | Function Call Statement | PAR / TC / VAL | 🟢 | Ignored result must satisfy `must_use`. |
| 9.6 | Statements Grammar Summary | PAR | 🟢 | |
| 9.7 | Statements Behavior Analysis | VAL | 🟡 | Per-stmt set of behaviors {Next, Break, Continue, Return}; reachability check. **Section F**. |
| **10** | Assertions | PAR / CE / VAL | 🟢 | `const_assert` only. |
| 10.1 | Const Assertion Statement | CE / VAL | 🟢 | Evaluate at shader-creation; must be `true`. |
| **11** | Functions | PAR / RES / TC / VAL | 🟡 | |
| 11.1 | Declaring a User-defined Function | PAR / RES | 🟢 | |
| 11.2 | Function Calls | TC | 🟡 | Argument types must match parameter types after conversion-rank tower. |
| 11.3 | `const` Functions | CE | 🟡 | Subset of functions usable in const-expr. v1 implements predicate + memoizing const-call evaluator. |
| 11.4 | Restrictions on Functions | VAL | 🔴 | **No recursion** (call-graph cycle check). |
| 11.4.1 | Alias Analysis | VAL | 🔴 | Originating-variable equivalence on pointer arguments. **Section E**. |
| **12** | Attributes | PAR / VAL | 🟡 | 17 attribute kinds — table-driven validator. |
| 12.1–14 | Per-attribute rules | VAL | 🟡 | `align`/`size` (struct member layout), `binding`/`group` (resource), `blend_src`/`builtin`/`location`/`interpolate`/`invariant` (entry I/O), `const`/`must_use` (function), `diagnostic` (filter), `id` (override), `workgroup_size` (compute). |
| 12.15 | Shader Stage Attributes | VAL | 🟢 | `@vertex`/`@fragment`/`@compute` mark entry-point. |
| **13** | Entry Points | VAL | 🟡 | |
| 13.1 | Shader Stages | VAL | 🟢 | |
| 13.2 | Entry Point Declaration | VAL | 🟡 | One stage attr; param/return shape rules per stage. |
| 13.3 | Shader Interface | VAL | 🟡 | Inter-stage I/O matching; resource interface; layout compatibility; runtime-array element count from binding. |
| **14** | Memory | TC / VAL | 🟡 | |
| 14.1 | Memory Locations | TC | 🟢 | |
| 14.2 | Memory Access Mode | TC / VAL | 🟢 | `read` / `write` / `read_write` per address-space. |
| 14.3 | Address Spaces | TC / VAL | 🟡 | 6 spaces × 5 attributes (sharing, default access, instantiation, ref/ptr in user code, `var` decl form). |
| 14.4 | Memory Layout | VAL | 🟡 | Align + size rules for host-shareable types; struct member layout; padding. **Section G**. |
| 14.5 | Memory Model | — | 🟢 | Runtime semantics; no v1 enforcement (we don't run shaders). |
| **15** | Execution | VAL / — | 🟡 | |
| 15.1 | Program Order Within an Invocation | — | 🟢 | Runtime semantics. |
| **15.2** | **Uniformity** | VAL | 🔴 | **Section H** — its own multi-week pass. |
| 15.3 | Compute Shaders and Workgroups | VAL | 🟢 | `workgroup_size` placement. |
| 15.4 | Fragment Shaders and Helper Invocations | VAL | 🟢 | |
| 15.5 | Subgroups | VAL | 🟢 | Behind `enable subgroups`. |
| 15.6 | Collective Operations | VAL | 🟡 | Barriers / derivatives / subgroup / quad — uniformity-tagged. |
| 15.7 | Floating Point Evaluation | TC / CE | 🟡 | IEEE-754-with-deviations. Domain restrictions per builtin. |
| **16** | Keyword and Token Summary | LEX | 🟢 | Source of truth for the LEX tables. |
| 16.1 | Keyword Summary | LEX | 🟢 | 27 entries. |
| 16.2 | Reserved Words | LEX | 🟢 | Reject as identifier; reserve for future spec. |
| 16.3 | Syntactic Tokens | LEX | 🟢 | Operators + punctuation. |
| **17** | Built-in Functions | TC / VAL | 🔴 | ~130 functions across 13 groups. **Section C** — overload resolution does the heavy lifting. |
| 17.1 | Constructor Built-in Functions | TC | 🟡 | Zero-value + value-constructor; spec-special since some are type-generators (`vec2(…)` etc.). |
| 17.2 | Bit Reinterpretation | TC | 🟢 | `bitcast`. |
| 17.3 | Logical | TC | 🟢 | `all` `any` `select`. |
| 17.4 | Array | TC | 🟢 | `arrayLength`. |
| 17.5 | Numeric | TC | 🟡 | 63 numeric functions. Domain restrictions per fn (§15.7.7). |
| 17.6 | Derivative | TC / VAL | 🟡 | `dpdx` family — uniformity-required. Fragment-only. |
| 17.7 | Texture | TC / VAL | 🟡 | 15 texture functions × texture kinds; offset must be const-expr; uniformity-tagged for sampling. |
| 17.8 | Atomic | TC / VAL | 🟢 | 8 atomic ops. |
| 17.9–10 | Data Pack / Unpack | TC | 🟢 | |
| 17.11 | Synchronization | TC / VAL | 🟡 | `workgroupBarrier` etc. — uniformity-required. |
| 17.12 | Subgroup | TC / VAL | 🟡 | Uniformity-required. |
| 17.13 | Quad | TC / VAL | 🟡 | Uniformity-required (filterable). |
| **18** | Grammar for Recursive Descent Parsing | PAR | 🟢 | The full BNF. Mechanical translation to recursive-descent + Pratt. |

---

## Cross-cutting hard problems

### A. Template-list discovery (§3.9)

`<` and `>` are ambiguous (`a < b` vs `vec2<f32>`). Spec algorithm:

1. Walk tokens left-to-right with a `(depth, expr-depth)` stack.
2. On `template_elaborated_ident <`, *speculatively* mark the `<` as a
   template-start and recurse: a matching `>` at the same depth, with no
   intervening `; { } : = != == || && ?`, confirms it.
3. On confirmation, retag the `<` and `>` as `_template_args_start` /
   `_template_args_end`.

**Decision for v1:** interleaved with the lexer (per spec's Note in §3.1)
— the parser exposes a `peek-template-list` hook to the lexer, which
inserts the synthetic disambiguation token.

### B. Abstract numeric types and materialization (§6.2.1)

`AbstractInt` and `AbstractFloat` are *unobservable* — they convert
implicitly to a concrete numeric type at well-defined materialization
points (assignment target, function argument, return, etc.). Each
const-expr carries an *abstract* type until a context demands materialize.

**Decision for v1:** every TypeInfo has a `concrete_target` slot that
materialize sets; CE preserves abstract until TC's site demands concrete.

### C. Overload resolution (§6.1.3, §17)

Operators and builtins are written as *overload sets* in the spec.
Resolution: filter by arity → filter by parameter compatibility under
conversion rank → pick min-rank → tie-break by abstract-prefer-concrete.

**Decision for v1:** generated overload table from the spec text;
resolver is a single pure function `resolve(name, arg_types) → SignatureID`.
Source-of-truth doc: [`docs/wgsl/07-built-in-kutuphanesi.md`](wgsl/07-built-in-kutuphanesi.md).

### D. Reference vs Pointer (§6.4.3)

WGSL's `ref<AS, T, AM>` (implicit, auto-loaded) and `ptr<AS, T, AM>`
(explicit, user-visible) are the source of subtle bugs. Every expression
node carries an `is_ref` bit; `&` lifts ref→ptr, `*` lowers ptr→ref;
*load* is invisible (happens on use of a Ref in a value context).

**Decision for v1:** AST tags expressions with effective type **and**
ref-bit; the type checker normalizes Refs at every value-use site.

### E. Alias analysis (§11.4.1)

For `fn f(p: ptr<…, T, …>, q: ptr<…, T, …>)` with `read_write` access:
the called function may not have *p* and *q* alias the same originating
variable. v1 tracks "originating variable" per pointer-typed expression
through the call graph.

### F. Behavior analysis (§9.7)

Each statement has a *behaviors* set ⊆ {Next, Break, Continue, Return,
Discard}. Body of fn must satisfy: behaviors ⊆ {Next, Return}, and a
non-void fn must have *Return* in *all* paths (no Next).

**Decision for v1:** a small algebra over behavior sets; one pass after
parse, before validator.

### G. Memory layout (§14.4)

Host-shareable types in `uniform` and `storage` need precise align/size
matching the host buffer. Rules involve struct/array recursion and
attribute overrides. v1 emits both nominal layout and inter-buffer
compatibility metadata in the diagnostic + module-summary output.

### H. Uniformity analysis (§15.2)

Whole-pass: builds a per-function uniformity graph (control + value
dependencies) and labels each program point with a uniformity proof
obligation. Required for `derivative_uniformity` and
`subgroup_uniformity` filterable rules. Tint and Naga both implement
this; both took weeks. **Phase 8 budget: 2 weeks dedicated.**

---

## Pass ↔ phase mapping

| Pass | Phase | Module |
|------|------:|--------|
| L0 | 1 | `src/{arena,source,span,diag,utf8,token}.c` |
| LEX | 2 | `src/lexer.c` (+ template-discovery interleave hook) |
| PAR | 3 | `src/parser/{core,expr,stmt,decl,helpers}.c` |
| TC types | 4 | `src/sema/types.c` |
| RES | 5 | `src/sema/resolver.c` |
| CE | 6 | `src/sema/consteval.c` |
| TC | 7 | `src/sema/check.c` |
| VAL | 8 | `src/validate/{recursion,uniformity,address,attrs,entry,layout,behavior,alias}.c` |
| LSP | 9 | `src/lsp_glue.c` |

Phase 8 splits into 8 source files because the validator is
the largest, most distinct chunk (~3–4 weeks). Each rule family
gets its own file so the test-per-file pattern stays clean.

---

## Open questions that block Phase 1

(Carried over from `PLAN.md`'s closing section.)

1. **UTF-8 + XID source** — ✓ **Locked.** Vendor the Decoder library
   (MIT, Unicode 17.0.0). Both archives in
   [`vendor/unicode/`](../vendor/unicode/): `libunicode.native.a`
   (arm64, 3.1 MB) for native gates, `libunicode.wasm.a` (1.3 MB) for
   the Phase 10 WASM build. Headers from canonical
   `/Users/2n/dev/personal/decoder/include/`. Provides UTF-8 decode,
   NFC normalization, and identifier-start / identifier-continue
   predicates that match WGSL §3.2/§3.7/§3.7.1.
2. **Diagnostic span format** — match `metal/`'s LSP-compatible spans
   exactly (1-based line/column + end-line/end-column + severity)? Yes —
   matches the editor integration goal. *Locked.*
3. **Spec source pin** — W3C CRD 7 May 2026 (the file in this repo).
   *Locked.*

## Coverage check

- 18/18 top-level sections placed.
- 27/27 keywords, 17/17 attributes, ~130/130 built-ins owned by a pass.
- 0 sections marked "v1-out" — the user's stated target is **full spec**.
- 8 cross-cutting hard problems flagged with explicit decisions.
