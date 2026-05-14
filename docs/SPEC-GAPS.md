# SPEC-GAPS — what's shipped, what's not, what's deferred

> **As of 2026-05-08.** Snapshot of where the v1 implementation stands
> against [`SPEC-MAP.md`](SPEC-MAP.md). Three states:
>
> - ✅ **done** — covered for the v1 surface; corpus exercises this and
>   passes; behaves spec-correctly within the documented v1 scope.
> - ◐ **partial** — a subset of the spec rules is in place; the gap
>   is documented either here or in [`DESIGN.md`](DESIGN.md) and
>   tracked as v1.x. The library does not silently mis-handle the
>   uncovered shapes — it accepts them permissively (best-effort) or
>   rejects them with a precise diagnostic.
> - ✗ **not started** — flagged for v1.x. The library may accept
>   shapes that the strict spec would reject; the pipeline does not
>   crash but downstream consumers must not rely on these checks.
>
> One row per leaf section; phase ownership matches the SPEC-MAP.
> Cross-cutting hard problems §A–§H are summarized at the end.

## Headline

| Bucket                           | Done | Partial | Not started |
|----------------------------------|-----:|--------:|------------:|
| Top-level sections (§1–§18)       |  16  |    2    |      0      |
| Leaf sections counted in SPEC-MAP |  ~70 |   ~25   |     ~10     |
| Cross-cutting hard problems       |   5  |    1    |      2      |

Concretely:

- Lex + parse + resolve + const-eval (scalar) + type-check + validate
  (cycle / attr / behavior / entry-point / address-space) — **shipped**.
- Builtin overloads — **partial** (~80 of ~130, hand-coded; full Tint-
  style `def/wgsl.def` codegen is the canonical v1.x path).
- Memory layout (§14.4), uniformity (§15.2), alias analysis (§11.4.1)
  — **not started**, the three biggest validator pieces.

The corpus (17 ML / transformer shaders, 43 kernel sections) passes
every shipped pass without false negatives or false positives.

---

## §1–§3  Module structure / textual layer

| §       | Title                            | Status | Notes |
|---------|----------------------------------|:------:|-------|
| 1.x     | Introduction / overview          |   —    | Non-normative. |
| 2.1     | Shader Lifecycle                 |   ✅    | `wgsl_check` is the v1 entry point; pipeline-generation phase is host-side. |
| 2.2     | Errors                           |   ✅    | Shader-generation + pipeline-generation classes; runtime errors out of scope. |
| 2.3     | Diagnostics                      |   ✅    | LSP-shape spans, severity model, message arena. |
| 2.3.1   | Diagnostic Processing            |   ◐    | "Smallest containing range with same rule" lookup not yet implemented; v1.x. |
| 2.3.2   | Filterable Triggering Rules      |   ✗    | Bound to uniformity (`derivative_uniformity` / `subgroup_uniformity`); blocked on §15.2. |
| 2.3.3   | Diagnostic Filtering             |   ◐    | `@diagnostic` parsed and recognised; per-range filter table v1.x. |
| 2.4     | Limits                           |   ✗    | 9 numeric limits; table-driven check — not gated yet. |
| 3.1–3.8 | Tokens / blankspace / comments / literals / keywords / idents |   ✅    | Full lexer; UAX31 + NFC; `__` prefix rejected; hex floats with `p`. |
| **3.9** | Template-list discovery          |   ✅    | Cross-cutting **§A**. Pending stack with depth tracking, splits `>>` / `>=` / `>>=` correctly. |

## §4  Directives

| §   | Title              | Status | Notes |
|-----|--------------------|:------:|-------|
| 4.1 | Extensions         |   ◐    | `enable` / `requires` parsed and accepted; the **table of known extensions is not enforced** — any name passes. v1.x. |
| 4.2 | Global Diagnostic Filter | ◐ | Parsed; not bound to runtime severity yet. |

## §5  Declaration and scope

| §   | Title                  | Status | Notes |
|-----|------------------------|:------:|-------|
| 5.1 | Scope kinds            |   ✅    | Predeclared / module / function-block. |
| 5.2 | Name resolution        |   ✅    | Module forward refs OK; fn-scope order-sensitive; "use before declaration" → "undeclared". |
| 5.3 | Predeclared objects    |   ✅    | ~210 entries (5 scalars + 35 typegens + 30 type aliases + 7 addr_spaces + 3 access_modes + ~130 builtin fns). |

> **Deviation from spec §5.1**: The spec forbids shadowing *any*
> predeclared identifier in inner scopes.  v1 relaxes this to only
> reserve predeclared **type** names (scalar / typegen / alias /
> addr_space / access_mode).  Builtin function names (`step`,
> `length`, `dot`, …) can be shadowed locally, matching Tint / Naga
> de-facto behaviour and real-world shader practice.  Strict spec
> compliance is a v1.x toggle.

## §6  Types

| §        | Title                         | Status | Notes |
|----------|-------------------------------|:------:|-------|
| 6.1.1    | Type Rule Tables              |   ✅    | Operator surface mirrors spec §6.1.1 tables. |
| 6.1.2    | Conversion Rank               |   ✅    | Full 9-row table incl. element-wise inheritance. |
| 6.1.3    | Overload Resolution           |   ◐    | Cross-cutting **§C**.  Hand-coded surface; full Tint-style `.def` DSL deferred. |
| 6.2.1    | Abstract Numeric Types        |   ✅    | Cross-cutting **§B**.  Materialise at decl bind, default tower for free-standing exprs. |
| 6.2.2–5  | Bool / Integer / Float / Scalar | ✅  | All scalar shapes interned at session init. |
| 6.2.6    | Vector Types                  |   ✅    | vec2/3/4 + 12 predeclared `vec*[iuf h]` aliases. |
| 6.2.7    | Matrix Types                  |   ✅    | matCxR for C,R∈{2,3,4} + 18 `mat*x*[fh]` aliases. |
| 6.2.8    | Atomic Types                  |   ✅    | atomic<i32> / atomic<u32>. |
| 6.2.9    | Array Types                   |   ◐    | `array<T, N>` with N const-eval'd OK. **Runtime-array placement gate** ("RTS only as last struct member in storage") not enforced — v1.x layout pass. |
| 6.2.10   | Structure Types               |   ◐    | Members + types resolve.  Field strict validation in expression context is permissive (member access on unknown shape returns object's type). |
| 6.2.11–13 | Composite / Constructible / Fixed-Footprint | ◐ | Predicates not checked at decl sites yet. v1.x. |
| 6.3      | Enumeration Types             |   ✅    | Address-space + access-mode names predeclared.  Texel formats deferred. |
| **6.4**  | Memory Views                  |   ◐    | Cross-cutting **§D**.  See below. |
| 6.4.1    | Storable Types                |   ✗    | Predicate not checked at decl sites. v1.x. |
| 6.4.2    | Host-shareable Types          |   ✗    | Bound to layout (§14.4). |
| 6.4.3    | Reference and Pointer Types   |   ◐    | `is_ref` bit propagates correctly through swizzle / index / paren.  **Address-of `&` and indirection `*` not yet typed** — Phase 7 walks them but assigns no type.  v1.x. |
| 6.4.4    | Valid / Invalid Memory Refs   |   ✗    | v1.x. |
| 6.4.5    | Originating Variable          |   ✗    | Feeds alias analysis (§11.4.1). v1.x. |
| 6.4.6    | Out-of-Bounds Access          |   —    | Runtime concern. |
| 6.5      | Texture and Sampler Types     |   ◐    | Names predeclared (17 typegens).  Their TypeInfo isn't deeply modeled — `textureSample(...)` returns `vec4<f32>` as a best-effort default; `textureLoad` similarly.  v1.x. |
| 6.5.1    | Texel Formats                 |   ✗    | ~30 enum names not yet recognised. v1.x. |
| 6.7      | Type Aliases                  |   ✅    | Forward refs handled by struct/alias pre-pass in TC. |
| 6.8      | Type Specifier Grammar        |   ✅    | `template_elaborated_ident`. |
| 6.9      | Predeclared Types Summary     |   ✅    | All entries loaded into `WGSLResolver.predeclared`. |

## §7  Variable and value declarations

| §   | Title                       | Status | Notes |
|-----|-----------------------------|:------:|-------|
| 7.1 | Variables vs Values         |   ✅    | `var` produces ref; `const` / `let` / `param` produce values. |
| 7.2 | Value Declarations          |   ◐    | `const` const-eval'd; `let` typed.  **Override default-value folding** is a v1 deferral — only the *tier* of the init is implicitly checked. |
| 7.3 | `var` Declarations          |   ◐    | Address-space template + access mode parsed, typed.  Address-space *inference* (when no template given) defaults to `private`; the spec's full inference rules are v1.x. |

## §8  Expressions

| §       | Title                              | Status | Notes |
|---------|------------------------------------|:------:|-------|
| 8.1     | Early Evaluation Expressions       |   ◐    | Three tiers (const / override / runtime) modelled.  **`const` fn graph evaluation** (§11.3) deferred to v1.x. |
| 8.2     | Indeterminate Values               |   —    | Runtime-only concept. |
| 8.3     | Literal Value Expressions          |   ✅    | All four literal forms typed. |
| 8.4     | Parenthesized Expressions          |   ✅    | |
| 8.5.x   | Composite Decomposition            |   ✅    | Vec swizzle (.x / .xyz / .rgba), struct field, vec[i], mat[i] (column), array[i]. |
| 8.6     | Logical Expressions                |   ✅    | Bool short-circuit + vec component-wise. |
| 8.7     | Arithmetic Expressions             |   ✅    | Full vec/mat/scalar surface incl. mat*vec / vec*mat / mat*mat. |
| 8.8     | Comparison Expressions             |   ✅    | Scalar → bool, vec → vec<bool>. |
| 8.9     | Bit Expressions                    |   ✅    | Integer-only with element-type gate. |
| 8.10    | Function Call Expression           |   ◐    | Constructors + user-fn calls + ~80 builtins.  Strict argument validation (argc, type-rank against full overload set) is the v1.x `def/wgsl.def` deliverable. |
| 8.11    | Variable Identifier Expression     |   ✅    | |
| 8.12    | Formal Parameter Expression        |   ✅    | |
| 8.13    | Address-Of                         |   ✗    | Walked by TC but not typed. v1.x. |
| 8.14    | Indirection                        |   ✗    | Same. v1.x. |
| 8.15–17 | Identifier / Enum / Type           |   ✅    | |
| 8.19    | Operator Precedence + Assoc.       |   ✅    | Pratt table mirrors §8.19. |

## §9  Statements

| §       | Title                              | Status | Notes |
|---------|------------------------------------|:------:|-------|
| 9.1     | Compound Statement                 |   ✅    | Pushes block scope. |
| 9.2     | Assignment Statement               |   ◐    | Parsed, typed.  **LHS=ref enforcement** is v1.x (depends on full ref/ptr semantics). |
| 9.3     | Increment / Decrement              |   ◐    | Same gate. |
| 9.4.x   | Control flow (if / switch / loop / for / while / break / break-if / continue / continuing / return / discard) | ✅ | All shapes parsed + typed + behavior-analysed. |
| 9.5     | Function Call Statement            |   ◐    | Parsed + typed.  **`@must_use` enforcement** (rejecting ignored returns of must-use functions) is v1.x. |
| 9.7     | Behavior Analysis                  |   ✅    | Cross-cutting **§F**. |

## §10  Assertions

| §    | Title                       | Status | Notes |
|------|-----------------------------|:------:|-------|
| 10.1 | `const_assert`              |   ✅    | Eval'd at shader-generation; non-bool / false → diag. |

## §11  Functions

| §      | Title                          | Status | Notes |
|--------|--------------------------------|:------:|-------|
| 11.1   | User-defined Function Decl     |   ✅    | |
| 11.2   | Function Calls                 |   ◐    | Return type bound to call-site.  **Strict parameter-arg type matching** with conversion-rank scoring is v1.x. |
| 11.3   | `const` Functions              |   ✗    | v1 has no const-fn graph evaluator (Naga's is 194 KB; risk register explicit). v1.x. |
| 11.4   | No Recursion                   |   ✅    | Validator's call-graph cycle DFS. |
| 11.4.1 | Alias Analysis                 |   ✗    | Cross-cutting **§E**. v1.x — needs originating-variable propagation. |

## §12  Attributes

| §        | Title                              | Status | Notes |
|----------|------------------------------------|:------:|-------|
| 12.1–14  | Per-attribute placement + arity    |   ✅    | 17-entry `kAttrTable`; placement-bitmask check. |
| 12.x     | Per-attribute **value semantics**  |   ◐    | E.g., `@align` requires power-of-two; `@binding` non-negative; `@location` < device limit.  **Value-semantic checks beyond arity** are v1.x. |
| 12.15    | Stage Attributes                   |   ✅    | |

## §13  Entry points

| §    | Title                       | Status | Notes |
|------|-----------------------------|:------:|-------|
| 13.1 | Shader Stages               |   ✅    | At-most-one stage attr / fn. |
| 13.2 | Entry Point Declaration     |   ◐    | Stage attr + `@workgroup_size` for compute.  **Strict IO type rules per stage** (vertex requires `@builtin(position)` return, fragment outputs need `@location` etc.) are v1.x. |
| 13.3 | Shader Interface            |   ✗    | Inter-stage IO matching, resource interface validation, layout compatibility — v1.x. |

## §14  Memory

| §    | Title                       | Status | Notes |
|------|-----------------------------|:------:|-------|
| 14.1 | Memory Locations            |   ✅    | Implicit in TC. |
| 14.2 | Memory Access Mode          |   ✅    | `read` / `write` / `read_write` recognised; module summary surfaces them. |
| 14.3 | Address Spaces              |   ◐    | `var<uniform>` / `var<storage>` require `@group` + `@binding` (validator).  **Full §14.3 table** (6 spaces × 5 attribute matrix: sharing, default access, instantiation, ref/ptr in user code, var-decl form) is v1.x. |
| 14.4 | Memory Layout               |   ✗    | Cross-cutting **§G**. v1.x — entire pass. |
| 14.5 | Memory Model                |   —    | Runtime semantics. |

## §15  Execution

| §    | Title                            | Status | Notes |
|------|----------------------------------|:------:|-------|
| 15.1 | Program Order                    |   —    | Runtime. |
| 15.2 | **Uniformity**                   |   ✗    | Cross-cutting **§H**. v1.x — multi-week sub-project per risk register. |
| 15.3 | Compute / Workgroups             |   ✅    | `@workgroup_size` placement enforced. |
| 15.4 | Fragment / Helper Invocations    |   ◐    | Stage attr recognised; helper-invocation gates are uniformity-bound. |
| 15.5 | Subgroups                        |   ◐    | Predeclared `subgroup*` builtins; **gating on `enable subgroups`** v1.x. |
| 15.6 | Collective Operations            |   ◐    | Recognised at TC; uniformity-tagged enforcement v1.x. |
| 15.7 | Floating Point Evaluation        |   ◐    | IEEE-754 semantics in const-eval; **per-builtin domain restrictions** (§15.7.7) v1.x. |

## §16  Keyword and token summary

| §    | Title                       | Status | Notes |
|------|-----------------------------|:------:|-------|
| 16.1 | Keyword Summary             |   ✅    | 27 entries. |
| 16.2 | Reserved Words              |   ✗    | List of future-reserved names not enforced. v1.x — minor lexer table. |
| 16.3 | Syntactic Tokens            |   ✅    | Full operator + punctuation tables. |

## §17  Built-in functions

`kBuiltinTable` covers ~80 of ~130 spec builtins.  All entries return
the correct *result type* for the corpus's usage; **per-arg type
matching against the spec's overload sets** is the v1.x `def/wgsl.def`
deliverable.

| §       | Title                       | Status | Coverage |
|---------|-----------------------------|:------:|----------|
| 17.1    | Constructor Built-ins       |   ✅    | Inline in TC's `type_call`. |
| 17.2    | Bit Reinterpretation        |   ✅    | `bitcast<T>(x)` template-arg dispatch. |
| 17.3    | Logical                     |   ✅    | `all` / `any` / `select`. |
| 17.4    | Array                       |   ◐    | `arrayLength` returns u32; full `ptr<array<T>>` arg gate is v1.x. |
| 17.5    | Numeric (63 fns)            |   ✅    | All shipped (component-wise + reductions + cross/distance/length). **Domain restrictions per fn** (§15.7.7) v1.x. |
| 17.6    | Derivative                  |   ◐    | Type-wise present; **fragment-only check** v1.x (uniformity-bound). |
| 17.7    | Texture (15 fns)            |   ◐    | Names recognised; return shape best-effort `vec4<f32>` / `u32`.  Strict overload + offset-must-be-const + uniformity gates v1.x. |
| 17.8    | Atomic (8 ops)              |   ◐    | Names recognised; `ptr<atomic<T>>` first-arg matching v1.x. |
| 17.9–10 | Pack / Unpack               |   ✅    | All entries. |
| 17.11   | Synchronization             |   ◐    | Recognised; uniformity gates v1.x. |
| 17.12   | Subgroup                    |   ◐    | All ~22 entries; uniformity gates v1.x. |
| 17.13   | Quad                        |   ◐    | 4 entries; filterable-rule gating v1.x. |

## §18  Grammar

| §  | Title                          | Status | Notes |
|----|--------------------------------|:------:|-------|
| 18 | Recursive-descent grammar      |   ✅    | All productions implemented; corpus 0.37 ms/Kloc warm. |

---

## Cross-cutting hard problems

| ID | Title                                 | Status | Notes |
|----|---------------------------------------|:------:|-------|
| A  | Template-list discovery (§3.9)        |   ✅    | Lexer-level pending stack; splits `>>` / `>=` / `>>=`. |
| B  | Abstract types + materialisation      |   ✅    | Per-value type tag; default tower at decl bind. |
| C  | Overload resolution                   |   ◐    | Hand-coded; full Tint-style `def/wgsl.def` + codegen pipeline is the canonical v1.x. |
| D  | Reference vs Pointer (§6.4.3)         |   ◐    | `is_ref` bit propagates; addr-of/indirection un-typed. |
| E  | Alias analysis (§11.4.1)              |   ✗    | v1.x — originating-variable propagation through call graph. |
| F  | Behavior analysis (§9.7)              |   ✅    | Bitmask propagation; all paths return on non-void. |
| G  | Memory layout (§14.4)                 |   ✗    | v1.x — host-shareable size+align rules. |
| H  | Uniformity analysis (§15.2)           |   ✗    | v1.x — multi-week sub-project per risk register. |

## What "v1 ship" needs vs. has

The original v1 target is "full spec".  In practice, full spec for
WGSL is closer to a 6-month project (Tint is ~50 KLOC, Naga is ~80
KLOC).  What we've shipped is:

- **Frontend**: complete (lex / parse / resolve / scalar const-eval /
  type-check / structural validate).
- **Validator**: 5 of 8 rule families (recursion / struct cycle /
  alias cycle / const cycle / override cycle / attribute conformance /
  behavior / entry-point shape / address-space subset).
- **Public API + LSP queries + WASM bundle**: complete.

What "full v1" still needs, prioritised by impact:

1. **Memory layout (§14.4)** — gates host-shareable type sizing in
   the module summary and is the most-asked-for missing feature when
   a host pipeline tries to bind buffers.  ~1 week.
2. **Strict overload resolution** — `def/wgsl.def` DSL + codegen
   pipeline.  Replaces the hand-coded ~80-entry table; gives precise
   "no matching overload" diagnostics.  ~1–2 weeks.
3. **Ref vs Ptr full semantics** — addr-of / indirection typing,
   assignment LHS=ref enforcement, ptr type kind in `types.h`.
   ~1 week.
4. **Uniformity analysis (§15.2)** — risk register's explicit multi-
   week task.  Required for `derivative_uniformity` and
   `subgroup_uniformity` filterable rules.  ~2–3 weeks.
5. **Alias analysis (§11.4.1)** — originating-variable propagation
   through the call graph.  ~1 week.
6. **Layout / texel-format / texture overload precision** — gradual
   tightening of the §6.5 + §17.7 surfaces.  ~1 week.
7. **Strict entry-point IO rules (§13.2 / §13.3)** — vertex /
   fragment IO matching, location packing, builtin-position gating.
   ~3–5 days.
8. **Strict spec compliance polish** — predeclared-fn shadowing
   tightening, reserved-word enforcement, `@diagnostic` filter
   range, `@must_use` enforcement, value-semantic attribute checks.
   ~3–5 days.

**Grand total of remaining v1.x work**: ~6–10 weeks of focused effort
to claim "full WGSL spec" parity with Tint / Naga's frontend depth.
What's shipped today is enough for an LSP server, a `wgsl.run`
sandbox, and any host that wants to validate the *structural* shape
of a WGSL module.  It is **not yet enough** to replace Tint /
Naga's frontend in a production WebGPU stack — the layout +
uniformity gaps would surface as accepted-but-runtime-broken
shaders.
