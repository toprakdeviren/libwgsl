# wgsl — Roadmap

> **Status**: Phase 0 opening (2026-05-07). Repo skeleton in. Phase 0
> closes with a WGSL-spec coverage map + reference-implementation
> structural review + a frozen public API surface (`wgsl.h` v1).

## Targets

Three product surfaces, one library:

1. **`libwgsl.a` / `wgsl.h`** — C API: source-in, AST + diagnostics +
   semantic tokens + validator verdict-out.
2. **LSP integration** — hover · go-to-def · semantic tokens · diagnostics.
   The C library is the engine; the LSP server is a thin wrapper around it.
3. **`wgsl.run`** — browser sandbox. Monaco editor (WGSL mode) + WebGPU
   live preview + share-via-URL. WASM build of the C library + a minimal
   JS shell.

Spec target: **W3C WGSL Recommendation** (full).

## Why this scope is hard but bounded

WGSL is a ~250-page spec with several non-obvious passes:

- **Template-list discovery** — the `<` / `>` disambiguation pre-pass
  (spec §10.3 *Template Lists*).
- **Abstract-type materialization** — `AbstractInt` / `AbstractFloat`
  feasibility tower across const-expressions.
- **Address-space inference and conformance** — storage / uniform /
  workgroup / private / function rules.
- **Uniformity analysis** — control-flow + value-flow combined; its
  own multi-week pass.
- **Builtin overload resolution** with abstract-type generics
  (the only generics WGSL has).
- **Structural validity** — no recursion, no alias / const / override
  cycles, no struct cycles.
- **Entry-point I/O shape rules** — vertex / fragment / compute.

The size is bounded because:

- No user-defined generics (only builtins are generic).
- No closures, no modules, no exceptions, no GC.
- Tight, finite type system (`f16 / f32 / i32 / u32 / bool` ×
  scalar / vector / matrix).
- Module scope is flat.
- Spec is precise and machine-checkable; reference implementations
  (Tint, Naga) define what "done" looks like.

## Phases

| #  | Goal | Done when… | Est. |
|----|------|-----------|------|
| 0a | Repo skeleton, README, PLAN, `.gitignore` | ✓ done | done |
| **0b** | **Spec-coverage map** — section-by-section verdict against W3C WGSL Rec; mark v1-in / v1-out / hard-cases | ✓ done — `docs/SPEC-MAP.md` shipped (18/18 §, 27 keywords, 17 attributes, ~130 builtins, 8 cross-cutting hard problems flagged) | done |
| **0c** | **Reference-impl pass review** — Tint + Naga module split; record what passes they have, in what order, with what data shapes | ✓ done — `docs/REF-REVIEW.md` shipped (3 big calls locked: Tint-style decorated-AST resolver, Tint-style `wgsl.def` DSL for builtins, Tint-style directed dependency graph for uniformity) | done |
| 0d | API freeze gate — public `wgsl.h` shape, return types, error model | ✓ done — [`include/wgsl.h`](../include/wgsl.h) v1 locked + [`docs/DESIGN.md`](DESIGN.md) rev 1 | done |
| 1 | Source / Token / AST / TypeInfo kernel structs + arena + UTF-8 / XID. **Sub-deliverables**: (a) `sizeof(WGSLNode)` gate; (b) TSan smoke (two threads, two sessions); (c) arena-churn baseline | ✓ done. (a) sizeof=48, align=8, 51 kinds · (b) TSan zero races on 4×10K × 4 arenas · (c) ~6 ns/alloc steady-state on M-arm64-1 (`docs/BENCH.md`). Modules in: **arena · source · span · diag · utf8 · ast (header) · token (header)**. 6 test suites, 6 PASS. | 1 wk |
| 2 | Lexer — id / number (with abstract-type tagging) / operators / attributes / comments / **template-list discovery pre-pass** (spec §10.3) | tokens emitted for the WebGPU "hello triangle" + a deeply-templated builtin-call sample | 1.5–2 wk |
| 3 | Parser — Pratt expressions + recursive-descent decls / stmts; attribute parsing; recovery for diagnostics | hello-triangle + 10 spec-doc examples parse + AST round-trip dump | 2 wk |
| **3.5** | **Phase 1 benchmark gate** — synthetic 1Kloc WGSL parse-time. Target: ≤ 2 ms cold cache (Naga's order of magnitude). | `docs/BENCH.md` with measured ms-per-1Kloc figures | 1 day |
| 4 | TypeInfo kinds + abstract types + conversion rank + type arena | type-resolve unit tests on the full numeric tower | 1 wk |
| 5 | Name resolver + module scope + predeclared identifiers + forward-ref handling | resolve unit tests across module scope, function scope, block scope | 1 wk |
| 6 | Const evaluator — const-expressions + override-expressions + materialization | const-eval unit tests covering the abstract → concrete tower | 1–2 wk |
| 7 | Type checker / overload resolver — builtin overload resolution with abstract generics; assignability; ref materialization | type-check unit tests including all builtin shapes | 2 wk |
| 8 | **Validator** — recursion + alias-cycle + entry-point + address-space + uniformity + attribute conformance | runs the WebGPU CTS WGSL validation subset green | 3–4 wk |
| 9 | LSP integration glue — semantic tokens, hover types, go-to-def, diagnostics with LSP-compatible spans | `wgsl-lsp` serves a VSCode extension shell | 1.5 wk |
| 10 | WASM build (Emscripten) — `make wasm` produces a JS bundle | `wgsl_compiler.{js,wasm}` boots in browser | 3–4 days |
| 11 | **`wgsl.run` sandbox** — monaco editor + WebGPU live preview + share-via-URL | hello-triangle live in a deployed page | 2 wk |
| 12 | **Spec-conformance push** — chase SPEC-MAP to 100% v1-in green | full v1 validator parity with Tint / Naga | open |
| —  | **v1 ship target — ~16–20 weeks** | three product surfaces work end-to-end | |

## Phase 0 deliverables — to be produced

- `docs/SPEC-MAP.md` — per-§ status of W3C WGSL Recommendation (v1-in /
  v1-out / hard-cases per section).
- `docs/REF-REVIEW.md` — Tint + Naga module split, pass list, data-shape review.
- `docs/DESIGN.md` rev 1 — public API + internal pass structure.
- `include/wgsl.h` v1 (locked) — public API surface, audience-tagged
  (Consumer / Backend / Primitives), msf-style.

## Architecture sketch (placeholder — finalized in 0d)

```
WGSL source
   │
   ▼
┌────────┐  ┌────────┐  ┌──────────┐  ┌─────────┐  ┌───────────┐  ┌───────────┐
│ Lexer  │→ │ Parser │→ │ Resolver │→ │ ConstEv │→ │ TypeCheck │→ │ Validator │
│ token  │  │  AST   │  │  scope   │  │ values  │  │ TypeInfo  │  │  verdict  │
└────────┘  └────────┘  └──────────┘  └─────────┘  └───────────┘  └───────────┘
                                                                       │
                                                                       ▼
                                                                   diagnostics
                                                                   semantic tokens
                                                                   module summary
```

Validator does NOT modify the AST; it produces a verdict + diagnostics.
LSP integration consumes intermediate state (resolver scope, type-check
types, parser tree) without re-running passes.

## Project shape (planned)

```
wgsl/
  README.md
  Makefile
  include/
    wgsl.h            Public API (single header, audience-tagged)
  src/
    arena.c           Phase 1
    source.c          Phase 1
    span.c            Phase 1
    diag.c            Phase 1
    utf8.c            Phase 1 — wraps vendor/unicode
    token.c           Phase 1
    lexer.c           Phase 2
    parser/           Phase 3 — core.c · expr.c · stmt.c · decl.c · helpers.c
    sema/             Phase 4–7 — types.c · resolver.c · consteval.c · check.c
                                  · overloads.gen.c (generated from def/wgsl.def)
    validate/         Phase 8 — recursion.c · uniformity.c · address.c · attrs.c
                                · entry.c · layout.c · behavior.c · alias.c
    lsp_glue.c        Phase 9
    wasm_api.c        Phase 10
  def/
    wgsl.def          Phase 7 — Tint-style builtin DSL (source of truth)
  tools/
    gen-overloads.c   Phase 7 — codegens overloads.gen.c from wgsl.def
  tests/              one-suite-per-phase, named like the phase number
  docs/
    PLAN.md, DESIGN.md, SPEC-MAP.md, REF-REVIEW.md, BENCH.md
  examples/           hello-triangle.wgsl, sky.wgsl, …
  vendor/
    unicode/          ✓ libunicode.{native,wasm}.a + headers (Unicode 17.0.0)
```

## Naming convention

Public C symbols: `wgsl_*`. Header: `wgsl.h`. Library: `libwgsl.a`.
Mirror of the `metal/` project's `msl_*` style — same pattern, no code share.

## Deliberately NOT in scope (v1)

- Code generation (WGSL → SPIR-V / MSL / HLSL). Validator and AST only.
- Source formatter. Easy add post-v1; not on the critical path.
- Bidirectional MSL ↔ WGSL coupling with `metal/`. The two projects
  share *patterns*, not code, in v1. If a `metal/` reverse path is
  desired later, it is post-v1 and lives behind a translation shim.
- Incremental / recompile-on-edit caching. v1 is one-shot per call.
- Shader debugger / step-through.

## Risk register

| Phase | Trigger | Reaction |
|-------|---------|----------|
| 2 | Template-list discovery (spec §10.3) is more entangled with parsing than the spec text suggests | Allow a controlled lexer / parser feedback loop instead of a strict pre-pass; document the deviation in `DESIGN.md` |
| 3.5 | Parse > 4 ms / 1Kloc cold cache | Profile dispatch hot path before Phase 4; redesign before continuing |
| 6 | Abstract-type materialization rules differ between WGSL spec and Tint / Naga in practice | Pick spec; document Tint / Naga deviations as known minor incompatibilities |
| 6 | Const-evaluator scope blows past 1–2 wk budget — Naga's `constant_evaluator.rs` is 194 KB | **New (from REF-REVIEW).** Cap v1 const-eval to what the spec demands at shader-creation time (`const_assert`, module-init, override-expr). Postpone full const-fn graph evaluation to v1.x. Document deviation in `DESIGN.md`. |
| 7 | Builtin overload table grows beyond hand-coded reach (~130 fns × N overloads) | **New (from REF-REVIEW).** Adopt Tint-style `def/wgsl.def` DSL + codegen → `src/sema/overloads.gen.c`. Source-of-truth is the `.def`; never hand-edit the `.gen.c`. |
| 8 | Uniformity analysis turns into its own multi-week sub-project | Acceptable — it is. Budget 2 weeks of Phase 8 explicitly for it. |
| 11 | WebGPU support gap on user's browsers | Fallback to a canvas-2D placeholder; do not block sandbox v1 ship on cross-browser parity |
| All | Project is extracted to its own git repo mid-stream | Keep all `wgsl/` work self-contained — no cross-references to `kavak/` or `miniswift/` source. Patterns may be borrowed; code may not. |

## Where we are

| Phase | Status |
|-------|--------|
| 0a — repo bones | ✓ done |
| 0b — spec coverage map | ✓ done — `docs/SPEC-MAP.md` |
| 0c — reference-impl review | ✓ done — `docs/REF-REVIEW.md` |
| 0d — API freeze | ✓ done — `include/wgsl.h` v1 + `docs/DESIGN.md` |
| **Phase 0** | **✓ closed — Phase 1 cleared to start** |
| 1 — Layer 0 kernel | ✓ done — arena · source · span · diag · utf8 · ast.h · token.h. 6/6 tests, TSan clean, ~6 ns/alloc |
| 2 — Lexer kernel | ✓ done. **Round A**: blankspace · comments (nested) · keywords (27) · idents (ASCII + non-ASCII XID) · numeric literals (all suffixes, leading-zero rule) · operators (max-munch through `>>=`/`<<=`) · punctuation. **Round B**: §3.9 template-list discovery on token stream — pending stack with depth tracking, splits `>>` / `>=` / `>>=` when leading `>` closes a pending template, second `>` of `>>` may close another (recursive close). 10/10 tests, TSan clean, hello-triangle parses to template-tagged stream. |
| 3 — Pratt parser + RD decl/stmt scaffold | ✓ done — Round A · Round B Iter 1 (fn-scope statements: if/while/for/loop+continuing+break_if/switch/break/continue/discard/var/const/const_assert + 11 assignment forms) · Iter 2 (module-scope: struct/alias/const/override/var w. `<addr_space, access>`/const_assert + directives enable/requires/diagnostic) · Iter 3 (corpus smoke: 17 ML/transformer shaders from examples/shaders/, 2971 LOC, 45 433 tokens, 291 top-level decls, 54 fns parse zero-error). **17/17 tests green, TSan clean.** |
| 3.5 — Parser benchmark gate | ✓ done — corpus 2 971 LOC / 45 433 tokens parses in **0.373 ms/Kloc warm** (gate ≤ 4 ms/Kloc, goal ≤ 2 ms/Kloc). Throughput ≈ 41 M tokens/sec. See `docs/BENCH.md`. |
| 4 — TypeInfo + abstract types + conversion rank | ✓ done — `WGSLTypeInfo` 24 B locked, 14 kinds, type store with 8 pre-interned scalars + on-demand interning for vec/mat/atomic/array, full 9-row §6.1.2 conversion-rank table (incl. element-wise inheritance for vec/mat/array), concretization (Abstract* → i32/f32 + recursive), formatter producing WGSL-shape strings. **19/19 tests green, TSan clean.** |
| 5 — Resolver + scope + predeclared identifiers | ✓ done — predeclared scope (5 scalars · 35 typegens · 30 type aliases · 7 addr_spaces · 3 access_modes · ~130 builtin fns), 2-pass module walk for forward refs, fn-block scope chain, identifier resolution stored in AST `payload[2]`. Member-access RHS deferred to TC; attribute-arg idents silent on miss. **Iter A+B**: 17 unit cases (predecl, fwd-ref, redecl, type-shadow guard, params/let/var, block + for-loop scope leak guards, undecl, member-rhs, templated-ident, struct). **Iter C**: 43/43 corpus sections across 17 ML shaders resolve clean (engine-style `_shared.wgsl` prepend + `// --- KERNEL: <name> ---` split). 21/21 tests green, TSan clean. |
| 6 — Const evaluator | ✓ done (v1 cap per risk register). **Iter A**: `WGSLValue` 32 B (bool / int64 / double / vec / mat slots), scalar literal parse (decimal · hex · suffixes), unary `- ! ~`, binary arith / comparison / logical / bitwise, §6.1.2 pair-promotion + §6.2.1 materialisation (target-driven + default tower). **Iter B**: module + fn-scope walker for `DECL_CONST` / `DECL_CONST_ASSERT`, symbol→value cache, EXPR_IDENT lookup, "is not a constant" diagnostic. **Iter C**: 43/43 corpus sections × 307 const decls fold clean. **v1 deferrals** (DESIGN.md): vec/mat constructor folding, member-access in const-expr, override default-value eval, user-defined `const fn` graph. 24/24 tests green, TSan clean. |
| 7 — Type checker / overload resolver | ✓ done (v1 surface). **Iter A**: type-spec resolution (scalar · vec · mat · atomic · array<T,N> with N from const-eval), module decl typing (const · let · var · override · alias · struct), fn param/return typing, scalar literal/ident/paren/unary/binary-numeric. **Iter B**: composite expressions — vec/mat/scalar constructors, vec swizzle (.x / .xyz / .rgba), struct field access, EXPR_INDEX (vec/mat/array), full vec×vec / vec×scalar / mat×scalar / mat×mat / mat×vec / vec×mat operator surface, user-fn return type. **Iter C**: hand-coded builtin overload table (~80 entries covering §17.2–17.13, incl. `bitcast<T>`, atomics, subgroup, texture, barriers); 43/43 corpus sections × **8 560 typed expressions** clean. **v1 deferrals** (DESIGN.md): Tint-style `def/wgsl.def` DSL + codegen → `overloads.gen.c`, EXPR_ADDR_OF / INDIRECTION + full ref/ptr semantics, mixed-mode vec ctors (vec3 from vec2+scalar), strict ctor argc validation, struct field validation in member access (currently permissive — Phase 8 will own). 27/27 tests green, TSan clean. |
| 8 — Validator | ✓ done (v1 surface). **Iter A**: 5 cycle-detection rule families (recursion · struct cycle · alias cycle · const cycle · override cycle) via shared DFS-coloring engine over per-decl-kind edge enumerators. **Iter B**: attribute conformance (17-entry placement + arity table covering §12.1–12.15) + §9.7 behavior-set analysis (Next/Break/Continue/Return propagation; non-void fn must return on all paths; switch absorbs Break into Next; loops always reach Next). **Iter C**: entry-point shape (§13: at-most-one stage attr per fn; `@compute` requires `@workgroup_size`; `@workgroup_size` only on `@compute`) + address-space conformance subset (§14.3: `var<uniform>` / `var<storage>` require `@group` + `@binding`). 43/43 corpus sections green, 35 unit cases. **v1 deferrals** (DESIGN.md): host-shareable layout (§14.4), uniformity analysis (§15.2 — explicit multi-week task per risk register), alias analysis (§11.4.1 — originating-variable propagation through call graph), strict entry-point IO type rules. 29/29 tests green, TSan clean. |
| 9 — LSP integration glue | ✓ done (public C API surface complete). **Iter A**: `WGSLResult` lifecycle behind `wgsl_check` / `wgsl_check_n` / `wgsl_free`; verdict + LSP-shape diagnostics (`wgsl_diagnostic*`); raw `wgsl_lex` with `WGSLLexTokenKind` classification (KEYWORD / TYPE / OPERATOR / PUNCT / TEMPLATE_DELIM / ATTRIBUTE / DIRECTIVE / COMMENT / WHITESPACE / NUMBER / IDENTIFIER); init/version glue. **Iter B**: `wgsl_semantic_tokens` overlays resolver bindings on raw IDENT tokens, upgrading them to STRUCT_NAME / FUNCTION_NAME / PARAMETER / FIELD / VAR / LET / CONST / OVERRIDE / TYPE_ALIAS / TYPE / BUILTIN_FUNC / BUILTIN_VALUE; `wgsl_hover_at` returns the resolved type-as-string for the symbol under a byte offset (use site or decl name); `wgsl_definition_at` returns the decl name span. **Iter C**: `wgsl_module_json` — JSON module summary (entry_points with workgroup_size const-resolution, resources with @group/@binding/address_space/access/type, structs with member types, overrides with @id), schema documented in `docs/MODULE_JSON.md`. Lazy-cached; idempotent. **v1.x deferrals** (DESIGN.md): VSCode extension shell + `wgsl-lsp` server binary; per-resource layout in JSON; vertex/fragment IO records. 32/32 tests green, TSan clean. |
| 10 — WASM build | ✓ done. `make wasm` drives Emscripten 4.0+ over the same `LIB_SRCS` as the native lib, links the prebuilt `vendor/unicode/libunicode.wasm.a`, and emits `wgsl_compiler.{js,wasm}` (14 KB JS + 178 KB wasm, MODULARIZE + ES-CommonJS, `ENVIRONMENT=node,web,worker`, ALLOW_MEMORY_GROWTH).  The full public C API surface is exported (19 fns + `_malloc`/`_free`).  Static-assert layout gates updated to be pointer-size-aware (`WGSLTypeInfo` is 24 B / 8-aligned on LP64, 20 B / 4-aligned on wasm32 — both intentional and exercised in CI).  `make wasm-test` runs a Node smoke (24 cwrap'd checks: spec_pin / unicode_version / empty / valid compute / parse-error / resolve-error / module JSON shape / `linear.wgsl` corpus parse). 33/33 tests overall green, TSan clean. |
| 11 — `wgsl.run` sandbox | ⏳ next |

## Open questions for Phase 0

1. ~~UTF-8 decoder source.~~ ✓ Locked — vendor the Decoder library
   (MIT, Unicode 17.0.0). Both archives shipped in
   [`vendor/unicode/`](../vendor/unicode/): `libunicode.native.a`
   (arm64) for native gates, `libunicode.wasm.a` for Phase 10. Headers
   pinned to `/Users/2n/dev/personal/decoder/include/`.
2. ~~Diagnostics format.~~ ✓ Locked (per SPEC-MAP) — match `metal/`'s
   LSP-style spans (1-based line/col + end-line/end-col + severity).
3. ~~WGSL spec source pin.~~ ✓ Locked — W3C Candidate Recommendation
   Draft, 7 May 2026 (the HTML in [`docs/`](.)).
