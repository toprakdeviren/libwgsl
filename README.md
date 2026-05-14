# libwgsl

[![CI](https://github.com/toprakdeviren/libwgsl/actions/workflows/ci.yml/badge.svg)](https://github.com/toprakdeviren/libwgsl/actions/workflows/ci.yml)

> A from-scratch C99 frontend for the
> [W3C WebGPU Shading Language](https://www.w3.org/TR/WGSL/) — lex,
> parse, resolve, const-eval, type-check, validate.  No dependencies
> outside a vendored Unicode tables archive.  Spec target: **WGSL
> Recommendation**, Candidate Recommendation Draft of 2026-05-07.

```c
#include <wgsl.h>

WGSLResult *r = wgsl_check(src);
if (wgsl_ok(r)) {
    use(wgsl_module_json(r));   // entry points, bindings, layout
} else {
    fprintf(stderr, "%s\n", wgsl_error(r));
}
wgsl_free(r);
```

## What it is

A trustworthy WGSL frontend.  Given a source string, libwgsl returns
an authoritative "is-this-spec-conformant" verdict, an LSP-grade
diagnostic stream, semantic tokens for syntax highlighters, and a
JSON module summary (entry points, resources, struct layouts,
override IDs).  It does not generate code — Tint and Naga already do
that, and the validator/AST/diagnostic state we produce flows straight
into editors and language servers.

## What it isn't

- **A backend.** No SPIR-V / MSL / HLSL emission.
- **A WebGPU runtime.** Pipeline generation, dispatch, and resource
  management belong to the host engine.
- **A formatter.** Reasonable add-on for v1.x; not on the v1 path.
- **An incremental compiler.** Every `wgsl_check` runs the full
  pipeline.  The corpus benchmark says ~0.37 ms / KLoc warm on an
  M-series CPU, so for the sizes WGSL targets (typically <2 KLoc per
  module) this is comfortably below the 16 ms frame budget anyway.

## Capabilities

| Pass            | Status | Notes |
|-----------------|:------:|-------|
| Lexer           |   ✅   | Full WGSL §3 token set, template-list discovery (§3.9). |
| Parser          |   ✅   | Pratt expressions + recursive-descent decls/stmts; corpus parses zero-error. |
| Type system     |   ✅   | Abstract-type tower (§6.1.2), conversion rank, type interner. |
| Resolver        |   ✅   | Predeclared scope, forward refs, kernel-split corpus convention. |
| Const evaluator |   ✅   | Scalar + vec/mat operators, materialisation, `const_assert`. |
| Type checker    |   ✅   | Generated builtin overload rows, composite + index expressions. |
| Validator       |   ✅   | Recursion / cycle families, attributes, behavior, entry points, address spaces, uniformity, alias analysis. |
| Module summary  |   ✅   | JSON: entry points, resources, structs, overrides; schema in `docs/MODULE_JSON.md`. |
| Memory layout   |   ✅   | Struct/resource layout validation and JSON layout summary. |
| Uniformity      |   ✅   | CTS-clean validator behavior for the current WGSL compile corpus. |
| Alias analysis  |   ✅   | Root-granularity call-site and module-scope alias checks. |

The current external conformance signal is WebGPU CTS WGSL
shader-module validation parity: 565,599 / 565,599 compile cases
matched in the development harness.

## Public C API

The whole surface lives in [`include/wgsl.h`](include/wgsl.h) and is
audience-tagged: process lifecycle, check, result, diagnostics,
semantic tokens, hover/navigation, project (`wgslconfig.toml`).  All
result strings are owned by the `WGSLResult` and live until
`wgsl_free`; every accessor handles `NULL` gracefully.

The library is one-shot: you give it source, you get a result back,
you free.  No global mutable state apart from one idempotent
process-wide Unicode-table init.

### `wgslconfig.toml` preamble injection

WGSL has no `#include`.  Engines and editors typically work around
this by string-concatenating helpers in front of every shader they
compile.  libwgsl ships a tiny TOML reader + glob matcher so that
convention can live in one place:

```toml
# wgslconfig.toml
[preamble]
files            = ["00_shared.wgsl", "00_infrastructure.wgsl"]
auto_inject_into = ["*.wgsl"]
```

`wgsl_project_open_from_string` parses the config; `wgsl_project_match`
returns the preamble file list for a given target path; the embedder
reads each file and feeds the concatenation to
`wgsl_check_with_preamble`.  A native-only convenience
`wgsl_check_in_project` does the file IO itself; it's compiled out
under `-DWGSL_NO_FS` for the WASM build.

The matcher excludes self-injection — a preamble file matching its
own glob does not get itself prepended.

## Build

```sh
make            # build .build/libwgsl.a
make test       # build + run native unit/API tests
make cli        # build .build/wgsl
make test-cli   # smoke-test the CLI
make example-embedder # build + run examples/embedder.c
make tsan       # build + run TSan smoke (tests/01-tsan/*)
make wasm       # cross-compile via Emscripten → .build/wasm/wgsl_compiler.{js,wasm}
make wasm-test  # Node smoke against the wasm bundle
```

The library is C99, single static archive, no system dependencies
beyond `libc`.  Vendored archive: Unicode 17.0.0 tables (MIT-
licensed, see `vendor/unicode/LICENSE`).

## Layout

```
libwgsl/
  include/wgsl.h         Public API (single header, audience-tagged)
  src/
    arena.c              Arena allocator
    source.c             Source + line index
    diag.c               Diagnostics
    utf8.c               UTF-8 / XID helpers
    token.c · lexer.c    Lex + template-list discovery
    parser.c             Pratt + recursive-descent
    types.c              TypeInfo + interner
    resolver.c           Name resolution
    consteval.c          Const evaluator
    layout.c             Layout helpers
    check/               Type checker / overload resolver
    validate/            Validator passes
    glob.c · toml.c      Project preamble layer (no FS)
    project.c
    wgsl.c               Public-API plumbing
  def/wgsl.def           Builtin overload source table
  tools/                 Builtin generator and native CLI
  vendor/unicode/        Pre-built Unicode 17.0.0 tables
  tests/                 One executable per `tests/NN-name/test_*.c`
  examples/              Small smoke/embedder examples
  docs/                  PLAN, DESIGN, BENCH, SPEC-MAP, SPEC-GAPS, …
```

## Status

The public API surface is in `include/wgsl.h` v0.2.0.  Current local
verification in the development tree:

- `make test`: 36 / 36 passed
- WebGPU CTS WGSL shader-module validation: 565,599 / 565,599 compile
  cases matched

This repository intentionally carries the core C library, native CLI,
examples, and tests.  Product surfaces such as the browser sandbox and
editor extension are kept out of this package.

## License

MIT.  See [`LICENSE`](LICENSE).  Vendored Unicode tables are MIT-
licensed and attributed in `vendor/unicode/LICENSE`.
