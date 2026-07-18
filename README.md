# libwgsl

[![CI](https://github.com/toprakdeviren/libwgsl/actions/workflows/ci.yml/badge.svg)](https://github.com/toprakdeviren/libwgsl/actions/workflows/ci.yml)

> A from-scratch C99 library for validating
> [WGSL](https://www.w3.org/TR/WGSL/), extracting shader interfaces, and
> powering editor and analysis tools. Native and WebAssembly builds share the
> same checked semantic core. MSL emission and CPU execution analysis are
> experimental. Spec profile: **WGSL Candidate Recommendation Draft,
> 2026-05-07**.

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

A small, embeddable WGSL frontend and analysis library. Given a source string, libwgsl checks it against the pinned WGSL profile and exposes structured diagnostics, semantic tokens, editor queries, and a JSON module summary with entry points, resources, layouts, overrides, and pipeline information.

The same checked result feeds native applications, WebAssembly clients,
language servers, reflection tooling, and the experimental execution analyzer.

Unsupported or partial experimental behavior is reported explicitly instead of being presented as a successful full result.

## What it isn't

- **A general shader backend.** The MSL emitter is target-focused and
  best-effort; there is no SPIR-V / HLSL / GLSL backend, and Tint/Naga
  remain the right production code generators.
- **A WebGPU runtime.** Pipeline generation, dispatch, and resource
  management belong to the host engine.
- **A multi-file IDE project system.** WGSL has no imports; the LSP and
  session APIs work on one module at a time.  `WGSLSession` caches
  byte-identical checks and tracks changed top-level declarations for
  editor workflows, but cross-file dependency management belongs to
  the host.

## Capabilities

| Pass            | Status | Notes |
|-----------------|:------:|-------|
| Lexer           |   ✅   | Full WGSL §3 token set, template-list discovery (§3.9). |
| Parser          |   ✅   | Pratt expressions + recursive-descent declarations and statements. |
| Type system     |   ✅   | Abstract-type tower (§6.1.2), conversion rank, type interner. |
| Resolver        |   ✅   | Predeclared scope, forward refs, kernel-split corpus convention. |
| Const evaluator |   ✅   | Scalar + vec/mat operators, materialisation, `const_assert`. |
| Type checker    |   ✅   | Generated builtin overload rows, composite + index expressions. |
| Validator       |   ✅   | Recursion / cycle families, attributes, behavior, entry points, address spaces, uniformity, alias analysis. |
| Module summary  |   ✅   | JSON: entry points, resources, structs, overrides; schema in `include/wgsl.h` and `docs/libwgsl.md`. |
| Memory layout   |   ✅   | Struct/resource layout validation and JSON layout summary. |
| Uniformity      |   ✅   | Entry-point and collective-operation uniformity analysis with regression fixtures. |
| Alias analysis  |   ✅   | Root-granularity call-site and module-scope alias checks. |
| MSL emitter      |   ◐   | Target-focused WGSL→MSL text with structured partial/unsupported diagnostics. |
| Interpreter      |   ◐   | N-lane CPU execution, buffer snapshots, races, divergence, and cost JSON. |
| ML analysis      |   ◐   | Compute-kernel pattern and shape hints for workbench use. |

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
make                   # build .build/libwgsl.a
make test              # build + run native unit/API tests
make test-all          # native, CLI, LSP wire, corpus, and differential gates
make test-golden       # run disk-backed interpreter/analysis fixtures
make cli               # build .build/wgsl
make test-cli          # smoke-test the CLI
make example-embedder  # build + run examples/embedder.c
make tsan              # build + run ThreadSanitizer tests (tests/01-tsan/*)
make wasm              # cross-compile via Emscripten → .build/wasm/wgsl_compiler.{js,wasm}
make wasm-size         # write .build/wasm*/size.txt
make wasm-test         # Node smoke against the wasm bundle
make wasm-simd         # build + test the SIMD128 wasm bundle
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
  docs/libwgsl.md        Consolidated technical reference
  docs/                  Additional technical documentation
```

## Project maturity

The public C API is currently version **0.2.0**. Validation, diagnostics,
reflection, editor queries, sessions, generated ABI bindings, and WebAssembly builds are supported surfaces. The MSL emitter, optimizer, CPU execution model, and ML-oriented analysis remain experimental and return explicit partial/unsupported results where applicable.

The CI badge reflects the current branch state. For a local checkout, run the relevant commands from the [Build](#build) section before relying on a surface for release or integration work.

This repository intentionally carries the core C library, native CLI,
examples, and tests.  Product surfaces such as the browser sandbox and
editor extension are kept out of this package.

## License

MIT.  See [`LICENSE`](LICENSE).  Vendored Unicode tables are MIT-
licensed and attributed in `vendor/unicode/LICENSE`.
