# Fuzzing, property tests & CTS-lite

| Target | How | Notes |
|--------|-----|--------|
| `test_property_format` | `make test` | format∘check metamorphic + session cache |
| `test_fuzz_mutational` | `make test` | deterministic mutational smoke (default 2500 iters) |
| `test_cts_lite` | `make test` | curated accept/reject/any fixtures |
| `fuzz_check` (optional) | `make fuzz` | libFuzzer harness over `wgsl_check` |
| long mutational run | `make fuzz-long` | `FUZZ_ITERS` (default 1e5), no libFuzzer |
| external CTS import | `make import-cts CTS_DIR=…` then `make test-cts-import` | optional |

## Mutational fuzzer (always available)

No special toolchain:

```bash
make test                              # includes 2500-iter smoke
FUZZ_ITERS=100000 make fuzz-long       # longer soak
FUZZ_ITERS=1000000 FUZZ_SEED=42 make fuzz-long
```

Mutations: byte flip, delete, insert, WGSL snippet inject, seed splice.
Each input exercises `wgsl_check`, `wgsl_format`, and `WGSLSession`.

## libFuzzer (optional)

Requires Clang with `-fsanitize=fuzzer,address`:

```bash
make fuzz
./.build/fuzz_check -max_total_time=30 tests/13-fuzz/seeds
./.build/fuzz_check -max_total_time=3600 -rss_limit_mb=2048 tests/13-fuzz/seeds
```

Without a fuzzer-capable toolchain, `make fuzz` prints SKIP (exit 0).
Use `make fuzz-long` instead.

## CTS-lite (in-tree)

`test_cts_lite` embeds curated cases inspired by CTS / real-shader shapes:

- compute entry + builtins
- fragment IO with `@location`
- arrays, control flow, f16, aliases
- expected rejects (type mismatch, missing location, call entry point)

## External CTS / shader tree import

Does **not** download. Point at a local tree of `*.wgsl` files (WebGPU CTS
checkout, Dawn tests, your studio corpus, …):

```bash
# e.g. after cloning https://github.com/gpuweb/cts (if it has .wgsl),
# or any directory of shaders:
make import-cts CTS_DIR=/path/to/shaders
make test-cts-import    # crash-free walk; accept/reject mix is fine
```

Env aliases: `WGSL_CTS_DIR`, `WGSL_CTS_MAX` (default 500 files).

Imported copies live in `.build/cts-import/` (gitignored build tree).

## Seeds

`tests/13-fuzz/seeds/` — tiny well-formed + broken WGSL for libFuzzer and
humans. The mutational harness also embeds its own seed table so `make test`
needs no filesystem walk.
