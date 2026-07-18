# ML analysis & golden reference suite

| Target | Purpose |
|--------|---------|
| `test_ml_kernel` | pattern, binding, and workgroup hint JSON |
| `test_golden_ref` | embedded golden needles + determinism |
| `test_golden_walk` / `make test-golden` | disk corpus (`cases/*.wgsl` + `*.expect.json`) |

## ML kernel (`wgsl_ml_analyze_json`)

```bash
wgsl ml shader.wgsl
```

Reports:

- `patterns[]` — softmax, attention, layernorm, gelu/silu/relu, matmul, …
- `bindings[]` — group/binding, address space, type, array_len, vec_width
- `shape_hints[]` — per-binding rank/shape; `array<vec4f,8>` reports `[8,4]`
- `shape_flow` — derived attention hints such as `seq_len`, `head_dim`,
  and `softmax_axis`, plus matmul `m`/`n`/`k`/`broadcast`, when fixed
  storage tensors provide enough evidence
- `workgroup_hints` — current size + suggested power-of-two X sweep
- `autotune_protocol` — versioned host protocol for measuring suggested
  `workgroup_x` candidates
- `advice[]` — human-readable tuning notes

## Golden reference (`wgsl_golden_run` / `test_golden_ref`)

Each embedded case is `(WGSL source, entry, default_len, expected JSON needles)`.

The interpreter is the **semantic oracle**: if a GPU backend disagrees on
these needles, either the backend or the fixture is wrong.

Also reuses `tests/12-diff/exec/*.wgsl` fixtures when run from the repo root.

```bash
make test                 # includes test_golden_ref + test_ml_kernel
```

## Disk golden corpus (`cases/`)

File-walk harness: every `cases/*.wgsl` with a sibling `*.expect.json` is
executed via `wgsl_interp` / `wgsl_golden_run`. **Adding a case = two files
only** (no C edit). Corpus size gate: **≥50** green cases.

See [`cases/README.md`](cases/README.md) for the expect schema, numeric
whitelist (`known_numeric_difference` / `abs_tol`), **hand-verified** lists
(batch 1 + batch 2), negative-test invariants, and the **texture skip**
report (interpreter texture execution gap).

```bash
make test-golden          # negative invariants + walk cases/; ≥50 green
```

Guardrail: do **not** blind-capture `interp > expect.json`. Expected
values must be hand-reasoned; on mismatch, report a bug instead of
embedding the wrong dump. Texture goldens are intentionally absent until
the interpreter has a real texture data plane.
