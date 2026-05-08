# BENCH — wgsl performance numbers

> Numbers carry a date and a hardware tag.  When a number changes
> meaningfully (good or bad), append a row; never overwrite history.
> Phase 3.5 will add the parser-cold-cache figure that the gate triggers
> on; this file already exists so Phase 1's (c) baseline has a home.

## Hardware tags

| Tag | Machine |
|-----|---------|
| **M-arm64-1** | Apple Silicon arm64 macOS, native build, `-O2` |

---

## Phase 1 (c) — arena-churn baseline

Workload (`tests/01-arena-churn/test_arena_churn.c`): 1000 iterations
of (one synthetic fn body of 50 statements, each with a depth-5
expression tree) + (one depth-30 binop chain). Total **993 000 nodes
+ child-arrays** in one arena, then destroy.

| Date | Tag | Allocs | Wall (ms) | ns/alloc | Throughput | Notes |
|------|-----|-------:|----------:|---------:|-----------:|-------|
| 2026-05-07 | M-arm64-1 | 993 000 | 12.8 (cold) → 6.1 (warm) | 12.9 → 6.1 | 2.9 GB/s → 6.1 GB/s | First run cold cache; steady-state ≤ 5 runs in. |

**Verdict:** ≤ 7 ns/alloc steady state.  Kernel hot path is not a
bottleneck; **per-kind pooling not warranted before Phase 5**.

Re-run threshold: revisit if any phase-test regresses past
**12 ns/alloc steady state** on M-arm64-1, or if a profile shows
arena-alloc on the top-3 sample list.

## Phase 3.5 — parser benchmark gate

Workload: real ML/transformer shader corpus under
[`examples/shaders/`](../examples/shaders/) — 17 files,
**2 971 LOC**, **45 433 tokens**.  Per-shader: slurp once, then run
`wgsl_tokenize` + `wgsl_parse` + arena destroy through 30 iterations
with a fresh arena every time; report the minimum wall as **warm**.
Iteration 1 is reported separately as **cold-ish** (file already in
page cache; arena freshly malloc'd).

Gate: aggregate **≤ 4 ms / Kloc** (PLAN.md risk row Phase 3.5).
Goal: ≤ 2 ms / Kloc (Naga's order of magnitude).

| Date | Tag | Workload | LOC | Tokens | Warm ms | Cold ms | Warm ms/Kloc | Cold ms/Kloc |
|------|-----|----------|----:|-------:|--------:|--------:|-------------:|-------------:|
| 2026-05-08 | M-arm64-1 | shader corpus (17 files) | 2 971 | 45 433 | **1.107** | 1.537 | **0.373** | 0.517 |

Per-shader detail (warm-min over 30 iters):

| Shader | LOC | Tokens | Warm ms | Cold ms | ms/Kloc |
|--------|----:|-------:|--------:|--------:|--------:|
| `_shared.wgsl`            |  92 |   579 | 0.0160 | 0.0340 | 0.174 |
| `activation.wgsl`         |  52 |   534 | 0.0150 | 0.0230 | 0.288 |
| `attention.wgsl`          | 296 | 3 029 | 0.0730 | 0.1110 | 0.247 |
| `backward_attention.wgsl` | 584 | 6 533 | 0.1860 | 0.2680 | 0.318 |
| `backward_ffn.wgsl`       |  90 |   959 | 0.0260 | 0.0390 | 0.289 |
| `cast.wgsl`               |  55 |   385 | 0.0110 | 0.0200 | 0.200 |
| `common.wgsl`             |  97 | 1 086 | 0.0330 | 0.0580 | 0.340 |
| `embedding.wgsl`          |  78 |   802 | 0.0200 | 0.0290 | 0.256 |
| `embedding_w16.wgsl`      |  39 |   340 | 0.0120 | 0.0380 | 0.308 |
| `linear.wgsl`             | 598 |15 868 | 0.3410 | 0.4250 | 0.570 |
| `linear_w16.wgsl`         | 293 | 7 933 | 0.1880 | 0.2270 | 0.642 |
| `loss.wgsl`               | 143 | 1 272 | 0.0300 | 0.0470 | 0.210 |
| `norm.wgsl`               | 104 | 1 222 | 0.0300 | 0.0380 | 0.288 |
| `norm_w16.wgsl`           |  41 |   502 | 0.0130 | 0.0230 | 0.317 |
| `optimizer.wgsl`          | 133 | 1 456 | 0.0380 | 0.0510 | 0.286 |
| `optimizer_8bit.wgsl`     | 171 | 1 836 | 0.0430 | 0.0610 | 0.251 |
| `rope.wgsl`               | 105 | 1 097 | 0.0320 | 0.0450 | 0.305 |

**Verdict:** gate cleared by **~11×**; goal cleared by **~5×**.
Throughput ≈ **41 M tokens / sec** on M-arm64-1.

Outliers worth noting (still well under gate):

- `linear.wgsl` and `linear_w16.wgsl` show the highest ms/Kloc.
  They have **27 tokens / LOC** (vs the ~10 typical for the corpus),
  so their LOC count understates the actual parser work.  Token-
  normalized, all 17 files cluster around **24 ns / token**.

Re-run threshold: revisit the dispatch hot path if any phase commit
pushes warm aggregate **past 1.0 ms / Kloc** on M-arm64-1, or if any
single shader exceeds **1.5 ms / Kloc**.
