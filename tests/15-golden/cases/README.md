# Golden reference cases (disk corpus)

Each case is a pair under this directory:

| File | Role |
|------|------|
| `NAME.wgsl` | Compute shader writing storage buffers (fixed inputs) |
| `NAME.expect.json` | Hand-reasoned expected buffers + run params |

**Adding a case does not require editing C.** Drop the two files and
`make test-golden` picks them up.

## expect.json shape

```json
{
  "entry": "main",
  "len": 4,
  "ok": true,
  "buffers": [
    { "name": "o", "values": ["1", "2", "3", "4"] }
  ]
}
```

Optional **known numeric difference** (f16 / FP imprecision whitelist):

```json
{
  "known_numeric_difference": true,
  "abs_tol": 0.001,
  "buffers": [ { "name": "o", "values": ["0.3", "0.333333"] } ]
}
```

When `known_numeric_difference` is true (or `abs_tol` / `tolerance` > 0),
the harness compares buffer values as doubles within `abs_tol`
(default `1e-3`) instead of exact JSON-needle match. Exact-match remains
the default for integer / dyadic cases.

## Guardrail — do not blind-capture

Golden expects are a **correctness oracle**, not a dump of “whatever
interp printed.”

1. Prefer **hand-computable** outputs (integer arithmetic, exact dyadic floats).
2. Compute the expected values on paper (or mental arithmetic).
3. Confirm with `./.build/wgsl interp --entry … --len … file.wgsl`.
4. If interp disagrees with the hand result, **report a bug** — do not
   paste the wrong dump into `.expect.json`.

## Negative-test invariant (harness)

`test_golden_walk` embeds three self-checks before the file walk:

| Check | Expected harness behavior |
|-------|---------------------------|
| Exact expect `999` vs real `42` | **FAIL** the case (reject) |
| Numeric expect `2.0` vs real `1.0`, `abs_tol=0.01` | **FAIL** the case |
| Numeric expect `1.005` vs real `1.0`, `abs_tol=0.01` | **PASS** the case |

If any invariant breaks, `make test-golden` fails even when all disk
cases are green.

## Texture cases — **skipped** (interp gap)

**Checked batch 2:** `textureSample` / `textureLoad` are **not** real
texture execution.

| Probe | Result |
|-------|--------|
| `textureSample(t, s, uv)` | returns `vec(0,0,0,0)`, `"ok":false` |
| `textureLoad(t, coords, level)` | returns `vec(0,0,0,0)`, `"ok":false` |
| Texture binding dump | `"type":"?"`, `"values":[]` (no texel backing store) |

This matches `docs/libwgsl.md`: texture types are predeclared
and `textureSample` is **best-effort `vec4<f32>`**, not a seeded image
data plane. There is **no** texture path in `src/interp/` (no sample/load
implementation beyond a zero vector).

**Decision:** no texture golden cases in this corpus. Embedding `0,0,0,0`
as an “oracle” would launder a stub into a false positive.

> **Report: interp texture execution gap.**
> Golden texture suite blocked until the interpreter can load real
> texel data (or an explicit seed API) and return non-stub results
> with `"ok":true`.

## Hand-verified cases

### Batch 1 (≥8)

| Case | Hand result | Reasoning |
|------|-------------|-----------|
| `store_42` | `[42]` | Constant store. |
| `int_arith` | `[35, 61, 12, 2]` | `(7+3)*4-5=35`; `(0xF<<2)\|1=61`; `abs(-12)=12`; `100%7=2`. |
| `mat2x2_mul` | `[23, 34, 31, 46]` | Column-major `A*B`: col0=`A*[5,6]=[23,34]`, col1=`A*[7,8]=[31,46]`. |
| `vec_swizzle` | `[4, 3, 2, 1]` | `vec4i(1,2,3,4).wzyx`. |
| `for_sum` | `[6]` | `0+1+2+3`. |
| `nested_loop` | `[9]` | Outer×inner `3×3` increment. |
| `atomic_add` | `[16]` | 16 lanes each `atomicAdd(1)`. |
| `subgroup_add` | eight × `28` | `0+1+…+7=28` broadcast to every lane. |

**hand-verified: store_42, int_arith, mat2x2_mul, vec_swizzle, for_sum, nested_loop, atomic_add, subgroup_add**

### Batch 2 (new ≥8)

| Case | Hand result | Reasoning |
|------|-------------|-----------|
| `multi_buffer` | `a=[10,20]`, `b=[30,200]` | `10+20`, `10*20`. |
| `atomic_cas_loop` | `c=7`, `o=[7,1]` | CAS `0→7` succeeds first try. |
| `atomic_workgroup` | `[8]` | 8 lanes `atomicAdd` on `var<workgroup>` atomic, barrier, load. |
| `mat3x3_diag_vec` | `[4, 10, 18]` | diag(1,2,3)×(4,5,6). |
| `mat2x2_determinant` | `[-2]` | cols `[1,3][2,4]` → `1*4-2*3`. |
| `bits_ops` | `[4, 2147483648, 240, 15]` | popcount(0xF); rev(1)=0x80000000; insert/extract nibble. |
| `continue_odds` | `[25]` | `1+3+5+7+9`. |
| `user_fn_sq` | `[58]` | `7²+3²=49+9`. |

**hand-verified (dilim-2): multi_buffer, atomic_cas_loop, atomic_workgroup, mat3x3_diag_vec, mat2x2_determinant, bits_ops, continue_odds, user_fn_sq**

Also hand-checked while authoring (not required for the ≥8 gate):
`atomic_max_or` (max→8, or→15), `atomic_exchange` (9→42), `atomic_xor` (0xF0=240),
`mat4x4_diag_vec` ([2,6,12,20]), `mat2x3_transpose`, subgroup exclusive/broadcast/max,
`break_if_sum` (10,5), `array_length` (7), `struct_storage` (1,2,21), `bitcast_i32_u32`,
`select_vec`, `storage_barrier_sum`, `pack4x8unorm` (255, 65280), `first_bits` (3,3).

## Bucket coverage (full suite, 50 cases)

| Bucket | Cases |
|--------|--------|
| Integer / float arith | `int_arith`, `float_arith`, `min_max_select`, `store_42`, `clamp_f32`, `bits_ops`, `bitcast_i32_u32`, `first_bits`, `pack4x8unorm` |
| matCxR + matmul | `mat2x2_mul`, `mat2x2_vec`, `mat2x2_determinant`, `mat3x3_diag_vec`, `mat4x4_diag_vec`, `mat2x3_transpose` |
| vec swizzle / materialize | `vec_swizzle`, `vec_materialize`, `select_vec` |
| for / while / nested / loop | `for_sum`, `while_sum`, `nested_loop`, `loop_break`, `break_if_sum`, `continue_odds` |
| switch / if | `switch_case`, `if_else`, `if_divergence`, `nested_if` |
| workgroup / storage barrier | `workgroup_barrier`, `storage_barrier_sum` |
| atomics (storage + workgroup + CAS) | `atomic_add`, `atomic_cas`, `atomic_cas_loop`, `atomic_workgroup`, `atomic_max_or`, `atomic_exchange`, `atomic_xor` |
| subgroup | `subgroup_add`, `subgroup_exclusive_add`, `subgroup_broadcast`, `subgroup_max` |
| f16 | `f16_exact`, `f16_dyadic`, `f16_imprecise` (whitelist) |
| multi-buffer / struct / private / fn | `multi_buffer`, `struct_storage`, `private_accum`, `user_fn_sq`, `array_length` |
| lane / SIMT | `lane_index` |
| texture | **none** — see gap report above |

## Run

```bash
make test-golden          # negative invariants + walk (≥50 cases)
make test                 # unit suite (52 tests; does not replace test-golden)
```
