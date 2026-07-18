# Execution-diff fixtures

Each case is a pair:

| File | Role |
|------|------|
| `NAME.wgsl` | Compute shader writing storage buffers |
| `NAME.expect.json` | Golden: entry, `len` (default_len), expected buffer values |

## Run

```bash
make test                 # includes C test_exec_diff (embedded + smoke files)
make test-exec-diff       # shell harness over this directory
make test-exec-diff-ref   # same harness against tools/wgpu-ref-runner.sh
```

## expect.json shape

```json
{
  "entry": "main",
  "len": 4,
  "ok": true,
  "buffers": [
    { "name": "B", "values": ["1", "2", "3", "4"] }
  ]
}
```

Values are **strings** matching `wgsl_interp` JSON dumps (decimal integer /
float spelling as emitted by the interpreter).  The harness compares buffers by
name.  Integer buffers are exact; `f32` buffers allow exact match, explicit
`abs_tol` / `tolerance`, or a small ULP tolerance.

## External backend (optional)

```bash
# Your wrapper must accept the same flags as:
#   wgsl interp --entry E --len N FILE
# and print JSON containing a "buffers" array comparable to ours.
export WGSL_EXEC_REF=tools/wgpu-ref-runner.sh
make test-exec-diff
```

The in-tree wgpu runner is intentionally minimal: `@group(0)` storage buffers,
`i32` / `u32` / `f32` arrays or `atomic<u32>`, one workgroup dispatch, then
readback.  With `WGSL_EXEC_REF` set, any reference-runner mismatch hard-fails.
Without it, the harness compares libwgsl against goldens (self-oracle) and
checks determinism (run twice).
