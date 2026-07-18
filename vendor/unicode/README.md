# vendor/unicode — Decoder Unicode surface

Vendored headers, platform-specific archives, and the narrow source-built
Unicode surface used by `wgsl` (Unicode 17.0.0).

Runtime use is deliberately small: `wgsl` delegates WGSL §3.7 identifier
start/continue predicates to Decoder-compatible XID tables, while UTF-8
step decoding and WGSL §3.2 Pattern_White_Space live in `src/utf8.c`.
WGSL §3.7.1 identifier comparison is raw code-point identity; Unicode
normalization is not used for equality.

## What we use

| WGSL spec § | What it needs | Decoder API |
|------------|--------------|-------------|
| §3.7 | Identifier start / continue (XID rule) | `decoder_is_identifier_start`, `decoder_is_identifier_continue` |
| §3.7.1 | Identifier equality | raw code-point identity in resolver; no Decoder normalization |
| LEX | UTF-8 byte stream -> code points | local `wgsl_utf8_step` |
| §3.2 | Pattern_White_Space classification | local 11-code-point table |

The other headers (`case.h`, `emoji.h`, `script.h`, `security.h`,
`segment.h`, `parallel.h`, `decoder.h` pipeline) are bundled because
they ship together but **are not used by `wgsl` v1**.

## Provenance

Upstream source: [`/Users/2n/dev/personal/decoder/`](file:///Users/2n/dev/personal/decoder/)
(MIT, © 2025 Ugur Toprakdeviren — same author as `wgsl`, license clean
for vendoring).

| Artefact | Source | Built | Notes |
|----------|--------|------:|-------|
| `include/*.h` | `decoder/include/` | n/a | Canonical headers (Apr 27 snapshot). |
| `libunicode.native.a` (3.1 MB, arm64 Mach-O) | `decoder/build/libunicode.a` | 2026-05-07 | Darwin host archive selected by `Makefile` on macOS. |
| `src/decoder_xid.c` | `tools/gen-unicode-xid.py` from `decoder/data/ucd/DerivedCoreProperties.txt` | 2026-07-18 | Non-Darwin native source shim for the Decoder ABI subset `wgsl` links. |
| `libunicode.wasm.a` (1.3 MB) | `miniswift/miniswift/vendor/decoder/` | 2026-04-09 | Older snapshot. Pre-dates the Indic / script-family additions in `properties.h` + `types.h`, but those APIs are not used by `wgsl` so this is safe. **Rebuild via `make wasm` in upstream when WASM build is wired up (Phase 10).** |
| `LICENSE` | `decoder/LICENSE` | n/a | MIT. |

## Re-vendor procedure

When upgrading:

```bash
cd /Users/2n/dev/personal/decoder
make clean && make            # native arm64
make wasm                     # WASM (requires emcc)
cp build/libunicode.a       <wgsl>/vendor/unicode/libunicode.native.a
cp build/wasm/libunicode.a  <wgsl>/vendor/unicode/libunicode.wasm.a
cp include/*.h              <wgsl>/vendor/unicode/include/
cp LICENSE                  <wgsl>/vendor/unicode/LICENSE
python3 <wgsl>/tools/gen-unicode-xid.py \
  data/ucd/DerivedCoreProperties.txt \
  <wgsl>/vendor/unicode/src/decoder_xid.c
```

Then update *Pinned* below and run the WGSL §3.x affected suites
(blankspace, identifier-start, identifier-continue, raw identifier compare).

## Pinned

- **Unicode version**: 17.0.0 (`decoder_get_unicode_version()` returns
  this string).
- **Headers snapshot**: 2026-04-27 (canonical `decoder/include/`).
- **Darwin native archive**: built 2026-05-07 from canonical source.
- **Non-Darwin native source shim**: generated 2026-07-18 from Unicode 17.0.0
  `DerivedCoreProperties.txt`; `make linux-compile` builds
  `.build/vendor/unicode/libunicode.native.a`.
- **WASM archive**: 2026-04-09 snapshot (older — see Re-vendor above).
- A future Decoder upgrade triggers a re-vendor + diff review against
  WGSL §3.x — not in-place edits.
