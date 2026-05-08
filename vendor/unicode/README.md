# vendor/unicode — Decoder Unicode library

Vendored binary + headers from the Decoder library (Unicode 17.0.0).
Used by `wgsl` for spec-mandated Unicode work that we will not
re-implement: UTF-8 decode, NFC normalization, and the
`identifier_start` / `identifier_continue` predicates that match WGSL
§3.7 (XID-equivalent).

## What we use

| WGSL spec § | What it needs | Decoder API |
|------------|--------------|-------------|
| §3.2 | Pattern_White_Space classification | `decoder_is_whitespace` (audit before lock) |
| §3.7 | Identifier start / continue (XID rule) | `decoder_is_identifier_start`, `decoder_is_identifier_continue` |
| §3.7.1 | NFC-form identifier comparison | `decoder_normalize`, `DECODER_NORM_NFC` |
| LEX | UTF-8 byte stream → code points | `encoding.h` API |

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
| `libunicode.native.a` (3.1 MB, arm64) | `decoder/build/libunicode.a` | 2026-05-07 | Host-arch build. Used by all native test gates. |
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
```

Then update *Pinned* below and run the WGSL §3.x affected suites
(blankspace, identifier-start, identifier-continue, NFC compare).

## Pinned

- **Unicode version**: 17.0.0 (`decoder_get_unicode_version()` returns
  this string).
- **Headers snapshot**: 2026-04-27 (canonical `decoder/include/`).
- **Native archive**: built 2026-05-07 from canonical source.
- **WASM archive**: 2026-04-09 snapshot (older — see Re-vendor above).
- A future Decoder upgrade triggers a re-vendor + diff review against
  WGSL §3.x — not in-place edits.
