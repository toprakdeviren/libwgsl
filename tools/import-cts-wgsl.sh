#!/usr/bin/env bash
# §5.5 — optional WebGPU CTS / external WGSL corpus import.
#
# Usage:
#   tools/import-cts-wgsl.sh [SOURCE_DIR]
#   WGSL_CTS_DIR=/path/to/cts tools/import-cts-wgsl.sh
#
# Walks SOURCE_DIR for `*.wgsl` files (max depth 12), copies up to
# WGSL_CTS_MAX (default 500) into `.build/cts-import/`, and writes a
# manifest.  Does not download anything — the CTS tree (or any shader
# folder) must already be local.
#
# After import:
#   make test-cts-import
#
# Exit codes:
#   0 — imported ≥1 file, or SOURCE missing and we soft-SKIP
#   1 — SOURCE given but unreadable / zero files found when required

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SRC="${1:-${WGSL_CTS_DIR:-}}"
OUT=".build/cts-import"
MAX="${WGSL_CTS_MAX:-500}"
mkdir -p "$OUT"
: > "$OUT/manifest.txt"

if [[ -z "$SRC" ]]; then
  echo "SKIP  import-cts-wgsl: no SOURCE_DIR / WGSL_CTS_DIR"
  echo "      usage: tools/import-cts-wgsl.sh /path/to/wgsl-tree"
  exit 0
fi

if [[ ! -d "$SRC" ]]; then
  echo "FAIL  import-cts-wgsl: not a directory: $SRC" >&2
  exit 1
fi

# Portable find (macOS bash 3.2 has no mapfile).
n=0
while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  base=$(basename "$f")
  # Disambiguate collisions with a short hash of the path.
  tag=$(printf '%s' "$f" | cksum | awk '{print $1}')
  dest="$OUT/${tag}_${base}"
  cp "$f" "$dest"
  printf '%s\t%s\n' "$dest" "$f" >> "$OUT/manifest.txt"
  n=$((n + 1))
done < <(find "$SRC" -maxdepth 12 -type f -name '*.wgsl' 2>/dev/null | head -n "$MAX")

if [[ "$n" -eq 0 ]]; then
  echo "FAIL  import-cts-wgsl: no *.wgsl under $SRC" >&2
  exit 1
fi

echo "import-cts-wgsl: copied $n file(s) → $OUT"
echo "manifest: $OUT/manifest.txt"
echo "run: make test-cts-import"
