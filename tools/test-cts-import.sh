#!/usr/bin/env bash
# §5.5 — run wgsl CLI / check over `.build/cts-import/*.wgsl`.
#
# Soft-SKIP when the import directory is empty (no prior
# tools/import-cts-wgsl.sh).  Crash or tool failure → hard fail.
# Accept/reject mix is expected; we only require crash-freedom and
# that every file produces a diagnostic stream (ok or not).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DIR=".build/cts-import"
if [[ ! -d "$DIR" ]] || ! compgen -G "$DIR/*.wgsl" >/dev/null; then
  echo "SKIP  test-cts-import: no imported *.wgsl (run tools/import-cts-wgsl.sh)"
  exit 0
fi

CLI=".build/wgsl"
if [[ ! -x "$CLI" ]]; then
  make -s cli >/dev/null
fi

ok=0
err=0
usage=0
total=0

for f in "$DIR"/*.wgsl; do
  total=$((total + 1))
  # CLI: `check FILE` → 0 ok, 1 validation fail, 2 usage/IO.
  set +e
  out=$("$CLI" check --quiet "$f" 2>&1)
  rc=$?
  set -e
  case $rc in
    0) ok=$((ok + 1)) ;;
    1) err=$((err + 1)) ;;
    *)
      usage=$((usage + 1))
      echo "FAIL  unexpected rc=$rc for $f" >&2
      echo "$out" >&2
      ;;
  esac
done

echo "test-cts-import: total=$total accept=$ok reject=$err other=$usage"
if [[ "$usage" -ne 0 ]]; then
  exit 1
fi
exit 0
