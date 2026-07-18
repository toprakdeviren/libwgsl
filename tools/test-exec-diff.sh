#!/usr/bin/env bash
# §5.4 — Execution differential: libwgsl_interp vs golden (+ optional REF).
#
# For each tests/12-diff/exec/*.wgsl with a sibling *.expect.json:
#   1. Run `.build/wgsl interp --entry … --len … FILE`
#   2. Assert expected buffers match by name (exact, or f32 abs/ULP tolerance)
#   3. Run a second time and require identical "buffers" slices (determinism)
#   4. If WGSL_EXEC_REF is set, run it the same way and hard-fail on mismatch
#
# Exit 0 on PASS.  Missing CLI → build it.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CLI="${WGSL_BIN:-.build/wgsl}"
FIXDIR="tests/12-diff/exec"
REF="${WGSL_EXEC_REF:-}"
COMPARE="$ROOT/tools/exec-diff-compare.py"

if [[ ! -x "$CLI" ]]; then
  make -s cli >/dev/null
fi
if [[ ! -x "$CLI" ]]; then
  echo "FAIL  test-exec-diff: $CLI missing" >&2
  exit 1
fi

shopt -s nullglob
wgsls=("$FIXDIR"/*.wgsl)
if [[ ${#wgsls[@]} -eq 0 ]]; then
  echo "FAIL  test-exec-diff: no fixtures in $FIXDIR" >&2
  exit 1
fi

pass=0
fail=0

json_get() {
  local file=$1 key=$2
  python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
v=d.get(sys.argv[2],"")
if isinstance(v,bool):
    print("true" if v else "false")
else:
    print(v)
' "$file" "$key" 2>/dev/null || true
}

check_buffers() {
  # $1 = expect.json path, $2 = label, stdin = execution JSON
  local exp=$1 label=$2
  "$COMPARE" "$exp" "$label"
}

buffers_slice() {
  python3 -c '
import sys
s=sys.stdin.read()
i=s.find("\"buffers\"")
if i<0:
    sys.exit(0)
j=s.find("\"anomalies\"", i)
sys.stdout.write(s[i: j if j>=0 else len(s)])
'
}

for f in "${wgsls[@]}"; do
  base=$(basename "$f" .wgsl)
  exp="$FIXDIR/${base}.expect.json"
  if [[ ! -f "$exp" ]]; then
    echo "SKIP  $base (no .expect.json)"
    continue
  fi

  entry=$(json_get "$exp" entry)
  [[ -z "$entry" ]] && entry=main
  len=$(json_get "$exp" len)
  [[ -z "$len" ]] && len=1

  set +e
  out1=$("$CLI" interp --entry "$entry" --len "$len" "$f" 2>&1)
  out2=$("$CLI" interp --entry "$entry" --len "$len" "$f" 2>&1)
  set -e

  case_ok=1

  want_ok=$(json_get "$exp" ok)
  if [[ "$want_ok" == "true" ]]; then
    if ! grep -q '"ok":true' <<<"$out1"; then
      echo "FAIL  $base: expected ok:true"
      case_ok=0
    fi
  fi

  if ! printf '%s' "$out1" | check_buffers "$exp" "$base/libwgsl"; then
    echo "FAIL  $base: buffer values mismatch"
    printf '%s\n' "$out1" | head -c 500
    echo
    case_ok=0
  fi

  b1=$(printf '%s' "$out1" | buffers_slice)
  b2=$(printf '%s' "$out2" | buffers_slice)
  if [[ -z "$b1" || "$b1" != "$b2" ]]; then
    echo "FAIL  $base: non-deterministic buffers across two runs"
    case_ok=0
  fi

  if [[ -n "$REF" ]]; then
    set +e
    ref_out=$("$REF" interp --entry "$entry" --len "$len" "$f" 2>&1)
    ref_rc=$?
    if [[ $ref_rc -ne 0 ]]; then
      ref_out=$("$REF" --entry "$entry" --len "$len" "$f" 2>&1)
      ref_rc=$?
    fi
    set -e
    if [[ $ref_rc -ne 0 ]]; then
      echo "FAIL  $base: WGSL_EXEC_REF exited $ref_rc"
      printf '%s\n' "$ref_out" | head -c 700
      echo
      case_ok=0
    elif ! printf '%s' "$ref_out" | check_buffers "$exp" "$base/ref"; then
      echo "FAIL  $base: WGSL_EXEC_REF disagreed with golden"
      printf '%s\n' "$ref_out" | head -c 700
      echo
      case_ok=0
    else
      echo "  ref-ok  $base"
    fi
  fi

  if [[ $case_ok -eq 1 ]]; then
    echo "PASS  $base"
    pass=$((pass + 1))
  else
    fail=$((fail + 1))
  fi
done

echo
echo "test-exec-diff: pass=$pass fail=$fail"
mkdir -p .build
{
  echo "exec_pass=$pass"
  echo "exec_fail=$fail"
  echo "exec_ref=${REF:-none}"
} > .build/test-exec-diff.summary

if [[ $fail -ne 0 ]]; then
  echo "FAIL  test-exec-diff"
  exit 1
fi
echo "PASS  test-exec-diff"
exit 0
