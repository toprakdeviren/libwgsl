#!/usr/bin/env bash
# §5.4 — Verdict differential: libwgsl vs naga (and tint if present).
#
# For each fixture under tests/12-diff/fixtures/:
#   - run `wgsl check` (accept = exit 0)
#   - run `naga --bulk-validate` (accept = exit 0)
#   - compare accept/reject; print a small conformance table
#
# Missing external tools → SKIP (exit 0), so CI without naga stays green.
# Override binary paths with WGSL_BIN / NAGA_BIN / TINT_BIN.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WGSL_BIN="${WGSL_BIN:-.build/wgsl}"
NAGA_BIN="${NAGA_BIN:-naga}"
TINT_BIN="${TINT_BIN:-tint}"
FIXDIR="tests/12-diff/fixtures"

if [[ ! -x "$WGSL_BIN" ]]; then
  if [[ -f Makefile ]]; then
    make cli >/dev/null
  fi
fi
if [[ ! -x "$WGSL_BIN" ]]; then
  echo "SKIP  test-diff: $WGSL_BIN not found (run make cli)"
  exit 0
fi

have_naga=0
have_tint=0
if command -v "$NAGA_BIN" >/dev/null 2>&1; then have_naga=1; fi
if command -v "$TINT_BIN" >/dev/null 2>&1; then have_tint=1; fi

if [[ $have_naga -eq 0 && $have_tint -eq 0 ]]; then
  echo "SKIP  test-diff: neither naga nor tint in PATH"
  exit 0
fi

shopt -s nullglob
files=("$FIXDIR"/*.wgsl)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "SKIP  test-diff: no fixtures in $FIXDIR"
  exit 0
fi

wgsl_ok() {
  "$WGSL_BIN" check --quiet "$1" >/dev/null 2>&1
}

naga_ok() {
  # naga validates when no output is requested; non-zero = reject.
  "$NAGA_BIN" "$1" >/dev/null 2>&1
}

tint_ok() {
  # tint's validate subcommand (when available)
  if "$TINT_BIN" "$1" --validate >/dev/null 2>&1; then return 0; fi
  if "$TINT_BIN" validate "$1" >/dev/null 2>&1; then return 0; fi
  return 1
}

pass=0
fail=0
agree_naga=0
disagree_naga=0
agree_tint=0
disagree_tint=0

printf "%-28s  %-6s" "fixture" "wgsl"
[[ $have_naga -eq 1 ]] && printf " %-6s %-6s" "naga" "Δn"
[[ $have_tint -eq 1 ]] && printf " %-6s %-6s" "tint" "Δt"
printf "\n"
printf "%-28s  %-6s" "-------" "----"
[[ $have_naga -eq 1 ]] && printf " %-6s %-6s" "----" "--"
[[ $have_tint -eq 1 ]] && printf " %-6s %-6s" "----" "--"
printf "\n"

for f in "${files[@]}"; do
  base=$(basename "$f")
  if wgsl_ok "$f"; then w="accept"; else w="reject"; fi
  line=$(printf "%-28s  %-6s" "$base" "$w")

  if [[ $have_naga -eq 1 ]]; then
    if naga_ok "$f"; then n="accept"; else n="reject"; fi
    if [[ "$w" == "$n" ]]; then d="ok"; agree_naga=$((agree_naga+1));
    else d="DIFF"; disagree_naga=$((disagree_naga+1)); fail=$((fail+1)); fi
    line+=$(printf " %-6s %-6s" "$n" "$d")
  fi

  if [[ $have_tint -eq 1 ]]; then
    if tint_ok "$f"; then t="accept"; else t="reject"; fi
    if [[ "$w" == "$t" ]]; then d="ok"; agree_tint=$((agree_tint+1));
    else d="DIFF"; disagree_tint=$((disagree_tint+1)); fail=$((fail+1)); fi
    line+=$(printf " %-6s %-6s" "$t" "$d")
  fi

  echo "$line"
  pass=$((pass+1))
done

echo
echo "fixtures: ${#files[@]}"
if [[ $have_naga -eq 1 ]]; then
  total=$((agree_naga + disagree_naga))
  pct=0
  [[ $total -gt 0 ]] && pct=$((agree_naga * 100 / total))
  echo "naga agreement: $agree_naga/$total (${pct}%)"
fi
if [[ $have_tint -eq 1 ]]; then
  total=$((agree_tint + disagree_tint))
  pct=0
  [[ $total -gt 0 ]] && pct=$((agree_tint * 100 / total))
  echo "tint agreement: $agree_tint/$total (${pct}%)"
fi

# Write machine-readable summary for the dashboard.
mkdir -p .build
{
  echo "fixtures=${#files[@]}"
  echo "naga_agree=$agree_naga"
  echo "naga_disagree=$disagree_naga"
  echo "tint_agree=$agree_tint"
  echo "tint_disagree=$disagree_tint"
} > .build/test-diff.summary

if [[ $fail -eq 0 ]]; then
  echo "PASS  test-diff"
  exit 0
fi
echo "FAIL  test-diff ($fail disagreement(s))"
# Verdict diffs against external frontends are informational until the
# corpus is curated; do not fail CI hard unless WGSL_DIFF_STRICT=1.
if [[ "${WGSL_DIFF_STRICT:-0}" == "1" ]]; then
  exit 1
fi
echo "  (non-strict mode — set WGSL_DIFF_STRICT=1 to gate)"
exit 0
