#!/usr/bin/env bash
# §5.6 — Conformance dashboard artefact.
#
# Runs the native suite + optional corpus/diff, writes
#   .build/conformance.md
# suitable as a CI artefact / badge source.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p .build

OUT=".build/conformance.md"
STAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

run_count() {
  # Capture "N passed, M failed" line from make test.
  local log
  log=$(mktemp)
  if make -s test >"$log" 2>&1; then
    tail -n 5 "$log" | tee -a .build/conformance_test.log >/dev/null
    grep -E '[0-9]+ passed' "$log" | tail -1 || echo "unknown"
    rm -f "$log"
    return 0
  else
    cat "$log" >> .build/conformance_test.log
    grep -E '[0-9]+ passed|[0-9]+ failed' "$log" | tail -1 || echo "failed"
    rm -f "$log"
    return 1
  fi
}

echo "Collecting conformance dashboard…"
: > .build/conformance_test.log

test_line="(not run)"
test_rc=1
if test_line=$(run_count); then test_rc=0; fi

corpus_line="SKIP"
if make -s test-corpus >/tmp/wgsl_corpus.log 2>&1; then
  corpus_line=$(grep -E '[0-9]+ passed' /tmp/wgsl_corpus.log | tail -1 || echo "ok")
else
  corpus_line=$(grep -E 'passed|failed|SKIP' /tmp/wgsl_corpus.log | tail -1 || echo "failed")
fi

diff_line="SKIP"
if [[ -x tools/test-diff.sh ]]; then
  if bash tools/test-diff.sh >/tmp/wgsl_diff.log 2>&1; then
    diff_line=$(grep -E 'agreement|PASS|SKIP' /tmp/wgsl_diff.log | paste -sd '; ' -)
  else
    diff_line=$(tail -3 /tmp/wgsl_diff.log | tr '\n' ' ')
  fi
fi

exec_line="SKIP"
if bash tools/test-exec-diff.sh >/tmp/wgsl_exec.log 2>&1; then
  exec_line=$(grep -E 'test-exec-diff:|PASS  test-exec' /tmp/wgsl_exec.log | tail -1 | tr -d '\r')
else
  exec_line=$(tail -1 /tmp/wgsl_exec.log 2>/dev/null | tr -d '\r' || echo failed)
fi

fuzz_line="SKIP"
if FUZZ_ITERS=500 .build/tests/13-fuzz/test_fuzz_mutational >/tmp/wgsl_fuzz.log 2>&1; then
  fuzz_line=$(tail -1 /tmp/wgsl_fuzz.log | tr -d '\r')
elif [[ -x .build/tests/13-fuzz/test_fuzz_mutational ]]; then
  fuzz_line=$(tail -1 /tmp/wgsl_fuzz.log 2>/dev/null | tr -d '\r' || echo "failed")
else
  # Build smoke target if missing.
  if make -s .build/tests/13-fuzz/test_fuzz_mutational >/dev/null 2>&1 && \
     FUZZ_ITERS=500 .build/tests/13-fuzz/test_fuzz_mutational >/tmp/wgsl_fuzz.log 2>&1; then
    fuzz_line=$(tail -1 /tmp/wgsl_fuzz.log | tr -d '\r')
  else
    fuzz_line="(not built)"
  fi
fi

cts_line="SKIP"
if [[ -d .build/cts-import ]] && compgen -G ".build/cts-import/*.wgsl" >/dev/null; then
  if bash tools/test-cts-import.sh >/tmp/wgsl_cts.log 2>&1; then
    cts_line=$(grep -E 'total=' /tmp/wgsl_cts.log | tail -1 || echo ok)
  else
    cts_line=$(tail -1 /tmp/wgsl_cts.log | tr -d '\r')
  fi
else
  cts_line="SKIP (no import; make import-cts CTS_DIR=…)"
fi

pin=$(grep -E 'WGSL_SPEC_PIN|CRD-' include/wgsl.h 2>/dev/null | head -1 || true)
[[ -z "$pin" ]] && pin="(see include/wgsl.h)"

{
  echo "# libwgsl conformance dashboard"
  echo
  echo "Generated: \`$STAMP\`"
  echo
  echo "| Surface | Result |"
  echo "|---------|--------|"
  echo "| Spec pin | \`$pin\` |"
  echo "| Native unit tests | \`$test_line\` (rc=$test_rc) |"
  echo "| Corpus | \`$corpus_line\` |"
  echo "| Diff (naga/tint) | \`$diff_line\` |"
  echo "| Exec-diff (golden) | \`$exec_line\` |"
  echo "| Fuzz mutational (500) | \`$fuzz_line\` |"
  echo "| CTS import | \`$cts_line\` |"
  echo
  if [[ -f .build/test-diff.summary ]]; then
    echo "## Diff summary"
    echo '```'
    cat .build/test-diff.summary
    echo '```'
    echo
  fi
  if [[ -f .build/test-exec-diff.summary ]]; then
    echo "## Exec-diff summary"
    echo '```'
    cat .build/test-exec-diff.summary
    echo '```'
    echo
  fi
  echo "## Notes"
  echo
  echo "- Corpus SKIP when \`examples/shaders\` is absent (§5.1)."
  echo "- Exec-diff: \`make test-exec-diff\`; optional \`WGSL_EXEC_REF\` (§5.4)."
  echo "- Long fuzz: \`make fuzz-long\` / libFuzzer \`make fuzz\` (§5.5)."
  echo "- CTS import: \`make import-cts CTS_DIR=/path/to/*.wgsl tree\` (§5.5)."
  echo "- Diff SKIP when neither \`naga\` nor \`tint\` is in PATH (§5.4)."
  echo "- Dashboard is informational; \`make test\` remains the hard gate."
} > "$OUT"

echo "Wrote $OUT"
cat "$OUT"
exit "$test_rc"
