#!/usr/bin/env bash
# §5.5 — long mutational fuzz run (no libFuzzer required).
#
# Builds/runs tests/13-fuzz/test_fuzz_mutational with a high iteration
# count.  Defaults:
#   FUZZ_ITERS=100000
#   FUZZ_SEED=0xC0FFEE
#
# Usage:
#   make fuzz-long
#   FUZZ_ITERS=1000000 tools/fuzz_long.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export FUZZ_ITERS="${FUZZ_ITERS:-100000}"
export FUZZ_SEED="${FUZZ_SEED:-0xC0FFEE}"

echo "fuzz-long: iters=$FUZZ_ITERS seed=$FUZZ_SEED"
make -s .build/tests/13-fuzz/test_fuzz_mutational
time .build/tests/13-fuzz/test_fuzz_mutational
