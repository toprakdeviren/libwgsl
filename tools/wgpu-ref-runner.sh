#!/usr/bin/env bash
# Optional real-backend oracle for tools/test-exec-diff.sh.
#
# Usage:
#   WGSL_EXEC_REF=tools/wgpu-ref-runner.sh make test-exec-diff

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec cargo run --quiet --manifest-path "$ROOT/tools/wgpu-ref-runner/Cargo.toml" -- "$@"
