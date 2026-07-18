#!/usr/bin/env bash
# §B1 — real Metal compile-test for graphics entry IO structuring.
#
# NOT part of default `make test` (must stay green without Metal).
#
# Policy (Darwin):
#   - metal tool present → hard-fail on compile error (real gate)
#   - metal tool missing → LOUD message + non-zero exit
#     (B1 UNVERIFIED — do not treat as pass)
# Linux/other: skip with message, exit 0.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

UNAME="$(uname -s 2>/dev/null || echo unknown)"
if [ "$UNAME" != "Darwin" ]; then
  echo "SKIP  test-msl-metal: not Darwin (uname=$UNAME); Metal N/A"
  exit 0
fi

if ! xcrun --find metal >/dev/null 2>&1; then
  echo "FAIL  test-msl-metal: MetalToolchain missing on Darwin."
  echo "      B1 UNVERIFIED — install with:"
  echo "        xcodebuild -downloadComponent MetalToolchain"
  echo "      Then re-run: make test-msl-metal"
  exit 1
fi

CLI="${CLI:-.build/wgsl}"
if [ ! -x "$CLI" ]; then
  echo "Building CLI…"
  make cli >/dev/null
fi

OUT=".build/msl-metal"
mkdir -p "$OUT"

# (a) VS builtin-only + position — free builtins, no stage_in
cat > "$OUT/b1_vs.wgsl" <<'EOF'
@vertex
fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
  return vec4f(0.0, 0.0, 0.0, 1.0);
}
EOF

# (b) FS @location input — load-bearing stage_in + body rewrite + color out
cat > "$OUT/b1_fs.wgsl" <<'EOF'
@fragment
fn fs(@location(0) color: vec4f) -> @location(0) vec4f {
  return color;
}
EOF

# §B3a control-flow fixtures (must compile; honesty: no UNSUPPORTED left)
cat > "$OUT/b3a_for.wgsl" <<'EOF'
@compute @workgroup_size(1)
fn cs_for(@builtin(global_invocation_id) gid: vec3u) {
  var acc = 0u;
  for (var i = 0u; i < 3u; i = i + 1u) {
    acc = acc + i;
  }
}
EOF

cat > "$OUT/b3a_loop.wgsl" <<'EOF'
@compute @workgroup_size(1)
fn cs_loop() {
  var i = 0;
  loop {
    if (i >= 3) { break; }
    continuing {
      i = i + 1;
      break if i >= 10;
    }
  }
}
EOF

cat > "$OUT/b3a_switch.wgsl" <<'EOF'
@compute @workgroup_size(1)
fn cs_switch(@builtin(local_invocation_index) li: u32) {
  var x = 0;
  switch (li) {
    case 0u, 1u: { x = 1; }
    default: { x = 2; }
  }
}
EOF

cat > "$OUT/b3a_fs_for.wgsl" <<'EOF'
@fragment
fn fs(@location(0) color: vec4f) -> @location(0) vec4f {
  var acc = color;
  for (var i = 0; i < 1; i = i + 1) {
    acc = color;
  }
  return acc;
}
EOF

# Nested break must compile (correctness fixed by nest counter)
cat > "$OUT/b3a_nest_switch.wgsl" <<'EOF'
@compute @workgroup_size(1)
fn cs_nest_sw(@builtin(local_invocation_index) li: u32) {
  var x = 0;
  loop {
    switch (li) {
      case 0u: { break; }
      default: { }
    }
    x = x + 10;
    if (x > 100) { break; }
    continuing { x = x + 1; }
  }
}
EOF

cat > "$OUT/b3a_nest_for.wgsl" <<'EOF'
@compute @workgroup_size(1)
fn cs_nest_for() {
  var x = 0;
  loop {
    for (var i = 0; i < 3; i = i + 1) {
      if (i == 1) { break; }
    }
    x = x + 1000;
    if (x > 5000) { break; }
    continuing { x = x + 1; }
  }
}
EOF

# §B3b resource data-plane: storage buffer as kernel param (not file-scope device)
cat > "$OUT/b3b_buffer.wgsl" <<'EOF'
@group(0) @binding(0) var<storage, read_write> B: array<u32>;
@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) gid: vec3u) {
  B[gid.x] = gid.x + 1u;
}
EOF

# §B3b texture + sampler entry params + textureSample
cat > "$OUT/b3b_texture.wgsl" <<'EOF'
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
@fragment
fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
  return textureSample(t, s, uv);
}
EOF

# §B-kalan textureSample* overload options
cat > "$OUT/b_msl_texture_options.wgsl" <<'EOF'
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
@fragment
fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
  let a = textureSampleLevel(t, s, uv, 0.0);
  let b = textureSampleBias(t, s, uv, 0.0);
  return a + b;
}
EOF

# §B-kalan storage texture type + textureStore
cat > "$OUT/b_msl_storage_texture.wgsl" <<'EOF'
@group(0) @binding(0) var t: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(1)
fn cs(@builtin(global_invocation_id) gid: vec3u) {
  textureStore(t, vec2u(gid.x, gid.y), vec4f(1.0));
}
EOF

# §B3b workgroup + barrier
cat > "$OUT/b3b_workgroup.wgsl" <<'EOF'
var<workgroup> tile : array<u32, 8>;
@group(0) @binding(0) var<storage, read_write> o : array<u32>;
@compute @workgroup_size(8)
fn cs(@builtin(local_invocation_id) l: vec3u) {
  tile[l.x] = l.x;
  workgroupBarrier();
  o[l.x] = tile[0];
}
EOF

compile_one() {
  local name="$1" wgsl="$2"
  local metal="$OUT/${name}.metal"
  local air="$OUT/${name}.air"
  echo "  emit   $wgsl → $metal"
  "$CLI" msl "$wgsl" > "$metal"
  # Honesty: supported fixtures must not leave comment-fallback markers
  if grep -q 'unsupported in MSL emit' "$metal" 2>/dev/null; then
    echo "FAIL  $name: MSL still contains unsupported fallback"
    exit 1
  fi
  echo "  metal  $metal → $air"
  xcrun metal -c "$metal" -o "$air"
  if [ ! -f "$air" ]; then
    echo "FAIL  no .air produced for $name"
    exit 1
  fi
  echo "  OK     $name → $(wc -c < "$air") bytes .air"
}

echo "test-msl-metal: xcrun metal present ($(xcrun metal --version 2>/dev/null | head -1))"
compile_one b1_vs "$OUT/b1_vs.wgsl"
compile_one b1_fs "$OUT/b1_fs.wgsl"
compile_one b3a_for "$OUT/b3a_for.wgsl"
compile_one b3a_loop "$OUT/b3a_loop.wgsl"
compile_one b3a_switch "$OUT/b3a_switch.wgsl"
compile_one b3a_fs_for "$OUT/b3a_fs_for.wgsl"
compile_one b3a_nest_switch "$OUT/b3a_nest_switch.wgsl"
compile_one b3a_nest_for "$OUT/b3a_nest_for.wgsl"
compile_one b3b_buffer "$OUT/b3b_buffer.wgsl"
compile_one b3b_texture "$OUT/b3b_texture.wgsl"
compile_one b_msl_texture_options "$OUT/b_msl_texture_options.wgsl"
compile_one b_msl_storage_texture "$OUT/b_msl_storage_texture.wgsl"
compile_one b3b_workgroup "$OUT/b3b_workgroup.wgsl"
echo "PASS  test-msl-metal  (13/13 .air)"
exit 0
