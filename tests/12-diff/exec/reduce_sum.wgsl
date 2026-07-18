@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(1)
fn main() {
  var acc = 0u;
  for (var i = 0u; i < 4u; i = i + 1u) {
    acc = acc + i;
  }
  o[0] = f32(acc);
}
