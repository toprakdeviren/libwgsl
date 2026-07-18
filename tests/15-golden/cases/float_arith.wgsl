@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(1)
fn main() {
  o[0] = 1.25 + 2.75;       // 4
  o[1] = 3.0 * 4.0 - 5.0;   // 7
  o[2] = 8.0 / 2.0;         // 4
}
