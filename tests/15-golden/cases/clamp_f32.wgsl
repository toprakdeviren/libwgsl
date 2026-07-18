@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(1)
fn main() {
  o[0] = clamp(1.5, 0.0, 1.0);
  o[1] = clamp(-2.0, 0.0, 1.0);
  o[2] = clamp(0.5, 0.0, 1.0);
}
