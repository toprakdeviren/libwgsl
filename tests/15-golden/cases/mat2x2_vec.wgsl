@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(1)
fn main() {
  let A = mat2x2f(1.0, 2.0, 3.0, 4.0);
  let v = A * vec2f(1.0, 1.0);
  o[0] = v.x;
  o[1] = v.y;
}
