@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(1)
fn main() {
  // columns [1,0,0], [0,2,0], [0,0,3]
  let M = mat3x3f(
    1.0, 0.0, 0.0,
    0.0, 2.0, 0.0,
    0.0, 0.0, 3.0
  );
  let v = M * vec3f(4.0, 5.0, 6.0); // [4, 10, 18]
  o[0] = v.x;
  o[1] = v.y;
  o[2] = v.z;
}
