@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(1)
fn main() {
  let M = mat4x4f(
    2.0, 0.0, 0.0, 0.0,
    0.0, 3.0, 0.0, 0.0,
    0.0, 0.0, 4.0, 0.0,
    0.0, 0.0, 0.0, 5.0
  );
  let v = M * vec4f(1.0, 2.0, 3.0, 4.0); // [2,6,12,20]
  o[0] = v.x;
  o[1] = v.y;
  o[2] = v.z;
  o[3] = v.w;
}
