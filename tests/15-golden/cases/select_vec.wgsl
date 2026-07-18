@group(0) @binding(0) var<storage, read_write> o: array<i32>;
@compute @workgroup_size(1)
fn main() {
  let a = vec3i(1, 2, 3);
  let b = vec3i(10, 20, 30);
  let c = select(a, b, vec3(true, false, true));
  o[0] = c.x;
  o[1] = c.y;
  o[2] = c.z; // 10, 2, 30
}
