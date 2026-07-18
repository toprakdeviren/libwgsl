@group(0) @binding(0) var<storage, read_write> o: array<i32>;
@compute @workgroup_size(1)
fn main() {
  let v = vec4i(1, 2, 3, 4);
  let s = v.wzyx;
  o[0] = s.x;
  o[1] = s.y;
  o[2] = s.z;
  o[3] = s.w;
}
