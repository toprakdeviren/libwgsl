@group(0) @binding(0) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(4)
fn main(@builtin(local_invocation_id) l: vec3u) {
  if (l.x < 2u) {
    o[l.x] = 111u;
  } else {
    o[l.x] = 222u;
  }
}
