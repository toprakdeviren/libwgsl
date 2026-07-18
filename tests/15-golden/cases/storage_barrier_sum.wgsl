@group(0) @binding(0) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(4)
fn main(@builtin(local_invocation_id) l: vec3u) {
  o[l.x] = l.x + 1u;
  storageBarrier();
  if (l.x == 0u) {
    var s = 0u;
    for (var i = 0u; i < 4u; i = i + 1u) {
      s = s + o[i];
    }
    o[0] = s; // 1+2+3+4 = 10
  }
}
