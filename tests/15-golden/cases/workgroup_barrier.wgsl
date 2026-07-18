var<workgroup> tile: array<u32, 8>;
@group(0) @binding(0) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(8)
fn main(@builtin(local_invocation_id) l: vec3u) {
  tile[l.x] = l.x + 1u;
  workgroupBarrier();
  if (l.x == 0u) {
    var s = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
      s = s + tile[i];
    }
    o[0] = s;
  }
}
