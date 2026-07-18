var<workgroup> c: atomic<u32>;
@group(0) @binding(0) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(8)
fn main(@builtin(local_invocation_id) l: vec3u) {
  atomicAdd(&c, 1u);
  workgroupBarrier();
  if (l.x == 0u) {
    o[0] = atomicLoad(&c);
  }
}
