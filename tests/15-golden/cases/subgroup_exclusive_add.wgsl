enable subgroups;
@group(0) @binding(0) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(8)
fn main(@builtin(local_invocation_id) l: vec3u) {
  o[l.x] = subgroupExclusiveAdd(l.x);
}
