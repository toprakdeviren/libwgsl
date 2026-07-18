@group(0) @binding(0) var<storage, read> buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(1)
fn main() {
  o[0] = arrayLength(&buf);
}
