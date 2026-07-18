@group(0) @binding(0) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(1)
fn main() {
  o[0] = bitcast<u32>(i32(-1)); // 0xFFFFFFFF
  o[1] = bitcast<u32>(1.0f);    // 0x3f800000
}
