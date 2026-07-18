@group(0) @binding(0) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(1)
fn main() {
  o[0] = countOneBits(0xFu);              // 4
  o[1] = reverseBits(0x1u);               // 0x80000000
  o[2] = insertBits(0u, 0xFu, 4u, 4u);    // 0xF0 = 240
  o[3] = extractBits(0xF0u, 4u, 4u);      // 0xF = 15
}
