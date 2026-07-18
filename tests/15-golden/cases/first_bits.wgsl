@group(0) @binding(0) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(1)
fn main() {
  o[0] = firstTrailingBit(0x8u); // bit 3
  o[1] = firstLeadingBit(0x8u);  // bit 3
}
