@group(0) @binding(0) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(1)
fn main() {
  // pack4x8unorm: component i → round(c*255) in byte i (LE)
  o[0] = pack4x8unorm(vec4f(1.0, 0.0, 0.0, 0.0)); // 255
  o[1] = pack4x8unorm(vec4f(0.0, 1.0, 0.0, 0.0)); // 255<<8 = 65280
}
