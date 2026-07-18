enable f16;
@group(0) @binding(0) var<storage, read_write> o: array<f16>;
@compute @workgroup_size(1)
fn main() {
  o[0] = 0.5h + 1.5h;  // 2
  o[1] = 4.0h * 0.25h; // 1
}
