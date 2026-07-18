enable f16;
@group(0) @binding(0) var<storage, read_write> o: array<f16>;
@compute @workgroup_size(1)
fn main() {
  o[0] = 0.1h + 0.2h;
  o[1] = 1.0h / 3.0h;
}
