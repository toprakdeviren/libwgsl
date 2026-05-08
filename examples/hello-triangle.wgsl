// WebGPU "hello triangle" — three vertices, one solid color.
//
// Used by tests/02-lexer-hello/ as Phase 2's done-when example.

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
    let pos = array<vec2f, 3>(
        vec2f( 0.0,  1.0),
        vec2f(-1.0, -1.0),
        vec2f( 1.0, -1.0),
    );
    return vec4f(pos[vid], 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(1.0, 0.0, 0.0, 1.0);
}
