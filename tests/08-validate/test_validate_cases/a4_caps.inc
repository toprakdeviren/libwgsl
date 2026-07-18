/* §A4 — silent correctness cliffs: over-cap honesty.
 *
 * Classification (spec first):
 *   - Resource count 64 / @location bitset 0..63 / builtin[16] were
 *     *internal analysis buffers*, not WGSL hard limits.  Fixed by
 *     growable tracking (under-cap behaviour unchanged).
 *   - Tests assert uniqueness still fires past the old walls.
 */
{
    /* 65 storage resources; two share (group,binding) at indices 0 and 64.
     * Old [64] table dropped #64 → silent miss of the dup. */
    char src[32 * 1024];
    size_t off = 0;
    for (int i = 0; i < 65; i++) {
        int g = 0, b = i;
        if (i == 64) { g = 0; b = 0; } /* collide with resource 0 */
        int n = snprintf(src + off, sizeof src - off,
            "@group(%d) @binding(%d) var<storage, read_write> r%d: atomic<u32>;\n",
            g, b, i);
        if (n < 0 || (size_t)n >= sizeof src - off) break;
        off += (size_t)n;
    }
    {
        int n = snprintf(src + off, sizeof src - off,
            "@compute @workgroup_size(1)\n"
            "fn main() {\n"
            "  atomicStore(&r0, 1u);\n"
            "  atomicStore(&r64, 2u);\n"
            "}\n");
        if (n > 0) off += (size_t)n;
    }
    src[off < sizeof src ? off : sizeof src - 1] = 0;
    P p; run(&p, src);
    CHECK(!p.ok, "§A4 resources>64: dup across old wall must fail validation");
    CHECK(has_error_with(&p.diag, "duplicate resource binding"),
          "§A4 resources>64: duplicate diagnostic fires for index 64 vs 0");
    done(&p);
}
{
    /* @location(64) twice — old uint64 bitset ignored l>=64. */
    P p; run(&p,
        "struct VSOut {\n"
        "  @location(64) a: f32,\n"
        "  @location(64) b: f32,\n"
        "  @builtin(position) p: vec4f,\n"
        "}\n"
        "@vertex fn vs() -> VSOut {\n"
        "  var o: VSOut;\n"
        "  o.a = 0.0; o.b = 0.0; o.p = vec4f(0.0);\n"
        "  return o;\n"
        "}\n");
    CHECK(!p.ok, "§A4 @location(64) dup must fail");
    CHECK(has_error_with(&p.diag, "duplicate @location"),
          "§A4 @location>=64 uniqueness tracked");
    done(&p);
}
{
    /* Negative @location — not a legal index. */
    P p; run(&p,
        "@fragment fn fs(@location(-1) c: f32) -> @location(0) vec4f {\n"
        "  return vec4f(c);\n"
        "}\n");
    CHECK(!p.ok, "§A4 negative @location must fail");
    CHECK(has_error_with(&p.diag, "non-negative"),
          "§A4 negative @location diagnostic");
    done(&p);
}
{
    /* Under-cap still works: classic @location(0) dup. */
    P p; run(&p,
        "struct VSOut {\n"
        "  @location(0) a: f32,\n"
        "  @location(0) b: f32,\n"
        "  @builtin(position) p: vec4f,\n"
        "}\n"
        "@vertex fn vs() -> VSOut {\n"
        "  var o: VSOut;\n"
        "  o.a = 0.0; o.b = 0.0; o.p = vec4f(0.0);\n"
        "  return o;\n"
        "}\n");
    CHECK(!p.ok, "§A4 under-cap @location(0) dup still fails");
    CHECK(has_error_with(&p.diag, "duplicate @location"),
          "§A4 under-cap duplicate diagnostic (no regression)");
    done(&p);
}
