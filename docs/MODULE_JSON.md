# `wgsl_module_json` — module summary schema

`wgsl_module_json(WGSLResult *)` returns a NUL-terminated JSON string
describing the module surface that downstream tools (host bindings,
LSP, pipeline-creation glue) need to wire WGSL into a runtime.  The
schema is intentionally narrow — it carries the shape of the module,
not its IR.

Strings are owned by the `WGSLResult` and live until `wgsl_free`.
The empty string is returned when `wgsl_ok` is false.

## Top-level shape

```json
{
  "spec_pin":     "CRD-2026-05-07",
  "entry_points": [ ... ],
  "resources":    [ ... ],
  "structs":      [ ... ],
  "overrides":    [ ... ]
}
```

`spec_pin` matches `WGSL_SPEC_PIN` at compile time (see [`PLAN.md`](PLAN.md)).

## `entry_points[]`

Each fn marked with `@vertex` / `@fragment` / `@compute` produces one
entry.  Compute entries carry their `@workgroup_size` arguments (as
literal integers — references to module-scope `const`s are resolved
through Phase 6's const-evaluator before emission).

```json
{
  "name":  "main",
  "stage": "compute",                 // "vertex" | "fragment" | "compute"
  "workgroup_size": [256, 1, 1]       // present only on `compute`
}
```

## `resources[]`

Module-scope `var<…>` declarations with `@group(N)` and `@binding(N)`
attributes.  `address_space` is `storage` / `uniform` / `handle`;
`access` is `read` / `write` / `read_write`, defaulted per the
spec's §14.3 table when no explicit access mode is given.

```json
{
  "name":          "X",
  "group":         0,
  "binding":       1,
  "address_space": "storage",
  "access":        "read_write",
  "type":          "array<f32>"
}
```

## `structs[]`

Every user-defined `struct` decl, with its members in source order.
Member types are formatted via `wgsl_type_format` (the same shape used
by hover).

```json
{
  "name": "Vertex",
  "members": [
    { "name": "pos", "type": "vec3<f32>" },
    { "name": "uv",  "type": "vec2<f32>" }
  ]
}
```

Layout (`@align`, `@size`, padding) is **not** in the v1 module summary
— it's a Phase 8 v1.x deliverable per [`DESIGN.md`](DESIGN.md).

## `overrides[]`

Pipeline-overridable constants (`override NAME: T = …;`).  The `id`
field carries the `@id(N)` value when present; otherwise null (the
runtime auto-assigns).

```json
{
  "name": "SCALE",
  "type": "f32",
  "id":   null
}
```

## v1 deferrals

- `vertex_attributes` / `fragment_outputs` / `interpolation` records —
  Phase 8 owns strict entry-point IO type rules; once those land, this
  schema gains location-tagged IO entries.
- Per-resource layout (host-shareable size + alignment) — same gating
  as struct layouts.
- Filterable diagnostic configuration mirrored from `@diagnostic`
  directives — useful for editor settings round-trip; deferred.
