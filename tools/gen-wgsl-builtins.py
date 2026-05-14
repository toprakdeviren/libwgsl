#!/usr/bin/env python3
"""Generate the builtin metadata table used by src/check/exprs.c.

The input DSL is intentionally small:

    builtin <name> <return> <family> <arity> <flags> <special>
    overload <name> <pattern> <return> <flags> <special>
    texture_matrix

Flags are comma-separated tokens from {const,must_use}; use '-' for none.
Return, family, and special are C enum suffixes, for example ARG0,
FLOAT_T, or TEXTURE.  Overload patterns are C enum suffixes for
WGSLBuiltinOverloadPattern.  texture_matrix expands the §17.7 texture
signature matrix into generated exact texture overload rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


RETURNS = {
    "VOID", "BOOL", "I32", "U32", "ARG0", "ARG0_ELEM", "TEMPLATE",
    "VEC4F", "VEC4U", "VEC4I", "VEC2F", "ATOMIC_CX_RESULT",
    "FREXP_RESULT", "MODF_RESULT",
}

FAMILIES = {"NONE", "FLOAT_T", "NUMERIC_T", "INTEGER_T"}

SPECIALS = {"NONE", "LDEXP", "MIX", "TEXTURE"}

PATTERNS = {
    "VOID_0",
    "ALL_ANY",
    "SELECT",
    "ARRAY_LENGTH",
    "WORKGROUP_UNIFORM_LOAD",
    "T_FLOAT_1",
    "T_FLOAT_2",
    "T_FLOAT_3",
    "T_NUMERIC_1",
    "T_NUMERIC_2",
    "T_NUMERIC_3",
    "T_SIGN_1",
    "T_INTEGER_1",
    "EXTRACT_BITS",
    "INSERT_BITS",
    "LDEXP",
    "MIX",
    "DOT",
    "DOT4_U32",
    "LENGTH",
    "DETERMINANT",
    "CROSS",
    "TRANSPOSE",
    "VEC_FLOAT_1",
    "VEC_FLOAT_2",
    "VEC_FLOAT_3",
    "REFRACT",
    "T_F32_1",
    "ARG_U32_1",
    "ARG_VEC4F_1",
    "ARG_VEC4I_1",
    "ARG_VEC4U_1",
    "ARG_VEC2F_1",
    "ATOMIC_LOAD",
    "ATOMIC_STORE",
    "ATOMIC_RMW",
    "ATOMIC_CX",
    "BOOL_1",
    "T_NUMERIC_U32_2",
}

FLAG_BITS = {
    "const": "WGSL_BIF_CONST",
    "must_use": "WGSL_BIF_MUST_USE",
}


def parse_flags(raw: str, path: Path, lineno: int) -> str:
    if raw == "-":
        return "0"
    bits: list[str] = []
    for item in raw.split(","):
        item = item.strip()
        if item not in FLAG_BITS:
            raise SystemExit(f"{path}:{lineno}: unknown flag '{item}'")
        bits.append(FLAG_BITS[item])
    return " | ".join(bits) if bits else "0"


BuiltinRow = tuple[str, str, str, str, str, str]
OverloadRow = tuple[str, str, str, str, str]
TextureRow = tuple[str, str, str, str, str]


SAMPLED_DIMS = [
    "1D",
    "2D",
    "2D_ARRAY",
    "3D",
    "CUBE",
    "CUBE_ARRAY",
]
SAMPLED_MS_DIMS = ["MULTISAMPLED_2D"]
DEPTH_DIMS = [
    "DEPTH_2D",
    "DEPTH_2D_ARRAY",
    "DEPTH_CUBE",
    "DEPTH_CUBE_ARRAY",
]
DEPTH_MS_DIMS = ["DEPTH_MULTISAMPLED_2D"]
STORAGE_DIMS = [
    "STORAGE_1D",
    "STORAGE_2D",
    "STORAGE_2D_ARRAY",
    "STORAGE_3D",
]
SAMPLED_ELEMS = ["F32", "I32", "U32"]
STORAGE_ELEMS = ["STORAGE_F32", "STORAGE_I32", "STORAGE_U32"]
FILTERABLE_SAMPLED_ELEMS = ["F32"]


def tex_dim(name: str) -> str:
    return f"WGSL_TEX_DIM_{name}"


def texture_allows_offset(dim: str) -> bool:
    return dim in {"2D", "2D_ARRAY", "3D", "DEPTH_2D", "DEPTH_2D_ARRAY"}


def add_texture_rows_for_offset_pair(
    out: list[TextureRow],
    name: str,
    base_kind: str,
    offset_kind: str,
    dim: str,
    elem: str,
    access: str = "WGSL_ACCESS_NONE",
) -> None:
    out.append((name, base_kind, tex_dim(dim), elem, access))
    if texture_allows_offset(dim):
        out.append((name, offset_kind, tex_dim(dim), elem, access))


def generate_texture_matrix() -> list[TextureRow]:
    rows: list[TextureRow] = []

    # textureSample family: filtering is only defined for f32 sampled
    # textures; integer sampled textures are valid for load/gather/dim
    # builtins, but not for sampler-based filtering.
    for name, base, offset in [
        ("textureSample", "SAMPLE", "SAMPLE_OFFSET"),
        ("textureSampleLevel", "SAMPLE_LEVEL", "SAMPLE_LEVEL_OFFSET"),
    ]:
        for dim in SAMPLED_DIMS:
            for elem in FILTERABLE_SAMPLED_ELEMS:
                add_texture_rows_for_offset_pair(rows, name, base, offset, dim, elem)
        for dim in DEPTH_DIMS:
            add_texture_rows_for_offset_pair(rows, name, base, offset, dim, "DEPTH")

    # Explicit LOD-bias / gradient sampling excludes 1D, depth, MS, and external.
    for name, base, offset in [
        ("textureSampleBias", "SAMPLE_BIAS", "SAMPLE_BIAS_OFFSET"),
        ("textureSampleGrad", "SAMPLE_GRAD", "SAMPLE_GRAD_OFFSET"),
    ]:
        for dim in ["2D", "2D_ARRAY", "3D", "CUBE", "CUBE_ARRAY"]:
            for elem in FILTERABLE_SAMPLED_ELEMS:
                add_texture_rows_for_offset_pair(rows, name, base, offset, dim, elem)

    for name, base, offset in [
        ("textureSampleCompare", "SAMPLE_COMPARE", "SAMPLE_COMPARE_OFFSET"),
        ("textureSampleCompareLevel", "SAMPLE_COMPARE_LEVEL", "SAMPLE_COMPARE_LEVEL_OFFSET"),
    ]:
        for dim in DEPTH_DIMS:
            add_texture_rows_for_offset_pair(rows, name, base, offset, dim, "DEPTH")

    rows.append(("textureSampleBaseClampToEdge", "SAMPLE_BASE_CLAMP", tex_dim("2D"), "F32", "WGSL_ACCESS_NONE"))
    rows.append(("textureSampleBaseClampToEdge", "SAMPLE_BASE_CLAMP", tex_dim("EXTERNAL"), "EXTERNAL", "WGSL_ACCESS_NONE"))

    # textureLoad excludes cube/cube-array textures; cube coordinates are
    # direction vectors, not integer texel coordinates.
    load_sampled_dims = ["1D", "2D", "2D_ARRAY", "3D"] + SAMPLED_MS_DIMS
    load_depth_dims = ["DEPTH_2D", "DEPTH_2D_ARRAY"] + DEPTH_MS_DIMS
    for dim in load_sampled_dims:
        for elem in SAMPLED_ELEMS:
            rows.append(("textureLoad", "LOAD", tex_dim(dim), elem, "WGSL_ACCESS_NONE"))
    for dim in load_depth_dims:
        rows.append(("textureLoad", "LOAD", tex_dim(dim), "DEPTH", "WGSL_ACCESS_NONE"))
    rows.append(("textureLoad", "LOAD", tex_dim("EXTERNAL"), "EXTERNAL", "WGSL_ACCESS_NONE"))
    for dim in STORAGE_DIMS:
        for elem in STORAGE_ELEMS:
            for access in ["WGSL_ACCESS_READ", "WGSL_ACCESS_READ_WRITE"]:
                rows.append(("textureLoad", "LOAD", tex_dim(dim), elem, access))

    # textureStore.
    for dim in STORAGE_DIMS:
        for elem in STORAGE_ELEMS:
            for access in ["WGSL_ACCESS_WRITE", "WGSL_ACCESS_READ_WRITE"]:
                rows.append(("textureStore", "STORE", tex_dim(dim), elem, access))

    # textureGather: component form for sampled textures; no-component form for depth.
    for dim in ["2D", "2D_ARRAY", "CUBE", "CUBE_ARRAY"]:
        for elem in SAMPLED_ELEMS:
            add_texture_rows_for_offset_pair(
                rows, "textureGather", "GATHER_COMPONENT",
                "GATHER_COMPONENT_OFFSET", dim, elem)
        add_texture_rows_for_offset_pair(
            rows, "textureGather", "GATHER_DEPTH",
            "GATHER_DEPTH_OFFSET", f"DEPTH_{dim}", "DEPTH")
        add_texture_rows_for_offset_pair(
            rows, "textureGatherCompare", "GATHER_COMPARE",
            "GATHER_COMPARE_OFFSET", f"DEPTH_{dim}", "DEPTH")

    # Queries.
    for dim in SAMPLED_DIMS + SAMPLED_MS_DIMS:
        for elem in SAMPLED_ELEMS:
            rows.append(("textureDimensions", "DIMENSIONS", tex_dim(dim), elem, "WGSL_ACCESS_NONE"))
            if dim not in SAMPLED_MS_DIMS:
                rows.append(("textureDimensions", "DIMENSIONS_LEVEL", tex_dim(dim), elem, "WGSL_ACCESS_NONE"))
    for dim in DEPTH_DIMS + DEPTH_MS_DIMS:
        rows.append(("textureDimensions", "DIMENSIONS", tex_dim(dim), "DEPTH", "WGSL_ACCESS_NONE"))
        if dim not in DEPTH_MS_DIMS:
            rows.append(("textureDimensions", "DIMENSIONS_LEVEL", tex_dim(dim), "DEPTH", "WGSL_ACCESS_NONE"))
    rows.append(("textureDimensions", "DIMENSIONS", tex_dim("EXTERNAL"), "EXTERNAL", "WGSL_ACCESS_NONE"))
    for dim in STORAGE_DIMS:
        rows.append(("textureDimensions", "DIMENSIONS", tex_dim(dim), "STORAGE_ANY", "WGSL_ACCESS_NONE"))

    for dim in ["2D_ARRAY", "CUBE_ARRAY"]:
        for elem in SAMPLED_ELEMS:
            rows.append(("textureNumLayers", "NUM_LAYERS", tex_dim(dim), elem, "WGSL_ACCESS_NONE"))
    for dim in ["DEPTH_2D_ARRAY", "DEPTH_CUBE_ARRAY"]:
        rows.append(("textureNumLayers", "NUM_LAYERS", tex_dim(dim), "DEPTH", "WGSL_ACCESS_NONE"))
    rows.append(("textureNumLayers", "NUM_LAYERS", tex_dim("STORAGE_2D_ARRAY"), "STORAGE_ANY", "WGSL_ACCESS_NONE"))

    for dim in SAMPLED_DIMS:
        for elem in SAMPLED_ELEMS:
            rows.append(("textureNumLevels", "NUM_LEVELS", tex_dim(dim), elem, "WGSL_ACCESS_NONE"))
    for dim in DEPTH_DIMS:
        rows.append(("textureNumLevels", "NUM_LEVELS", tex_dim(dim), "DEPTH", "WGSL_ACCESS_NONE"))

    for dim in SAMPLED_MS_DIMS:
        for elem in SAMPLED_ELEMS:
            rows.append(("textureNumSamples", "NUM_SAMPLES", tex_dim(dim), elem, "WGSL_ACCESS_NONE"))
    for dim in DEPTH_MS_DIMS:
        rows.append(("textureNumSamples", "NUM_SAMPLES", tex_dim(dim), "DEPTH", "WGSL_ACCESS_NONE"))

    return rows


def parse(path: Path) -> tuple[list[BuiltinRow], list[OverloadRow], list[TextureRow]]:
    rows: list[BuiltinRow] = []
    overloads: list[OverloadRow] = []
    texture_rows: list[TextureRow] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if parts[0] == "texture_matrix":
            if len(parts) != 1:
                raise SystemExit(f"{path}:{lineno}: expected 'texture_matrix'")
            if texture_rows:
                raise SystemExit(f"{path}:{lineno}: duplicate texture_matrix")
            texture_rows = generate_texture_matrix()
            continue
        if parts[0] == "overload":
            if len(parts) != 6:
                raise SystemExit(
                    f"{path}:{lineno}: expected 'overload <name> <pattern> "
                    "<return> <flags> <special>'"
                )
            _, name, pattern, ret, flags, special = parts
            if pattern not in PATTERNS:
                raise SystemExit(f"{path}:{lineno}: unknown overload pattern '{pattern}'")
            if ret not in RETURNS:
                raise SystemExit(f"{path}:{lineno}: unknown return '{ret}'")
            if special not in SPECIALS:
                raise SystemExit(f"{path}:{lineno}: unknown special '{special}'")
            overloads.append((name, pattern, ret, parse_flags(flags, path, lineno), special))
            continue

        if len(parts) != 7 or parts[0] != "builtin":
            raise SystemExit(
                f"{path}:{lineno}: expected 'builtin <name> <return> "
                "<family> <arity> <flags> <special>' or 'overload <name> "
                "<pattern> <return> <flags> <special>'"
            )
        _, name, ret, fam, arity, flags, special = parts
        if ret not in RETURNS:
            raise SystemExit(f"{path}:{lineno}: unknown return '{ret}'")
        if fam not in FAMILIES:
            raise SystemExit(f"{path}:{lineno}: unknown family '{fam}'")
        if special not in SPECIALS:
            raise SystemExit(f"{path}:{lineno}: unknown special '{special}'")
        try:
            arity_int = int(arity, 10)
        except ValueError as exc:
            raise SystemExit(f"{path}:{lineno}: arity must be an integer") from exc
        if arity_int < 0 or arity_int > 4:
            raise SystemExit(f"{path}:{lineno}: arity must be in 0..4")
        rows.append((name, ret, fam, str(arity_int), parse_flags(flags, path, lineno), special))
    return rows, overloads, texture_rows


def render(
    rows: list[BuiltinRow],
    overloads: list[OverloadRow],
    texture_rows: list[TextureRow],
    source: Path,
) -> str:
    out = [
        "/* Generated by tools/gen-wgsl-builtins.py from def/wgsl.def. */",
        "/* Do not edit by hand; edit the DSL and run `make gen-builtins`. */",
        "static const WGSLBuiltinEntry kBuiltinTable[] = {",
    ]
    width = max(len(row[0]) for row in rows) if rows else 1
    for name, ret, fam, arity, flags, special in rows:
        out.append(
            f'    {{ "{name}",{" " * (width - len(name))} '
            f"BR_{ret}, PF_{fam}, {arity}, {flags}, BS_{special} }},"
        )
    out.extend(["};", ""])
    out.append("static const WGSLBuiltinOverload kBuiltinOverloads[] = {")
    width = max((len(row[0]) for row in overloads), default=1)
    for name, pattern, ret, flags, special in overloads:
        out.append(
            f'    {{ "{name}",{" " * (width - len(name))} '
            f"BOP_{pattern}, BR_{ret}, {flags}, BS_{special} }},"
        )
    out.extend(["};", ""])
    out.append("static const WGSLTextureOverload kTextureOverloads[] = {")
    width = max((len(row[0]) for row in texture_rows), default=1)
    for name, kind, dim, elem, access in texture_rows:
        out.append(
            f'    {{ "{name}",{" " * (width - len(name))} '
            f"BTOV_{kind}, {dim}, BTEXEL_{elem}, {access} }},"
        )
    out.extend(["};", ""])
    return "\n".join(out)


def render_names(rows: list[BuiltinRow], source: Path) -> str:
    out = [
        "/* Generated by tools/gen-wgsl-builtins.py from def/wgsl.def. */",
        "/* Do not edit by hand; edit the DSL and run `make gen-builtins`. */",
        "static const char *kPredeclaredBuiltinNames[] = {",
    ]
    seen = set()
    for name, *_ in rows:
        if name in seen:
            continue
        seen.add(name)
        out.append(f'    "{name}",')
    out.extend(["};", ""])
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--names-output", type=Path)
    ns = ap.parse_args(argv)
    rows, overloads, texture_rows = parse(ns.input)
    ns.output.write_text(render(rows, overloads, texture_rows, ns.input), encoding="utf-8")
    if ns.names_output:
        ns.names_output.write_text(render_names(rows, ns.input), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
