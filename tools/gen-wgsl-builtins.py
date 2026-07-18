#!/usr/bin/env python3
"""Generate the builtin metadata tables used by src/check/exprs.c (§4.2).

Single source of truth: def/wgsl.def

    builtin <name> <return> <family> <arity> <flags> <special>
    overload <name> <pattern> <return> <flags> <special>   # optional override
    texture_matrix

Most non-texture builtins get their kBuiltinOverloads row *auto-derived*
from NAME_PATTERN (name → exact §17 matcher pattern).  Explicit
`overload` lines override the auto row (same name) or add extra rows.

Texture signatures come only from `texture_matrix` → kTextureOverloads.

Output is sorted by name so the C side can binary-search.
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
    "VOID_0", "ALL_ANY", "SELECT", "ARRAY_LENGTH", "WORKGROUP_UNIFORM_LOAD",
    "T_FLOAT_1", "T_FLOAT_2", "T_FLOAT_3",
    "T_NUMERIC_1", "T_NUMERIC_2", "T_NUMERIC_3",
    "T_SIGN_1", "T_INTEGER_1",
    "EXTRACT_BITS", "INSERT_BITS", "LDEXP", "MIX", "DOT", "DOT4_U32",
    "LENGTH", "DETERMINANT", "CROSS", "TRANSPOSE",
    "VEC_FLOAT_1", "VEC_FLOAT_2", "VEC_FLOAT_3", "REFRACT",
    "T_F32_1", "ARG_U32_1", "ARG_VEC4F_1", "ARG_VEC4I_1", "ARG_VEC4U_1",
    "ARG_VEC2F_1",
    "ATOMIC_LOAD", "ATOMIC_STORE", "ATOMIC_RMW", "ATOMIC_CX",
    "BOOL_1", "T_NUMERIC_U32_2",
}

FLAG_BITS = {
    "const": "WGSL_BIF_CONST",
    "must_use": "WGSL_BIF_MUST_USE",
}

# §A7 — name → const-eval fold kind (WGSL_CFOLD_*).  Drives generated
# dispatch; keeps consteval off the strcmp cascade.  Names absent here
# (textures, atomics, barriers, …) get WGSL_CFOLD_NONE.
NAME_CFOLD: dict[str, str] = {
    "bitcast": "BITCAST",
    "pack4x8snorm": "PACK", "pack4x8unorm": "PACK",
    "pack4xI8": "PACK", "pack4xU8": "PACK",
    "pack4xI8Clamp": "PACK", "pack4xU8Clamp": "PACK",
    "pack2x16snorm": "PACK", "pack2x16unorm": "PACK", "pack2x16float": "PACK",
    "unpack4x8snorm": "UNPACK", "unpack4x8unorm": "UNPACK",
    "unpack4xI8": "UNPACK", "unpack4xU8": "UNPACK",
    "unpack2x16snorm": "UNPACK", "unpack2x16unorm": "UNPACK",
    "unpack2x16float": "UNPACK",
    "dot4U8Packed": "DOT4", "dot4I8Packed": "DOT4",
    "any": "ANY_ALL", "all": "ANY_ALL",
    "abs": "ABS_SIGN", "sign": "ABS_SIGN",
    "ceil": "UNARY_FLOAT", "cos": "UNARY_FLOAT", "cosh": "UNARY_FLOAT",
    "exp": "UNARY_FLOAT", "exp2": "UNARY_FLOAT", "floor": "UNARY_FLOAT",
    "fract": "UNARY_FLOAT", "log": "UNARY_FLOAT", "log2": "UNARY_FLOAT",
    "round": "UNARY_FLOAT", "sin": "UNARY_FLOAT", "sinh": "UNARY_FLOAT",
    "sqrt": "UNARY_FLOAT", "tan": "UNARY_FLOAT", "tanh": "UNARY_FLOAT",
    "asin": "UNARY_FLOAT", "acos": "UNARY_FLOAT", "atan": "UNARY_FLOAT",
    "asinh": "UNARY_FLOAT", "acosh": "UNARY_FLOAT", "atanh": "UNARY_FLOAT",
    "trunc": "UNARY_FLOAT", "inverseSqrt": "UNARY_FLOAT",
    "saturate": "UNARY_FLOAT", "radians": "UNARY_FLOAT", "degrees": "UNARY_FLOAT",
    "quantizeToF16": "UNARY_FLOAT",
    "min": "BINARY_NUMERIC", "max": "BINARY_NUMERIC",
    "pow": "BINARY_NUMERIC", "atan2": "BINARY_NUMERIC", "step": "BINARY_NUMERIC",
    "clamp": "TERNARY_NUMERIC", "fma": "TERNARY_NUMERIC",
    "mix": "TERNARY_NUMERIC", "smoothstep": "TERNARY_NUMERIC",
    "ldexp": "LDEXP",
    "length": "LENGTH",
    "distance": "DISTANCE_DOT", "dot": "DISTANCE_DOT",
    "cross": "CROSS",
    "normalize": "NORMALIZE",
    "reflect": "REFLECT",
    "faceForward": "FACE_FORWARD",
    "refract": "REFRACT",
    "transpose": "TRANSPOSE",
    "determinant": "DETERMINANT",
    "extractBits": "BITFIELD", "insertBits": "BITFIELD",
    "reverseBits": "INTEGER_BITS", "countOneBits": "INTEGER_BITS",
    "countLeadingZeros": "INTEGER_BITS", "countTrailingZeros": "INTEGER_BITS",
    "firstLeadingBit": "INTEGER_BITS", "firstTrailingBit": "INTEGER_BITS",
    "select": "SELECT",
    "frexp": "DECOMPOSE_FLOAT", "modf": "DECOMPOSE_FLOAT",
}

# name → (BOP pattern suffix, BR return suffix).  Flags/special come from
# the matching `builtin` row.  Texture builtins are omitted (texture_matrix).
NAME_PATTERN: dict[str, tuple[str, str]] = {
    "all": ("ALL_ANY", "BOOL"),
    "any": ("ALL_ANY", "BOOL"),
    "select": ("SELECT", "ARG0"),
    "arrayLength": ("ARRAY_LENGTH", "U32"),
    "abs": ("T_NUMERIC_1", "ARG0"),
    "ceil": ("T_FLOAT_1", "ARG0"),
    "cos": ("T_FLOAT_1", "ARG0"),
    "cosh": ("T_FLOAT_1", "ARG0"),
    "exp": ("T_FLOAT_1", "ARG0"),
    "exp2": ("T_FLOAT_1", "ARG0"),
    "floor": ("T_FLOAT_1", "ARG0"),
    "fract": ("T_FLOAT_1", "ARG0"),
    "log": ("T_FLOAT_1", "ARG0"),
    "log2": ("T_FLOAT_1", "ARG0"),
    "round": ("T_FLOAT_1", "ARG0"),
    "sin": ("T_FLOAT_1", "ARG0"),
    "sinh": ("T_FLOAT_1", "ARG0"),
    "sqrt": ("T_FLOAT_1", "ARG0"),
    "tan": ("T_FLOAT_1", "ARG0"),
    "tanh": ("T_FLOAT_1", "ARG0"),
    "asin": ("T_FLOAT_1", "ARG0"),
    "acos": ("T_FLOAT_1", "ARG0"),
    "atan": ("T_FLOAT_1", "ARG0"),
    "asinh": ("T_FLOAT_1", "ARG0"),
    "acosh": ("T_FLOAT_1", "ARG0"),
    "atanh": ("T_FLOAT_1", "ARG0"),
    "trunc": ("T_FLOAT_1", "ARG0"),
    "sign": ("T_SIGN_1", "ARG0"),
    "normalize": ("VEC_FLOAT_1", "ARG0"),
    "inverseSqrt": ("T_FLOAT_1", "ARG0"),
    "saturate": ("T_FLOAT_1", "ARG0"),
    "radians": ("T_FLOAT_1", "ARG0"),
    "degrees": ("T_FLOAT_1", "ARG0"),
    "quantizeToF16": ("T_F32_1", "ARG0"),
    "reverseBits": ("T_INTEGER_1", "ARG0"),
    "countOneBits": ("T_INTEGER_1", "ARG0"),
    "countLeadingZeros": ("T_INTEGER_1", "ARG0"),
    "countTrailingZeros": ("T_INTEGER_1", "ARG0"),
    "firstLeadingBit": ("T_INTEGER_1", "ARG0"),
    "firstTrailingBit": ("T_INTEGER_1", "ARG0"),
    "extractBits": ("EXTRACT_BITS", "ARG0"),
    "insertBits": ("INSERT_BITS", "ARG0"),
    "min": ("T_NUMERIC_2", "ARG0"),
    "max": ("T_NUMERIC_2", "ARG0"),
    "pow": ("T_FLOAT_2", "ARG0"),
    "atan2": ("T_FLOAT_2", "ARG0"),
    "step": ("T_FLOAT_2", "ARG0"),
    "ldexp": ("LDEXP", "ARG0"),
    "modf": ("T_FLOAT_1", "MODF_RESULT"),
    "frexp": ("T_FLOAT_1", "FREXP_RESULT"),
    "clamp": ("T_NUMERIC_3", "ARG0"),
    "fma": ("T_FLOAT_3", "ARG0"),
    "mix": ("MIX", "ARG0"),
    "smoothstep": ("T_FLOAT_3", "ARG0"),
    "faceForward": ("VEC_FLOAT_3", "ARG0"),
    "reflect": ("VEC_FLOAT_2", "ARG0"),
    "refract": ("REFRACT", "ARG0"),
    "dot": ("DOT", "ARG0_ELEM"),
    "dot4U8Packed": ("DOT4_U32", "U32"),
    "dot4I8Packed": ("DOT4_U32", "I32"),
    "distance": ("T_FLOAT_2", "ARG0_ELEM"),
    "length": ("LENGTH", "ARG0_ELEM"),
    "determinant": ("DETERMINANT", "ARG0_ELEM"),
    "cross": ("CROSS", "ARG0"),
    "transpose": ("TRANSPOSE", "ARG0"),
    "dpdx": ("T_F32_1", "ARG0"),
    "dpdy": ("T_F32_1", "ARG0"),
    "fwidth": ("T_F32_1", "ARG0"),
    "dpdxCoarse": ("T_F32_1", "ARG0"),
    "dpdyCoarse": ("T_F32_1", "ARG0"),
    "fwidthCoarse": ("T_F32_1", "ARG0"),
    "dpdxFine": ("T_F32_1", "ARG0"),
    "dpdyFine": ("T_F32_1", "ARG0"),
    "fwidthFine": ("T_F32_1", "ARG0"),
    "atomicLoad": ("ATOMIC_LOAD", "ARG0"),
    "atomicAdd": ("ATOMIC_RMW", "ARG0"),
    "atomicSub": ("ATOMIC_RMW", "ARG0"),
    "atomicMax": ("ATOMIC_RMW", "ARG0"),
    "atomicMin": ("ATOMIC_RMW", "ARG0"),
    "atomicAnd": ("ATOMIC_RMW", "ARG0"),
    "atomicOr": ("ATOMIC_RMW", "ARG0"),
    "atomicXor": ("ATOMIC_RMW", "ARG0"),
    "atomicExchange": ("ATOMIC_RMW", "ARG0"),
    "atomicStore": ("ATOMIC_STORE", "VOID"),
    "atomicCompareExchangeWeak": ("ATOMIC_CX", "ATOMIC_CX_RESULT"),
    "pack4x8snorm": ("ARG_VEC4F_1", "U32"),
    "pack4x8unorm": ("ARG_VEC4F_1", "U32"),
    "pack4xI8": ("ARG_VEC4I_1", "U32"),
    "pack4xU8": ("ARG_VEC4U_1", "U32"),
    "pack4xI8Clamp": ("ARG_VEC4I_1", "U32"),
    "pack4xU8Clamp": ("ARG_VEC4U_1", "U32"),
    "pack2x16snorm": ("ARG_VEC2F_1", "U32"),
    "pack2x16unorm": ("ARG_VEC2F_1", "U32"),
    "pack2x16float": ("ARG_VEC2F_1", "U32"),
    "unpack4x8snorm": ("ARG_U32_1", "VEC4F"),
    "unpack4x8unorm": ("ARG_U32_1", "VEC4F"),
    "unpack4xI8": ("ARG_U32_1", "VEC4I"),
    "unpack4xU8": ("ARG_U32_1", "VEC4U"),
    "unpack2x16snorm": ("ARG_U32_1", "VEC2F"),
    "unpack2x16unorm": ("ARG_U32_1", "VEC2F"),
    "unpack2x16float": ("ARG_U32_1", "VEC2F"),
    "storageBarrier": ("VOID_0", "VOID"),
    "textureBarrier": ("VOID_0", "VOID"),
    "workgroupBarrier": ("VOID_0", "VOID"),
    "workgroupUniformLoad": ("WORKGROUP_UNIFORM_LOAD", "ARG0"),
    "subgroupAdd": ("T_NUMERIC_1", "ARG0"),
    "subgroupMul": ("T_NUMERIC_1", "ARG0"),
    "subgroupMax": ("T_NUMERIC_1", "ARG0"),
    "subgroupMin": ("T_NUMERIC_1", "ARG0"),
    "subgroupAnd": ("T_INTEGER_1", "ARG0"),
    "subgroupOr": ("T_INTEGER_1", "ARG0"),
    "subgroupXor": ("T_INTEGER_1", "ARG0"),
    "subgroupExclusiveAdd": ("T_NUMERIC_1", "ARG0"),
    "subgroupExclusiveMul": ("T_NUMERIC_1", "ARG0"),
    "subgroupInclusiveAdd": ("T_NUMERIC_1", "ARG0"),
    "subgroupInclusiveMul": ("T_NUMERIC_1", "ARG0"),
    "subgroupBroadcast": ("T_NUMERIC_U32_2", "ARG0"),
    "subgroupBroadcastFirst": ("T_NUMERIC_1", "ARG0"),
    "subgroupShuffle": ("T_NUMERIC_U32_2", "ARG0"),
    "subgroupShuffleDown": ("T_NUMERIC_U32_2", "ARG0"),
    "subgroupShuffleUp": ("T_NUMERIC_U32_2", "ARG0"),
    "subgroupShuffleXor": ("T_NUMERIC_U32_2", "ARG0"),
    "subgroupBallot": ("BOOL_1", "VEC4U"),
    "subgroupAll": ("BOOL_1", "BOOL"),
    "subgroupAny": ("BOOL_1", "BOOL"),
    "subgroupElect": ("VOID_0", "BOOL"),
    "quadBroadcast": ("T_NUMERIC_U32_2", "ARG0"),
    "quadSwapDiagonal": ("T_NUMERIC_1", "ARG0"),
    "quadSwapX": ("T_NUMERIC_1", "ARG0"),
    "quadSwapY": ("T_NUMERIC_1", "ARG0"),
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


SAMPLED_DIMS = ["1D", "2D", "2D_ARRAY", "3D", "CUBE", "CUBE_ARRAY"]
SAMPLED_MS_DIMS = ["MULTISAMPLED_2D"]
DEPTH_DIMS = ["DEPTH_2D", "DEPTH_2D_ARRAY", "DEPTH_CUBE", "DEPTH_CUBE_ARRAY"]
DEPTH_MS_DIMS = ["DEPTH_MULTISAMPLED_2D"]
STORAGE_DIMS = ["STORAGE_1D", "STORAGE_2D", "STORAGE_2D_ARRAY", "STORAGE_3D"]
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

    for name, base, offset in [
        ("textureSample", "SAMPLE", "SAMPLE_OFFSET"),
        ("textureSampleLevel", "SAMPLE_LEVEL", "SAMPLE_LEVEL_OFFSET"),
    ]:
        for dim in SAMPLED_DIMS:
            for elem in FILTERABLE_SAMPLED_ELEMS:
                add_texture_rows_for_offset_pair(rows, name, base, offset, dim, elem)
        for dim in DEPTH_DIMS:
            add_texture_rows_for_offset_pair(rows, name, base, offset, dim, "DEPTH")

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

    for dim in STORAGE_DIMS:
        for elem in STORAGE_ELEMS:
            for access in ["WGSL_ACCESS_WRITE", "WGSL_ACCESS_READ_WRITE"]:
                rows.append(("textureStore", "STORE", tex_dim(dim), elem, access))

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


def derive_overloads(
    rows: list[BuiltinRow],
    explicit: list[OverloadRow],
) -> list[OverloadRow]:
    """Build kBuiltinOverloads: auto from NAME_PATTERN, overridden by explicit."""
    by_name: dict[str, BuiltinRow] = {r[0]: r for r in rows}
    explicit_names = {o[0] for o in explicit}

    derived: list[OverloadRow] = []
    missing: list[str] = []
    for name, ret, fam, arity, flags, special in rows:
        # Texture family (except barrier) uses kTextureOverloads only.
        if name.startswith("texture") and name != "textureBarrier":
            continue
        if name == "bitcast":
            continue  # template-driven, no fixed BOP row
        if name in explicit_names:
            continue  # explicit wins
        if name not in NAME_PATTERN:
            missing.append(name)
            continue
        pat, pret = NAME_PATTERN[name]
        if pat not in PATTERNS:
            raise SystemExit(f"NAME_PATTERN[{name}] unknown pattern {pat}")
        if pret not in RETURNS:
            raise SystemExit(f"NAME_PATTERN[{name}] unknown return {pret}")
        # Prefer pattern's return when it differs (sign still ARG0 etc.)
        derived.append((name, pat, pret, flags, special))

    if missing:
        raise SystemExit(
            "builtins without NAME_PATTERN and without explicit overload "
            f"(add to NAME_PATTERN or write overload lines): {missing}"
        )

    # Explicit rows last so they can add *extra* patterns; also replace
    # any auto for the same (name, pattern) pair.
    merged: dict[tuple[str, str], OverloadRow] = {}
    for o in derived:
        merged[(o[0], o[1])] = o
    for o in explicit:
        merged[(o[0], o[1])] = o

    out = list(merged.values())
    out.sort(key=lambda r: (r[0], r[1]))
    return out


def render(
    rows: list[BuiltinRow],
    overloads: list[OverloadRow],
    texture_rows: list[TextureRow],
) -> str:
    # Sort base table by name for bsearch.
    rows_sorted = sorted(rows, key=lambda r: r[0])

    out = [
        "/* Generated by tools/gen-wgsl-builtins.py from def/wgsl.def. */",
        "/* Do not edit by hand; edit the DSL and run `make gen-builtins`. */",
        "/* §A7: single owner = def/wgsl.def (+ NAME_PATTERN / NAME_CFOLD). */",
        "",
        f"#define WGSL_BUILTIN_TABLE_COUNT {len(rows_sorted)}",
        f"#define WGSL_BUILTIN_OVERLOAD_COUNT {len(overloads)}",
        f"#define WGSL_TEXTURE_OVERLOAD_COUNT {len(texture_rows)}",
        "",
        "const WGSLBuiltinEntry kBuiltinTable[WGSL_BUILTIN_TABLE_COUNT] = {",
    ]
    width = max(len(row[0]) for row in rows_sorted) if rows_sorted else 1
    for name, ret, fam, arity, flags, special in rows_sorted:
        cfold = NAME_CFOLD.get(name, "NONE")
        out.append(
            f'    {{ "{name}",{" " * (width - len(name))} '
            f"BR_{ret}, PF_{fam}, {arity}, {flags}, BS_{special}, "
            f"WGSL_CFOLD_{cfold} }},"
        )
    out.extend(["};", ""])

    out.append(
        "const WGSLBuiltinOverload "
        "kBuiltinOverloads[WGSL_BUILTIN_OVERLOAD_COUNT] = {"
    )
    width = max((len(row[0]) for row in overloads), default=1)
    for name, pattern, ret, flags, special in overloads:
        out.append(
            f'    {{ "{name}",{" " * (width - len(name))} '
            f"BOP_{pattern}, BR_{ret}, {flags}, BS_{special} }},"
        )
    out.extend(["};", ""])

    # Name → contiguous range in kBuiltinOverloads (table is sorted by name).
    index: list[tuple[str, int, int]] = []
    i = 0
    while i < len(overloads):
        name = overloads[i][0]
        j = i + 1
        while j < len(overloads) and overloads[j][0] == name:
            j += 1
        index.append((name, i, j - i))
        i = j

    # WGSLBuiltinOverloadIndex typedef lives in exprs_priv.h (multi-TU).
    out.append(
        "/* Sorted by name; binary-searchable. begin/count into kBuiltinOverloads. */"
    )
    out.append(
        f"#define WGSL_BUILTIN_OVERLOAD_INDEX_COUNT {len(index)}"
    )
    out.append(
        "const WGSLBuiltinOverloadIndex "
        "kBuiltinOverloadIndex[WGSL_BUILTIN_OVERLOAD_INDEX_COUNT] = {"
    )
    for name, begin, count in index:
        out.append(f'    {{ "{name}", {begin}, {count} }},')
    out.extend(["};", ""])

    out.append(
        "const WGSLTextureOverload "
        "kTextureOverloads[WGSL_TEXTURE_OVERLOAD_COUNT] = {"
    )
    width = max((len(row[0]) for row in texture_rows), default=1)
    for name, kind, dim, elem, access in texture_rows:
        out.append(
            f'    {{ "{name}",{" " * (width - len(name))} '
            f"BTOV_{kind}, {dim}, BTEXEL_{elem}, {access} }},"
        )
    out.extend(["};", ""])

    return "\n".join(out)


def render_cfold(rows: list[BuiltinRow]) -> str:
    """§A7 — self-contained consteval dispatch table (no check/ types)."""
    rows_sorted = sorted(rows, key=lambda r: r[0])
    cfold_code = {
        "NONE": 0, "BITCAST": 1, "PACK": 2, "UNPACK": 3, "DOT4": 4,
        "ANY_ALL": 5, "UNARY_FLOAT": 6, "ABS_SIGN": 7, "BINARY_NUMERIC": 8,
        "TERNARY_NUMERIC": 9, "LDEXP": 10, "LENGTH": 11, "DISTANCE_DOT": 12,
        "CROSS": 13, "NORMALIZE": 14, "REFLECT": 15, "FACE_FORWARD": 16,
        "REFRACT": 17, "TRANSPOSE": 18, "DETERMINANT": 19, "BITFIELD": 20,
        "INTEGER_BITS": 21, "SELECT": 22, "DECOMPOSE_FLOAT": 23,
    }
    out = [
        "/* Generated by tools/gen-wgsl-builtins.py from def/wgsl.def. */",
        "/* §A7 consteval dispatch: sorted name → WGSL_CFOLD_* ordinal. */",
        "/* Do not edit by hand. */",
        "",
        "#ifndef WGSL_BUILTINS_CFOLD_GEN_H",
        "#define WGSL_BUILTINS_CFOLD_GEN_H",
        "",
        "#include <stdint.h>",
        "",
        "typedef struct { const char *name; uint8_t cfold; } WGSLBuiltinCFoldEntry;",
        f"#define WGSL_BUILTIN_CFOLD_COUNT {len(rows_sorted)}",
        "static const WGSLBuiltinCFoldEntry "
        "kBuiltinCFoldTable[WGSL_BUILTIN_CFOLD_COUNT] = {",
    ]
    width = max(len(row[0]) for row in rows_sorted) if rows_sorted else 1
    for name, *_ in rows_sorted:
        ck = NAME_CFOLD.get(name, "NONE")
        code = cfold_code[ck]
        out.append(
            f'    {{ "{name}",{" " * (width - len(name))} {code} }},'
        )
    out.extend(["};", "", "#endif /* WGSL_BUILTINS_CFOLD_GEN_H */", ""])
    return "\n".join(out)


def render_names(rows: list[BuiltinRow]) -> str:
    # Keep insertion order from def for resolver stability, but unique.
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
    out.append(f"#define WGSL_PREDECLARED_BUILTIN_COUNT {len(seen)}")
    out.append("")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--names-output", type=Path)
    ap.add_argument(
        "--cfold-output", type=Path,
        help="§A7 consteval name→fold table (src/check/builtins_cfold.gen.h)",
    )
    ns = ap.parse_args(argv)
    rows, explicit_overloads, texture_rows = parse(ns.input)

    table_names = [r[0] for r in rows]
    if len(table_names) != len(set(table_names)):
        dups = sorted({n for n in table_names if table_names.count(n) > 1})
        raise SystemExit(f"duplicate builtin base rows in def: {dups}")

    overloads = derive_overloads(rows, explicit_overloads)

    # Every non-texture builtin except bitcast must have ≥1 overload row.
    for name, *_ in rows:
        if name == "bitcast":
            continue
        if name.startswith("texture") and name != "textureBarrier":
            continue
        if not any(o[0] == name for o in overloads):
            raise SystemExit(f"no overload row derived for builtin '{name}'")

    ns.output.write_text(
        render(rows, overloads, texture_rows), encoding="utf-8"
    )
    if ns.names_output:
        ns.names_output.write_text(render_names(rows), encoding="utf-8")
    cfold_path = ns.cfold_output
    if cfold_path is None:
        cfold_path = ns.output.parent / "builtins_cfold.gen.h"
    cfold_path.write_text(render_cfold(rows), encoding="utf-8")

    print(
        f"gen-builtins: {len(rows)} builtins, {len(overloads)} overloads "
        f"({len(explicit_overloads)} explicit), "
        f"{len(texture_rows)} texture rows, cfold→{cfold_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
