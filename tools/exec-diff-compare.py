#!/usr/bin/env python3
"""Compare an execution JSON buffer dump against a small expect.json file."""

from __future__ import annotations

import json
import math
import struct
import sys
from typing import Any


def load_first_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start >= 0:
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(obj, dict):
            return obj
        start = text.find("{", start + 1)
    raise ValueError("no JSON object found in output")


def as_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(v) for v in values]


def is_floaty(value: str) -> bool:
    lower = value.lower()
    return (
        "." in value
        or "e" in lower
        or lower in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
    )


def f32_ordered_bits(value: float) -> int:
    bits = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    if bits & 0x80000000:
        return 0x80000000 - (bits & 0x7FFFFFFF)
    return 0x80000000 + bits


def ulp_distance(a: float, b: float) -> int:
    if not (math.isfinite(a) and math.isfinite(b)):
        return 0 if a == b else 2**32
    return abs(f32_ordered_bits(a) - f32_ordered_bits(b))


def numeric_match(want: str, got: str, abs_tol: float, ulp_tol: int) -> bool:
    try:
        want_f = float(want)
        got_f = float(got)
    except ValueError:
        return False
    if abs(want_f - got_f) <= abs_tol:
        return True
    return ulp_distance(want_f, got_f) <= ulp_tol


def compare(expect: dict[str, Any], got: dict[str, Any], label: str) -> int:
    failures = 0
    want_ok = expect.get("ok")
    if want_ok is True and got.get("ok") is not True:
        print(f"FAIL  {label}: expected ok:true", file=sys.stderr)
        failures += 1
    if want_ok is False and got.get("ok") is True:
        print(f"FAIL  {label}: expected ok:false", file=sys.stderr)
        failures += 1

    got_buffers = {
        str(buf.get("name", "")): buf
        for buf in got.get("buffers", [])
        if isinstance(buf, dict)
    }
    known_numeric = bool(expect.get("known_numeric_difference"))
    abs_tol = float(expect.get("abs_tol", expect.get("tolerance", 0.0)) or 0.0)
    exp_ulp = expect.get("ulp_tol", expect.get("ulps", None))
    default_float_abs_tol = float(expect.get("f32_abs_tol", 1.0e-6))

    for want_buf in expect.get("buffers", []):
        if not isinstance(want_buf, dict):
            continue
        name = str(want_buf.get("name", ""))
        got_buf = got_buffers.get(name)
        if got_buf is None:
            print(f"FAIL  {label}: missing buffer {name!r}", file=sys.stderr)
            failures += 1
            continue

        want_values = as_strings(want_buf.get("values", []))
        got_values = as_strings(got_buf.get("values", []))
        if len(want_values) != len(got_values):
            print(
                f"FAIL  {label}: buffer {name!r} length mismatch "
                f"want={len(want_values)} got={len(got_values)}",
                file=sys.stderr,
            )
            failures += 1
            continue

        ty = str(got_buf.get("type", ""))
        float_buffer = "f32" in ty or "f16" in ty or any(is_floaty(v) for v in want_values + got_values)
        value_abs_tol = abs_tol
        value_ulp_tol = int(exp_ulp if exp_ulp is not None else 0)
        if float_buffer:
            value_abs_tol = max(value_abs_tol, default_float_abs_tol)
            if exp_ulp is None:
                value_ulp_tol = 4

        for idx, (want, got_value) in enumerate(zip(want_values, got_values)):
            if want == got_value:
                continue
            if (known_numeric or float_buffer) and numeric_match(
                want, got_value, value_abs_tol, value_ulp_tol
            ):
                continue
            print(
                f"FAIL  {label}: buffer {name!r}[{idx}] want={want!r} got={got_value!r}",
                file=sys.stderr,
            )
            failures += 1

    return failures


def main() -> int:
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("usage: exec-diff-compare.py EXPECT_JSON [LABEL]", file=sys.stderr)
        return 2
    label = sys.argv[2] if len(sys.argv) == 3 else sys.argv[1]
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        expect = json.load(f)
    try:
        got = load_first_json(sys.stdin.read())
    except ValueError as exc:
        print(f"FAIL  {label}: {exc}", file=sys.stderr)
        return 1
    return 1 if compare(expect, got, label) else 0


if __name__ == "__main__":
    raise SystemExit(main())
