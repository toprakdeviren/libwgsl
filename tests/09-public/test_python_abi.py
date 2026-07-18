#!/usr/bin/env python3
"""§2.4 — smoke-test generated Python ctypes bindings against lib-shared."""
from __future__ import annotations

import os
import sys

# Repo-root bindings/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from wgsl_abi import Client, ABI_VERSION  # noqa: E402


def main() -> int:
    fail = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal fail
        if not cond:
            fail += 1
            print(f"FAIL  {msg}", file=sys.stderr)

    with Client() as api:
        check(api.abi.abi_version == ABI_VERSION, "abi_version")
        check(api.abi.ptr_size in (4, 8), "ptr_size")
        pin = api.spec_pin()
        check(isinstance(pin, str) and len(pin) > 0, "spec_pin non-empty")

        with api.check("const N: u32 = 4u;\nfn f() {}\n") as r:
            check(r.ok, "check ok")
            check(r.diagnostic_count() == 0, "no diags")
            j = r.module_json()
            check("entry_points" in j or j == "", "module_json reachable")

        with api.check("fn f() { let x: i32 = true; }\n") as r:
            check(not r.ok, "type error not ok")
            check(r.diagnostic_count() >= 1, "has diagnostics")
            d0 = r.diagnostic(0)
            import ctypes
            msg = (
                ctypes.string_at(d0.message).decode()
                if d0.message
                else ""
            )
            check(len(msg) > 0, "diag message non-empty")

        with api.session_new() as s:
            r1 = s.check("const A = 1;\n")
            check(r1.ok, "session check ok")
            r2 = s.check("const A = 1;\n")  # cache hit path
            check(r2.ok, "session cache check ok")

    print(
        f"{'FAIL' if fail else 'PASS'}  test_python_abi  "
        f"({fail} failure{'s' if fail != 1 else ''})",
        file=sys.stderr,
    )
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
