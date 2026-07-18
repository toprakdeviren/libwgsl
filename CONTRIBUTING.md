# Contributing to libwgsl

This repo values honest, verified progress over broad claims. A change is done
when its acceptance evidence is in the tree and the documented boundary is
truthful.

## Ground Rules

- Keep `docs/TODO.md` as the open-work list.  Remove an item only when its
  acceptance criteria are verified.
- Keep `docs/libwgsl.md` as the consolidated reference.  When behavior changes,
  update the capability table, limits, and command list in the same patch.
- Push policy: do not push from local agent work unless the maintainer explicitly
  asks for it.  Local review/merge belongs to the maintainer.
- Do not silently broaden claims.  If a feature is partial, say exactly where:
  supported subset, missing cases, and the test that proves the subset.
- Generated bindings must be checked with `python3 tools/gen-abi-bindings.py --check`
  after public ABI changes.

## Partial Features

A partial feature can land when it is useful and bounded, but it must not
masquerade as a finished feature. Before calling it complete, either:

- close the documented gaps with tests, or
- leave the open-work item in place and document the precise frontier in
  `docs/libwgsl.md`.

Good partial-feature language names the contract: "supports X/Y/Z, rejects or
reports A/B, does not yet implement C." Bad partial-feature language says
"full", "complete", or "production-ready" without a test and a frontier.

## Pre-Review Checklist

Run the narrow test for the changed area first, then the broad gate when feasible:

```sh
python3 tools/gen-abi-bindings.py --check
git diff --check
make test-all
make wasm-test
make wasm-simd
```

If a command is skipped because a tool is unavailable, say so in the handoff and
keep the related TODO open unless another gate proves the same behavior.
