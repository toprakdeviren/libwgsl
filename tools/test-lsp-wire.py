#!/usr/bin/env python3
"""LSP stdio Content-Length wire coverage for tools/wgsl_lsp."""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LSP = os.environ.get("LSP", os.path.join(ROOT, ".build", "wgsl_lsp"))
URI = "file:///wire/main.wgsl"

SRC = """@group(0) @binding(0) var<storage, read_write> B: array<f32>;

fn helper(a: u32) -> u32 {
    return a;
}

@compute @workgroup_size(16, 16)
fn matmul_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dead = 1u;
    let keep = helper(gid.x);
    B[keep] = f32(keep);
}
"""


covered: set[str] = set()
next_id = 1
notifications: list[dict] = []
proc: subprocess.Popen[bytes] | None = None
phase = "startup"


def fail(msg: str) -> None:
    global proc
    stderr = ""
    if proc is not None:
        try:
            proc.kill()
            _, err = proc.communicate(timeout=2)
            stderr = err.decode("utf-8", "replace")
        except Exception:
            pass
    if stderr:
        msg = f"{msg}\n--- lsp stderr ---\n{stderr}"
    msg = f"{msg}\nphase: {phase}\ncovered: {', '.join(sorted(covered))}"
    print(f"FAIL  test-lsp-wire: {msg}", file=sys.stderr)
    sys.exit(1)


def frame(body: dict) -> bytes:
    raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
    return b"Content-Length: " + str(len(raw)).encode("ascii") + b"\r\n\r\n" + raw


def send(method: str, params: object | None = None, *, request: bool = True) -> int | None:
    global next_id, proc
    if proc is None or proc.stdin is None:
        fail("server process not started")
    covered.add(method)
    msg: dict[str, object] = {"jsonrpc": "2.0", "method": method}
    req_id: int | None = None
    if request:
        req_id = next_id
        next_id += 1
        msg["id"] = req_id
    if params is not None:
        msg["params"] = params
    os.write(proc.stdin.fileno(), frame(msg))
    return req_id


def read_exact(n: int, deadline: float) -> bytes:
    global proc
    if proc is None or proc.stdout is None:
        fail("server stdout not available")
    fd = proc.stdout.fileno()
    out = bytearray()
    while len(out) < n:
        remain = deadline - time.monotonic()
        if remain <= 0:
            fail("timed out while reading response body")
        ready, _, _ = select.select([fd], [], [], remain)
        if not ready:
            fail("timed out while reading response body")
        chunk = os.read(fd, n - len(out))
        if not chunk:
            fail("server closed stdout")
        out.extend(chunk)
    return bytes(out)


def read_message(timeout: float = 5.0) -> dict:
    global proc
    if proc is None or proc.stdout is None:
        fail("server stdout not available")
    fd = proc.stdout.fileno()
    deadline = time.monotonic() + timeout
    header = bytearray()
    while b"\r\n\r\n" not in header:
        remain = deadline - time.monotonic()
        if remain <= 0:
            fail("timed out while reading response header")
        ready, _, _ = select.select([fd], [], [], remain)
        if not ready:
            fail("timed out while reading response header")
        b = os.read(fd, 1)
        if not b:
            fail("server closed stdout")
        header.extend(b)
    header_text = header.decode("ascii", "replace")
    length = None
    for line in header_text.split("\r\n"):
        if line.lower().startswith("content-length:"):
            length = int(line.split(":", 1)[1].strip())
            break
    if length is None:
        fail(f"missing Content-Length header: {header_text!r}")
    body = read_exact(length, deadline)
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON body: {exc}: {body!r}")
    raise AssertionError("unreachable")


def wait_response(req_id: int) -> dict:
    while True:
        msg = read_message()
        if "method" in msg:
            notifications.append(msg)
            continue
        if msg.get("id") == req_id:
            if "error" in msg:
                fail(f"request {req_id} returned error {msg['error']}")
            return msg


def wait_notification(method: str, predicate=lambda _m: True) -> dict:
    for i, msg in enumerate(notifications):
        if msg.get("method") == method and predicate(msg):
            return notifications.pop(i)
    while True:
        msg = read_message()
        if msg.get("method") == method and predicate(msg):
            return msg
        notifications.append(msg)


def request(method: str, params: object | None = None) -> object:
    req_id = send(method, params, request=True)
    assert req_id is not None
    return wait_response(req_id).get("result")


def notify(method: str, params: object | None = None) -> None:
    send(method, params, request=False)


def pos_at(src: str, needle: str, offset: int = 0) -> dict[str, int]:
    idx = src.index(needle) + offset
    return pos_index(src, idx)


def pos_index(src: str, idx: int) -> dict[str, int]:
    line = src.count("\n", 0, idx)
    bol = src.rfind("\n", 0, idx)
    return {"line": line, "character": idx if bol < 0 else idx - bol - 1}


def range_for(src: str, needle: str) -> dict[str, dict[str, int]]:
    return {
        "start": pos_at(src, needle),
        "end": pos_at(src, needle, len(needle)),
    }


def doc_params(src: str, needle: str, offset: int = 0) -> dict:
    return {"textDocument": {"uri": URI}, "position": pos_at(src, needle, offset)}


def assert_true(cond: object, msg: str) -> None:
    if not cond:
        fail(msg)


def main() -> None:
    global phase, proc
    if not os.path.exists(LSP):
        subprocess.check_call(["make", "lsp"], cwd=ROOT, stdout=subprocess.DEVNULL)
    proc = subprocess.Popen(
        [LSP],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    phase = "initialize"
    init = request(
        "initialize",
        {
            "capabilities": {
                "workspace": {"workspaceFolders": True},
                "textDocument": {"semanticTokens": {"requests": {"full": {"delta": True}}}},
            },
            "processId": None,
            "rootUri": "file:///wire",
            "workspaceFolders": [{"uri": "file:///wire", "name": "wire"}],
        },
    )
    caps = init["capabilities"]
    assert_true(caps["textDocumentSync"]["change"] == 2, "initialize: incremental sync")
    assert_true(caps["documentFormattingProvider"], "initialize: formatting provider")
    assert_true(caps["documentRangeFormattingProvider"], "initialize: range formatting provider")
    assert_true(caps["semanticTokensProvider"]["full"]["delta"], "initialize: semantic delta")
    assert_true(caps["workspace"]["workspaceFolders"]["supported"], "initialize: workspace folders")

    phase = "didOpen diagnostics"
    notify("initialized", {})
    notify(
        "textDocument/didOpen",
        {"textDocument": {"uri": URI, "languageId": "wgsl", "version": 1, "text": SRC}},
    )
    diag = wait_notification(
        "textDocument/publishDiagnostics",
        lambda m: m.get("params", {}).get("uri") == URI,
    )
    sources = {d.get("source") for d in diag["params"]["diagnostics"]}
    assert_true("libwgsl.optimize" in sources, "diagnostics: optimizer hints")
    assert_true("libwgsl.ml" in sources, "diagnostics: ML hints")

    phase = "didChange diagnostics"
    src2 = SRC.replace("helper(gid.x)", "helper(gid.y)")
    notify(
        "textDocument/didChange",
        {
            "textDocument": {"uri": URI, "version": 2},
            "contentChanges": [{"range": range_for(SRC, "gid.x"), "text": "gid.y"}],
        },
    )
    diag2 = wait_notification(
        "textDocument/publishDiagnostics",
        lambda m: m.get("params", {}).get("uri") == URI,
    )
    assert_true(
        not any(d.get("severity") == 1 for d in diag2["params"]["diagnostics"]),
        "didChange: partial edit kept document valid",
    )

    phase = "language requests"
    hover = request("textDocument/hover", doc_params(src2, "keep ="))
    assert_true(hover and "u32" in hover["contents"]["value"], "hover")

    definition = request("textDocument/definition", doc_params(src2, "helper(gid.y)"))
    assert_true(definition and "range" in definition, "definition")

    completion = request("textDocument/completion", doc_params(src2, "helper(gid.y)", 3))
    assert_true(isinstance(completion, list), "completion")

    symbols = request("textDocument/documentSymbol", {"textDocument": {"uri": URI}})
    assert_true(any(s.get("name") == "matmul_main" for s in symbols), "documentSymbol")

    references = request("textDocument/references", doc_params(src2, "keep);"))
    assert_true(isinstance(references, list) and len(references) >= 2, "references")

    prep = request("textDocument/prepareRename", doc_params(src2, "keep);"))
    assert_true(prep and "start" in prep, "prepareRename")

    rename = request(
        "textDocument/rename",
        {**doc_params(src2, "keep);"), "newName": "kept"},
    )
    assert_true(rename and URI in rename["changes"], "rename")

    folding = request("textDocument/foldingRange", {"textDocument": {"uri": URI}})
    assert_true(isinstance(folding, list) and folding, "foldingRange")

    sig = request("textDocument/signatureHelp", doc_params(src2, "gid.y", 1))
    assert_true(sig and sig["signatures"], "signatureHelp")

    full_range = {"start": {"line": 0, "character": 0}, "end": pos_index(src2, len(src2))}
    actions = request(
        "textDocument/codeAction",
        {"textDocument": {"uri": URI}, "range": full_range, "context": {"diagnostics": []}},
    )
    assert_true(any(a.get("kind") == "quickfix.opt.apply" for a in actions), "codeAction")

    formatting = request("textDocument/formatting", {"textDocument": {"uri": URI}, "options": {}})
    assert_true(formatting and "fn helper" in formatting[0]["newText"], "formatting")

    range_fmt = request(
        "textDocument/rangeFormatting",
        {"textDocument": {"uri": URI}, "range": range_for(src2, "return a;"), "options": {}},
    )
    assert_true(
        range_fmt and "fn helper" in range_fmt[0]["newText"] and "matmul_main" not in range_fmt[0]["newText"],
        "rangeFormatting",
    )

    sem = request("textDocument/semanticTokens/full", {"textDocument": {"uri": URI}})
    assert_true(sem["data"], "semanticTokens/full")

    delta = request(
        "textDocument/semanticTokens/full/delta",
        {"textDocument": {"uri": URI}, "previousResultId": sem["resultId"]},
    )
    assert_true(delta["edits"] and "data" in delta["edits"][0], "semanticTokens/full/delta")

    phase = "workspace close shutdown"
    notify(
        "workspace/didChangeWorkspaceFolders",
        {"event": {"added": [{"uri": "file:///wire/extra", "name": "extra"}], "removed": []}},
    )
    notify("textDocument/didClose", {"textDocument": {"uri": URI}})
    wait_notification(
        "textDocument/publishDiagnostics",
        lambda m: m.get("params", {}).get("uri") == URI and not m["params"]["diagnostics"],
    )

    shutdown = request("shutdown", None)
    assert_true(shutdown is None, "shutdown")
    notify("exit", None)
    assert proc is not None
    try:
        rc = proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        fail("server did not exit")
    if rc != 0:
        fail(f"server exited with {rc}")

    assert_true(len(covered) >= 16, f"covered only {len(covered)} methods")
    print(f"PASS  test-lsp-wire  ({len(covered)} methods over stdio)")


if __name__ == "__main__":
    main()
