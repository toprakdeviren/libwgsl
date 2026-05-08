/**
 * Phase 10 — WASM build smoke test.
 *
 * Loads `.build/wasm/wgsl_compiler.{js,wasm}`, wraps the public C API
 * via Emscripten's `cwrap`, and exercises the same surfaces the
 * native suite does:
 *
 *   - process lifecycle: `wgsl_spec_pin` / `wgsl_unicode_version`
 *   - `wgsl_check` on empty, valid, and intentionally-broken sources
 *   - diagnostic accessors round-trip
 *   - module JSON shape on a compute entry point
 *   - corpus parse: feed `03_linear.wgsl` straight through (the multi-
 *     kernel file with unique resource names is the only corpus
 *     shader that compiles cleanly without `_shared.wgsl` prepended)
 *
 * Run via `make wasm-test`.  Requires `node`.
 */

const path  = require('path');
const fs    = require('fs');
const WGSL  = require('../../.build/wasm/wgsl_compiler.js');

let pass = 0, fail = 0;
function check(cond, msg) {
    if (cond) { pass += 1; }
    else      { fail += 1; console.error('FAIL  ' + msg); }
}

WGSL().then(M => {
    /* — Wrap the public surface — */
    const wgsl_init             = M.cwrap('wgsl_init',             null,     []);
    const wgsl_spec_pin         = M.cwrap('wgsl_spec_pin',         'string', []);
    const wgsl_unicode_version  = M.cwrap('wgsl_unicode_version',  'string', []);
    const wgsl_check            = M.cwrap('wgsl_check',            'number', ['string']);
    const wgsl_free             = M.cwrap('wgsl_free',             null,     ['number']);
    const wgsl_ok               = M.cwrap('wgsl_ok',               'number', ['number']);
    const wgsl_error            = M.cwrap('wgsl_error',            'string', ['number']);
    const wgsl_module_json      = M.cwrap('wgsl_module_json',      'string', ['number']);
    const wgsl_module_json_len  = M.cwrap('wgsl_module_json_len',  'number', ['number']);
    const wgsl_diagnostic_count = M.cwrap('wgsl_diagnostic_count', 'number', ['number']);
    const wgsl_diagnostic       = M.cwrap('wgsl_diagnostic',       'number', ['number', 'number']);

    wgsl_init();

    /* — Process lifecycle constants — */
    const pin = wgsl_spec_pin();
    const uv  = wgsl_unicode_version();
    check(typeof pin === 'string' && pin.length > 0, 'spec_pin non-empty');
    check(typeof uv  === 'string' && uv.length > 0,  'unicode_version non-empty');
    check(uv === '17.0.0',                            'unicode_version = 17.0.0');

    /* — Empty source — */
    {
        const r = wgsl_check('');
        check(r !== 0,                          'empty src: non-zero handle');
        check(wgsl_ok(r) === 1,                 'empty src: ok = 1');
        check(wgsl_diagnostic_count(r) === 0,   'empty src: 0 diags');
        wgsl_free(r);
    }

    /* — Simple compute shader — */
    {
        const src =
            '@compute @workgroup_size(64)\n' +
            'fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n' +
            '  let x: i32 = 1 + 2;\n' +
            '  _ = x;\n' +
            '}\n';
        const r = wgsl_check(src);
        check(wgsl_ok(r) === 1, 'compute src: ok');
        const json = wgsl_module_json(r);
        check(json.length > 0,                            'json: non-empty');
        check(json.includes('"stage":"compute"'),         'json: stage compute');
        check(json.includes('"workgroup_size":[64,1,1]'), 'json: wgs literal');
        check(json.includes('"name":"main"'),             'json: name main');
        check(wgsl_module_json_len(r) === json.length,    'json len matches');
        wgsl_free(r);
    }

    /* — Parse error — */
    {
        const r = wgsl_check('fn { }');
        check(wgsl_ok(r) === 0,               'parse err: ok = 0');
        check(wgsl_diagnostic_count(r) >= 1, 'parse err: ≥1 diag');
        const msg = wgsl_error(r);
        check(typeof msg === 'string' && msg.length > 0,
              'parse err: error msg non-empty');

        /* Diagnostic struct (size = 4 + 4*4 + 2 ptrs).  Read line / column
         * via `getValue` to confirm the LSP-shape fields are reachable.
         * Diagnostic struct layout (LP32 / wasm32):
         *   0    severity   (i32)
         *   4    line       (u32)
         *   8    column     (u32)
         *   12   end_line   (u32)
         *   16   end_column (u32)
         *   20   message    (i32 pointer)
         *   24   rule       (i32 pointer) */
        const dptr = wgsl_diagnostic(r, 0);
        check(dptr !== 0, 'diagnostic[0]: non-NULL pointer');
        if (dptr !== 0) {
            const sev = M.getValue(dptr,      'i32');
            const lin = M.getValue(dptr + 4,  'i32');
            const col = M.getValue(dptr + 8,  'i32');
            check(sev === 1,        'diag severity = ERROR (1)');
            check(lin >= 1,         'diag line >= 1');
            check(col >= 1,         'diag column >= 1');
        }
        wgsl_free(r);
    }

    /* — Resolve error surfaces 'undeclared' — */
    {
        const r = wgsl_check('fn f() { _ = ghost; }\n');
        check(wgsl_ok(r) === 0, 'resolve err: ok = 0');
        check(wgsl_error(r).indexOf('undeclared') >= 0,
              'resolve err: "undeclared" present');
        wgsl_free(r);
    }

    /* — Real-corpus shader (single-file, self-contained) — */
    {
        const src = fs.readFileSync(
            path.resolve(__dirname, '../../examples/shaders/03_linear.wgsl'),
            'utf8');
        const r = wgsl_check(src);
        if (wgsl_ok(r) !== 1) {
            console.error('  03_linear.wgsl error:', wgsl_error(r));
        }
        check(wgsl_ok(r) === 1,  '03_linear.wgsl: ok via wasm');
        const json = wgsl_module_json(r);
        check(json.includes('"entry_points"'), '03_linear.wgsl: json shape');
        check(json.includes('"stage":"compute"'), '03_linear.wgsl: compute entries');
        wgsl_free(r);
    }

    /* ── Extension-side FFI surface ──────────────────────────────────
     *
     * The VS Code extension calls `wgsl_hover_at_into` /
     * `wgsl_definition_at_into` / `wgsl_semantic_tokens` and reads the
     * struct fields at fixed offsets.  Validate the wasm32 layout
     * here so the extension can rely on it. */
    const wgsl_hover_at_into = M.cwrap('wgsl_hover_at_into', null,
                                       ['number', 'number', 'number']);
    const wgsl_definition_at_into = M.cwrap('wgsl_definition_at_into', null,
                                            ['number', 'number', 'number']);
    const wgsl_semantic_tokens = M.cwrap('wgsl_semantic_tokens', 'number',
                                         ['string', 'number', 'number']);
    const wgsl_lex_free        = M.cwrap('wgsl_lex_free',        null,     ['number']);

    /* hover_at_into struct shape: 0:present 4:type_repr* 8:sig*
     * 12:doc* 16:offset 20:length (24 bytes). */
    {
        const src = 'fn f() { let x: i32 = 5; let y = x + 1; }\n';
        const r = wgsl_check(src);
        const offX = src.indexOf('x: i32');     /* hover at decl-x */
        const buf = M._malloc(24);
        wgsl_hover_at_into(r, offX, buf);
        const present  = M.getValue(buf, 'i32');
        const typePtr  = M.getValue(buf + 4, 'i32');
        const sOffset  = M.getValue(buf + 16, 'i32');
        const sLength  = M.getValue(buf + 20, 'i32');
        check(present === 1,                'hover_into: present');
        check(M.UTF8ToString(typePtr) === 'i32',
                                            'hover_into: typeRepr = "i32"');
        check(sOffset === offX,             'hover_into: symbol offset');
        check(sLength === 1,                'hover_into: symbol length');
        M._free(buf);
        wgsl_free(r);
    }

    /* definition_at_into struct shape: 0:present 4:offset 8:length
     * (12 bytes). */
    {
        const src = 'fn add(a: i32) -> i32 { return a; }\n' +
                    'fn caller() { _ = add(1); }\n';
        const r = wgsl_check(src);
        const useOff  = src.indexOf('add(1)');
        const declOff = src.indexOf('add');     /* first occurrence = decl */
        const buf = M._malloc(12);
        wgsl_definition_at_into(r, useOff, buf);
        const present = M.getValue(buf,     'i32');
        const dOff    = M.getValue(buf + 4, 'i32');
        const dLen    = M.getValue(buf + 8, 'i32');
        check(present === 1,        'def_into: present');
        check(dOff === declOff,     'def_into: offset → decl name');
        check(dLen === 3,           'def_into: length === 3');
        M._free(buf);
        wgsl_free(r);
    }

    /* semantic_tokens struct shape: 0:kind 4:offset 8:length 12:line
     * 16:column (20 bytes per token). */
    {
        const src = 'fn add(a: i32) -> i32 { return a; }\n';
        const ptrPtr = M._malloc(4);
        const cntPtr = M._malloc(4);
        const ok = wgsl_semantic_tokens(src, ptrPtr, cntPtr);
        check(ok === 1, 'sem_tokens: rc = 1');
        const arrPtr = M.getValue(ptrPtr, 'i32');
        const count  = M.getValue(cntPtr, 'i32');
        check(count > 0, 'sem_tokens: count > 0');

        let sawFnName = false, sawType = false, sawParam = false;
        const FUNCTION_NAME = 15, TYPE = 3, PARAMETER = 16;
        for (let i = 0; i < count; i++) {
            const t = arrPtr + i * 20;
            const kind   = M.getValue(t,      'i32');
            const offset = M.getValue(t + 4,  'i32');
            const length = M.getValue(t + 8,  'i32');
            if (kind === FUNCTION_NAME && length === 3 &&
                src.substr(offset, 3) === 'add') sawFnName = true;
            if (kind === TYPE && length === 3 &&
                src.substr(offset, 3) === 'i32') sawType = true;
            if (kind === PARAMETER && length === 1 && src[offset] === 'a')
                sawParam = true;
        }
        check(sawFnName, 'sem_tokens: `add` → FUNCTION_NAME');
        check(sawType,   'sem_tokens: `i32` → TYPE');
        check(sawParam,  'sem_tokens: `a` → PARAMETER');

        wgsl_lex_free(arrPtr);
        M._free(ptrPtr);
        M._free(cntPtr);
    }

    if (fail === 0) {
        console.log(`PASS  wasm-smoke  (${pass} checks)`);
        process.exit(0);
    } else {
        console.error(`FAIL  wasm-smoke  ${fail}/${pass + fail} checks failed`);
        process.exit(1);
    }
}).catch(err => {
    console.error('FAIL  wasm-smoke loader:', err && err.stack || err);
    process.exit(1);
});
