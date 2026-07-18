/**
 * Phase 10 — WASM build smoke test.
 *
 * Loads the Emscripten bundle through the single supported binding
 * (`bindings/wgsl.js`) so struct offsets come from `wgsl_abi_layout`
 * rather than hand-copied constants.
 *
 * Run via `make wasm-test`.  Requires `node`.
 */
'use strict';

const path = require('path');
const fs = require('fs');
const { load } = require('../../bindings/wgsl');

let pass = 0, fail = 0;
function check(cond, msg) {
    if (cond) pass += 1;
    else { fail += 1; console.error('FAIL  ' + msg); }
}

load(process.env.WGSL_WASM_JS).then(api => {
    const pin = api.specPin();
    const uv = api.unicodeVersion();
    check(typeof pin === 'string' && pin.length > 0, 'spec_pin non-empty');
    check(typeof uv === 'string' && uv.length > 0, 'unicode_version non-empty');
    check(uv === '17.0.0', 'unicode_version = 17.0.0');

    /* ABI table from C — wasm32 expectations. */
    const A = api.abi;
    check(A.abiVersion === 1, 'abi version 1');
    check(A.ptrSize === 4, 'wasm32 ptr_size = 4');
    check(A.diagnostic.size === 28, 'diag size 28 (wasm32)');
    check(A.hover.size === 24, 'hover size 24 (wasm32)');
    check(A.definition.size === 12, 'definition size 12');
    check(A.lexToken.size === 20, 'lex token size 20');
    check(A.interp.stepCap === 2000000, 'interp step cap');
    check(A.interp.bufferValuesCap === 4096, 'interp buffer values cap');
    check(A.interp.maxLanes === 256, 'interp max lanes');

    /* Empty source */
    {
        const r = api.check('');
        check(r !== 0, 'empty src: non-zero handle');
        check(api.ok(r), 'empty src: ok');
        check(api.diagnosticCount(r) === 0, 'empty src: 0 diags');
        api.free(r);
    }

    /* Simple compute shader */
    {
        const src =
            '@compute @workgroup_size(64)\n' +
            'fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n' +
            '  let x: i32 = 1 + 2;\n' +
            '  _ = x;\n' +
            '}\n';
        const r = api.check(src);
        check(api.ok(r), 'compute src: ok');
        const json = api.moduleJson(r);
        check(json.length > 0, 'json: non-empty');
        check(json.includes('"stage":"compute"'), 'json: stage compute');
        check(json.includes('"workgroup_size":[64,1,1]'), 'json: wgs literal');
        check(json.includes('"name":"main"'), 'json: name main');
        check(api.moduleJsonLen(r) === json.length, 'json len matches');
        api.free(r);
    }

    /* Parse error + diagnostic via ABI offsets */
    {
        const r = api.check('fn { }');
        check(!api.ok(r), 'parse err: ok = false');
        check(api.diagnosticCount(r) >= 1, 'parse err: ≥1 diag');
        check(typeof api.error(r) === 'string' && api.error(r).length > 0,
            'parse err: error msg non-empty');
        const d = api.diagnostic(r, 0);
        check(!!d, 'diagnostic[0] present');
        if (d) {
            check(d.severity === 1, 'diag severity = ERROR (1)');
            check(d.line >= 1, 'diag line >= 1');
            check(d.column >= 1, 'diag column >= 1');
        }
        api.free(r);
    }

    /* Resolve error */
    {
        const r = api.check('fn f() { _ = ghost; }\n');
        check(!api.ok(r), 'resolve err: ok = false');
        check(api.error(r).indexOf('undeclared') >= 0,
            'resolve err: "undeclared" present');
        api.free(r);
    }

    /* hello-triangle.wgsl */
    {
        const src = fs.readFileSync(
            path.resolve(__dirname, '../../examples/hello-triangle.wgsl'),
            'utf8');
        const r = api.check(src);
        if (!api.ok(r)) console.error('  hello-triangle.wgsl error:', api.error(r));
        check(api.ok(r), 'hello-triangle.wgsl: ok via wasm');
        const json = api.moduleJson(r);
        check(json.includes('"entry_points"'), 'hello-triangle.wgsl: json shape');
        check(json.includes('"stage":"vertex"'), 'hello-triangle.wgsl: vertex entry');
        check(json.includes('"stage":"fragment"'), 'hello-triangle.wgsl: fragment entry');
        api.free(r);
    }

    /* hover / definition / semantic tokens via binding helpers */
    {
        const src = 'fn f() { let x: i32 = 5; let y = x + 1; }\n';
        const r = api.check(src);
        const offX = src.indexOf('x: i32');
        const h = api.hoverAt(r, offX);
        check(h.present === true, 'hover_into: present');
        check(h.typeRepr === 'i32', 'hover_into: typeRepr = "i32"');
        check(h.symbolOffset === offX, 'hover_into: symbol offset');
        check(h.symbolLength === 1, 'hover_into: symbol length');
        api.free(r);
    }
    {
        const src = 'fn add(a: i32) -> i32 { return a; }\n' +
                    'fn caller() { _ = add(1); }\n';
        const r = api.check(src);
        const useOff = src.indexOf('add(1)');
        const declOff = src.indexOf('add');
        const d = api.definitionAt(r, useOff);
        check(d.present === true, 'def_into: present');
        check(d.offset === declOff, 'def_into: offset → decl name');
        check(d.length === 3, 'def_into: length === 3');
        api.free(r);
    }
    {
        const src = 'fn add(a: i32) -> i32 { return a; }\n';
        const toks = api.semanticTokens(src);
        check(toks.length > 0, 'sem_tokens: count > 0');
        const FUNCTION_NAME = 15, TYPE = 3, PARAMETER = 16;
        let sawFnName = false, sawType = false, sawParam = false;
        for (const t of toks) {
            if (t.kind === FUNCTION_NAME && t.length === 3 &&
                src.substr(t.offset, 3) === 'add') sawFnName = true;
            if (t.kind === TYPE && t.length === 3 &&
                src.substr(t.offset, 3) === 'i32') sawType = true;
            if (t.kind === PARAMETER && t.length === 1 && src[t.offset] === 'a')
                sawParam = true;
        }
        check(sawFnName, 'sem_tokens: `add` → FUNCTION_NAME');
        check(sawType, 'sem_tokens: `i32` → TYPE');
        check(sawParam, 'sem_tokens: `a` → PARAMETER');
    }

    /* Formatter + range formatter wrappers */
    {
        const src = 'enable f16;const A:i32=1;fn f(){let x=1;}fn g(){let y=2;}';
        const fmt = api.format(src);
        check(fmt.includes('enable f16;\n\nconst A: i32 = 1;\n\nfn f()'),
            'format: canonical top-level blank lines');
        const start = src.indexOf('y=2');
        const rf = api.formatRange(src, start, start);
        check(rf.text.includes('fn g()') && !rf.text.includes('fn f()'),
            'formatRange: only touched top-level unit');
        check(rf.text.includes('let y = 2;'), 'formatRange: formatted body');
        check(rf.editStart === src.indexOf('fn g'), 'formatRange: editStart');
        check(rf.editEnd === src.length, 'formatRange: editEnd');
    }

    /* Product-surface JSON helpers: optimize, ML, MSL */
    {
        const optSrc =
            '@compute @workgroup_size(1)\n' +
            'fn main() {\n' +
            '  let dead = 1;\n' +
            '  let keep = 2;\n' +
            '  _ = keep;\n' +
            '}\n';
        const report = api.optimize(optSrc);
        check(Array.isArray(report.findings) && report.findings.length >= 1,
            'optimize: findings array');
        const applied = api.optimizeApply(optSrc);
        check(!applied.includes('let dead'), 'optimizeApply: removes unused local');
        check(applied.includes('let keep'), 'optimizeApply: preserves used local');
    }
    {
        const mlSrc =
            '@group(0) @binding(0) var<storage, read_write> B: array<f32>;\n' +
            '@compute @workgroup_size(16, 16)\n' +
            'fn matmul_main(@builtin(global_invocation_id) gid: vec3<u32>) {\n' +
            '  B[gid.x] = f32(gid.x);\n' +
            '}\n';
        const ml = api.mlAnalyze(mlSrc);
        check(ml.kind === 'ml_kernel', 'mlAnalyze: kind');
        check(Array.isArray(ml.patterns) && ml.patterns.includes('matmul'),
            'mlAnalyze: matmul pattern');
    }
    {
        const mslSrc =
            '@vertex fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {\n' +
            '  return vec4f(0.0, 0.0, 0.0, 1.0);\n' +
            '}\n';
        const msl = api.toMsl(mslSrc);
        check(msl.includes('vertex') || msl.includes('Generated by libwgsl'),
            'toMsl: emits MSL text');
    }

    if (fail === 0) {
        console.log(`PASS  wasm-smoke  (${pass} checks)`);
        process.exit(0);
    }
    console.error(`FAIL  wasm-smoke  ${fail}/${pass + fail} checks failed`);
    process.exit(1);
}).catch(err => {
    console.error('FAIL  wasm-smoke loader:', err && err.stack || err);
    process.exit(1);
});
