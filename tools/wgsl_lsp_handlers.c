/**
 * @file wgsl_lsp_handlers.c — LSP request/notification handlers.
 */
#include "wgsl_lsp_priv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void handle_initialize(const char *id) {
    const char *caps =
        "{"
        "\"capabilities\":{"
          "\"textDocumentSync\":{\"openClose\":true,\"change\":2},"
          "\"hoverProvider\":true,"
          "\"definitionProvider\":true,"
          "\"completionProvider\":{\"triggerCharacters\":[\".\"]},"
          "\"documentSymbolProvider\":true,"
          "\"referencesProvider\":true,"
          "\"renameProvider\":{\"prepareProvider\":true},"
          "\"foldingRangeProvider\":true,"
          "\"signatureHelpProvider\":{\"triggerCharacters\":[\"(\",\",\"]},"
          "\"codeActionProvider\":true,"
          "\"documentFormattingProvider\":true,"
          "\"documentRangeFormattingProvider\":true,"
          "\"semanticTokensProvider\":{"
            "\"legend\":{"
              "\"tokenTypes\":[\"type\",\"function\",\"variable\",\"parameter\","
                "\"property\",\"keyword\",\"number\",\"operator\",\"comment\","
                "\"enumMember\"],"
              "\"tokenModifiers\":[]"
            "},"
            "\"full\":{\"delta\":true},"
            "\"range\":false"
          "},"
          "\"workspace\":{\"workspaceFolders\":{\"supported\":true,"
            "\"changeNotifications\":true}}"
        "},"
        "\"serverInfo\":{\"name\":\"libwgsl\",\"version\":\"" WGSL_VERSION_STRING "\"}"
        "}";
    reply_raw(id, caps);
}

Doc *doc_from_params(const char *params) {
    char uri[1024];
    if (!json_get_string(params, "uri", uri, sizeof uri)) {
        /* textDocument.uri nested */
        const char *td = json_val(params, "textDocument");
        if (!td || !json_get_string(td, "uri", uri, sizeof uri))
            return NULL;
    }
    return doc_find(uri);
}

uint32_t offset_from_params(Doc *d, const char *params) {
    if (!d || !d->text) return 0;
    const char *pos = json_val(params, "position");
    if (!pos) return 0;
    unsigned line = (unsigned)json_get_int(pos, "line", 0);
    unsigned ch   = (unsigned)json_get_int(pos, "character", 0);
    return pos_to_offset(d->text, d->text_len, line, ch);
}

static void put_range(SB *s, const char *text, size_t len,
                      uint32_t off, uint32_t n)
{
    unsigned sl, sc, el, ec;
    offset_to_pos(text, len, off, &sl, &sc);
    offset_to_pos(text, len, off + n, &el, &ec);
    lsp_sb_puts(s, "{\"start\":{\"line\":"); lsp_sb_put_u(s, sl);
    lsp_sb_puts(s, ",\"character\":"); lsp_sb_put_u(s, sc);
    lsp_sb_puts(s, "},\"end\":{\"line\":"); lsp_sb_put_u(s, el);
    lsp_sb_puts(s, ",\"character\":"); lsp_sb_put_u(s, ec);
    lsp_sb_puts(s, "}}");
}

static int range_offsets_from_json(Doc *d, const char *json,
                                   uint32_t *start, uint32_t *end)
{
    if (!d || !d->text || !json || !start || !end) return 0;
    const char *range = json_val(json, "range");
    if (!range) return 0;
    const char *rstart = json_val(range, "start");
    const char *rend = json_val(range, "end");
    if (!rstart || !rend) return 0;
    unsigned sl = (unsigned)json_get_int(rstart, "line", 0);
    unsigned sc = (unsigned)json_get_int(rstart, "character", 0);
    unsigned el = (unsigned)json_get_int(rend, "line", sl);
    unsigned ec = (unsigned)json_get_int(rend, "character", sc);
    *start = pos_to_offset(d->text, d->text_len, sl, sc);
    *end = pos_to_offset(d->text, d->text_len, el, ec);
    return 1;
}

static void handle_hover(const char *id, const char *params) {
    Doc *d = doc_from_params(params);
    if (!d || !doc_result(d)) { reply_null(id); return; }
    uint32_t off = offset_from_params(d, params);
    WGSLHover h = wgsl_hover_at(doc_result(d), off);
    if (!h.present) { reply_null(id); return; }
    SB r; memset(&r, 0, sizeof r);
    lsp_sb_puts(&r, "{\"contents\":{\"kind\":\"plaintext\",\"value\":");
    SB val; memset(&val, 0, sizeof val);
    if (h.type_repr && h.type_repr[0]) lsp_sb_puts(&val, h.type_repr);
    if (h.signature && h.signature[0]) {
        if (val.n) lsp_sb_puts(&val, "\n");
        lsp_sb_puts(&val, h.signature);
    }
    lsp_sb_json_str(&r, val.b ? val.b : "");
    free(val.b);
    lsp_sb_puts(&r, "},\"range\":");
    put_range(&r, d->text, d->text_len, h.symbol_offset, h.symbol_length);
    lsp_sb_putc(&r, '}');
    reply_raw(id, r.b);
    free(r.b);
}

static void handle_definition(const char *id, const char *params) {
    Doc *d = doc_from_params(params);
    if (!d || !doc_result(d)) { reply_null(id); return; }
    uint32_t off = offset_from_params(d, params);
    WGSLDefinition def = wgsl_definition_at(doc_result(d), off);
    if (!def.present) { reply_null(id); return; }
    SB r; memset(&r, 0, sizeof r);
    lsp_sb_puts(&r, "{\"uri\":"); lsp_sb_json_str(&r, d->uri);
    lsp_sb_puts(&r, ",\"range\":");
    put_range(&r, d->text, d->text_len, def.offset, def.length);
    lsp_sb_putc(&r, '}');
    reply_raw(id, r.b);
    free(r.b);
}

static int outline_to_lsp_kind(WGSLOutlineKind k) {
    switch (k) {
    case WGSL_OUTLINE_FUNCTION:   return 12; /* Function */
    case WGSL_OUTLINE_STRUCT:     return 23; /* Struct */
    case WGSL_OUTLINE_VARIABLE:   return 13; /* Variable */
    case WGSL_OUTLINE_CONSTANT:   return 14; /* Constant */
    case WGSL_OUTLINE_OVERRIDE:   return 14;
    case WGSL_OUTLINE_TYPE_ALIAS: return 5;  /* Class */
    case WGSL_OUTLINE_FIELD:      return 8;  /* Field */
    case WGSL_OUTLINE_PARAMETER:  return 13; /* Variable */
    }
    return 13;
}

static void handle_document_symbol(const char *id, const char *params) {
    Doc *d = doc_from_params(params);
    if (!d || !doc_result(d)) { reply_raw(id, "[]"); return; }
    WGSLOutlineSymbol *syms = NULL;
    int n = 0;
    wgsl_document_symbols(doc_result(d), &syms, &n);
    SB r; memset(&r, 0, sizeof r);
    lsp_sb_putc(&r, '[');
    for (int i = 0; i < n; i++) {
        if (i) lsp_sb_putc(&r, ',');
        char name[256];
        size_t nl = syms[i].name_length;
        if (nl >= sizeof name) nl = sizeof name - 1;
        if (d->text && syms[i].name_offset + nl <= d->text_len)
            memcpy(name, d->text + syms[i].name_offset, nl);
        else nl = 0;
        name[nl] = 0;
        lsp_sb_puts(&r, "{\"name\":"); lsp_sb_json_str(&r, name);
        lsp_sb_puts(&r, ",\"kind\":"); lsp_sb_put_i(&r, outline_to_lsp_kind(syms[i].kind));
        lsp_sb_puts(&r, ",\"range\":");
        put_range(&r, d->text, d->text_len, syms[i].range_offset, syms[i].range_length);
        lsp_sb_puts(&r, ",\"selectionRange\":");
        put_range(&r, d->text, d->text_len, syms[i].name_offset, syms[i].name_length);
        lsp_sb_putc(&r, '}');
    }
    lsp_sb_putc(&r, ']');
    reply_raw(id, r.b ? r.b : "[]");
    free(r.b);
    wgsl_outline_free(syms);
}

static void handle_completion(const char *id, const char *params) {
    Doc *d = doc_from_params(params);
    if (!d || !doc_result(d)) { reply_raw(id, "[]"); return; }
    uint32_t off = offset_from_params(d, params);
    WGSLCompletionItem *items = NULL;
    int n = 0;
    wgsl_completions_at(doc_result(d), off, &items, &n);
    SB r; memset(&r, 0, sizeof r);
    lsp_sb_putc(&r, '[');
    for (int i = 0; i < n && i < 200; i++) { /* cap response size */
        if (i) lsp_sb_putc(&r, ',');
        int kind = 6; /* Variable */
        switch (items[i].kind) {
        case WGSL_COMPLETION_KEYWORD:  kind = 14; break;
        case WGSL_COMPLETION_TYPE:     kind = 25; break;
        case WGSL_COMPLETION_FUNCTION: kind = 3;  break;
        case WGSL_COMPLETION_FIELD:    kind = 5;  break;
        case WGSL_COMPLETION_VALUE:    kind = 12; break;
        default: break;
        }
        lsp_sb_puts(&r, "{\"label\":"); lsp_sb_json_str(&r, items[i].label);
        lsp_sb_puts(&r, ",\"kind\":"); lsp_sb_put_i(&r, kind);
        if (items[i].detail && items[i].detail[0]) {
            lsp_sb_puts(&r, ",\"detail\":"); lsp_sb_json_str(&r, items[i].detail);
        }
        lsp_sb_putc(&r, '}');
    }
    lsp_sb_putc(&r, ']');
    reply_raw(id, r.b ? r.b : "[]");
    free(r.b);
    wgsl_completions_free(items, n);
}

static void handle_references(const char *id, const char *params) {
    Doc *d = doc_from_params(params);
    if (!d || !doc_result(d)) { reply_raw(id, "[]"); return; }
    uint32_t off = offset_from_params(d, params);
    WGSLReference *refs = NULL;
    int n = 0;
    wgsl_references_at(doc_result(d), off, &refs, &n);
    SB r; memset(&r, 0, sizeof r);
    lsp_sb_putc(&r, '[');
    for (int i = 0; i < n; i++) {
        if (i) lsp_sb_putc(&r, ',');
        lsp_sb_puts(&r, "{\"uri\":"); lsp_sb_json_str(&r, d->uri);
        lsp_sb_puts(&r, ",\"range\":");
        put_range(&r, d->text, d->text_len, refs[i].offset, refs[i].length);
        lsp_sb_putc(&r, '}');
    }
    lsp_sb_putc(&r, ']');
    reply_raw(id, r.b ? r.b : "[]");
    free(r.b);
    wgsl_references_free(refs);
}

static void handle_prepare_rename(const char *id, const char *params) {
    Doc *d = doc_from_params(params);
    if (!d || !doc_result(d)) { reply_null(id); return; }
    uint32_t off = offset_from_params(d, params);
    uint32_t no = 0, nl = 0;
    if (!wgsl_prepare_rename(doc_result(d), off, &no, &nl)) { reply_null(id); return; }
    SB r; memset(&r, 0, sizeof r);
    put_range(&r, d->text, d->text_len, no, nl);
    reply_raw(id, r.b);
    free(r.b);
}

static void handle_rename(const char *id, const char *params) {
    Doc *d = doc_from_params(params);
    if (!d || !doc_result(d)) { reply_null(id); return; }
    uint32_t off = offset_from_params(d, params);
    char new_name[128];
    if (!json_get_string(params, "newName", new_name, sizeof new_name)) {
        reply_null(id); return;
    }
    WGSLReference *refs = NULL;
    int n = 0;
    wgsl_references_at(doc_result(d), off, &refs, &n);
    if (n == 0) { reply_null(id); return; }
    SB r; memset(&r, 0, sizeof r);
    lsp_sb_puts(&r, "{\"changes\":{");
    lsp_sb_json_str(&r, d->uri);
    lsp_sb_puts(&r, ":[");
    for (int i = 0; i < n; i++) {
        if (i) lsp_sb_putc(&r, ',');
        lsp_sb_puts(&r, "{\"range\":");
        put_range(&r, d->text, d->text_len, refs[i].offset, refs[i].length);
        lsp_sb_puts(&r, ",\"newText\":");
        lsp_sb_json_str(&r, new_name);
        lsp_sb_putc(&r, '}');
    }
    lsp_sb_puts(&r, "]}}");
    reply_raw(id, r.b);
    free(r.b);
    wgsl_references_free(refs);
}

static void handle_folding(const char *id, const char *params) {
    Doc *d = doc_from_params(params);
    if (!d || !doc_result(d)) { reply_raw(id, "[]"); return; }
    WGSLFoldingRange *fr = NULL;
    int n = 0;
    wgsl_folding_ranges(doc_result(d), &fr, &n);
    SB r; memset(&r, 0, sizeof r);
    lsp_sb_putc(&r, '[');
    for (int i = 0; i < n; i++) {
        unsigned sl, sc, el, ec;
        offset_to_pos(d->text, d->text_len, fr[i].start_offset, &sl, &sc);
        offset_to_pos(d->text, d->text_len, fr[i].end_offset, &el, &ec);
        if (el > 0 && ec == 0) el--; /* exclusive end on blank → prev line */
        if (i) lsp_sb_putc(&r, ',');
        lsp_sb_puts(&r, "{\"startLine\":"); lsp_sb_put_u(&r, sl);
        lsp_sb_puts(&r, ",\"endLine\":"); lsp_sb_put_u(&r, el);
        lsp_sb_putc(&r, '}');
    }
    lsp_sb_putc(&r, ']');
    reply_raw(id, r.b ? r.b : "[]");
    free(r.b);
    wgsl_folding_free(fr);
}

static void handle_signature_help(const char *id, const char *params) {
    Doc *d = doc_from_params(params);
    if (!d || !doc_result(d)) { reply_null(id); return; }
    uint32_t off = offset_from_params(d, params);
    WGSLSignatureHelp sh = wgsl_signature_help_at(doc_result(d), off);
    if (!sh.present) { reply_null(id); return; }
    SB r; memset(&r, 0, sizeof r);
    lsp_sb_puts(&r, "{\"signatures\":[{\"label\":");
    lsp_sb_json_str(&r, sh.label ? sh.label : "");
    lsp_sb_puts(&r, "}],\"activeSignature\":0,\"activeParameter\":");
    lsp_sb_put_u(&r, sh.active_parameter);
    lsp_sb_putc(&r, '}');
    reply_raw(id, r.b);
    free(r.b);
}

static void handle_code_action(const char *id, const char *params) {
    Doc *d = doc_from_params(params);
    if (!d || !doc_result(d)) { reply_raw(id, "[]"); return; }
    WGSLCodeAction *acts = NULL;
    int n = 0;
    wgsl_code_actions(doc_result(d), &acts, &n);
    SB r; memset(&r, 0, sizeof r);
    lsp_sb_putc(&r, '[');
    for (int i = 0; i < n; i++) {
        if (i) lsp_sb_putc(&r, ',');
        lsp_sb_puts(&r, "{\"title\":"); lsp_sb_json_str(&r, acts[i].title);
        lsp_sb_puts(&r, ",\"kind\":"); lsp_sb_json_str(&r, acts[i].kind ? acts[i].kind : "quickfix");
        lsp_sb_puts(&r, ",\"edit\":{\"changes\":{");
        lsp_sb_json_str(&r, d->uri);
        lsp_sb_puts(&r, ":[");
        for (int e = 0; e < acts[i].edit_count; e++) {
            if (e) lsp_sb_putc(&r, ',');
            uint32_t so = acts[i].edits[e].start_offset;
            uint32_t eo = acts[i].edits[e].end_offset;
            lsp_sb_puts(&r, "{\"range\":");
            put_range(&r, d->text, d->text_len, so,
                      eo >= so ? eo - so : 0);
            lsp_sb_puts(&r, ",\"newText\":");
            lsp_sb_json_str(&r, acts[i].edits[e].new_text ? acts[i].edits[e].new_text : "");
            lsp_sb_putc(&r, '}');
        }
        lsp_sb_puts(&r, "]}}}");
    }
    lsp_sb_putc(&r, ']');
    reply_raw(id, r.b ? r.b : "[]");
    free(r.b);
    wgsl_code_actions_free(acts, n);
}

static void put_text_edit_array(SB *r, Doc *d, uint32_t start, uint32_t end,
                                const char *new_text)
{
    lsp_sb_puts(r, "[{\"range\":");
    put_range(r, d->text, d->text_len, start, end >= start ? end - start : 0);
    lsp_sb_puts(r, ",\"newText\":");
    lsp_sb_json_str(r, new_text ? new_text : "");
    lsp_sb_puts(r, "}]");
}

static void handle_formatting(const char *id, const char *params) {
    Doc *d = doc_from_params(params);
    if (!d || !d->text) { reply_raw(id, "[]"); return; }
    char *fmt = wgsl_format_n(d->text, d->text_len);
    if (!fmt) { reply_raw(id, "[]"); return; }
    SB r; memset(&r, 0, sizeof r);
    put_text_edit_array(&r, d, 0, (uint32_t)d->text_len, fmt);
    reply_raw(id, r.b ? r.b : "[]");
    free(r.b);
    wgsl_free_string(fmt);
}

static void handle_range_formatting(const char *id, const char *params) {
    Doc *d = doc_from_params(params);
    if (!d || !d->text) { reply_raw(id, "[]"); return; }
    uint32_t start = 0, end = 0;
    if (!range_offsets_from_json(d, params, &start, &end)) {
        reply_raw(id, "[]");
        return;
    }
    uint32_t edit_start = start, edit_end = end;
    char *fmt = wgsl_format_range_n(d->text, d->text_len, start, end,
                                    &edit_start, &edit_end);
    if (!fmt) { reply_raw(id, "[]"); return; }
    SB r; memset(&r, 0, sizeof r);
    put_text_edit_array(&r, d, edit_start, edit_end, fmt);
    reply_raw(id, r.b ? r.b : "[]");
    free(r.b);
    wgsl_free_string(fmt);
}

static int semantic_type_index(WGSLLexTokenKind k) {
    switch (k) {
    case WGSL_LEX_TYPE:
    case WGSL_LEX_STRUCT_NAME:
    case WGSL_LEX_TYPE_ALIAS:      return 0; /* type */
    case WGSL_LEX_FUNCTION_NAME:
    case WGSL_LEX_BUILTIN_FUNC:    return 1; /* function */
    case WGSL_LEX_VAR:
    case WGSL_LEX_LET:
    case WGSL_LEX_CONST:
    case WGSL_LEX_OVERRIDE:
    case WGSL_LEX_IDENTIFIER:      return 2; /* variable */
    case WGSL_LEX_PARAMETER:       return 3; /* parameter */
    case WGSL_LEX_FIELD:           return 4; /* property */
    case WGSL_LEX_KEYWORD:
    case WGSL_LEX_ATTRIBUTE:
    case WGSL_LEX_DIRECTIVE:       return 5; /* keyword */
    case WGSL_LEX_NUMBER:          return 6; /* number */
    case WGSL_LEX_OPERATOR:
    case WGSL_LEX_TEMPLATE_DELIM:  return 7; /* operator */
    case WGSL_LEX_COMMENT:         return 8; /* comment */
    case WGSL_LEX_BUILTIN_VALUE:   return 9; /* enumMember */
    default:                       return -1;
    }
}

static uint32_t utf16_len_span(const char *text, size_t text_len,
                               uint32_t off, uint32_t byte_len)
{
    if (!text || off >= text_len) return byte_len;
    size_t i = off;
    size_t end = (size_t)off + byte_len;
    if (end > text_len) end = text_len;
    uint32_t units = 0;
    while (i < end) {
        unsigned char c = (unsigned char)text[i];
        uint32_t cp = c;
        size_t adv = 1;
        if ((c & 0xE0u) == 0xC0u && i + 1 < end) {
            cp = ((uint32_t)(c & 0x1Fu) << 6) |
                 (uint32_t)(text[i + 1] & 0x3Fu);
            adv = 2;
        } else if ((c & 0xF0u) == 0xE0u && i + 2 < end) {
            cp = ((uint32_t)(c & 0x0Fu) << 12) |
                 ((uint32_t)(text[i + 1] & 0x3Fu) << 6) |
                 (uint32_t)(text[i + 2] & 0x3Fu);
            adv = 3;
        } else if ((c & 0xF8u) == 0xF0u && i + 3 < end) {
            cp = ((uint32_t)(c & 0x07u) << 18) |
                 ((uint32_t)(text[i + 1] & 0x3Fu) << 12) |
                 ((uint32_t)(text[i + 2] & 0x3Fu) << 6) |
                 (uint32_t)(text[i + 3] & 0x3Fu);
            adv = 4;
        }
        units += cp > 0xFFFFu ? 2u : 1u;
        i += adv;
    }
    return units;
}

static size_t append_semantic_data(SB *r, Doc *d) {
    WGSLLexToken *tokens = NULL;
    int n = 0;
    if (!d || !d->text ||
        !wgsl_semantic_tokens(d->text, &tokens, &n))
        return 0;
    int first = 1;
    uint32_t prev_line = 0, prev_start = 0;
    size_t int_count = 0;
    for (int i = 0; i < n; i++) {
        int type = semantic_type_index(tokens[i].kind);
        if (type < 0 || tokens[i].length == 0) continue;
        uint32_t line = tokens[i].line ? tokens[i].line - 1 : 0;
        uint32_t start = tokens[i].column ? tokens[i].column - 1 : 0;
        uint32_t dline = first ? line : line - prev_line;
        uint32_t dstart = dline ? start : start - prev_start;
        uint32_t len = utf16_len_span(d->text, d->text_len,
                                      tokens[i].offset, tokens[i].length);
        if (!first) lsp_sb_putc(r, ',');
        lsp_sb_put_u(r, dline); lsp_sb_putc(r, ',');
        lsp_sb_put_u(r, dstart); lsp_sb_putc(r, ',');
        lsp_sb_put_u(r, len); lsp_sb_putc(r, ',');
        lsp_sb_put_u(r, (unsigned long)type); lsp_sb_puts(r, ",0");
        first = 0;
        prev_line = line;
        prev_start = start;
        int_count += 5;
    }
    wgsl_lex_free(tokens);
    return int_count;
}

static void handle_semantic_tokens_full(const char *id, const char *params) {
    Doc *d = doc_from_params(params);
    if (!d || !d->text) {
        reply_raw(id, "{\"data\":[]}");
        return;
    }
    SB r; memset(&r, 0, sizeof r);
    lsp_sb_puts(&r, "{\"resultId\":");
    char rid[32];
    snprintf(rid, sizeof rid, "%d", d->version);
    lsp_sb_json_str(&r, rid);
    lsp_sb_puts(&r, ",\"data\":[");
    d->semantic_data_count = append_semantic_data(&r, d);
    lsp_sb_puts(&r, "]}");
    reply_raw(id, r.b ? r.b : "{\"data\":[]}");
    free(r.b);
}

static void handle_semantic_tokens_delta(const char *id, const char *params) {
    Doc *d = doc_from_params(params);
    if (!d || !d->text) {
        reply_raw(id, "{\"edits\":[]}");
        return;
    }
    size_t old_count = d->semantic_data_count;
    SB data; memset(&data, 0, sizeof data);
    size_t new_count = append_semantic_data(&data, d);
    d->semantic_data_count = new_count;
    SB r; memset(&r, 0, sizeof r);
    lsp_sb_puts(&r, "{\"resultId\":");
    char rid[32];
    snprintf(rid, sizeof rid, "%d", d->version);
    lsp_sb_json_str(&r, rid);
    lsp_sb_puts(&r, ",\"edits\":[{\"start\":0,\"deleteCount\":");
    lsp_sb_put_u(&r, old_count);
    lsp_sb_puts(&r, ",\"data\":[");
    lsp_sb_puts(&r, data.b ? data.b : "");
    lsp_sb_puts(&r, "]}]}");
    free(data.b);
    reply_raw(id, r.b ? r.b : "{\"edits\":[]}");
    free(r.b);
}

/* Decode a JSON string value starting at the opening quote into a malloc'd buffer. */
char *json_decode_string_at(const char *v) {
    if (!v || *v != '"') return NULL;
    size_t cap = 4096, n = 0;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;
    v++;
    while (*v) {
        if (n + 8 >= cap) {
            cap *= 2;
            char *g = (char *)realloc(out, cap);
            if (!g) { free(out); return NULL; }
            out = g;
        }
        if (*v == '\\' && v[1]) {
            v++;
            char c = *v++;
            if (c == 'n') out[n++] = '\n';
            else if (c == 't') out[n++] = '\t';
            else if (c == 'r') out[n++] = '\r';
            else if (c == '"' || c == '\\' || c == '/') out[n++] = c;
            else if (c == 'u' && v[0] && v[1] && v[2] && v[3]) {
                v += 4;
                out[n++] = '?';
            } else out[n++] = c;
        } else if (*v == '"') {
            out[n] = 0;
            return out;
        } else out[n++] = *v++;
    }
    free(out);
    return NULL;
}

static void handle_did_open(const char *params) {
    const char *td = json_val(params, "textDocument");
    if (!td) return;
    char uri[1024];
    if (!json_get_string(td, "uri", uri, sizeof uri)) return;
    const char *tv = json_val(td, "text");
    char *full = json_decode_string_at(tv);
    Doc *d = doc_ensure(uri);
    if (!d) { free(full); return; }
    d->version = (int)json_get_int(td, "version", 0);
    doc_set_text(d, full ? full : "");
    free(full);
    publish_diags(d);
}

static void handle_did_change(const char *params) {
    const char *td = json_val(params, "textDocument");
    if (!td) return;
    char uri[1024];
    if (!json_get_string(td, "uri", uri, sizeof uri)) return;
    Doc *d = doc_find(uri);
    if (!d) d = doc_ensure(uri);
    if (!d) return;
    d->version = (int)json_get_int(td, "version", d->version);
    const char *cc = json_val(params, "contentChanges");
    if (!cc) return;
    const char *item = strchr(cc, '{');
    if (!item) return;
    const char *tv = json_val(item, "text");
    char *text = json_decode_string_at(tv);
    if (!text) return;
    uint32_t start = 0, end = 0;
    if (range_offsets_from_json(d, item, &start, &end))
        (void)doc_apply_change(d, start, end, text);
    else
        doc_set_text(d, text);
    free(text);
    publish_diags(d);
}

static void handle_did_close(const char *params) {
    const char *td = json_val(params, "textDocument");
    char uri[1024];
    if (td && json_get_string(td, "uri", uri, sizeof uri)) {
        /* clear diags */
        Doc *d = doc_find(uri);
        if (d) {
            SB p; memset(&p, 0, sizeof p);
            lsp_sb_puts(&p, "{\"uri\":"); lsp_sb_json_str(&p, uri);
            lsp_sb_puts(&p, ",\"diagnostics\":[]}");
            notify("textDocument/publishDiagnostics", p.b);
            free(p.b);
        }
        doc_close(uri);
    }
}

void dispatch(const char *msg) {
    char method[128];
    char id[128];
    json_get_id_raw(msg, id, sizeof id);
    int has_id = (strcmp(id, "null") != 0);
    if (!json_get_string(msg, "method", method, sizeof method)) {
        /* response to our request — ignore */
        return;
    }
    const char *params = json_val(msg, "params");
    if (!params) params = "{}";

    if (strcmp(method, "initialize") == 0 && has_id) {
        handle_initialize(id);
    } else if (strcmp(method, "initialized") == 0) {
        /* ack nothing */
    } else if (strcmp(method, "shutdown") == 0 && has_id) {
        g_shutdown = 1;
        reply_null(id);
    } else if (strcmp(method, "exit") == 0) {
        exit(g_shutdown ? 0 : 1);
    } else if (strcmp(method, "textDocument/didOpen") == 0) {
        handle_did_open(params);
    } else if (strcmp(method, "textDocument/didChange") == 0) {
        handle_did_change(params);
    } else if (strcmp(method, "textDocument/didClose") == 0) {
        handle_did_close(params);
    } else if (strcmp(method, "textDocument/hover") == 0 && has_id) {
        handle_hover(id, params);
    } else if (strcmp(method, "textDocument/definition") == 0 && has_id) {
        handle_definition(id, params);
    } else if (strcmp(method, "textDocument/completion") == 0 && has_id) {
        handle_completion(id, params);
    } else if (strcmp(method, "textDocument/documentSymbol") == 0 && has_id) {
        handle_document_symbol(id, params);
    } else if (strcmp(method, "textDocument/references") == 0 && has_id) {
        handle_references(id, params);
    } else if (strcmp(method, "textDocument/prepareRename") == 0 && has_id) {
        handle_prepare_rename(id, params);
    } else if (strcmp(method, "textDocument/rename") == 0 && has_id) {
        handle_rename(id, params);
    } else if (strcmp(method, "textDocument/foldingRange") == 0 && has_id) {
        handle_folding(id, params);
    } else if (strcmp(method, "textDocument/signatureHelp") == 0 && has_id) {
        handle_signature_help(id, params);
    } else if (strcmp(method, "textDocument/codeAction") == 0 && has_id) {
        handle_code_action(id, params);
    } else if (strcmp(method, "textDocument/formatting") == 0 && has_id) {
        handle_formatting(id, params);
    } else if (strcmp(method, "textDocument/rangeFormatting") == 0 && has_id) {
        handle_range_formatting(id, params);
    } else if (strcmp(method, "textDocument/semanticTokens/full") == 0 && has_id) {
        handle_semantic_tokens_full(id, params);
    } else if (strcmp(method, "textDocument/semanticTokens/full/delta") == 0 && has_id) {
        handle_semantic_tokens_delta(id, params);
    } else if (strcmp(method, "workspace/didChangeWorkspaceFolders") == 0) {
        /* notification: no response */
    } else if (has_id) {
        /* method not found — return null result rather than error */
        reply_null(id);
    }
}
