/**
 * §2.4 — public ABI layout table must match real sizeof/offsetof of
 * the structs declared in wgsl.h (so foreign bindings can trust
 * `wgsl_abi_layout()` instead of hand-copied offsets).
 */
#include "wgsl.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

static int fail;

#define CHECK(cond, msg) do {                                   \
    if (!(cond)) { fail++; fprintf(stderr, "FAIL  %s\n", msg); } \
} while (0)

int main(void) {
    const WGSLAbiLayout *L = wgsl_abi_layout();
    CHECK(L != NULL, "wgsl_abi_layout non-NULL");
    CHECK(L->abi_version == WGSL_ABI_LAYOUT_VERSION, "abi_version");
    CHECK(L->ptr_size == (uint32_t)sizeof(void *), "ptr_size");

    CHECK(L->diagnostic_size == sizeof(WGSLDiagnostic), "diag size");
    CHECK(L->diagnostic_severity == offsetof(WGSLDiagnostic, severity), "diag.severity");
    CHECK(L->diagnostic_line == offsetof(WGSLDiagnostic, line), "diag.line");
    CHECK(L->diagnostic_column == offsetof(WGSLDiagnostic, column), "diag.column");
    CHECK(L->diagnostic_end_line == offsetof(WGSLDiagnostic, end_line), "diag.end_line");
    CHECK(L->diagnostic_end_column == offsetof(WGSLDiagnostic, end_column), "diag.end_column");
    CHECK(L->diagnostic_message == offsetof(WGSLDiagnostic, message), "diag.message");
    CHECK(L->diagnostic_rule == offsetof(WGSLDiagnostic, rule), "diag.rule");

    CHECK(L->lex_token_size == sizeof(WGSLLexToken), "lex size");
    CHECK(L->lex_token_kind == offsetof(WGSLLexToken, kind), "lex.kind");
    CHECK(L->lex_token_offset == offsetof(WGSLLexToken, offset), "lex.offset");
    CHECK(L->lex_token_length == offsetof(WGSLLexToken, length), "lex.length");
    CHECK(L->lex_token_line == offsetof(WGSLLexToken, line), "lex.line");
    CHECK(L->lex_token_column == offsetof(WGSLLexToken, column), "lex.column");

    CHECK(L->hover_size == sizeof(WGSLHover), "hover size");
    CHECK(L->hover_present == offsetof(WGSLHover, present), "hover.present");
    CHECK(L->hover_type_repr == offsetof(WGSLHover, type_repr), "hover.type_repr");
    CHECK(L->hover_signature == offsetof(WGSLHover, signature), "hover.signature");
    CHECK(L->hover_doc == offsetof(WGSLHover, doc), "hover.doc");
    CHECK(L->hover_symbol_offset == offsetof(WGSLHover, symbol_offset), "hover.symbol_offset");
    CHECK(L->hover_symbol_length == offsetof(WGSLHover, symbol_length), "hover.symbol_length");

    CHECK(L->definition_size == sizeof(WGSLDefinition), "def size");
    CHECK(L->definition_present == offsetof(WGSLDefinition, present), "def.present");
    CHECK(L->definition_offset == offsetof(WGSLDefinition, offset), "def.offset");
    CHECK(L->definition_length == offsetof(WGSLDefinition, length), "def.length");

    CHECK(L->interp_step_cap == WGSL_INTERP_STEP_CAP, "interp step cap");
    CHECK(L->interp_buffer_values_cap == WGSL_INTERP_BUFFER_VALUES_CAP, "interp buf cap");
    CHECK(L->interp_max_lanes == WGSL_INTERP_MAX_LANES, "interp max lanes");

    fprintf(stderr, "%s  test_abi_layout  (%d failure%s)  ptr_size=%u diag=%u hover=%u\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s",
            L->ptr_size, L->diagnostic_size, L->hover_size);
    return fail ? 1 : 0;
}
