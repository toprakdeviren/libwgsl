#include "wgsl.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const char *severity_name(WGSLDiagSeverity severity) {
    switch (severity) {
    case WGSL_DIAG_ERROR:   return "error";
    case WGSL_DIAG_WARNING: return "warning";
    case WGSL_DIAG_INFO:    return "info";
    case WGSL_DIAG_HINT:    return "hint";
    }
    return "unknown";
}

static void print_diagnostics(const char *label, const WGSLResult *r) {
    int count = wgsl_diagnostic_count(r);
    for (int i = 0; i < count; i++) {
        const WGSLDiagnostic *d = wgsl_diagnostic(r, i);
        if (!d) continue;
        fprintf(stderr, "%s:%u:%u: %s: %s\n",
                label,
                d->line,
                d->column,
                severity_name(d->severity),
                d->message ? d->message : "");
    }
}

static int find_offset(const char *src, const char *needle) {
    const char *p = strstr(src, needle);
    return p ? (int)(p - src) : -1;
}

int main(void) {
    const char *shader =
        "@compute @workgroup_size(1)\n"
        "fn main() {\n"
        "  let value: i32 = 1 + 2;\n"
        "  _ = value;\n"
        "}\n";

    wgsl_init();

    printf("wgsl %s, spec %s, Unicode %s\n",
           WGSL_VERSION_STRING, wgsl_spec_pin(), wgsl_unicode_version());

    WGSLResult *r = wgsl_check(shader);
    if (!r) {
        fprintf(stderr, "wgsl_check returned NULL\n");
        wgsl_shutdown();
        return 2;
    }

    if (!wgsl_ok(r)) {
        print_diagnostics("shader", r);
        wgsl_free(r);
        wgsl_shutdown();
        return 1;
    }

    puts("\nmodule json:");
    fwrite(wgsl_module_json(r), 1, wgsl_module_json_len(r), stdout);
    fputc('\n', stdout);

    int value_use = find_offset(shader, "_ = value");
    if (value_use >= 0) {
        uint32_t value_offset = (uint32_t)value_use + 4u;
        WGSLHover hover = wgsl_hover_at(r, value_offset);
        if (hover.present) {
            printf("\nhover: %.*s => %s\n",
                   (int)hover.symbol_length,
                   shader + hover.symbol_offset,
                   hover.type_repr ? hover.type_repr : "");
        }

        WGSLDefinition def = wgsl_definition_at(r, value_offset);
        if (def.present) {
            printf("definition: %.*s at byte %u\n",
                   (int)def.length,
                   shader + def.offset,
                   def.offset);
        }
    }

    WGSLLexToken *tokens = NULL;
    int token_count = 0;
    if (wgsl_semantic_tokens(shader, &tokens, &token_count)) {
        int function_names = 0;
        int locals = 0;
        for (int i = 0; i < token_count; i++) {
            if (tokens[i].kind == WGSL_LEX_FUNCTION_NAME) function_names++;
            if (tokens[i].kind == WGSL_LEX_LET) locals++;
        }
        printf("semantic tokens: %d total, %d function name(s), %d local-let token(s)\n",
               token_count, function_names, locals);
        wgsl_lex_free(tokens);
    }

    wgsl_free(r);

    const char *bad_shader = "fn f() { let x: i32 = true; }\n";
    WGSLResult *bad = wgsl_check(bad_shader);
    if (!bad) {
        fprintf(stderr, "wgsl_check returned NULL for invalid shader\n");
        wgsl_shutdown();
        return 2;
    }
    if (wgsl_ok(bad)) {
        fprintf(stderr, "expected invalid shader to fail validation\n");
        wgsl_free(bad);
        wgsl_shutdown();
        return 1;
    }
    puts("\nexpected diagnostic:");
    fflush(stdout);
    print_diagnostics("bad_shader", bad);
    wgsl_free(bad);

    wgsl_shutdown();
    return 0;
}
