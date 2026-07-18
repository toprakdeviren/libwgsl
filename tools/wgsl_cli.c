#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wgsl.h"
#include "internal/arena.h"
#include "internal/ast_dump.h"
#include "internal/diag.h"
#include "internal/lexer.h"
#include "internal/parser.h"
#include "internal/source.h"

typedef struct {
    char  *bytes;
    size_t length;
} InputBuffer;

static const char *diag_severity_name(WGSLDiagSeverity severity) {
    switch (severity) {
    case WGSL_DIAG_ERROR:   return "error";
    case WGSL_DIAG_WARNING: return "warning";
    case WGSL_DIAG_INFO:    return "info";
    case WGSL_DIAG_HINT:    return "hint";
    }
    return "unknown";
}

static void print_usage(FILE *out, const char *prog) {
    fprintf(out,
        "usage:\n"
        "  %s check [--quiet] [--json] FILE...\n"
        "  %s json FILE\n"
        "  %s ast FILE\n"
        "  %s interp [--entry NAME] [--gx N] [--gy N] [--gz N]\n"
        "           [--len N] [--seeds JSON] FILE\n"
        "  %s debug [--entry NAME] [--gx N] [--gy N] [--gz N]\n"
        "          [--len N] [--seeds JSON] FILE\n"
        "  %s optimize [--apply] FILE\n"
        "  %s msl FILE\n"
        "  %s ml FILE\n"
        "  %s version\n"
        "\n"
        "FILE may be '-' to read from stdin.\n"
        "\n"
        "exit codes:\n"
        "  0  valid input / command succeeded\n"
        "  1  WGSL validation failed (check) or interp ok:false\n"
        "  2  usage, I/O, or internal CLI error\n",
        prog, prog, prog, prog, prog, prog, prog, prog, prog);
}

static int read_stream(FILE *f, InputBuffer *out, char *err, size_t err_len) {
    size_t capacity = 16384;
    size_t length = 0;
    char *bytes = (char *)malloc(capacity + 1);
    if (!bytes) {
        snprintf(err, err_len, "out of memory");
        return 0;
    }

    for (;;) {
        if (length == capacity) {
            size_t next_capacity = capacity * 2;
            char *next = (char *)realloc(bytes, next_capacity + 1);
            if (!next) {
                free(bytes);
                snprintf(err, err_len, "out of memory");
                return 0;
            }
            bytes = next;
            capacity = next_capacity;
        }

        size_t n = fread(bytes + length, 1, capacity - length, f);
        length += n;
        if (n == 0) {
            if (ferror(f)) {
                free(bytes);
                snprintf(err, err_len, "read failed: %s", strerror(errno));
                return 0;
            }
            break;
        }
    }

    bytes[length] = '\0';
    out->bytes = bytes;
    out->length = length;
    return 1;
}

static int read_file(const char *path, InputBuffer *out, char *err, size_t err_len) {
    memset(out, 0, sizeof(*out));
    if (strcmp(path, "-") == 0) {
        return read_stream(stdin, out, err, err_len);
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        snprintf(err, err_len, "open failed: %s", strerror(errno));
        return 0;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        int ok = read_stream(f, out, err, err_len);
        fclose(f);
        return ok;
    }

    long end = ftell(f);
    if (end < 0) {
        fclose(f);
        snprintf(err, err_len, "tell failed: %s", strerror(errno));
        return 0;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        snprintf(err, err_len, "seek failed: %s", strerror(errno));
        return 0;
    }

    size_t length = (size_t)end;
    char *bytes = (char *)malloc(length + 1);
    if (!bytes) {
        fclose(f);
        snprintf(err, err_len, "out of memory");
        return 0;
    }

    size_t n = fread(bytes, 1, length, f);
    if (n != length) {
        int read_error = ferror(f);
        free(bytes);
        fclose(f);
        snprintf(err, err_len, "read failed: %s",
                 read_error ? strerror(errno) : "short read");
        return 0;
    }
    fclose(f);

    bytes[length] = '\0';
    out->bytes = bytes;
    out->length = length;
    return 1;
}

static void json_write_string(FILE *out, const char *s) {
    fputc('"', out);
    if (s) {
        for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
            switch (*p) {
            case '"':  fputs("\\\"", out); break;
            case '\\': fputs("\\\\", out); break;
            case '\b': fputs("\\b", out); break;
            case '\f': fputs("\\f", out); break;
            case '\n': fputs("\\n", out); break;
            case '\r': fputs("\\r", out); break;
            case '\t': fputs("\\t", out); break;
            default:
                if (*p < 0x20) {
                    fprintf(out, "\\u%04x", (unsigned)*p);
                } else {
                    fputc((int)*p, out);
                }
                break;
            }
        }
    }
    fputc('"', out);
}

static void print_diagnostic_text(const char *path, const WGSLDiagnostic *d) {
    const char *message = d && d->message ? d->message : "";
    const char *rule = d && d->rule ? d->rule : "";
    fprintf(stderr, "%s:%u:%u: %s", path, d->line, d->column,
            diag_severity_name(d->severity));
    if (rule[0]) {
        fprintf(stderr, "[%s]", rule);
    }
    fprintf(stderr, ": %s\n", message);
}

static void print_result_diagnostics_text(const char *path, const WGSLResult *r) {
    int count = wgsl_diagnostic_count(r);
    for (int i = 0; i < count; i++) {
        const WGSLDiagnostic *d = wgsl_diagnostic(r, i);
        if (d) {
            print_diagnostic_text(path, d);
        }
    }
}

static void print_diag_bag_text(const char *path, const WGSLDiagBag *bag) {
    size_t count = wgsl_diag_count(bag);
    for (size_t i = 0; i < count; i++) {
        const WGSLDiagnostic *d = wgsl_diag_at(bag, i);
        if (d) {
            print_diagnostic_text(path, d);
        }
    }
}

static void write_diagnostic_json(FILE *out, const WGSLDiagnostic *d) {
    fprintf(out, "{\"severity\":");
    json_write_string(out, diag_severity_name(d->severity));
    fprintf(out, ",\"line\":%u,\"column\":%u,\"endLine\":%u,\"endColumn\":%u",
            d->line, d->column, d->end_line, d->end_column);
    fprintf(out, ",\"rule\":");
    json_write_string(out, d->rule ? d->rule : "");
    fprintf(out, ",\"message\":");
    json_write_string(out, d->message ? d->message : "");
    fputc('}', out);
}

static void write_check_result_json(FILE *out, const char *path, const WGSLResult *r) {
    int count = wgsl_diagnostic_count(r);
    fprintf(out, "{\"file\":");
    json_write_string(out, path);
    fprintf(out, ",\"ok\":%s,\"diagnostics\":[", wgsl_ok(r) ? "true" : "false");
    for (int i = 0; i < count; i++) {
        const WGSLDiagnostic *d = wgsl_diagnostic(r, i);
        if (i > 0) {
            fputc(',', out);
        }
        if (d) {
            write_diagnostic_json(out, d);
        } else {
            fputs("null", out);
        }
    }
    fputs("]}", out);
}

static int run_check(int argc, char **argv, const char *prog) {
    int json = 0;
    int quiet = 0;
    int file_start = -1;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--") == 0) {
            file_start = i + 1;
            break;
        }
        if (strcmp(argv[i], "--json") == 0) {
            json = 1;
            continue;
        }
        if (strcmp(argv[i], "--quiet") == 0 || strcmp(argv[i], "-q") == 0) {
            quiet = 1;
            continue;
        }
        if (argv[i][0] == '-' && strcmp(argv[i], "-") != 0) {
            fprintf(stderr, "%s: unknown option for check: %s\n", prog, argv[i]);
            return 2;
        }
        file_start = i;
        break;
    }

    if (file_start < 0 || file_start >= argc) {
        print_usage(stderr, prog);
        return 2;
    }

    int any_invalid = 0;
    int any_hard_error = 0;
    if (json) {
        fputs("[", stdout);
    }

    for (int i = file_start; i < argc; i++) {
        char err[256];
        InputBuffer input;
        if (!read_file(argv[i], &input, err, sizeof(err))) {
            any_hard_error = 1;
            if (json) {
                if (i > file_start) {
                    fputc(',', stdout);
                }
                fprintf(stdout, "{\"file\":");
                json_write_string(stdout, argv[i]);
                fprintf(stdout, ",\"ok\":false,\"diagnostics\":[{\"severity\":\"error\","
                                "\"line\":0,\"column\":0,\"endLine\":0,\"endColumn\":0,"
                                "\"rule\":\"\",\"message\":");
                json_write_string(stdout, err);
                fputs("}]}", stdout);
            } else if (!quiet) {
                fprintf(stderr, "%s: %s\n", argv[i], err);
            }
            continue;
        }

        WGSLResult *r = wgsl_check_n(input.bytes, input.length);
        if (!r) {
            any_hard_error = 1;
            if (json) {
                if (i > file_start) {
                    fputc(',', stdout);
                }
                fprintf(stdout, "{\"file\":");
                json_write_string(stdout, argv[i]);
                fputs(",\"ok\":false,\"diagnostics\":[{\"severity\":\"error\","
                      "\"line\":0,\"column\":0,\"endLine\":0,\"endColumn\":0,"
                      "\"rule\":\"\",\"message\":\"wgsl_check_n returned null\"}]}",
                      stdout);
            } else if (!quiet) {
                fprintf(stderr, "%s: wgsl_check_n returned null\n", argv[i]);
            }
            free(input.bytes);
            continue;
        }

        int ok = wgsl_ok(r);
        if (!ok) {
            any_invalid = 1;
        }

        if (json) {
            if (i > file_start) {
                fputc(',', stdout);
            }
            write_check_result_json(stdout, argv[i], r);
        } else if (!quiet) {
            if (ok) {
                fprintf(stdout, "%s: ok\n", argv[i]);
            }
            print_result_diagnostics_text(argv[i], r);
        }

        wgsl_free(r);
        free(input.bytes);
    }

    if (json) {
        fputs("]\n", stdout);
    }

    if (any_hard_error) {
        return 2;
    }
    return any_invalid ? 1 : 0;
}

static int run_json(int argc, char **argv, const char *prog) {
    if (argc != 3) {
        print_usage(stderr, prog);
        return 2;
    }

    char err[256];
    InputBuffer input;
    if (!read_file(argv[2], &input, err, sizeof(err))) {
        fprintf(stderr, "%s: %s\n", argv[2], err);
        return 2;
    }

    WGSLResult *r = wgsl_check_n(input.bytes, input.length);
    if (!r) {
        free(input.bytes);
        fprintf(stderr, "%s: wgsl_check_n returned null\n", argv[2]);
        return 2;
    }

    int rc = 0;
    if (!wgsl_ok(r)) {
        print_result_diagnostics_text(argv[2], r);
        rc = 1;
    } else {
        print_result_diagnostics_text(argv[2], r);
        const char *json = wgsl_module_json(r);
        size_t json_len = wgsl_module_json_len(r);
        if (!json || json_len == 0) {
            fprintf(stderr, "%s: module JSON is empty\n", argv[2]);
            rc = 2;
        } else {
            fwrite(json, 1, json_len, stdout);
            fputc('\n', stdout);
        }
    }

    wgsl_free(r);
    free(input.bytes);
    return rc;
}

static int run_ast(int argc, char **argv, const char *prog) {
    if (argc != 3) {
        print_usage(stderr, prog);
        return 2;
    }

    char err[256];
    InputBuffer input;
    if (!read_file(argv[2], &input, err, sizeof(err))) {
        fprintf(stderr, "%s: %s\n", argv[2], err);
        return 2;
    }

    int rc = 0;
    WGSLSource source;
    if (!wgsl_source_init(&source, input.bytes, input.length)) {
        free(input.bytes);
        fprintf(stderr, "%s: out of memory while indexing source\n", argv[2]);
        return 2;
    }

    WGSLArena arena;
    WGSLDiagBag diag;
    WGSLLexResult lex = {0};
    WGSLAst ast = {0};
    wgsl_arena_init(&arena);
    wgsl_diag_init(&diag);

    if (!wgsl_tokenize(&source, &arena, &diag, &lex)) {
        rc = wgsl_diag_has_error(&diag) ? 1 : 2;
    } else if (!wgsl_parse(&lex, &source, &arena, &diag, &ast)) {
        rc = 1;
    }

    if (ast.root) {
        wgsl_ast_dump(ast.root, &source, stdout);
    }
    print_diag_bag_text(argv[2], &diag);

    wgsl_diag_destroy(&diag);
    wgsl_arena_destroy(&arena);
    wgsl_source_destroy(&source);
    free(input.bytes);
    return rc;
}

/* §6.3 — ML-kernel analysis JSON. */
static int run_ml(int argc, char **argv, const char *prog) {
    if (argc != 3) {
        fprintf(stderr, "%s ml: usage: %s ml FILE\n", prog, prog);
        return 2;
    }
    char err[256];
    InputBuffer input;
    if (!read_file(argv[2], &input, err, sizeof err)) {
        fprintf(stderr, "%s: %s\n", argv[2], err);
        return 2;
    }
    char *json = wgsl_ml_analyze_json_src(input.bytes);
    free(input.bytes);
    if (!json) {
        fprintf(stderr, "%s: ml_analyze failed\n", argv[2]);
        return 2;
    }
    fputs(json, stdout);
    if (json[0] && json[strlen(json) - 1] != '\n') fputc('\n', stdout);
    wgsl_free_string(json);
    return 0;
}

/* §6.1 — WGSL → MSL. */
static int run_msl(int argc, char **argv, const char *prog) {
    if (argc != 3) {
        fprintf(stderr, "%s msl: usage: %s msl FILE\n", prog, prog);
        return 2;
    }
    char err[256];
    InputBuffer input;
    if (!read_file(argv[2], &input, err, sizeof err)) {
        fprintf(stderr, "%s: %s\n", argv[2], err);
        return 2;
    }
    char *msl = wgsl_to_msl_n(input.bytes, input.length);
    free(input.bytes);
    if (!msl) {
        fprintf(stderr, "%s: wgsl_to_msl failed\n", argv[2]);
        return 2;
    }
    fputs(msl, stdout);
    if (msl[0] && msl[strlen(msl) - 1] != '\n') fputc('\n', stdout);
    wgsl_free_string(msl);
    return 0;
}

/* §6.2 — optimization analysis / safe apply. */
static int run_optimize(int argc, char **argv, const char *prog) {
    int apply = 0;
    const char *file = NULL;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--apply") == 0) {
            apply = 1;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "%s optimize: unknown option %s\n", prog, argv[i]);
            return 2;
        } else if (!file) {
            file = argv[i];
        } else {
            fprintf(stderr, "%s optimize: extra argument %s\n", prog, argv[i]);
            return 2;
        }
    }
    if (!file) {
        fprintf(stderr, "%s optimize: missing FILE\n", prog);
        return 2;
    }
    char err[256];
    InputBuffer input;
    if (!read_file(file, &input, err, sizeof err)) {
        fprintf(stderr, "%s: %s\n", file, err);
        return 2;
    }
    int rc = 0;
    if (apply) {
        char *out = wgsl_optimize_apply(input.bytes);
        free(input.bytes);
        if (!out) {
            fprintf(stderr, "%s: optimize_apply failed\n", file);
            return 2;
        }
        fputs(out, stdout);
        if (out[0] && out[strlen(out) - 1] != '\n') fputc('\n', stdout);
        wgsl_free_string(out);
    } else {
        char *json = wgsl_optimize_json_src(input.bytes);
        free(input.bytes);
        if (!json) {
            fprintf(stderr, "%s: optimize_json failed\n", file);
            return 2;
        }
        fputs(json, stdout);
        if (json[0] && json[strlen(json) - 1] != '\n') fputc('\n', stdout);
        wgsl_free_string(json);
    }
    return rc;
}

/* §5.4 — run the CPU interpreter; print the full JSON trace. */
static int run_interp(int argc, char **argv, const char *prog) {
    const char *entry = "main";
    unsigned gx = 0, gy = 0, gz = 0, len = 1;
    const char *seeds = NULL;
    const char *file = NULL;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--entry") == 0 && i + 1 < argc) {
            entry = argv[++i];
        } else if (strcmp(argv[i], "--gx") == 0 && i + 1 < argc) {
            gx = (unsigned)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--gy") == 0 && i + 1 < argc) {
            gy = (unsigned)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--gz") == 0 && i + 1 < argc) {
            gz = (unsigned)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--len") == 0 && i + 1 < argc) {
            len = (unsigned)strtoul(argv[++i], NULL, 10);
            if (len == 0) len = 1;
        } else if (strcmp(argv[i], "--seeds") == 0 && i + 1 < argc) {
            seeds = argv[++i];
        } else if (strcmp(argv[i], "-") == 0 && !file) {
            file = argv[i];
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "%s interp: unknown option %s\n", prog, argv[i]);
            print_usage(stderr, prog);
            return 2;
        } else if (!file) {
            file = argv[i];
        } else {
            fprintf(stderr, "%s interp: extra argument %s\n", prog, argv[i]);
            return 2;
        }
    }
    if (!file) {
        fprintf(stderr, "%s interp: missing FILE\n", prog);
        print_usage(stderr, prog);
        return 2;
    }

    char err[256];
    InputBuffer input;
    if (!read_file(file, &input, err, sizeof err)) {
        fprintf(stderr, "%s: %s\n", file, err);
        return 2;
    }

    char *json = wgsl_interp(input.bytes, entry, gx, gy, gz, len, seeds);
    free(input.bytes);
    if (!json) {
        fprintf(stderr, "%s: wgsl_interp returned null\n", file);
        return 2;
    }
    fputs(json, stdout);
    if (json[0] && json[strlen(json) - 1] != '\n') fputc('\n', stdout);

    int rc = 0;
    if (strstr(json, "\"ok\":false") != NULL) rc = 1;
    wgsl_free_string(json);
    return rc;
}

/* Differentiator wedge — CI JSON oracle + short explanations. */
static int run_debug(int argc, char **argv, const char *prog) {
    const char *entry = "main";
    unsigned gx = 0, gy = 0, gz = 0, len = 1;
    const char *seeds = NULL;
    const char *file = NULL;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--entry") == 0 && i + 1 < argc) {
            entry = argv[++i];
        } else if (strcmp(argv[i], "--gx") == 0 && i + 1 < argc) {
            gx = (unsigned)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--gy") == 0 && i + 1 < argc) {
            gy = (unsigned)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--gz") == 0 && i + 1 < argc) {
            gz = (unsigned)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--len") == 0 && i + 1 < argc) {
            len = (unsigned)strtoul(argv[++i], NULL, 10);
            if (len == 0) len = 1;
        } else if (strcmp(argv[i], "--seeds") == 0 && i + 1 < argc) {
            seeds = argv[++i];
        } else if (strcmp(argv[i], "-") == 0 && !file) {
            file = argv[i];
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "%s debug: unknown option %s\n", prog, argv[i]);
            print_usage(stderr, prog);
            return 2;
        } else if (!file) {
            file = argv[i];
        } else {
            fprintf(stderr, "%s debug: extra argument %s\n", prog, argv[i]);
            return 2;
        }
    }
    if (!file) {
        fprintf(stderr, "%s debug: missing FILE\n", prog);
        print_usage(stderr, prog);
        return 2;
    }

    char err[256];
    InputBuffer input;
    if (!read_file(file, &input, err, sizeof err)) {
        fprintf(stderr, "%s: %s\n", file, err);
        return 2;
    }

    char *json = wgsl_debug_oracle(input.bytes, entry, gx, gy, gz, len, seeds);
    free(input.bytes);
    if (!json) {
        fprintf(stderr, "%s: wgsl_debug_oracle returned null\n", file);
        return 2;
    }
    fputs(json, stdout);
    if (json[0] && json[strlen(json) - 1] != '\n') fputc('\n', stdout);

    int rc = strstr(json, "\"verdict\":\"fail\"") ? 1 : 0;
    wgsl_free_string(json);
    return rc;
}

int main(int argc, char **argv) {
    const char *prog = argc > 0 && argv[0] ? argv[0] : "wgsl";

    wgsl_init();

    int rc = 0;
    if (argc < 2 || strcmp(argv[1], "help") == 0 ||
        strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        print_usage(argc < 2 ? stderr : stdout, prog);
        rc = argc < 2 ? 2 : 0;
    } else if (strcmp(argv[1], "version") == 0 || strcmp(argv[1], "--version") == 0) {
        printf("wgsl %s (%s, Unicode %s)\n",
               WGSL_VERSION_STRING, wgsl_spec_pin(), wgsl_unicode_version());
    } else if (strcmp(argv[1], "check") == 0) {
        rc = run_check(argc, argv, prog);
    } else if (strcmp(argv[1], "json") == 0) {
        rc = run_json(argc, argv, prog);
    } else if (strcmp(argv[1], "ast") == 0) {
        rc = run_ast(argc, argv, prog);
    } else if (strcmp(argv[1], "interp") == 0) {
        rc = run_interp(argc, argv, prog);
    } else if (strcmp(argv[1], "debug") == 0) {
        rc = run_debug(argc, argv, prog);
    } else if (strcmp(argv[1], "optimize") == 0) {
        rc = run_optimize(argc, argv, prog);
    } else if (strcmp(argv[1], "msl") == 0) {
        rc = run_msl(argc, argv, prog);
    } else if (strcmp(argv[1], "ml") == 0) {
        rc = run_ml(argc, argv, prog);
    } else {
        fprintf(stderr, "%s: unknown command: %s\n", prog, argv[1]);
        print_usage(stderr, prog);
        rc = 2;
    }

    wgsl_shutdown();
    return rc;
}
