/**
 * TOML subset parser unit tests — sections, scalar strings, arrays,
 * comments, escapes, error recovery.
 */
#include "internal/toml.h"

#include <stdio.h>
#include <string.h>

static int fail = 0;

#define CHECK(cond, name)                                                   \
    do {                                                                    \
        if (!(cond)) { fprintf(stderr, "  fail: %s\n", name); fail += 1; }  \
    } while (0)

int main(void) {
    /* Empty input. */
    {
        WGSLToml *t = wgsl_toml_parse("", 0, NULL, 0);
        CHECK(t != NULL,                                  "empty: parses");
        CHECK(wgsl_toml_get_string(t, "x", "y") == NULL,  "empty: missing key");
        wgsl_toml_free(t);
    }

    /* Single root-level key. */
    {
        const char src[] = "name = \"alpha\"\n";
        WGSLToml *t = wgsl_toml_parse(src, sizeof src - 1, NULL, 0);
        CHECK(t != NULL, "root: parses");
        const char *v = wgsl_toml_get_string(t, "", "name");
        CHECK(v && strcmp(v, "alpha") == 0, "root: name == 'alpha'");
        wgsl_toml_free(t);
    }

    /* Section + scalar + array. */
    {
        const char src[] =
            "# project config\n"
            "[preamble]\n"
            "file = \"00_shared.wgsl\"\n"
            "files = [\"a.wgsl\", \"b.wgsl\"]\n"
            "auto_inject_into = [\"src/*.wgsl\"]\n";
        WGSLToml *t = wgsl_toml_parse(src, sizeof src - 1, NULL, 0);
        CHECK(t != NULL, "preamble: parses");
        CHECK(strcmp(wgsl_toml_get_string(t, "preamble", "file"), "00_shared.wgsl") == 0,
              "preamble.file");
        int n = 0;
        const char *const *arr = wgsl_toml_get_string_array(t, "preamble", "files", &n);
        CHECK(arr && n == 2 && strcmp(arr[0], "a.wgsl") == 0 && strcmp(arr[1], "b.wgsl") == 0,
              "preamble.files");
        const char *const *gl = wgsl_toml_get_string_array(t, "preamble", "auto_inject_into", &n);
        CHECK(gl && n == 1 && strcmp(gl[0], "src/*.wgsl") == 0,
              "preamble.auto_inject_into");
        wgsl_toml_free(t);
    }

    /* Empty array. */
    {
        const char src[] = "[s]\nks = []\n";
        WGSLToml *t = wgsl_toml_parse(src, sizeof src - 1, NULL, 0);
        int n = -1;
        const char *const *arr = wgsl_toml_get_string_array(t, "s", "ks", &n);
        CHECK(arr != NULL && n == 0, "empty array");
        wgsl_toml_free(t);
    }

    /* Trailing comma in array. */
    {
        const char src[] = "[s]\na = [\"x\", \"y\",]\n";
        WGSLToml *t = wgsl_toml_parse(src, sizeof src - 1, NULL, 0);
        int n = 0;
        const char *const *arr = wgsl_toml_get_string_array(t, "s", "a", &n);
        CHECK(arr && n == 2, "trailing comma");
        wgsl_toml_free(t);
    }

    /* String escapes. */
    {
        const char src[] = "x = \"line\\n2\\twith\\\"quote\"\n";
        WGSLToml *t = wgsl_toml_parse(src, sizeof src - 1, NULL, 0);
        const char *v = wgsl_toml_get_string(t, "", "x");
        CHECK(v && strcmp(v, "line\n2\twith\"quote") == 0, "escapes");
        wgsl_toml_free(t);
    }

    /* Dotted section name → flattened. */
    {
        const char src[] = "[a.b]\nx = \"v\"\n";
        WGSLToml *t = wgsl_toml_parse(src, sizeof src - 1, NULL, 0);
        const char *v = wgsl_toml_get_string(t, "a.b", "x");
        CHECK(v && strcmp(v, "v") == 0, "dotted section");
        wgsl_toml_free(t);
    }

    /* Error: missing close quote. */
    {
        char err[128] = {0};
        const char src[] = "x = \"unterminated\n";
        WGSLToml *t = wgsl_toml_parse(src, sizeof src - 1, err, sizeof err);
        CHECK(t == NULL,        "error returns NULL");
        CHECK(err[0] != '\0',   "error message populated");
        wgsl_toml_free(t);
    }

    /* Error: array of tables not yet supported. */
    {
        char err[128] = {0};
        const char src[] = "[[preamble]]\nfile = \"a.wgsl\"\n";
        WGSLToml *t = wgsl_toml_parse(src, sizeof src - 1, err, sizeof err);
        CHECK(t == NULL, "rejects [[…]]");
        wgsl_toml_free(t);
    }

    if (fail == 0) {
        printf("PASS  test_toml  (root, sections, arrays, escapes, errors)\n");
        return 0;
    }
    fprintf(stderr, "FAIL  test_toml  %d check(s) failed\n", fail);
    return 1;
}
