/**
 * Project loader unit tests — config parse + match() resolution.
 */
#include "internal/project.h"
#include "wgsl.h"

#include <stdio.h>
#include <string.h>

static int fail = 0;

#define CHECK(cond, name)                                                   \
    do {                                                                    \
        if (!(cond)) { fprintf(stderr, "  fail: %s\n", name); fail += 1; }  \
    } while (0)

int main(void) {
    /* Empty config: project opens, but match returns nothing. */
    {
        WGSLProject *p = wgsl_project_open_from_string("", 0, NULL, 0);
        CHECK(p != NULL, "empty: opens");
        int n = -1;
        CHECK(wgsl_project_match(p, "anything.wgsl", &n) == NULL,
              "empty: match() returns NULL");
        CHECK(n == 0, "empty: match() count == 0");
        wgsl_project_close(p);
    }

    /* file = single + auto_inject_into. */
    {
        const char src[] =
            "[preamble]\n"
            "file = \"00_shared.wgsl\"\n"
            "auto_inject_into = [\"kernels/*.wgsl\"]\n";
        WGSLProject *p = wgsl_project_open_from_string(src, sizeof src - 1, NULL, 0);
        CHECK(p != NULL, "single file: opens");

        int n = 0;
        const char *const *files = wgsl_project_match(p, "kernels/foo.wgsl", &n);
        CHECK(files && n == 1 && strcmp(files[0], "00_shared.wgsl") == 0,
              "single file: kernels/foo.wgsl matches");

        files = wgsl_project_match(p, "other/foo.wgsl", &n);
        CHECK(files == NULL && n == 0, "single file: other path skipped");
        wgsl_project_close(p);
    }

    /* files = list, multiple inject globs. */
    {
        const char src[] =
            "[preamble]\n"
            "files = [\"00_shared.wgsl\", \"01_helpers.wgsl\"]\n"
            "auto_inject_into = [\"kernels/*.wgsl\", \"util/*.wgsl\"]\n";
        WGSLProject *p = wgsl_project_open_from_string(src, sizeof src - 1, NULL, 0);
        CHECK(p != NULL, "list: opens");

        int n = 0;
        const char *const *files = wgsl_project_match(p, "kernels/a.wgsl", &n);
        CHECK(files && n == 2, "list: 2 preamble files");
        CHECK(files && strcmp(files[0], "00_shared.wgsl") == 0,   "list: order [0]");
        CHECK(files && strcmp(files[1], "01_helpers.wgsl") == 0,  "list: order [1]");

        files = wgsl_project_match(p, "util/b.wgsl", &n);
        CHECK(files && n == 2, "list: second glob hits");

        files = wgsl_project_match(p, "examples/c.wgsl", &n);
        CHECK(files == NULL, "list: no glob hits → no inject");
        wgsl_project_close(p);
    }

    /* file + files: file goes first. */
    {
        const char src[] =
            "[preamble]\n"
            "file  = \"00_shared.wgsl\"\n"
            "files = [\"01_extra.wgsl\"]\n"
            "auto_inject_into = [\"*.wgsl\"]\n";
        WGSLProject *p = wgsl_project_open_from_string(src, sizeof src - 1, NULL, 0);
        int n = 0;
        const char *const *files = wgsl_project_match(p, "norm.wgsl", &n);
        CHECK(files && n == 2, "file+files: count == 2");
        CHECK(files && strcmp(files[0], "00_shared.wgsl") == 0,  "file goes first");
        CHECK(files && strcmp(files[1], "01_extra.wgsl") == 0,   "files appended");
        wgsl_project_close(p);
    }

    /* Self-preamble exclusion: the preamble file matching its own
     * inject glob does not get itself injected. */
    {
        const char src[] =
            "[preamble]\n"
            "files = [\"00_shared.wgsl\"]\n"
            "auto_inject_into = [\"*.wgsl\"]\n";
        WGSLProject *p = wgsl_project_open_from_string(src, sizeof src - 1, NULL, 0);
        int n = -1;
        const char *const *files = wgsl_project_match(p, "00_shared.wgsl", &n);
        CHECK(files == NULL && n == 0,
              "self-preamble: shared file does not inject into itself");
        files = wgsl_project_match(p, "kernel.wgsl", &n);
        CHECK(files != NULL && n == 1,
              "self-preamble: other files still get the preamble");
        wgsl_project_close(p);
    }

    /* Preamble composition: preamble[k] receives preamble[0..k-1]
     * regardless of whether it matches the inject globs.  A later
     * preamble that depends on an earlier one (e.g. infrastructure
     * helpers calling shared `flat_id`) must resolve cleanly when
     * the user opens it directly. */
    {
        const char src[] =
            "[preamble]\n"
            "files = [\"00_shared.wgsl\", \"01_infra.wgsl\", \"02_extra.wgsl\"]\n"
            "auto_inject_into = [\"*.wgsl\"]\n";
        WGSLProject *p = wgsl_project_open_from_string(src, sizeof src - 1, NULL, 0);
        int n = -1;

        const char *const *files = wgsl_project_match(p, "00_shared.wgsl", &n);
        CHECK(files == NULL && n == 0,
              "preamble[0]: empty prefix");

        files = wgsl_project_match(p, "01_infra.wgsl", &n);
        CHECK(files != NULL && n == 1 && strcmp(files[0], "00_shared.wgsl") == 0,
              "preamble[1]: gets preamble[0] only");

        files = wgsl_project_match(p, "02_extra.wgsl", &n);
        CHECK(files != NULL && n == 2 &&
              strcmp(files[0], "00_shared.wgsl") == 0 &&
              strcmp(files[1], "01_infra.wgsl") == 0,
              "preamble[2]: gets preamble[0..1]");

        files = wgsl_project_match(p, "kernel.wgsl", &n);
        CHECK(files != NULL && n == 3, "non-preamble: full set");
        wgsl_project_close(p);
    }

    /* No `auto_inject_into` → no preamble for any file. */
    {
        const char src[] =
            "[preamble]\n"
            "file = \"00_shared.wgsl\"\n";
        WGSLProject *p = wgsl_project_open_from_string(src, sizeof src - 1, NULL, 0);
        int n = -1;
        CHECK(wgsl_project_match(p, "norm.wgsl", &n) == NULL,
              "no globs: match() returns NULL");
        CHECK(n == 0, "no globs: count == 0");
        wgsl_project_close(p);
    }

    /* check_with_preamble: preamble + main concat compiles cleanly. */
    {
        const char preamble[] = "fn helper() -> f32 { return 1.0; }\n";
        const char main_src[] =
            "@compute @workgroup_size(1)\n"
            "fn cs() { let v = helper(); }\n";
        WGSLResult *r = wgsl_check_with_preamble(
            preamble, sizeof preamble - 1,
            main_src, sizeof main_src - 1);
        CHECK(r != NULL,  "preamble: returns result");
        if (r && !wgsl_ok(r)) {
            fprintf(stderr, "    diag: %s\n", wgsl_error(r));
        }
        CHECK(r && wgsl_ok(r), "preamble: compiles ok");
        wgsl_free(r);
    }

    /* check_with_preamble: missing helper without preamble fails. */
    {
        const char main_src[] =
            "@compute @workgroup_size(1)\n"
            "fn cs() { let v = helper(); }\n";
        WGSLResult *r = wgsl_check_with_preamble(
            NULL, 0, main_src, sizeof main_src - 1);
        CHECK(r && !wgsl_ok(r), "no preamble: fails as expected");
        wgsl_free(r);
    }

    if (fail == 0) {
        printf("PASS  test_project  (config + match + check_with_preamble)\n");
        return 0;
    }
    fprintf(stderr, "FAIL  test_project  %d check(s) failed\n", fail);
    return 1;
}
