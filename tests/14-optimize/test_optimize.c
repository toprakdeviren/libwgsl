/**
 * Optimization analysis + safe apply.
 */
#include "wgsl.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail;
#define CHECK(c, m) do { if (!(c)) { \
  fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, m); fail++; } } while (0)

static int has_suffix(const char *name, const char *suffix) {
    size_t nl = strlen(name), sl = strlen(suffix);
    return nl >= sl && memcmp(name + nl - sl, suffix, sl) == 0;
}

static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0 || n > (1 << 20)) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *s = (char *)malloc((size_t)n + 1);
    if (!s) { fclose(f); return NULL; }
    size_t got = fread(s, 1, (size_t)n, f);
    fclose(f);
    s[got] = 0;
    return s;
}

static void check_apply_then_check_file(const char *path, int *count) {
    char *src = slurp(path);
    CHECK(src != NULL, "corpus: readable");
    if (!src) return;
    WGSLResult *before = wgsl_check(src);
    CHECK(before != NULL && wgsl_ok(before), "corpus: input check ok");
    if (before) wgsl_free(before);

    char *out = wgsl_optimize_apply(src);
    CHECK(out != NULL, "corpus: apply");
    if (out) {
        WGSLResult *after = wgsl_check(out);
        if (!after || !wgsl_ok(after)) {
            fprintf(stderr, "FAIL corpus apply-check: %s\n%s\n", path, out);
            fail++;
        }
        if (after) wgsl_free(after);
        wgsl_free_string(out);
    }
    free(src);
    *count += 1;
}

static void check_apply_then_check_dir(const char *dir_path, int *count) {
    DIR *d = opendir(dir_path);
    CHECK(d != NULL, "corpus: opendir");
    if (!d) return;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (!has_suffix(ent->d_name, ".wgsl")) continue;
        char path[512];
        snprintf(path, sizeof path, "%s/%s", dir_path, ent->d_name);
        check_apply_then_check_file(path, count);
    }
    closedir(d);
}

int main(void) {
    /* DCE: code after return */
    {
        const char *src =
            "fn f() {\n"
            "  return;\n"
            "  let dead: i32 = 1;\n"
            "}\n";
        char *j = wgsl_optimize_json_src(src);
        CHECK(j != NULL, "dce: json");
        CHECK(strstr(j, "\"dce\"") != NULL, "dce: pass listed");
        CHECK(strstr(j, "unreachable_after_terminal") != NULL, "dce: finding");
        CHECK(strstr(j, "\"dce\":1") != NULL || strstr(j, "\"dce\": 1") != NULL ||
              strstr(j, "unreachable") != NULL, "dce: summary or finding");
        wgsl_free_string(j);

        char *out = wgsl_optimize_apply(src);
        CHECK(out != NULL, "dce apply");
        CHECK(strstr(out, "dead") == NULL, "dce apply: removed dead let");
        CHECK(strstr(out, "return") != NULL, "dce apply: kept return");
        wgsl_free_string(out);
    }

    /* Unused local + unused function */
    {
        const char *src =
            "fn helper() {}\n"
            "@compute @workgroup_size(1)\n"
            "fn main() {\n"
            "  let x: i32 = 1;\n"
            "  let y: i32 = 2;\n"
            "  _ = y;\n"
            "}\n";
        char *j = wgsl_optimize_json_src(src);
        CHECK(j != NULL, "unused: json");
        CHECK(strstr(j, "unused_local") != NULL, "unused: local x");
        CHECK(strstr(j, "unused_function") != NULL, "unused: helper");
        wgsl_free_string(j);

        char *out = wgsl_optimize_apply(src);
        CHECK(out != NULL, "unused apply");
        CHECK(strstr(out, "let x") == NULL, "unused apply: removed local x");
        CHECK(strstr(out, "helper") == NULL, "unused apply: removed helper");
        CHECK(strstr(out, "_ = y") != NULL, "unused apply: kept live y");
        WGSLResult *r = wgsl_check(out);
        CHECK(r != NULL && wgsl_ok(r), "unused apply: check ok");
        if (r) wgsl_free(r);
        wgsl_free_string(out);
    }

    /* Const branch */
    {
        const char *src =
            "fn f() {\n"
            "  if (true) { let a = 1; _ = a; }\n"
            "  else { let b = 2; _ = b; }\n"
            "}\n";
        char *j = wgsl_optimize_json_src(src);
        CHECK(j != NULL, "branch: json");
        CHECK(strstr(j, "const_branch") != NULL, "branch: pass");
        CHECK(strstr(j, "if_true") != NULL, "branch: if_true");
        wgsl_free_string(j);

        char *out = wgsl_optimize_apply(src);
        CHECK(out != NULL, "branch apply");
        CHECK(strstr(out, "if (true)") == NULL, "branch apply: removed if");
        CHECK(strstr(out, "let a") != NULL, "branch apply: kept then");
        CHECK(strstr(out, "let b") == NULL, "branch apply: removed else");
        WGSLResult *r = wgsl_check(out);
        CHECK(r != NULL && wgsl_ok(r), "branch apply: check ok");
        if (r) wgsl_free(r);
        wgsl_free_string(out);
    }

    /* CSE */
    {
        const char *src =
            "fn f(a: i32, b: i32) {\n"
            "  let x = a + b * 2;\n"
            "  let y = a + b * 2;\n"
            "  _ = x + y;\n"
            "}\n";
        char *j = wgsl_optimize_json_src(src);
        CHECK(j != NULL, "cse: json");
        CHECK(strstr(j, "common_subexpression") != NULL ||
              strstr(j, "\"cse\"") != NULL, "cse: finding or pass");
        wgsl_free_string(j);

        char *out = wgsl_optimize_apply(src);
        CHECK(out != NULL, "cse apply");
        CHECK(strstr(out, "wgsl_opt_cse") != NULL, "cse apply: temp let");
        CHECK(strstr(out, "let y = a + b * 2") == NULL,
              "cse apply: replaced repeated expr");
        WGSLResult *r = wgsl_check(out);
        CHECK(r != NULL && wgsl_ok(r), "cse apply: check ok");
        if (r) wgsl_free(r);
        wgsl_free_string(out);
    }

    /* Entry point is not "unused" */
    {
        const char *src =
            "@compute @workgroup_size(1)\n"
            "fn main() {}\n";
        char *j = wgsl_optimize_json_src(src);
        CHECK(j != NULL, "entry: json");
        CHECK(strstr(j, "unused_function") == NULL, "entry: not unused");
        wgsl_free_string(j);
    }

    /* Clean module: empty findings ok */
    {
        const char *src = "fn f() { let x = 1; _ = x; }\n";
        char *j = wgsl_optimize_json_src(src);
        CHECK(j != NULL, "clean: json");
        CHECK(strstr(j, "\"findings\":[") != NULL, "clean: findings array");
        wgsl_free_string(j);
    }

    /* Corpus apply∘check: every in-tree compute fixture remains valid WGSL
     * after the safe rewrite pipeline. */
    {
        int corpus = 0;
        check_apply_then_check_dir("tests/12-diff/exec", &corpus);
        check_apply_then_check_dir("tests/15-golden/cases", &corpus);
        CHECK(corpus >= 50, "corpus: walked enough fixtures");
    }

    fprintf(stderr, "%s  test_optimize  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
