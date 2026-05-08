/**
 * Glob matcher unit tests — `*`, `**`, and literal segments.
 */
#include "internal/glob.h"

#include <stdio.h>
#include <string.h>

static int fail = 0;

#define CHECK(cond, name)                                                   \
    do {                                                                    \
        if (!(cond)) { fprintf(stderr, "  fail: %s\n", name); fail += 1; }  \
    } while (0)

int main(void) {
    /* Literal. */
    CHECK( wgsl_glob_match("foo.wgsl",       "foo.wgsl"),           "literal hit");
    CHECK(!wgsl_glob_match("foo.wgsl",       "bar.wgsl"),           "literal miss");
    CHECK(!wgsl_glob_match("foo",            "foox"),               "literal partial");

    /* Single `*`. */
    CHECK( wgsl_glob_match("*.wgsl",         "foo.wgsl"),           "* matches base");
    CHECK( wgsl_glob_match("*.wgsl",         ".wgsl"),              "* matches empty");
    CHECK(!wgsl_glob_match("*.wgsl",         "a/b.wgsl"),           "* refuses /");
    CHECK( wgsl_glob_match("kernels/*.wgsl", "kernels/foo.wgsl"),   "dir + *");
    CHECK(!wgsl_glob_match("kernels/*.wgsl", "kernels/sub/x.wgsl"), "dir + * no /");

    /* Double `**`. */
    CHECK( wgsl_glob_match("**", "anything"),                       "** matches all");
    CHECK( wgsl_glob_match("**", "a/b/c"),                          "** crosses /");
    CHECK( wgsl_glob_match("**", ""),                               "** matches empty");

    /* `**` at root + suffix. */
    CHECK( wgsl_glob_match("**/last", "last"),                      "** root + literal");
    CHECK( wgsl_glob_match("**/last", "a/last"),                    "** root + 1 dir");
    CHECK( wgsl_glob_match("**/last", "a/b/c/last"),                "** root + many dirs");
    CHECK(!wgsl_glob_match("**/last", "lastly"),                    "** root + suffix tight");

    /* The flagship case from the README — pattern built piecewise so
     * the literal byte sequence isn't ever spelled inside a block
     * comment by this file. */
    {
        const char *pat = "**";
        char buf[16]; strcpy(buf, pat); strcat(buf, "/"); strcat(buf, "*.wgsl");
        CHECK( wgsl_glob_match(buf, "foo.wgsl"),               "doublestar+glob: top-level");
        CHECK( wgsl_glob_match(buf, "kernels/foo.wgsl"),       "doublestar+glob: nested");
        CHECK( wgsl_glob_match(buf, "a/b/c/foo.wgsl"),         "doublestar+glob: deeply nested");
        CHECK(!wgsl_glob_match(buf, "foo.txt"),                "doublestar+glob: wrong ext");
        CHECK(!wgsl_glob_match(buf, "kernels/foo.txt"),        "doublestar+glob: wrong ext nested");
    }

    /* Mixed wildcards. */
    CHECK( wgsl_glob_match("src/*/file.wgsl", "src/a/file.wgsl"),    "single dir wildcard");
    CHECK(!wgsl_glob_match("src/*/file.wgsl", "src/a/b/file.wgsl"),  "single dir wildcard fails on nested");

    if (fail == 0) {
        printf("PASS  test_glob  (literal, *, **, mixed)\n");
        return 0;
    }
    fprintf(stderr, "FAIL  test_glob  %d check(s) failed\n", fail);
    return 1;
}
