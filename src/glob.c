/**
 * @file glob.c — minimal glob matcher (`*`, `**`, literal).
 *
 * Recursive backtracking — for the patterns we accept (a few `*`/`**`
 * wildcards) the recursion depth is bounded by the wildcard count,
 * not by the path length.  Our config globs are typically one or two
 * wildcards (`doublestar-slash glob (any depth, any *.wgsl)`), so this is fast in practice.
 */
#include "internal/glob.h"

bool wgsl_glob_match(const char *pattern, const char *path) {
    /* End-of-pattern: only matches end-of-path. */
    if (pattern[0] == '\0') return path[0] == '\0';

    /* `**` — match any number of characters including `/`. */
    if (pattern[0] == '*' && pattern[1] == '*') {
        const char *next = pattern + 2;
        if (*next == '/') next++;     /* `**​/` collapses to "0+ dirs" */
        for (const char *p = path; ; p++) {
            if (wgsl_glob_match(next, p)) return true;
            if (*p == '\0') return false;
        }
    }

    /* `*` — match any number of characters except `/`. */
    if (pattern[0] == '*') {
        for (const char *p = path; ; p++) {
            if (wgsl_glob_match(pattern + 1, p)) return true;
            if (*p == '\0' || *p == '/') return false;
        }
    }

    /* Literal byte. */
    if (path[0] != '\0' && pattern[0] == path[0]) {
        return wgsl_glob_match(pattern + 1, path + 1);
    }
    return false;
}
