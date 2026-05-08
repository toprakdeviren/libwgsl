/**
 * @file glob.h — minimal glob matcher (`*`, `**`, literal).
 *
 * No `?`, no `[…]` ranges, no escapes — the WGSL project config only
 * needs path globs of the `doublestar-slash glob (any depth, any *.wgsl)` shape and we'd rather hand-roll
 * a tight matcher than vendor a library for it.
 *
 * Semantics:
 *
 *   `*`        matches any run of characters except `/`.
 *   `**`       matches any run of characters including `/`.  An
 *              optional trailing `/` after `**` is consumed; this
 *              lets `doublestar-slash glob (any depth, any *.wgsl)` match top-level files, not just
 *              files in a subdirectory.
 *   anything   else matches literally (case-sensitive).
 *
 * Paths are forward-slash separated; callers are responsible for
 * normalising platform separators before calling.
 */
#ifndef WGSL_INTERNAL_GLOB_H
#define WGSL_INTERNAL_GLOB_H

#include <stdbool.h>

/** Returns true iff `path` matches `pattern` under the rules above. */
bool wgsl_glob_match(const char *pattern, const char *path);

#endif /* WGSL_INTERNAL_GLOB_H */
