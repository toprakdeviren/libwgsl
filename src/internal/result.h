/**
 * @file result.h — opaque WGSLResult full definition for internal TUs.
 *
 * Public headers only expose `typedef struct WGSLResult WGSLResult`.
 * Modules that implement the public API (module_json, lex/semantic,
 * hover, session, …) include this header so they can read the pipeline
 * state without living as #include fragments inside wgsl.c.
 */
#ifndef WGSL_INTERNAL_RESULT_H
#define WGSL_INTERNAL_RESULT_H

#include "internal/arena.h"
#include "internal/ast.h"
#include "internal/check.h"
#include "internal/consteval.h"
#include "internal/diag.h"
#include "internal/lexer.h"
#include "internal/parser.h"
#include "internal/resolver.h"
#include "internal/source.h"
#include "internal/types.h"
#include "internal/validate.h"

#include "wgsl.h"

struct WGSLResult {
    WGSLArena          arena;        /* AST + arena strings              */
    char              *src_copy;     /* heap-owned source bytes          */
    size_t             src_len;
    WGSLSource         source;
    WGSLDiagBag        diag;
    WGSLLexResult      lex;
    WGSLAst            ast;
    WGSLTypeStore      types;
    WGSLResolver       res;
    WGSLConstEvaluator cev;
    WGSLTypeChecker    tc;
    WGSLValidator      val;

    int                pipeline_ok;  /* lex && parse && error-free       */

    /* Lazy cache for `wgsl_module_json`. */
    char              *json_cache;
    size_t             json_cache_len;
    int                json_built;
};

/* Staged pipeline (session partial typecheck).  `wgsl_check_n` is the
 * full sequence; session may interleave fingerprinting between parse
 * and typecheck to decide which function bodies to skip. */

/** Lex → parse → resolve → const-eval.  Typecheck/validate not run. */
WGSLResult *wgsl_pipeline_begin(const char *src, size_t src_len);

/** Typecheck (+ optional body-skip plan).  No-op if parse failed. */
int wgsl_pipeline_typecheck(WGSLResult *r, const WGSLTypecheckOpts *opts);

/** Validate pass.  No-op if parse failed. */
int wgsl_pipeline_validate(WGSLResult *r);

/** Set `pipeline_ok` from pass results + diagnostic bag. */
void wgsl_pipeline_finalize(WGSLResult *r, int early_ok);

#endif /* WGSL_INTERNAL_RESULT_H */
