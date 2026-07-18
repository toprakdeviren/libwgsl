/**
 * @file module_json_priv.h — multi-TU surface for module JSON reflection.
 *
 * `module_json.c` was split into src/api/module_json/ TUs (json_buf,
 * entry_usage, entry_points, pipelines, module_emit). Shared buffer types
 * and cross-TU emit helpers live here.
 */
#ifndef WGSL_INTERNAL_MODULE_JSON_PRIV_H
#define WGSL_INTERNAL_MODULE_JSON_PRIV_H

#include "wgsl.h"
#include "internal/result.h"
#include "internal/layout.h"
#include "internal/token.h"
#include "internal/types.h"
#include "internal/ast.h"
#include "internal/check.h"
#include "internal/consteval.h"
#include "internal/resolver.h"
#include "internal/source.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* JSON assembly buffer. */

typedef struct {
    char  *buf;
    size_t len;
    size_t cap;
    int    oom;
} JsonBuf;

/* entry-scope usage analysis. */

enum {
    REFLECT_USE_SUBGROUPS  = 1u << 0,
    REFLECT_USE_F16        = 1u << 1,
};

typedef struct {
    WGSLResult      *r;
    const WGSLNode **fns;       /* reachable fn decls: worklist + visited set */
    uint32_t         fn_count, fn_cap;
    WGSLSymbol     **vars;      /* referenced module-scope resource vars       */
    uint32_t         var_count, var_cap;
    uint32_t         uses;      /* REFLECT_USE_* bitset                        */
    int              oom;       /* alloc failure or depth truncation           */
} EntryUsage;

typedef struct {
    long long group;
    long long binding;
    WGSLSymbol *sym;
} ReflectBindRef;

/* json_buf.c. */

void jb_grow(JsonBuf *j, size_t need);
void jb_putc(JsonBuf *j, char c);
void jb_putn(JsonBuf *j, const char *s, size_t n);
void jb_puts(JsonBuf *j, const char *s);
void jb_put_int(JsonBuf *j, long long v);
void jb_put_jstr(JsonBuf *j, const char *s, size_t n);
void jb_put_span(JsonBuf *j, const WGSLSource *src, uint32_t off, uint32_t len);
void jb_put_type(JsonBuf *j, WGSLTypeInfo *t);
void jb_put_resource_type(WGSLResult *r, JsonBuf *j, WGSLTypeInfo *t);
long long wgsl_attr_int_arg(WGSLResult *r, WGSLNode *arg);
void jb_put_wgs_dim(WGSLResult *r, JsonBuf *j, WGSLNode *arg);
WGSLNode *json_find_attr(WGSLResult *r, const WGSLNode *parent,
                         uint32_t first, uint32_t count, const char *want);
const char *json_stage_for(WGSLResult *r, WGSLNode *fn, uint32_t fn_attr_count);

/* entry_usage.c. */

void eu_collect(EntryUsage *eu, const WGSLNode *entry_fn);
void reflect_fill_bind_refs(WGSLResult *r, EntryUsage *eu,
                            ReflectBindRef **out_refs, uint32_t *out_n);
uint64_t reflect_workgroup_bytes(WGSLResult *r, EntryUsage *eu);
void emit_layout_resource_fields(WGSLResult *r, JsonBuf *j, WGSLSymbol *sym);
void emit_entry_usage(WGSLResult *r, JsonBuf *j, const WGSLNode *fn,
                      const char *stage);

/* entry_points.c. */

void jb_put_ident_name(WGSLResult *r, JsonBuf *j, const WGSLNode *id);
void emit_entry_io(WGSLResult *r, JsonBuf *j, const WGSLNode *fn);
void emit_entry_points(WGSLResult *r, JsonBuf *j);

/* pipelines.c. */

/* Returns 1 if the V×F product exceeded REFLECT_PIPELINE_RENDER_CAP and
 * render pipelines were omitted (`analysis_truncated`). */
int emit_pipelines(WGSLResult *r, JsonBuf *j);

/* module_emit.c. */

void emit_resources(WGSLResult *r, JsonBuf *j);
void emit_structs(WGSLResult *r, JsonBuf *j);
void emit_overrides(WGSLResult *r, JsonBuf *j);
void emit_directives(WGSLResult *r, JsonBuf *j);
void build_module_json(WGSLResult *r);

const char *wgsl_module_json(const WGSLResult *r);
size_t wgsl_module_json_len(const WGSLResult *r);

#endif /* WGSL_INTERNAL_MODULE_JSON_PRIV_H */
