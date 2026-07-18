/**
 * @file msl.c — WGSL to Metal Shading Language (MSL) emitter.
 *
 * Walks a checked (or parseable) WGSL AST and emits MSL text.
 * Type / builtin / attribute tables are the reverse of the
 * Metal→WGSL maps in the sibling metal-to-wgsl project (read-only
 * reference; no code dependency).
 *
 * Current coverage:
 *   - scalars/vecs/mats (incl. predeclared aliases)
 *   - structs, functions, lets/vars/consts
 *   - @vertex / @fragment / @compute → vertex / fragment / kernel
 *   - @builtin / @location / @group+@binding → [[…]] attributes
 *   - address spaces: storage→device, uniform→constant,
 *     workgroup→threadgroup, private/function→thread
 *   - common control flow + expressions
 *   - entry-IO lowering (stage_in / free builtins)
 *   - resource data-plane: module-scope storage/uniform/workgroup/
 *     texture/sampler → entry parameters (never silent module-scope
 *     device globals — those do not compile as Metal)
 *
 * Unsupported / partial constructs still best-effort emit, but
 * append *typed* WGSLMslDiagCode findings (not message scraping).
 */
#include "wgsl.h"
#include "internal/result.h"
#include "internal/ast.h"
#include "internal/token.h"
#include "internal/types.h"
#include "internal/check.h"
#include "internal/resolver.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Entry-body rewrite: stage_in parameter uses become in.field references. */
#define MSL_RW_MAX 32
#define MSL_SLOT_MAX 32
#define MSL_RES_MAX  64

/* Module-scope resources lowered to entry parameters. */
typedef enum {
    MSL_RES_BUFFER = 0,   /* var<storage> → device T* [[buffer(N)]] */
    MSL_RES_CONSTANT,     /* var<uniform> → constant T* [[buffer(N)]] */
    MSL_RES_THREADGROUP,  /* var<workgroup> → threadgroup T* [[threadgroup(N)]] */
    MSL_RES_TEXTURE,      /* handle texture → [[texture(N)]] */
    MSL_RES_SAMPLER,      /* handle sampler → [[sampler(N)]] */
    MSL_RES_PRIVATE       /* var<private> → thread local at entry body start */
} MslResKind;

typedef struct {
    const WGSLNode *decl;
    MslResKind      kind;
    int             read_only;   /* storage access mode read */
    int             has_binding;
    long long       binding;
    const WGSLNode *type_spec;
    char            name[64];
    uint32_t        name_off, name_len;
} MslRes;

typedef struct {
    char  *buf;
    size_t len, cap;
    int    indent;
    int    at_line_start;
    int    oom;
    const char *src;
    size_t src_len;
    const WGSLResult *res; /* may be NULL if emit from parse-only */
    /* Active only while emitting a graphics entry body. */
    const WGSLSymbol *rw_sym[MSL_RW_MAX];
    char              rw_to[MSL_RW_MAX][72];
    int               rw_n;
    /* Structured findings (owned by WGSLMslEmit after finish). */
    WGSLMslDiag *diags;
    int          ndiags, dcap;
    int          incomplete;
    /* loop+continuing lowers continue/break via do-while flags.
     * `loop_break_nest`: depth of nested switch/for/while/loop *inside*
     * the desugared body.  Flag-rewrite applies only when nest==0 so
     * break targets the nearest WGSL loop-or-switch, not the outer loop. */
    int          loop_cf_depth;     /* nested loop-with-continuing count */
    int          loop_in_body;      /* 1 while emitting body of such a loop */
    int          loop_break_nest;   /* nested breakables inside that body */
    /* Resource data-plane (module-scope → entry params). */
    MslRes       resources[MSL_RES_MAX];
    int          nresources;
} M;

struct WGSLMslEmit {
    char       *msl;
    int         incomplete;
    WGSLMslDiag *diags;
    int         ndiags;
};

#include "msl_common.inc"
#include "msl_types_resources.inc"
#include "msl_expr_stmt.inc"
#include "msl_entry.inc"
#include "msl_public.inc"
