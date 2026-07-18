/**
 * Builtin overload cross-check.
 *
 * For a wide corpus of (builtin, args) calls, records:
 *   accepted?  (typecheck has no ERROR diags on the call)
 *   return_type kind string
 *   is_const flag from kBuiltinTable (WGSL_BIF_CONST)
 *
 * Golden fingerprint of the full report is locked.  Refactors of the
 * overload motor must keep this fingerprint byte-identical (or update
 * the golden with a documented intentional change + finding).
 *
 * Scope: check-level accept/reject + result type kind + constness.
 * Texture builtins are covered lightly (BS_TEXTURE path); numeric bulk
 * covers the former dual-motor surface.
 */
#include "internal/check.h"
#include "internal/lexer.h"
#include "internal/parser.h"
#include "internal/resolver.h"
#include "internal/consteval.h"
#include "internal/types.h"
#include "internal/exprs_priv.h"
#include "internal/utf8.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail;
#define CHECK(c, m) do { if (!(c)) { \
  fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, m); fail++; } } while (0)

typedef struct {
    const char *name;
    const char *args;       /* inside call parens, e.g. "1.0" or "vec3f(1,2,3)" */
    const char *enable;     /* optional preamble, e.g. "enable f16;\n" */
} Case;

/* Representative valid + invalid shapes across families. */
static const Case k_cases[] = {
    /* numeric T */
    { "abs", "1", NULL },
    { "abs", "1.0", NULL },
    { "abs", "1u", NULL },
    { "abs", "vec3f(1.0)", NULL },
    { "abs", "true", NULL }, /* invalid */
    { "min", "1, 2", NULL },
    { "min", "1.0, 2.0", NULL },
    { "min", "1, 2.0", NULL },
    { "min", "1", NULL }, /* arity */
    { "max", "vec2i(1,2), vec2i(3,4)", NULL },
    { "clamp", "1, 0, 2", NULL },
    { "clamp", "1.0, 0.0, 2.0", NULL },
    { "clamp", "1, 0", NULL }, /* arity */
    /* float T */
    { "sin", "1.0", NULL },
    { "sin", "1", NULL }, /* abstract int → promote */
    { "sin", "1u", NULL }, /* invalid u32 */
    { "sin", "vec2f(0.0)", NULL },
    { "pow", "2.0, 3.0", NULL },
    { "pow", "2, 3", NULL },
    { "atan2", "1.0, 2.0", NULL },
    { "log2", "32", NULL },
    { "log2", "32u", NULL }, /* invalid */
    /* integer bits */
    { "countOneBits", "0xFu", NULL },
    { "countOneBits", "1.0", NULL }, /* invalid */
    { "reverseBits", "1u", NULL },
    { "extractBits", "0xFFu, 0u, 4u", NULL },
    { "insertBits", "0u, 0xFu, 4u, 4u", NULL },
    /* sign family (no u32) */
    { "sign", "1", NULL },
    { "sign", "1.0", NULL },
    { "sign", "1u", NULL }, /* invalid */
    /* select / any / all */
    { "select", "1, 2, true", NULL },
    { "select", "1.0, 2.0, false", NULL },
    { "select", "1, 2, 3", NULL }, /* invalid cond */
    { "any", "true", NULL },
    { "any", "vec3(true, false, true)", NULL },
    { "all", "vec2(true, true)", NULL },
    { "all", "1", NULL }, /* invalid */
    /* geometric */
    { "dot", "vec3f(1,2,3), vec3f(4,5,6)", NULL },
    { "dot", "1.0, 2.0", NULL }, /* invalid scalar */
    { "cross", "vec3f(1,0,0), vec3f(0,1,0)", NULL },
    { "cross", "vec2f(1,0), vec2f(0,1)", NULL }, /* invalid */
    { "length", "vec3f(3,4,0)", NULL },
    { "length", "3.0", NULL },
    { "normalize", "vec3f(0,3,4)", NULL },
    { "normalize", "1.0", NULL }, /* invalid scalar */
    { "distance", "vec2f(0,0), vec2f(3,4)", NULL },
    { "reflect", "vec3f(1,0,0), vec3f(0,1,0)", NULL },
    { "faceForward", "vec3f(1,0,0), vec3f(0,1,0), vec3f(0,0,1)", NULL },
    { "refract", "vec3f(1,0,0), vec3f(0,1,0), 0.5", NULL },
    /* matrix */
    { "transpose", "mat2x3f(1,2,3,4,5,6)", NULL },
    { "determinant", "mat2x2f(1,2,3,4)", NULL },
    { "determinant", "mat2x3f(1,2,3,4,5,6)", NULL }, /* invalid non-square */
    /* mix / ldexp / frexp */
    { "mix", "1.0, 2.0, 0.5", NULL },
    { "mix", "vec3f(1,2,3), vec3f(4,5,6), 0.5", NULL },
    { "ldexp", "1.0, 2", NULL },
    { "ldexp", "1.0, 2u", NULL },
    { "frexp", "1.5", NULL },
    { "modf", "1.5", NULL },
    /* pack / unpack */
    { "pack4x8unorm", "vec4f(1,0,0,0)", NULL },
    { "pack4x8unorm", "1.0", NULL }, /* invalid */
    { "unpack4x8unorm", "255u", NULL },
    { "unpack4x8unorm", "1.0", NULL }, /* invalid */
    { "dot4U8Packed", "1u, 2u", NULL },
    /* f16 */
    { "abs", "1.0h", "enable f16;\n" },
    { "sin", "1.0h", "enable f16;\n" },
    /* barriers / void */
    { "workgroupBarrier", "", NULL },
    { "storageBarrier", "", NULL },
    /* arrayLength — needs storage ptr; expect fail in free fn */
    { "arrayLength", "&x", NULL },
    /* subgroup (enable) */
    { "subgroupAdd", "1u", "enable subgroups;\n" },
    { "subgroupAdd", "1u", NULL }, /* missing enable still types but may error */
    { "subgroupMax", "vec2u(1,2)", "enable subgroups;\n" },
    /* bitcast */
    { "bitcast", "1i", NULL }, /* needs template — special case below */
};

typedef struct {
    WGSLArena arena;
    WGSLSource src;
    WGSLDiagBag diag;
    WGSLLexResult lex;
    WGSLAst ast;
    WGSLTypeStore types;
    WGSLResolver res;
    WGSLConstEvaluator cev;
    WGSLTypeChecker tc;
} Pipe;

static void pipe_done(Pipe *p) {
    wgsl_typecheck_destroy(&p->tc);
    wgsl_consteval_destroy(&p->cev);
    wgsl_resolver_destroy(&p->res);
    wgsl_types_destroy(&p->types);
    wgsl_diag_destroy(&p->diag);
    wgsl_source_destroy(&p->src);
    wgsl_arena_destroy(&p->arena);
}

static int pipe_run(Pipe *p, const char *text) {
    memset(p, 0, sizeof *p);
    wgsl_arena_init(&p->arena);
    wgsl_diag_init(&p->diag);
    wgsl_source_init(&p->src, text, strlen(text));
    if (!wgsl_tokenize(&p->src, &p->arena, &p->diag, &p->lex)) return 0;
    if (!wgsl_parse(&p->lex, &p->src, &p->arena, &p->diag, &p->ast)) return 0;
    wgsl_types_init(&p->types, &p->arena);
    if (!wgsl_resolve(&p->ast, &p->src, &p->arena, &p->diag, &p->types, &p->res))
        return 0;
    (void)wgsl_consteval(&p->ast, &p->src, &p->arena, &p->diag, &p->types,
                         &p->res, &p->cev);
    return wgsl_typecheck(&p->ast, &p->src, &p->arena, &p->diag, &p->types,
                          &p->res, &p->cev, &p->tc);
}

static const WGSLNode *find_call_named(const WGSLNode *n, const WGSLSource *src,
                                       const char *want)
{
    if (!n) return NULL;
    if (n->kind == WGSL_NODE_EXPR_CALL && n->child_count >= 1) {
        const WGSLNode *cal = n->children[0];
        const WGSLNode *id = cal;
        if (cal && cal->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
            cal->child_count >= 1)
            id = cal->children[0];
        if (id && id->kind == WGSL_NODE_EXPR_IDENT) {
            uint32_t off = wgsl_node_name_span(id).offset;
            uint32_t len = wgsl_node_name_span(id).length;
            if (len == strlen(want) &&
                memcmp(src->bytes + off, want, len) == 0)
                return n;
        }
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        const WGSLNode *h = find_call_named(n->children[i], src, want);
        if (h) return h;
    }
    return NULL;
}

static void type_kind_label(const WGSLTypeInfo *t, char *buf, size_t cap) {
    if (!t) { snprintf(buf, cap, "null"); return; }
    const char *k = wgsl_type_kind_name((WGSLTypeKind)t->kind);
    if (t->kind == WGSL_TYPE_VEC && t->ref) {
        snprintf(buf, cap, "vec%u<%s>", (unsigned)t->width,
                 wgsl_type_kind_name((WGSLTypeKind)((const WGSLTypeInfo *)t->ref)->kind));
    } else if (t->kind == WGSL_TYPE_MAT && t->ref) {
        snprintf(buf, cap, "mat%ux%u<%s>", (unsigned)t->width, (unsigned)t->rows,
                 wgsl_type_kind_name((WGSLTypeKind)((const WGSLTypeInfo *)t->ref)->kind));
    } else {
        snprintf(buf, cap, "%s", k ? k : "?");
    }
}

static void append_line(char **buf, size_t *len, size_t *cap, const char *line) {
    size_t n = strlen(line);
    if (*len + n + 2 > *cap) {
        size_t nc = *cap ? *cap * 2 : 4096;
        while (nc < *len + n + 2) nc *= 2;
        char *p = (char *)realloc(*buf, nc);
        if (!p) return;
        *buf = p; *cap = nc;
    }
    memcpy(*buf + *len, line, n);
    *len += n;
    (*buf)[(*len)++] = '\n';
    (*buf)[*len] = 0;
}

static int table_is_const(const char *name) {
    const WGSLBuiltinEntry *e = find_builtin_entry(name, (uint32_t)strlen(name));
    return e && (e->flags & WGSL_BIF_CONST) ? 1 : 0;
}

/* FNV-1a 64 of the report — compact golden. */
static uint64_t fnv1a64(const char *s, size_t n) {
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; i++) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211ull;
    }
    return h;
}

int main(void) {
    wgsl_utf8_init();

    char *report = NULL;
    size_t rlen = 0, rcap = 0;
    int ncases = (int)(sizeof k_cases / sizeof k_cases[0]);
    int n_ok = 0, n_fail = 0;

    for (int i = 0; i < ncases; i++) {
        const Case *c = &k_cases[i];
        char src[1024];
        if (strcmp(c->name, "bitcast") == 0) {
            snprintf(src, sizeof src,
                     "%s"
                     "fn f() {\n"
                     "  let r = bitcast<i32>(%s);\n"
                     "  _ = r;\n"
                     "}\n",
                     c->enable ? c->enable : "",
                     c->args[0] ? c->args : "1i");
        } else if (strcmp(c->name, "arrayLength") == 0) {
            snprintf(src, sizeof src,
                     "@group(0) @binding(0) var<storage, read> buf: array<f32>;\n"
                     "fn f() {\n"
                     "  let r = arrayLength(&buf);\n"
                     "  _ = r;\n"
                     "}\n");
        } else if (c->args[0] == 0) {
            snprintf(src, sizeof src,
                     "%s"
                     "@compute @workgroup_size(1)\n"
                     "fn f() { %s(); }\n",
                     c->enable ? c->enable : "", c->name);
        } else {
            snprintf(src, sizeof src,
                     "%s"
                     "fn f() {\n"
                     "  let r = %s(%s);\n"
                     "  _ = r;\n"
                     "}\n",
                     c->enable ? c->enable : "", c->name, c->args);
        }

        Pipe p;
        int tc_ok = pipe_run(&p, src);
        const WGSLNode *call = find_call_named(p.ast.root, &p.src, c->name);
        int accepted = 0;
        char tlab[96] = "none";
        if (call) {
            WGSLTypeInfo *ty = wgsl_typecheck_type_of(&p.tc, (WGSLNode *)call);
            type_kind_label(ty, tlab, sizeof tlab);
            /* accepted = typed call with no overload/shape ERROR diags. */
            accepted = 0;
            if (ty && ty->kind != WGSL_TYPE_INVALID && ty->kind != WGSL_TYPE_VOID) {
                int overload_fail = 0;
                for (size_t d = 0; d < p.diag.count; d++) {
                    const char *m = p.diag.items[d].message;
                    if (!m) continue;
                    if (p.diag.items[d].severity != WGSL_DIAG_ERROR) continue;
                    if (strstr(m, "no matching overload") ||
                        strstr(m, "no generated") ||
                        strstr(m, "does not accept"))
                    {
                        if (strstr(m, c->name) || strstr(m, "no matching overload") ||
                            strstr(m, "no generated"))
                            overload_fail = 1;
                    }
                    if (strstr(m, c->name) &&
                        (strstr(m, "requires") || strstr(m, "expects") ||
                         strstr(m, "argument")))
                        overload_fail = 1;
                }
                accepted = !overload_fail;
            } else if (ty && ty->kind == WGSL_TYPE_VOID) {
                /* barriers etc. */
                accepted = (p.diag.error_count == 0) || tc_ok;
            }
            (void)tc_ok;
        }
        int is_const = table_is_const(c->name);
        char line[256];
        snprintf(line, sizeof line, "%s(%s)|acc=%d|ret=%s|const=%d",
                 c->name, c->args, accepted, tlab, is_const);
        append_line(&report, &rlen, &rcap, line);
        if (accepted) n_ok++; else n_fail++;
        pipe_done(&p);
    }

    /* Also enumerate every kBuiltinTable name + is_const (coverage gate). */
    {
        char line[128];
        snprintf(line, sizeof line, "#table_count=%d", WGSL_BUILTIN_TABLE_COUNT);
        append_line(&report, &rlen, &rcap, line);
        int const_n = 0;
        for (int i = 0; i < WGSL_BUILTIN_TABLE_COUNT; i++) {
            const WGSLBuiltinEntry *e = &kBuiltinTable[i];
            int c = (e->flags & WGSL_BIF_CONST) ? 1 : 0;
            if (c) const_n++;
            snprintf(line, sizeof line, "T|%s|const=%d|cfold=%u",
                     e->name, c, (unsigned)e->cfold);
            append_line(&report, &rlen, &rcap, line);
        }
        snprintf(line, sizeof line, "#const_builtins=%d", const_n);
        append_line(&report, &rlen, &rcap, line);
    }

    uint64_t h = fnv1a64(report, rlen);
    int lines = 0;
    for (size_t i = 0; i < rlen; i++) if (report[i] == '\n') lines++;

    /* Write report for inspection / golden update. */
    {
        FILE *f = fopen("tests/07-check/goldens/builtin_overload_crosscheck.report.txt", "w");
        if (!f) {
            /* generate dir once */
            if (system("mkdir -p tests/07-check/goldens") != 0) {
                /* best effort; fopen below decides whether we can write */
            }
            f = fopen("tests/07-check/goldens/builtin_overload_crosscheck.report.txt", "w");
        }
        if (f) {
            fprintf(f, "# fnv=%016llx lines=%d cases=%d ok=%d fail=%d\n",
                    (unsigned long long)h, lines, ncases, n_ok, n_fail);
            fwrite(report, 1, rlen, f);
            fclose(f);
        }
    }

    /* Load golden fingerprint if present. */
    {
        FILE *g = fopen("tests/07-check/goldens/builtin_overload_crosscheck.fnv", "r");
        if (!g) {
            /* First pin: write golden. */
            if (system("mkdir -p tests/07-check/goldens") != 0) {
                /* best effort; fopen below decides whether we can write */
            }
            g = fopen("tests/07-check/goldens/builtin_overload_crosscheck.fnv", "w");
            if (g) {
                fprintf(g, "%016llx\n%d\n", (unsigned long long)h, lines);
                fclose(g);
                fprintf(stderr,
                        "PINNED  crosscheck golden fnv=%016llx lines=%d "
                        "(wrote tests/07-check/goldens/builtin_overload_crosscheck.fnv)\n",
                        (unsigned long long)h, lines);
            }
        } else {
            unsigned long long gh = 0;
            int gl = 0;
            if (fscanf(g, "%llx\n%d", &gh, &gl) != 2) {
                CHECK(0, "golden fnv parse");
            }
            fclose(g);
            if ((uint64_t)gh != h || gl != lines) {
                fprintf(stderr,
                        "FAIL  crosscheck fingerprint mismatch\n"
                        "  golden fnv=%016llx lines=%d\n"
                        "  got    fnv=%016llx lines=%d\n"
                        "  see tests/07-check/goldens/builtin_overload_crosscheck.report.txt\n",
                        gh, gl, (unsigned long long)h, lines);
                fail++;
            } else {
                fprintf(stderr,
                        "PASS  crosscheck fingerprint match "
                        "fnv=%016llx lines=%d cases=%d\n",
                        (unsigned long long)h, lines, ncases);
            }
        }
    }

    /* Structural gates. */
    CHECK(WGSL_BUILTIN_TABLE_COUNT >= 140, "table has full builtin set");
    CHECK(ncases >= 50, "corpus has ≥50 call shapes");
    CHECK(n_ok > 0 && n_fail > 0, "corpus has both accept and reject");

    /* Every const-foldable numeric name has non-NONE cfold metadata. */
    {
        static const char *must_fold[] = {
            "abs", "sin", "min", "clamp", "dot", "select", "pack4x8unorm",
            "countOneBits", "transpose", "frexp", NULL
        };
        for (int i = 0; must_fold[i]; i++) {
            const WGSLBuiltinEntry *e =
                find_builtin_entry(must_fold[i], (uint32_t)strlen(must_fold[i]));
            CHECK(e && e->cfold != WGSL_CFOLD_NONE,
                  "const builtin has def-driven cfold");
        }
    }

    free(report);
    fprintf(stderr, "%s  test_builtin_overload_crosscheck  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
