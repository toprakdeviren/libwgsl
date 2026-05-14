/**
 * Phase 4 — TypeInfo + interning + conversion rank + concretization.
 *
 * Covers:
 *   - sizeof(WGSLTypeInfo) gate (24 B, 8-byte aligned)
 *   - all 8 pre-interned scalars (void / bool / abstract*2 / i32 / u32 / f32 / f16)
 *   - vec / mat / atomic / array interning: same-shape calls return same ptr
 *   - predicates: scalar / numeric / integer / float / abstract / vector / matrix
 *   - conversion rank table from WGSL §6.1.2 (incl. composite-element rules)
 *   - concretization: AbstractInt → i32, AbstractFloat → f32, recursive
 *   - format: produces correct WGSL-shape strings
 */
#include "internal/types.h"
#include "internal/arena.h"
#include "internal/resolver.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int fail = 0;
#define CHECK(cond, msg) do {                                          \
    if (!(cond)) {                                                     \
        fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);  \
        fail += 1;                                                     \
    }                                                                  \
} while (0)

static void check_format(const WGSLTypeInfo *t, const char *want, const char *tag) {
    char buf[128];
    size_t n = wgsl_type_format(t, buf, sizeof buf);
    if (n != strlen(want) || strcmp(buf, want) != 0) {
        fprintf(stderr, "FAIL  %s: got \"%s\" (n=%zu), want \"%s\"\n",
                tag, buf, n, want);
        fail += 1;
    }
}

static void check_rank(const WGSLTypeInfo *src, const WGSLTypeInfo *dst,
                       int want, const char *tag) {
    int got = wgsl_type_conversion_rank(src, dst);
    if (got != want) {
        fprintf(stderr, "FAIL  %s: rank=%d, want=%d\n", tag, got, want);
        fail += 1;
    }
}

int main(void) {
    /* Layout gates */
    CHECK(sizeof(WGSLTypeInfo) == 24,    "sizeof(WGSLTypeInfo) == 24");
    CHECK(_Alignof(WGSLTypeInfo) == 8,   "alignof(WGSLTypeInfo) == 8");
    CHECK(WGSL_TYPE_KIND_COUNT > 10 && WGSL_TYPE_KIND_COUNT < (1u << 16),
          "kind enum count is sane");

    WGSLArena arena; wgsl_arena_init(&arena);
    WGSLTypeStore s;
    int init_ok = wgsl_types_init(&s, &arena);
    CHECK(init_ok == 1, "types init");

    /* Pre-interned scalars exist and are unique. */
    WGSLTypeInfo *t_bool  = wgsl_type_scalar(&s, WGSL_TYPE_BOOL);
    WGSLTypeInfo *t_i32   = wgsl_type_scalar(&s, WGSL_TYPE_I32);
    WGSLTypeInfo *t_u32   = wgsl_type_scalar(&s, WGSL_TYPE_U32);
    WGSLTypeInfo *t_f32   = wgsl_type_scalar(&s, WGSL_TYPE_F32);
    WGSLTypeInfo *t_f16   = wgsl_type_scalar(&s, WGSL_TYPE_F16);
    WGSLTypeInfo *t_aint  = wgsl_type_scalar(&s, WGSL_TYPE_ABSTRACT_INT);
    WGSLTypeInfo *t_afloat= wgsl_type_scalar(&s, WGSL_TYPE_ABSTRACT_FLOAT);
    WGSLTypeInfo *t_void  = wgsl_type_scalar(&s, WGSL_TYPE_VOID);
    CHECK(t_bool && t_i32 && t_u32 && t_f32 && t_f16 && t_aint && t_afloat && t_void,
          "all pre-interned scalars present");
    CHECK(t_bool != t_i32 && t_i32 != t_u32 && t_u32 != t_f32 && t_f32 != t_f16,
          "scalars are unique pointers");
    CHECK(wgsl_type_scalar(&s, WGSL_TYPE_F32) == t_f32, "scalar lookup is stable");

    /* Interning: vec3<f32> identity. */
    WGSLTypeInfo *vec3f_a = wgsl_type_vec(&s, 3, t_f32);
    WGSLTypeInfo *vec3f_b = wgsl_type_vec(&s, 3, t_f32);
    WGSLTypeInfo *vec4f   = wgsl_type_vec(&s, 4, t_f32);
    WGSLTypeInfo *vec3i   = wgsl_type_vec(&s, 3, t_i32);
    CHECK(vec3f_a == vec3f_b, "vec3<f32> interning: same ptr");
    CHECK(vec3f_a != vec4f,   "vec3<f32> != vec4<f32>");
    CHECK(vec3f_a != vec3i,   "vec3<f32> != vec3<i32>");

    /* Interning: mat3x3<f32> identity. */
    WGSLTypeInfo *mat3x3_a = wgsl_type_mat(&s, 3, 3, t_f32);
    WGSLTypeInfo *mat3x3_b = wgsl_type_mat(&s, 3, 3, t_f32);
    WGSLTypeInfo *mat3x4   = wgsl_type_mat(&s, 3, 4, t_f32);
    WGSLTypeInfo *mat4x3   = wgsl_type_mat(&s, 4, 3, t_f32);
    CHECK(mat3x3_a == mat3x3_b, "mat3x3 interning: same ptr");
    CHECK(mat3x3_a != mat3x4,   "mat3x3 != mat3x4");
    CHECK(mat3x4   != mat4x3,   "mat3x4 != mat4x3 (cols/rows distinct)");

    /* Atomic interning. */
    WGSLTypeInfo *atom_u32_a = wgsl_type_atomic(&s, t_u32);
    WGSLTypeInfo *atom_u32_b = wgsl_type_atomic(&s, t_u32);
    WGSLTypeInfo *atom_i32   = wgsl_type_atomic(&s, t_i32);
    CHECK(atom_u32_a == atom_u32_b, "atomic<u32> interning");
    CHECK(atom_u32_a != atom_i32,   "atomic<u32> != atomic<i32>");

    /* Array interning: fixed and runtime forms are distinct. */
    WGSLTypeInfo *arr4_a   = wgsl_type_array(&s, t_f32, 4);
    WGSLTypeInfo *arr4_b   = wgsl_type_array(&s, t_f32, 4);
    WGSLTypeInfo *arr_rs_a = wgsl_type_array(&s, t_f32, 0);
    WGSLTypeInfo *arr_rs_b = wgsl_type_array(&s, t_f32, 0);
    WGSLTypeInfo *arr8     = wgsl_type_array(&s, t_f32, 8);
    CHECK(arr4_a == arr4_b,           "array<f32, 4> interning");
    CHECK(arr_rs_a == arr_rs_b,       "array<f32> (runtime) interning");
    CHECK(arr4_a != arr_rs_a,         "fixed != runtime");
    CHECK(arr4_a != arr8,             "different lengths != ");

    /* Override-sized arrays keep full override-symbol identity. */
    int override_a = 0;
    int override_b = 0;
    WGSLTypeInfo *arr_ov_a1 = wgsl_type_array_override(&s, t_f32, &override_a);
    WGSLTypeInfo *arr_ov_a2 = wgsl_type_array_override(&s, t_f32, &override_a);
    WGSLTypeInfo *arr_ov_b  = wgsl_type_array_override(&s, t_f32, &override_b);
    CHECK(arr_ov_a1 == arr_ov_a2, "same override symbol interns same array");
    CHECK(arr_ov_a1 != arr_ov_b,  "different override symbols intern distinct arrays");
    CHECK(arr_ov_a1 != arr_rs_a,  "override-sized array != runtime array");
    CHECK(arr_ov_a1->ref == t_f32, "override-sized array keeps element in ref");

    /* Predicates. */
    CHECK(wgsl_type_is_scalar(t_bool),  "bool is scalar");
    CHECK(wgsl_type_is_scalar(t_f32),   "f32 is scalar");
    CHECK(!wgsl_type_is_scalar(vec3f_a),"vec3<f32> is not scalar");
    CHECK(wgsl_type_is_integer(t_i32),  "i32 is integer");
    CHECK(wgsl_type_is_integer(t_aint), "AbstractInt is integer");
    CHECK(!wgsl_type_is_integer(t_f32), "f32 not integer");
    CHECK(wgsl_type_is_float(t_f32),    "f32 is float");
    CHECK(wgsl_type_is_float(t_afloat), "AbstractFloat is float");
    CHECK(wgsl_type_is_numeric(t_i32),  "i32 is numeric");
    CHECK(!wgsl_type_is_numeric(t_bool),"bool not numeric");
    CHECK(wgsl_type_is_abstract(t_aint),    "AbstractInt is abstract");
    CHECK(wgsl_type_is_abstract(t_afloat),  "AbstractFloat is abstract");
    CHECK(!wgsl_type_is_abstract(t_i32),    "i32 is concrete");

    /* Composite of abstract is abstract. */
    WGSLTypeInfo *vec3_aint = wgsl_type_vec(&s, 3, t_aint);
    CHECK(wgsl_type_is_abstract(vec3_aint), "vec3<AbstractInt> is abstract");
    CHECK(!wgsl_type_is_abstract(vec3f_a),  "vec3<f32> is concrete");

    CHECK(wgsl_type_is_vector(vec3f_a),    "vec is vector");
    CHECK(wgsl_type_is_matrix(mat3x3_a),   "mat is matrix");
    CHECK(!wgsl_type_is_matrix(vec3f_a),   "vec is not matrix");

    /* Conversion rank table — WGSL §6.1.2 — full coverage. */
    check_rank(t_f32,    t_f32,    0,  "f32 → f32");
    check_rank(t_i32,    t_i32,    0,  "i32 → i32");
    check_rank(t_afloat, t_f32,    1,  "AbstractFloat → f32");
    check_rank(t_afloat, t_f16,    2,  "AbstractFloat → f16");
    check_rank(t_aint,   t_i32,    3,  "AbstractInt → i32");
    check_rank(t_aint,   t_u32,    4,  "AbstractInt → u32");
    check_rank(t_aint,   t_afloat, 5,  "AbstractInt → AbstractFloat");
    check_rank(t_aint,   t_f32,    6,  "AbstractInt → f32");
    check_rank(t_aint,   t_f16,    7,  "AbstractInt → f16");

    /* No automatic conversions: */
    check_rank(t_i32,    t_u32,    -1, "i32 → u32 (none)");
    check_rank(t_u32,    t_i32,    -1, "u32 → i32 (none)");
    check_rank(t_i32,    t_f32,    -1, "i32 → f32 (none)");
    check_rank(t_f32,    t_i32,    -1, "f32 → i32 (none)");
    check_rank(t_f32,    t_f16,    -1, "f32 → f16 (none, non-abstract)");
    check_rank(t_f32,    t_afloat, -1, "f32 → AbstractFloat (none)");
    check_rank(t_bool,   t_i32,    -1, "bool → i32 (none)");

    /* Composite element-wise rules. */
    check_rank(vec3_aint, vec3f_a, 6, "vec3<AbstractInt> → vec3<f32> (= 6)");
    WGSLTypeInfo *vec3_af = wgsl_type_vec(&s, 3, t_afloat);
    check_rank(vec3_af,   vec3f_a, 1, "vec3<AbstractFloat> → vec3<f32> (= 1)");
    check_rank(vec3_aint, vec4f,  -1, "vec3 → vec4 (width mismatch)");

    WGSLTypeInfo *mat3x3_aint = wgsl_type_mat(&s, 3, 3, t_aint);
    WGSLTypeInfo *mat3x3_f32  = wgsl_type_mat(&s, 3, 3, t_f32);
    check_rank(mat3x3_aint, mat3x3_f32, 6, "mat3x3<AbstractInt> → mat3x3<f32>");
    check_rank(mat3x3_f32,  mat3x4,    -1, "mat dim mismatch (none)");

    WGSLTypeInfo *arr4_aint = wgsl_type_array(&s, t_aint, 4);
    WGSLTypeInfo *arr4_i32  = wgsl_type_array(&s, t_i32, 4);
    check_rank(arr4_aint, arr4_i32, 3, "array<AbstractInt, 4> → array<i32, 4>");
    check_rank(arr4_aint, arr8,    -1, "array length mismatch (none)");
    /* runtime-sized arrays do not participate in the conversion rule */
    check_rank(arr_rs_a, wgsl_type_array(&s, t_aint, 0), -1,
               "runtime array conversion rejected");

    /* Concretization. */
    CHECK(wgsl_type_concretize(&s, t_aint)    == t_i32, "AbstractInt → i32");
    CHECK(wgsl_type_concretize(&s, t_afloat)  == t_f32, "AbstractFloat → f32");
    CHECK(wgsl_type_concretize(&s, t_f32)     == t_f32, "concrete pass-through");
    CHECK(wgsl_type_concretize(&s, vec3_aint) == wgsl_type_vec(&s, 3, t_i32),
          "vec3<AbstractInt> → vec3<i32>");
    CHECK(wgsl_type_concretize(&s, mat3x3_aint) == wgsl_type_mat(&s, 3, 3, t_i32),
          "mat3x3<AbstractInt> → mat3x3<i32>");
    CHECK(wgsl_type_concretize(&s, arr4_aint) == wgsl_type_array(&s, t_i32, 4),
          "array<AbstractInt, 4> → array<i32, 4>");
    CHECK(wgsl_type_concretize(&s, vec3f_a)   == vec3f_a,
          "vec3<f32> already concrete");

    /* Format strings — WGSL-shape for a human reader. */
    check_format(t_f32,        "f32",                    "fmt f32");
    check_format(t_aint,       "AbstractInt",            "fmt aint");
    check_format(t_afloat,     "AbstractFloat",          "fmt afloat");
    check_format(t_void,       "void",                   "fmt void");
    check_format(vec3f_a,      "vec3<f32>",              "fmt vec");
    check_format(mat3x3_a,     "mat3x3<f32>",            "fmt mat3x3");
    check_format(mat3x4,       "mat3x4<f32>",            "fmt mat3x4");
    check_format(atom_u32_a,   "atomic<u32>",            "fmt atomic");
    check_format(arr4_a,       "array<f32, 4>",          "fmt array fixed");
    check_format(arr_rs_a,     "array<f32>",             "fmt array runtime");
    check_format(wgsl_type_ptr(&s, WGSL_AS_FUNCTION, t_i32, WGSL_ACCESS_READ_WRITE),
                 "ptr<function, i32, read_write>",       "fmt ptr");
    check_format(wgsl_type_ref(&s, WGSL_AS_STORAGE, t_u32, WGSL_ACCESS_READ),
                 "ref<storage, u32, read>",              "fmt ref");
    /* Nested */
    WGSLTypeInfo *arr_of_vec = wgsl_type_array(&s, vec4f, 16);
    check_format(arr_of_vec,   "array<vec4<f32>, 16>",   "fmt nested");

    /* Format size-probing: cap=0 returns required length, doesn't write. */
    char dummy = 0xAA;
    size_t need = wgsl_type_format(arr_of_vec, &dummy, 0);
    CHECK(need == strlen("array<vec4<f32>, 16>"), "fmt size-probe length");
    CHECK(dummy == (char)0xAA,                     "fmt size-probe doesn't touch buffer");

    wgsl_types_destroy(&s);
    wgsl_arena_destroy(&arena);

    if (fail == 0) {
        printf("PASS  test_types  "
               "sizeof=%zu, kinds=%d, %d-rank table verified\n",
               sizeof(WGSLTypeInfo), (int)WGSL_TYPE_KIND_COUNT, 9);
        return 0;
    }
    fprintf(stderr, "FAIL  test_types  %d check(s) failed\n", fail);
    return 1;
}
