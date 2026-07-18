/**
 * @file templates.c — template-list disambiguation pass.
 */
#include "internal/lexer.h"
#include "internal/token.h"
#include "internal/arena.h"

#include <stdlib.h>
#include <string.h>

/*
 *  WGSL §3.9 — template-list discovery on a pre-tokenized stream.
 */

/* Token kinds whose lexical form matches `ident_pattern_token`.  The
 * §3.9 algorithm pushes a candidate when one of these is followed by
 * `<`. */
static int is_ident_shaped(WGSLTokenKind k) {
    if (k == WGSL_TOK_IDENT)      return 1;
    if (k == WGSL_TOK_UNDERSCORE) return 1;
    /* All keywords match ident_pattern_token lexically. */
    if (k >= WGSL_TOK_KW_ALIAS && k <= WGSL_TOK_KW_WHILE) return 1;
    return 0;
}

static int is_trivia(WGSLTokenKind k) {
    return k == WGSL_TOK_BLANKSPACE
        || k == WGSL_TOK_LINE_COMMENT
        || k == WGSL_TOK_BLOCK_COMMENT;
}

/* Pending stack entry — points into the **output** array (so the LESS
 * we may eventually retag is reachable by index). */
typedef struct {
    size_t out_idx;     /* index of the LESS token in out[] */
    int    depth;       /* nesting at the time of push       */
} Pending;

#define WGSL_TPL_STACK_LIMIT 256

typedef struct {
    WGSLToken *out;
    size_t     count;
    size_t     capacity;
} TplOut;

static int tpl_grow(TplOut *o) {
    if (o->capacity > SIZE_MAX / 2) return 0;
    size_t new_cap = o->capacity ? o->capacity * 2 : 64;
    if (new_cap > SIZE_MAX / sizeof *o->out) return 0;
    WGSLToken *grown = (WGSLToken *)realloc(o->out, new_cap * sizeof *grown);
    if (!grown) return 0;
    o->out = grown;
    o->capacity = new_cap;
    return 1;
}

static int tpl_append(TplOut *o, WGSLToken t) {
    if (o->count == o->capacity && !tpl_grow(o)) return 0;
    o->out[o->count++] = t;
    return 1;
}

/* Try to close a pending template at the current depth using the
 * just-appended token at out[idx], retagging it to TEMPLATE_END and
 * the matching LESS to TEMPLATE_START.  Returns 1 if a close happened. */
static int tpl_try_close(TplOut *o, Pending *stack, int *sp,
                         int depth, size_t close_idx)
{
    if (*sp > 0 && stack[*sp - 1].depth == depth) {
        size_t less_idx = stack[*sp - 1].out_idx;
        o->out[less_idx].kind  = WGSL_TOK_TEMPLATE_START;
        o->out[close_idx].kind = WGSL_TOK_TEMPLATE_END;
        *sp -= 1;
        return 1;
    }
    return 0;
}

/* Split `>>`, `>=`, `>>=`: the leading `>` closes a pending template,
 * the remainder is emitted as a fresh follow-up token.  Caller has
 * already determined a close is wanted; this routine writes both
 * tokens and recurses into a second close attempt for `>>` only.
 * Returns 1 on success, 0 on OOM. */
static int tpl_split_greater(
    TplOut *o, Pending *stack, int *sp, int depth,
    WGSLToken orig, WGSLTokenKind tail_kind, uint32_t tail_length)
{
    /* Emit head as TEMPLATE_END (length 1) */
    WGSLToken head = orig;
    head.kind        = WGSL_TOK_TEMPLATE_END;
    head.span.length = 1;
    if (!tpl_append(o, head)) return 0;
    size_t head_idx = o->count - 1;
    /* Retag matching LESS */
    if (*sp > 0) {
        size_t less_idx = stack[*sp - 1].out_idx;
        o->out[less_idx].kind = WGSL_TOK_TEMPLATE_START;
        *sp -= 1;
    }

    /* Emit tail */
    WGSLToken tail = orig;
    tail.kind        = tail_kind;
    tail.span.offset = orig.span.offset + 1;
    tail.span.length = tail_length;
    /* Column of the tail = column of head + 1 (same line guaranteed:
     * `>>` `>=` `>>=` are single-line tokens). */
    tail.column      = orig.column + 1;
    if (!tpl_append(o, tail)) return 0;

    /* For `>>`, the tail is GREATER; it might *also* close another
     * pending template at the same depth. */
    if (tail_kind == WGSL_TOK_GREATER) {
        size_t tail_idx = o->count - 1;
        (void)head_idx;
        tpl_try_close(o, stack, sp, depth, tail_idx);
    }
    return 1;
}

int wgsl_discover_templates(WGSLLexResult *result, WGSLArena *arena,
                            WGSLDiagBag *diag, const WGSLSource *source) {
    if (!result || !arena || result->count == 0) return 1;

    TplOut o;
    o.out = NULL;
    o.count = 0;
    o.capacity = 0;
    if (!tpl_grow(&o)) return 0;

    Pending stack[WGSL_TPL_STACK_LIMIT];
    int sp = 0;
    int depth = 0;
    int prev_ident_shaped = 0;
    int reported_overflow = 0;   /* emit the too-deeply-nested error once */

    for (size_t i = 0; i < result->count; i++) {
        WGSLToken t = result->tokens[i];
        WGSLTokenKind k = t.kind;

        if (is_trivia(k)) {
            if (!tpl_append(&o, t)) goto oom;
            continue;
        }

        if (is_ident_shaped(k)) {
            if (!tpl_append(&o, t)) goto oom;
            prev_ident_shaped = 1;
            continue;
        }

        switch (k) {
        case WGSL_TOK_LESS: {
            if (!tpl_append(&o, t)) goto oom;
            if (prev_ident_shaped) {
                if (sp < WGSL_TPL_STACK_LIMIT) {
                    stack[sp].out_idx = o.count - 1;
                    stack[sp].depth   = depth;
                    sp += 1;
                } else if (!reported_overflow && diag && source) {
                    /* §3.9 candidate `<` that we cannot track — instead of
                     * silently dropping it (which yields a confusing
                     * downstream generic error), report the real cause. */
                    reported_overflow = 1;
                    wgsl_diag_emit_at(diag, source, WGSL_DIAG_ERROR,
                                      t.span.offset, t.span.length, NULL,
                                      "template list too deeply nested "
                                      "(exceeds limit of %d)",
                                      WGSL_TPL_STACK_LIMIT);
                }
            }
            break;
        }
        case WGSL_TOK_LESS_LESS:
        case WGSL_TOK_LESS_EQUAL:
        case WGSL_TOK_LESS_LESS_EQUAL:
            /* Operators — never templates.  Just append. */
            if (!tpl_append(&o, t)) goto oom;
            break;

        case WGSL_TOK_GREATER:
            if (!tpl_append(&o, t)) goto oom;
            tpl_try_close(&o, stack, &sp, depth, o.count - 1);
            break;

        case WGSL_TOK_GREATER_GREATER:
            if (sp > 0 && stack[sp - 1].depth == depth) {
                if (!tpl_split_greater(&o, stack, &sp, depth, t,
                                       WGSL_TOK_GREATER, 1)) goto oom;
            } else {
                if (!tpl_append(&o, t)) goto oom;
            }
            break;

        case WGSL_TOK_GREATER_EQUAL:
            if (sp > 0 && stack[sp - 1].depth == depth) {
                if (!tpl_split_greater(&o, stack, &sp, depth, t,
                                       WGSL_TOK_EQUAL, 1)) goto oom;
                /* §3.9: the trailing `=` is a lone assignment.
                 * Spec assignment rule clears BOTH the pending stack
                 * AND the nesting depth. */
                sp = 0;
                depth = 0;
            } else {
                if (!tpl_append(&o, t)) goto oom;
            }
            break;

        case WGSL_TOK_GREATER_GREATER_EQUAL:
            if (sp > 0 && stack[sp - 1].depth == depth) {
                /* Close inner template with leading `>`; emit `>=` tail. */
                if (!tpl_split_greater(&o, stack, &sp, depth, t,
                                       WGSL_TOK_GREATER_EQUAL, 2)) goto oom;
                /* §3.9: re-examine the tail's leading `>` for closing
                 * a second pending template at the same depth (the
                 * `A<B<C>>=D` case).  If we close another, the trailing
                 * `=` is then a lone assignment and clears state. */
                if (sp > 0 && stack[sp - 1].depth == depth) {
                    /* Split the just-emitted `>=` tail at o.out[o.count - 1]
                     * into TEMPLATE_END + EQUAL. */
                    size_t tail_idx = o.count - 1;
                    WGSLToken tail = o.out[tail_idx];
                    o.out[tail_idx].kind        = WGSL_TOK_TEMPLATE_END;
                    o.out[tail_idx].span.length = 1;
                    /* Retag matching LESS as TEMPLATE_START + pop. */
                    size_t less_idx = stack[sp - 1].out_idx;
                    o.out[less_idx].kind = WGSL_TOK_TEMPLATE_START;
                    sp -= 1;
                    /* Append the trailing `=` as a fresh EQUAL token. */
                    WGSLToken eq = tail;
                    eq.kind        = WGSL_TOK_EQUAL;
                    eq.span.offset = tail.span.offset + 1;
                    eq.span.length = 1;
                    eq.column      = tail.column + 1;
                    if (!tpl_append(&o, eq)) goto oom;
                    /* `=` clears state (spec). */
                    sp = 0;
                    depth = 0;
                }
            } else {
                if (!tpl_append(&o, t)) goto oom;
            }
            break;

        case WGSL_TOK_LPAREN:
        case WGSL_TOK_LBRACKET:
            if (!tpl_append(&o, t)) goto oom;
            depth += 1;
            break;

        case WGSL_TOK_RPAREN:
        case WGSL_TOK_RBRACKET:
            if (!tpl_append(&o, t)) goto oom;
            /* Pop pending entries that were opened at or below this
             * depth — they cannot become templates anymore. */
            while (sp > 0 && stack[sp - 1].depth >= depth) sp -= 1;
            if (depth > 0) depth -= 1;
            break;

        case WGSL_TOK_SEMICOLON:
        case WGSL_TOK_LBRACE:
        case WGSL_TOK_COLON:
            if (!tpl_append(&o, t)) goto oom;
            sp = 0;
            depth = 0;
            break;

        case WGSL_TOK_AMP_AMP:
        case WGSL_TOK_PIPE_PIPE:
            /* Expression boundary at current depth. */
            if (!tpl_append(&o, t)) goto oom;
            while (sp > 0 && stack[sp - 1].depth >= depth) sp -= 1;
            break;

        case WGSL_TOK_EQUAL:
            /* §3.9: a lone `=` is an assignment, which cannot appear
             * inside an expression and therefore cannot appear in a
             * template list.  Spec clears BOTH the pending stack AND
             * the nesting depth. */
            if (!tpl_append(&o, t)) goto oom;
            sp = 0;
            depth = 0;
            break;

        default:
            if (!tpl_append(&o, t)) goto oom;
            break;
        }

        prev_ident_shaped = 0;
    }

    /* Copy the rebuilt stream into the arena, replacing the old. */
    WGSLToken *arena_out =
        (o.count > SIZE_MAX / sizeof *arena_out) ? NULL :
        (WGSLToken *)wgsl_arena_alloc(arena, o.count * sizeof *arena_out);
    if (!arena_out) goto oom;
    memcpy(arena_out, o.out, o.count * sizeof *arena_out);
    free(o.out);

    result->tokens = arena_out;
    result->count  = o.count;
    return 1;

oom:
    free(o.out);
    return 0;
}
