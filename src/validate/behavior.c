/**
 * @file behavior.c — §9.7 behavior-set analysis.
 *
 * Computes per-statement {Next, Break, Continue, Return} behavior sets
 * and enforces the spec's body-level requirements:
 *   - function body must have behavior ⊆ {Next, Return}
 *   - non-void function must Return on all paths
 *   - obviously infinite loops (empty body behavior) are rejected
 *   - continuing block must not contain Continue / Return
 *
 * Co-extensive with WGSL §9.7.
 */
#include "internal/validate.h"
#include "internal/validate_priv.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal/resolver.h"
#include "internal/types.h"
#include "internal/ast.h"

/* ── Behavior-set analysis (§9.7) ──────────────────────────────────── */

typedef uint8_t Behavior;

static Behavior beh_stmt(WGSLValidator *v, WGSLNode *n);

static Behavior beh_compound(WGSLValidator *v, WGSLNode *n) {
    Behavior b = WGSL_BEHAVIOR_NEXT;
    for (uint32_t i = 0; i < n->child_count; i++) {
        WGSLNode *c = n->children[i];
        if (!c) continue;
        Behavior s = beh_stmt(v, c);
        if ((b & WGSL_BEHAVIOR_NEXT) == 0) {
            /* Still validate unreachable statements: each statement's
             * own behavior must be legal, but unreachable behavior does
             * not affect the enclosing sequence. */
            continue;
        }
        b = (Behavior)((b & ~WGSL_BEHAVIOR_NEXT) | s);
    }
    return b;
}

static Behavior beh_stmt(WGSLValidator *v, WGSLNode *n) {
    if (!n) return WGSL_BEHAVIOR_NEXT;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_STMT_RETURN:   return WGSL_BEHAVIOR_RETURN;
    case WGSL_NODE_STMT_BREAK:    return WGSL_BEHAVIOR_BREAK;
    case WGSL_NODE_STMT_CONTINUE: return WGSL_BEHAVIOR_CONTINUE;
    case WGSL_NODE_STMT_DISCARD:  return WGSL_BEHAVIOR_NEXT;

    case WGSL_NODE_STMT_COMPOUND: return beh_compound(v, n);

    case WGSL_NODE_STMT_IF: {
        /* children: [cond, then-compound, (else_if|else)*] */
        Behavior result = 0;
        int saw_else = 0;
        for (uint32_t i = 1; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if (c->kind == WGSL_NODE_STMT_ELSE) saw_else = 1;
            if (c->kind == WGSL_NODE_STMT_COMPOUND ||
                c->kind == WGSL_NODE_STMT_ELSE_IF ||
                c->kind == WGSL_NODE_STMT_ELSE)
            {
                result = (Behavior)(result | beh_stmt(v, c));
            }
        }
        if (!saw_else) result = (Behavior)(result | WGSL_BEHAVIOR_NEXT);
        return result;
    }
    case WGSL_NODE_STMT_ELSE_IF: {
        /* children: [cond, then-compound, ...] — same as STMT_IF but
         * already inside an `if` chain. */
        Behavior result = 0;
        int saw_else = 0;
        for (uint32_t i = 1; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if (c->kind == WGSL_NODE_STMT_ELSE) saw_else = 1;
            if (c->kind == WGSL_NODE_STMT_COMPOUND ||
                c->kind == WGSL_NODE_STMT_ELSE_IF ||
                c->kind == WGSL_NODE_STMT_ELSE)
            {
                result = (Behavior)(result | beh_stmt(v, c));
            }
        }
        if (!saw_else) result = (Behavior)(result | WGSL_BEHAVIOR_NEXT);
        return result;
    }
    case WGSL_NODE_STMT_ELSE: {
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (c && c->kind == WGSL_NODE_STMT_COMPOUND) {
                return beh_stmt(v, c);
            }
        }
        return WGSL_BEHAVIOR_NEXT;
    }

    case WGSL_NODE_STMT_LOOP: {
        /* §9.7 — `loop { body }`'s behavior is derived from the body's
         * (sequenced statements) plus the continuing block (if any):
         *   - Break (from body or `break_if` in continuing) → Next
         *   - Continue restarts → drop
         *   - Next means body fell through → loop iterates again, drop
         *   - Return propagates
         * If the resulting set is empty, the loop has no exit edge —
         * spec rejects (`loop { }`, `loop { discard; }`).
         *
         * The parser stores loop body as flat children: every direct
         * child is a body statement, with the (optional) STMT_CONTINUING
         * appearing last.  Walk them sequentially exactly like a
         * compound, then fold the continuing's Break/Return in. */
        Behavior body = WGSL_BEHAVIOR_NEXT;
        WGSLNode *continuing = NULL;
        uint32_t continuing_index = n->child_count;
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if (c->kind == WGSL_NODE_STMT_CONTINUING) {
                continuing = c;
                continuing_index = i;
                break;
            }
        }
        for (uint32_t i = 0; i < continuing_index; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if (c->kind == WGSL_NODE_STMT_CONTINUING) {
                continue;
            }
            if ((body & WGSL_BEHAVIOR_NEXT) == 0) break;
            Behavior s = beh_stmt(v, c);
            body = (Behavior)((body & ~WGSL_BEHAVIOR_NEXT) | s);
        }
        Behavior cont = 0;
        if (continuing &&
            (body & (WGSL_BEHAVIOR_NEXT | WGSL_BEHAVIOR_CONTINUE)))
        {
            cont = beh_stmt(v, continuing);
        }
        Behavior r = 0;
        if (body & WGSL_BEHAVIOR_BREAK)  r |= WGSL_BEHAVIOR_NEXT;
        if (body & WGSL_BEHAVIOR_RETURN) r |= WGSL_BEHAVIOR_RETURN;
        if (cont & WGSL_BEHAVIOR_BREAK)  r |= WGSL_BEHAVIOR_NEXT;
        if (cont & WGSL_BEHAVIOR_RETURN) r |= WGSL_BEHAVIOR_RETURN;
        if (r == 0) {
            wgsl_val_error(v, n,
                "loop has no exit edge (empty behavior set); spec §9.7 "
                "rejects obviously infinite loops");
            r = WGSL_BEHAVIOR_NEXT;   /* keep walking */
        }
        return r;
    }
    case WGSL_NODE_STMT_WHILE: {
        /* §9.7 — `while` always has an implicit break edge via the
         * condition (the condition can evaluate false). */
        Behavior body = 0;
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (c && c->kind == WGSL_NODE_STMT_COMPOUND) {
                body = (Behavior)(body | beh_stmt(v, c));
            }
        }
        Behavior r = WGSL_BEHAVIOR_NEXT;
        if (body & WGSL_BEHAVIOR_RETURN) r |= WGSL_BEHAVIOR_RETURN;
        return r;
    }
    case WGSL_NODE_STMT_FOR: {
        uint32_t flags = (uint32_t)n->payload[0];
        int has_cond = (flags & 2u) != 0;
        Behavior body = 0;
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (c && c->kind == WGSL_NODE_STMT_COMPOUND) {
                body = (Behavior)(body | beh_stmt(v, c));
            }
        }
        if (has_cond) {
            Behavior r = WGSL_BEHAVIOR_NEXT;
            if (body & WGSL_BEHAVIOR_RETURN) r |= WGSL_BEHAVIOR_RETURN;
            return r;
        }

        Behavior r = 0;
        if (body & WGSL_BEHAVIOR_BREAK)  r |= WGSL_BEHAVIOR_NEXT;
        if (body & WGSL_BEHAVIOR_RETURN) r |= WGSL_BEHAVIOR_RETURN;
        if (r == 0) {
            wgsl_val_error(v, n,
                "for loop has no exit edge (empty behavior set); spec "
                "§9.7 rejects obviously infinite loops");
            r = WGSL_BEHAVIOR_NEXT;
        }
        return r;
    }

    case WGSL_NODE_STMT_SWITCH: {
        /* children: [selector, case_clauses...] */
        Behavior result = 0;
        int has_default = 0;
        for (uint32_t i = 1; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c || c->kind != WGSL_NODE_STMT_CASE_CLAUSE) continue;
            int is_default_alone = (int)((c->payload[0] >> 32) & 1u);
            if (is_default_alone) has_default = 1;
            result = (Behavior)(result | beh_stmt(v, c));
        }
        if (!has_default) result = (Behavior)(result | WGSL_BEHAVIOR_NEXT);
        /* Switch absorbs Break into Next (a `break` falls through past
         * the switch). */
        if (result & WGSL_BEHAVIOR_BREAK) {
            result = (Behavior)((result & ~WGSL_BEHAVIOR_BREAK) | WGSL_BEHAVIOR_NEXT);
        }
        return result;
    }
    case WGSL_NODE_STMT_CASE_CLAUSE: {
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (c && c->kind == WGSL_NODE_STMT_COMPOUND) {
                return beh_stmt(v, c);
            }
        }
        return WGSL_BEHAVIOR_NEXT;
    }

    case WGSL_NODE_STMT_BREAK_IF:    return WGSL_BEHAVIOR_BREAK | WGSL_BEHAVIOR_NEXT;
    case WGSL_NODE_STMT_CONTINUING: {
        /* The continuing block's children are statements (no wrapping
         * compound).  Walk them so a `break_if` inside surfaces Break
         * to the enclosing loop's behavior calc. */
        Behavior b = WGSL_BEHAVIOR_NEXT;
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if ((b & WGSL_BEHAVIOR_NEXT) == 0) break;
            Behavior s = beh_stmt(v, c);
            b = (Behavior)((b & ~WGSL_BEHAVIOR_NEXT) | s);
        }
        return b;
    }

    /* Plain statements continue to next. */
    default:
        return WGSL_BEHAVIOR_NEXT;
    }
}

/* Validate every user fn's body satisfies §9.7's body constraint:
 *   behaviors ⊆ {Next, Return}; non-void fn must have Return without
 *   Next (i.e. all paths return). */
void validate_behavior(WGSLValidator *v, WGSLNode *tu) {
    if (!tu) return;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *fn = tu->children[i];
        if (!fn || fn->kind != WGSL_NODE_DECL_FUNCTION) continue;

        uint32_t HR = (uint32_t)(fn->payload[2] >> 32);
        if (fn->child_count == 0) continue;
        WGSLNode *body = fn->children[fn->child_count - 1];
        if (!body || body->kind != WGSL_NODE_STMT_COMPOUND) continue;

        Behavior b = beh_stmt(v, body);

        if (b & WGSL_BEHAVIOR_BREAK) {
            wgsl_val_error(v, body,
                "function body must not have `break` outside a loop or switch");
            continue;
        }
        if (b & WGSL_BEHAVIOR_CONTINUE) {
            wgsl_val_error(v, body,
                "function body must not have `continue` outside a loop");
            continue;
        }
        if (HR && (b & WGSL_BEHAVIOR_NEXT)) {
            const char *fname = "<anonymous>";
            uint32_t fname_len = 11;
            uint32_t no = (uint32_t)(fn->payload[0] & 0xFFFFFFFFu);
            uint32_t nl = (uint32_t)(fn->payload[0] >> 32);
            if (nl > 0) {
                fname = v->src->bytes + no;
                fname_len = nl;
            }
            wgsl_val_error(v, body,
                "function '%.*s' has a return type but does not return on all paths",
                (int)fname_len, fname);
        }
    }
}
