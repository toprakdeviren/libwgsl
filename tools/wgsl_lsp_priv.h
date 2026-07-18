#ifndef WGSL_LSP_PRIV_H
#define WGSL_LSP_PRIV_H

#include "wgsl.h"

#include <stddef.h>
#include <stdint.h>

typedef struct { char *b; size_t n, cap; } SB;

typedef struct {
    char        *uri;
    char        *text;
    size_t       text_len;
    WGSLSession *session;
    int          version;
    size_t       semantic_data_count;
} Doc;

extern Doc *docs;
extern int  ndocs, cdocs;
extern int  g_shutdown;
extern SB   outb;

void lsp_sb_clear(SB *s);
int  lsp_sb_grow(SB *s, size_t need);
void lsp_sb_putn(SB *s, const char *p, size_t n);
void lsp_sb_puts(SB *s, const char *p);
void lsp_sb_putc(SB *s, char c);
void lsp_sb_put_u(SB *s, unsigned long v);
void lsp_sb_put_i(SB *s, long v);
void lsp_sb_json_str(SB *s, const char *p);

Doc *doc_find(const char *uri);
Doc *doc_ensure(const char *uri);
const WGSLResult *doc_result(Doc *d);
void doc_set_text(Doc *d, const char *text);
int  doc_apply_change(Doc *d, uint32_t start, uint32_t end, const char *text);
void doc_close(const char *uri);

void offset_to_pos(const char *text, size_t len, uint32_t off,
                   unsigned *line, unsigned *ch);
uint32_t pos_to_offset(const char *text, size_t len, unsigned line, unsigned ch);

void send_msg(const char *body);
void reply_raw(const char *id_json, const char *result_json);
void reply_null(const char *id_json);
void notify(const char *method, const char *params_json);

const char *json_skip_ws(const char *p);
const char *json_val(const char *json, const char *key);
long json_get_int(const char *json, const char *key, long def);
int json_get_string(const char *json, const char *key, char *out, size_t cap);
void json_get_id_raw(const char *json, char *out, size_t cap);
char *json_decode_string_at(const char *v);

void publish_diags(Doc *d);
Doc *doc_from_params(const char *params);
uint32_t offset_from_params(Doc *d, const char *params);

void dispatch(const char *msg);

#endif
