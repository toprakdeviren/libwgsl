/**
 * wgsl_lsp — minimal stdio JSON-RPC language server over libwgsl.
 */
#include "wgsl_lsp_priv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Doc *docs;
int  ndocs, cdocs;
int  g_shutdown;
SB   outb;

static int read_message(char **out, size_t *out_len) {
    *out = NULL; *out_len = 0;
    char line[256];
    long content_length = -1;
    for (;;) {
        if (!fgets(line, sizeof line, stdin)) return 0;
        if (line[0] == '\r' || line[0] == '\n') break;
        if (strncmp(line, "Content-Length:", 15) == 0)
            content_length = strtol(line + 15, NULL, 10);
    }
    if (content_length < 0 || content_length > 64 * 1024 * 1024) return 0;
    char *buf = (char *)malloc((size_t)content_length + 1);
    if (!buf) return 0;
    size_t got = fread(buf, 1, (size_t)content_length, stdin);
    if (got != (size_t)content_length) { free(buf); return 0; }
    buf[content_length] = 0;
    *out = buf;
    *out_len = (size_t)content_length;
    return 1;
}

int main(void) {
    /* Line-buffer stderr for logs; stdout is the LSP stream. */
    setvbuf(stdout, NULL, _IOFBF, 0);
    setvbuf(stderr, NULL, _IOLBF, 0);
    wgsl_init();
    fprintf(stderr, "libwgsl language server %s (spec %s)\n",
            WGSL_VERSION_STRING, WGSL_SPEC_PIN);

    char *msg = NULL;
    size_t len = 0;
    while (read_message(&msg, &len)) {
        dispatch(msg);
        free(msg);
        msg = NULL;
    }
    /* cleanup */
    while (ndocs > 0) doc_close(docs[0].uri);
    free(docs);
    free(outb.b);
    wgsl_shutdown();
    return g_shutdown ? 0 : 0;
}
