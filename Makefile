# wgsl — Phase 1 build.

CC      ?= cc
AR      ?= ar
CFLAGS  ?= -O2 -g -Wall -Wextra -Werror -Wno-unused-parameter \
           -std=c11 -fno-strict-aliasing -fvisibility=hidden
INCLUDES = -Iinclude -Isrc -Ivendor/unicode/include

BUILD_DIR  = .build
LIB_NAME   = libwgsl.a
LIB        = $(BUILD_DIR)/$(LIB_NAME)

UNICODE_LIB = vendor/unicode/libunicode.native.a

# Threads (used by Phase 1 (b) TSan smoke).
PTHREAD_LIBS = -lpthread

# ThreadSanitizer build — separate output tree so it doesn't shadow
# the regular .a / tests.  Same sources, instrumented.
TSAN_BUILD_DIR = .build/tsan
TSAN_CFLAGS    = $(CFLAGS) -fsanitize=thread -fno-omit-frame-pointer -DNDEBUG
TSAN_LDFLAGS   = -fsanitize=thread

# Sources for the static library.  Phase 1 picks them up explicitly
# so an accidentally-added file does not silently get archived.
LIB_SRCS = \
    src/arena.c \
    src/source.c \
    src/diag.c \
    src/utf8.c \
    src/token.c \
    src/lexer.c \
    src/parser.c \
    src/ast_dump.c \
    src/types.c \
    src/resolver.c \
    src/consteval.c \
    src/check.c \
    src/validate.c \
    src/glob.c \
    src/toml.c \
    src/project.c \
    src/wgsl.c

LIB_OBJS = $(LIB_SRCS:src/%.c=$(BUILD_DIR)/%.o)

# One test executable per `tests/NN-name/test_*.c`.
TEST_SRCS = $(wildcard tests/*/test_*.c)
TEST_BINS = $(TEST_SRCS:tests/%.c=$(BUILD_DIR)/tests/%)

.PHONY: all lib test tsan wasm wasm-test wasm-corpus clean help

all: lib

lib: $(LIB)

$(LIB): $(LIB_OBJS)
	@mkdir -p $(@D)
	@$(AR) rcs $@ $^
	@echo "  AR      $@"

$(BUILD_DIR)/%.o: src/%.c
	@mkdir -p $(@D)
	@$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<
	@echo "  CC      $<"

# Tests link against libwgsl.a; libunicode is added too even when unused
# (linker only pulls in what's referenced).
test: $(TEST_BINS)
	@echo
	@echo "  Running $(words $(TEST_BINS)) test(s):"
	@pass=0; fail=0; \
	 for t in $(TEST_BINS); do \
	    if $$t; then pass=$$((pass+1)); else fail=$$((fail+1)); fi; \
	 done; \
	 echo "  $$pass passed, $$fail failed"; \
	 [ $$fail -eq 0 ]

$(BUILD_DIR)/tests/%: tests/%.c $(LIB)
	@mkdir -p $(@D)
	@$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LIB) $(UNICODE_LIB) $(PTHREAD_LIBS)
	@echo "  LD      $@"

# ─────────────────────────────────────────────────────────────────────
#  ThreadSanitizer build — `make tsan`
# ─────────────────────────────────────────────────────────────────────

TSAN_LIB_OBJS = $(LIB_SRCS:src/%.c=$(TSAN_BUILD_DIR)/%.o)
TSAN_LIB      = $(TSAN_BUILD_DIR)/$(LIB_NAME)

# Only the tsan-marked tests run under TSan.  Everything else stays
# in the regular `.build/` tree.
TSAN_TEST_SRCS = $(wildcard tests/01-tsan/test_*.c)
TSAN_TEST_BINS = $(TSAN_TEST_SRCS:tests/%.c=$(TSAN_BUILD_DIR)/tests/%)

tsan: $(TSAN_TEST_BINS)
	@echo
	@echo "  Running $(words $(TSAN_TEST_BINS)) test(s) under TSan:"
	@pass=0; fail=0; \
	 for t in $(TSAN_TEST_BINS); do \
	    if TSAN_OPTIONS="halt_on_error=1 second_deadlock_stack=1" $$t; \
	      then pass=$$((pass+1)); else fail=$$((fail+1)); fi; \
	 done; \
	 echo "  $$pass passed, $$fail failed"; \
	 [ $$fail -eq 0 ]

$(TSAN_BUILD_DIR)/%.o: src/%.c
	@mkdir -p $(@D)
	@$(CC) $(TSAN_CFLAGS) $(INCLUDES) -c -o $@ $<
	@echo "  CC[ts]  $<"

$(TSAN_LIB): $(TSAN_LIB_OBJS)
	@mkdir -p $(@D)
	@$(AR) rcs $@ $^
	@echo "  AR[ts]  $@"

$(TSAN_BUILD_DIR)/tests/%: tests/%.c $(TSAN_LIB)
	@mkdir -p $(@D)
	@$(CC) $(TSAN_CFLAGS) $(TSAN_LDFLAGS) $(INCLUDES) \
	    -o $@ $< $(TSAN_LIB) $(UNICODE_LIB) $(PTHREAD_LIBS)
	@echo "  LD[ts]  $@"

# ─────────────────────────────────────────────────────────────────────
#  WASM build — `make wasm` produces wgsl_compiler.{js,wasm}
# ─────────────────────────────────────────────────────────────────────
#
# Drives Emscripten over the same source list as the native lib; links
# against the prebuilt `vendor/unicode/libunicode.wasm.a`.  The output
# is a self-contained ES module (`MODULARIZE=1`, `EXPORT_NAME=WGSL`)
# usable in Node, web workers, and the browser.

EMCC             ?= emcc
WASM_BUILD_DIR    = $(BUILD_DIR)/wasm
WASM_BUNDLE       = $(WASM_BUILD_DIR)/wgsl_compiler.js
UNICODE_LIB_WASM  = vendor/unicode/libunicode.wasm.a

EM_CFLAGS = -O2 -Wall -Wextra -Wno-unused-parameter \
            -std=c11 -fno-strict-aliasing -fvisibility=hidden \
            -DNDEBUG -DWGSL_NO_FS

# Public surface from include/wgsl.h, prefixed with `_` per emcc convention.
EM_EXPORTS = _wgsl_init,_wgsl_shutdown,\
_wgsl_spec_pin,_wgsl_unicode_version,\
_wgsl_check,_wgsl_check_n,_wgsl_free,\
_wgsl_ok,_wgsl_error,_wgsl_module_json,_wgsl_module_json_len,\
_wgsl_diagnostic_count,_wgsl_diagnostic,\
_wgsl_lex,_wgsl_semantic_tokens,_wgsl_lex_free,\
_wgsl_hover_at,_wgsl_definition_at,\
_wgsl_hover_at_into,_wgsl_definition_at_into,\
_wgsl_project_open_from_string,_wgsl_project_close,_wgsl_project_match,\
_wgsl_check_with_preamble,\
_malloc,_free

EM_RUNTIME_METHODS = cwrap,ccall,UTF8ToString,stringToUTF8,\
lengthBytesUTF8,getValue,setValue,HEAP8,HEAPU8,HEAPU32

EM_LDFLAGS = \
    -s WASM=1 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME=WGSL \
    -s EXPORT_ES6=0 \
    -s ENVIRONMENT=node,web,worker \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INITIAL_MEMORY=16MB \
    -s STACK_SIZE=2MB \
    -s EXPORTED_FUNCTIONS='[$(EM_EXPORTS)]' \
    -s EXPORTED_RUNTIME_METHODS='[$(EM_RUNTIME_METHODS)]' \
    --no-entry

WASM_LIB_OBJS = $(LIB_SRCS:src/%.c=$(WASM_BUILD_DIR)/%.o)

wasm: $(WASM_BUNDLE)

$(WASM_BUNDLE): $(WASM_LIB_OBJS) $(UNICODE_LIB_WASM)
	@mkdir -p $(@D)
	@$(EMCC) $(EM_CFLAGS) $(EM_LDFLAGS) -o $@ \
	    $(WASM_LIB_OBJS) $(UNICODE_LIB_WASM)
	@echo "  EMLD    $@"

$(WASM_BUILD_DIR)/%.o: src/%.c
	@mkdir -p $(@D)
	@$(EMCC) $(EM_CFLAGS) $(INCLUDES) -c -o $@ $<
	@echo "  EMCC    $<"

# Smoke-test the bundle in Node.  Verifies the public C API survives
# the Emscripten round-trip and `wgsl_check` answers correctly.
NODE ?= node
wasm-test: $(WASM_BUNDLE)
	@$(NODE) tests/10-wasm/smoke.js

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "Targets:"
	@echo "  make            — build $(LIB_NAME)"
	@echo "  make test       — build + run all tests under tests/*"
	@echo "  make tsan       — build + run TSan smoke (tests/01-tsan/*)"
	@echo "  make wasm       — build wgsl_compiler.{js,wasm} via Emscripten"
	@echo "  make wasm-test  — run Node smoke test against the wasm bundle"
	@echo "  make clean      — remove $(BUILD_DIR)"
