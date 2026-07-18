# libwgsl build.

CC      ?= cc
AR      ?= ar
JAVAC   ?= javac
JAVA    ?= java
CFLAGS  ?= -O2 -g -Wall -Wextra -Werror -Wno-unused-parameter \
           -std=c11 -fno-strict-aliasing -fvisibility=hidden -fPIC
INCLUDES = -Iinclude -Isrc -Ivendor/unicode/include
DEPFLAGS = -MMD -MP

BUILD_DIR  = .build
LIB_NAME   = libwgsl.a
LIB        = $(BUILD_DIR)/$(LIB_NAME)
WGSL_BIN  ?= wgsl
CLI_SRC    = tools/wgsl_cli.c
CLI_OBJ    = $(BUILD_DIR)/tools/wgsl_cli.o
CLI_BIN    = $(BUILD_DIR)/$(WGSL_BIN)
LSP_SRCS   = tools/wgsl_lsp.c tools/wgsl_lsp_core.c tools/wgsl_lsp_handlers.c
LSP_OBJS   = $(LSP_SRCS:tools/%.c=$(BUILD_DIR)/tools/%.o)
LSP_BIN    = $(BUILD_DIR)/wgsl_lsp
EMBEDDER_EXAMPLE = $(BUILD_DIR)/examples/embedder

UNAME_S := $(shell uname -s 2>/dev/null || echo unknown)
UNICODE_LIB_DARWIN = vendor/unicode/libunicode.native.a
UNICODE_SRC_DIR    = vendor/unicode/src
UNICODE_BUILD_DIR  = $(BUILD_DIR)/vendor/unicode
UNICODE_SRC_SRCS   = $(UNICODE_SRC_DIR)/decoder_xid.c
UNICODE_SRC_OBJS   = $(UNICODE_SRC_SRCS:$(UNICODE_SRC_DIR)/%.c=$(UNICODE_BUILD_DIR)/%.o)
UNICODE_SRC_LIB    = $(UNICODE_BUILD_DIR)/libunicode.native.a
ifeq ($(UNAME_S),Darwin)
  UNICODE_LIB ?= $(UNICODE_LIB_DARWIN)
  MATH_LIBS ?=
else
  UNICODE_LIB ?= $(UNICODE_SRC_LIB)
  MATH_LIBS ?= -lm
endif

# Thread support for ThreadSanitizer tests.
PTHREAD_LIBS = -lpthread

# ThreadSanitizer build — separate output tree so it doesn't shadow
# the regular .a / tests.  Same sources, instrumented.
TSAN_BUILD_DIR = .build/tsan
TSAN_CFLAGS    = $(CFLAGS) -fsanitize=thread -fno-omit-frame-pointer -DNDEBUG
TSAN_LDFLAGS   = -fsanitize=thread

# Sources for the static library. Keep this explicit so an accidentally-added
# file does not silently get archived.
LIB_SRCS = \
    src/arena.c \
    src/source.c \
    src/diag.c \
    src/utf8.c \
    src/token.c \
    src/lexer/scan.c \
    src/lexer/templates.c \
    src/parser.c \
    src/parser/helpers.c \
    src/parser/expressions_attrs.c \
    src/parser/statements.c \
    src/parser/decls_entry.c \
    src/ast_dump.c \
    src/types/store.c \
    src/types/predicates.c \
    src/types/format.c \
    src/resolver.c \
    src/consteval.c \
    src/consteval/materialize.c \
    src/consteval/binary_ops.c \
    src/consteval/unary_binop.c \
    src/consteval/builtin_helpers.c \
    src/consteval/constructors.c \
    src/consteval/bitcast_pack.c \
    src/consteval/numeric_scalar.c \
    src/consteval/numeric_geom.c \
    src/consteval/numeric_matrix.c \
    src/consteval/user_fn.c \
    src/consteval/bits_select_dispatch.c \
    src/consteval/eval_dispatch.c \
    src/check/check.c \
    src/check/check/loop_functions.c \
    src/check/check/statements.c \
    src/check/check/entry.c \
    src/check/types.c \
    src/check/exprs.c \
    src/check/exprs/core_atoms.c \
    src/check/exprs/core_binary.c \
    src/check/exprs/core_member_index.c \
    src/check/exprs/overloads.c \
    src/check/exprs/atomic_pack.c \
    src/check/exprs/texture_shape.c \
    src/check/exprs/domain_checks.c \
    src/check/exprs/texture_overloads.c \
    src/check/exprs/generated_overloads.c \
    src/check/exprs/builtin_calls.c \
    src/check/exprs/constructors.c \
    src/check/exprs/ref_dispatch.c \
    src/check/decls_init.c \
    src/check/decls_typing.c \
    src/layout.c \
    src/validate/validate.c \
    src/validate/attrs_table.c \
    src/validate/attrs_walk.c \
    src/validate/attrs_api.c \
    src/validate/io.c \
    src/validate/behavior.c \
    src/validate/access_static.c \
    src/validate/access_alias.c \
    src/validate/access_discard.c \
    src/validate/layout.c \
    src/validate/uniformity.c \
    src/validate/uniformity/state.c \
    src/validate/uniformity/behavior_flow.c \
    src/validate/uniformity/call_reqs.c \
    src/validate/uniformity/call_analyze.c \
    src/validate/uniformity/pointers_block.c \
    src/glob.c \
    src/toml.c \
    src/project.c \
    src/interp/helpers.c \
    src/interp/exec_leaf.c \
    src/interp/compile.c \
    src/interp/vm.c \
    src/interp/coop_builtins.c \
    src/interp/run.c \
    src/interp_band.c \
    src/format.c \
    src/wgsl.c \
    src/api/module_json/json_buf.c \
    src/api/module_json/entry_usage.c \
    src/api/module_json/entry_points.c \
    src/api/module_json/pipelines.c \
    src/api/module_json/module_emit.c \
    src/api/lex_semantic.c \
    src/api/queries_project.c \
    src/api/lsp/helpers.c \
    src/api/lsp/outline_refs.c \
    src/api/lsp/completions.c \
    src/api/lsp/signature.c \
    src/api/lsp/code_actions.c \
    src/api/session.c \
    src/optimize/analyze.c \
    src/optimize/apply.c \
    src/emit/msl.c \
    src/analyze/ml_kernel.c \
    src/analyze/debug_oracle.c \
    src/analyze/golden_ref.c

LIB_OBJS = $(LIB_SRCS:src/%.c=$(BUILD_DIR)/%.o)

# One test executable per `tests/NN-name/test_*.c`.
# Split into two groups:
#   - TEST_SRCS         : unit / API tests, run by `make test`
#   - CORPUS_TEST_SRCS  : corpus walks over `examples/shaders/` (slower,
#                         and treats real-world shader fidelity as a
#                         separate failure surface).  Run via
#                         `make test-corpus`.
ALL_TEST_SRCS    = $(wildcard tests/*/test_*.c)
CORPUS_TEST_SRCS = $(filter %_corpus.c %_parser_bench.c,$(ALL_TEST_SRCS))
# Disk golden walk: separate target (`make test-golden`), not `make test`.
GOLDEN_WALK_SRC  = tests/15-golden/test_golden_walk.c
GOLDEN_WALK_BIN  = $(BUILD_DIR)/tests/15-golden/test_golden_walk
TEST_SRCS        = $(filter-out $(CORPUS_TEST_SRCS) $(GOLDEN_WALK_SRC),$(ALL_TEST_SRCS))
TEST_BINS        = $(TEST_SRCS:tests/%.c=$(BUILD_DIR)/tests/%)
CORPUS_TEST_BINS = $(CORPUS_TEST_SRCS:tests/%.c=$(BUILD_DIR)/tests/%)

# Shared library for foreign bindings (ctypes / dlopen).
ifeq ($(UNAME_S),Darwin)
  SHARED_EXT = dylib
  SHARED_LDFLAGS = -dynamiclib -install_name @rpath/libwgsl.$(SHARED_EXT)
else
  SHARED_EXT = so
  SHARED_LDFLAGS = -shared
endif
SHARED = $(BUILD_DIR)/libwgsl.$(SHARED_EXT)

JNI_DIR           = bindings/jni
JNI_BUILD_DIR     = $(BUILD_DIR)/bindings/jni
JNI_CLASSES_DIR   = $(JNI_BUILD_DIR)/classes
JNI_CLASSES_STAMP = $(JNI_BUILD_DIR)/classes.stamp
JNI_HEADER        = $(JNI_DIR)/wgsl_jni.h
JNI_SRC           = $(JNI_DIR)/wgsl_jni.c
JNI_JAVA_SRCS     = $(JNI_DIR)/WGSL.java $(JNI_DIR)/WGSLTest.java
JNI_LIB           = $(JNI_BUILD_DIR)/libwgsl_jni.$(SHARED_EXT)
ifeq ($(UNAME_S),Darwin)
  JNI_LDFLAGS = -dynamiclib
  JNI_OS_INC  = darwin
else
  JNI_LDFLAGS = -shared
  JNI_OS_INC  = linux
endif

.PHONY: all lib lib-shared cli lsp test test-cli test-lsp-wire example-embedder test-corpus \
        test-diff test-exec-diff test-exec-diff-ref test-golden test-all tsan wasm wasm-size wasm-test wasm-simd fuzz fuzz-long \
        test-cts-import import-cts gen-builtins gen-abi \
        jni test-jni test-python-abi test-rust-abi test-msl-metal linux-compile linux-test dashboard clean help

# Default goal must stay first — do not put file rules above `all`.
all: lib

# Gitignored generated headers. Consumer TUs list them as
# prerequisites so `make clean && make` (fresh clone) self-bootstraps.
BUILTINS_GEN_H   = src/check/builtins.gen.h
BUILTINS_NAMES_H = src/builtins.names.gen.h
BUILTINS_CFOLD_H = src/check/builtins_cfold.gen.h

$(BUILTINS_GEN_H) $(BUILTINS_NAMES_H) $(BUILTINS_CFOLD_H): def/wgsl.def tools/gen-wgsl-builtins.py
	python3 tools/gen-wgsl-builtins.py def/wgsl.def $(BUILTINS_GEN_H) \
		--names-output $(BUILTINS_NAMES_H) \
		--cfold-output $(BUILTINS_CFOLD_H)
	@echo "  GEN     $(BUILTINS_GEN_H) + $(BUILTINS_NAMES_H) + $(BUILTINS_CFOLD_H)"

lib: $(LIB)

lib-shared: $(SHARED)

cli: $(CLI_BIN)

lsp: $(LSP_BIN)

gen-builtins: $(BUILTINS_GEN_H) $(BUILTINS_NAMES_H)

# Multi-lang ABI bindings (Python ctypes + Rust FFI).  Field list must track
# WGSLAbiLayout in include/wgsl.h — generator owns the single source of truth.
gen-abi:
	python3 tools/gen-abi-bindings.py

$(LIB): $(LIB_OBJS)
	@mkdir -p $(@D)
	@$(AR) rcs $@ $^
	@echo "  AR      $@"

$(UNICODE_SRC_LIB): $(UNICODE_SRC_OBJS)
	@mkdir -p $(@D)
	@rm -f $@
	@$(AR) rcs $@ $^
	@echo "  AR      $@"

$(UNICODE_BUILD_DIR)/%.o: $(UNICODE_SRC_DIR)/%.c
	@mkdir -p $(@D)
	@$(CC) $(CFLAGS) $(DEPFLAGS) -MF $@.d $(INCLUDES) -c -o $@ $<
	@echo "  CC      $<"

$(SHARED): $(LIB_OBJS) $(UNICODE_LIB)
	@mkdir -p $(@D)
	@$(CC) $(CFLAGS) $(SHARED_LDFLAGS) -o $@ $(LIB_OBJS) $(UNICODE_LIB) $(MATH_LIBS) $(PTHREAD_LIBS)
	@echo "  SHARED  $@"

$(BUILD_DIR)/%.o: src/%.c
	@mkdir -p $(@D)
	@$(CC) $(CFLAGS) $(DEPFLAGS) $(INCLUDES) -c -o $@ $<
	@echo "  CC      $<"

# Consumer edges: overloads.c → builtins.gen.h, resolver.c → names.gen.h.
$(BUILD_DIR)/check/exprs/overloads.o: $(BUILTINS_GEN_H)
$(BUILD_DIR)/resolver.o: $(BUILTINS_NAMES_H)

$(CLI_OBJ): $(CLI_SRC)
	@mkdir -p $(@D)
	@$(CC) $(CFLAGS) $(DEPFLAGS) $(INCLUDES) -c -o $@ $<
	@echo "  CC      $<"

$(CLI_BIN): $(CLI_OBJ) $(LIB) $(UNICODE_LIB)
	@mkdir -p $(@D)
	@$(CC) $(CFLAGS) -o $@ $(CLI_OBJ) $(LIB) $(UNICODE_LIB) $(MATH_LIBS) $(PTHREAD_LIBS)
	@echo "  LD      $@"

$(BUILD_DIR)/tools/%.o: tools/%.c
	@mkdir -p $(@D)
	@$(CC) $(CFLAGS) $(DEPFLAGS) $(INCLUDES) -c -o $@ $<
	@echo "  CC      $<"

$(LSP_BIN): $(LSP_OBJS) $(LIB) $(UNICODE_LIB)
	@mkdir -p $(@D)
	@$(CC) $(CFLAGS) -o $@ $(LSP_OBJS) $(LIB) $(UNICODE_LIB) $(MATH_LIBS) $(PTHREAD_LIBS)
	@echo "  LD      $@"

example-embedder: $(EMBEDDER_EXAMPLE)
	@$(EMBEDDER_EXAMPLE)

$(EMBEDDER_EXAMPLE): examples/embedder.c $(LIB) $(UNICODE_LIB)
	@mkdir -p $(@D)
	@$(CC) $(CFLAGS) $(DEPFLAGS) -MF $@.d $(INCLUDES) -o $@ $< \
	    $(LIB) $(UNICODE_LIB) $(MATH_LIBS) $(PTHREAD_LIBS)
	@echo "  LD      $@"

# Ban raw payload[3] bit-decodes outside the sole decoder header.
# Parser remains the encoder (writes `payload[i] = …`); pass-specific
# `wgsl_node_resolved_symbol` does not bit-shift.  Matches the historical
# `& 0xFFFFFFFF` / `>> 32` (and case-clause `>> 33`) hand-decode form.
.PHONY: check-payload-accessors
check-payload-accessors:
	@if ! command -v rg >/dev/null 2>&1; then \
	  echo "  SKIP    check-payload-accessors (rg not found)"; \
	  exit 0; \
	fi; \
	bad=$$(rg -n --pcre2 \
	  'payload\[[0-2]\][^;\n]*(&\s*0xFFFFFFFFu?\b|>>\s*3[23]\b)' \
	  src tests -g '*.c' -g '*.h' -g '!src/internal/ast.h' 2>/dev/null \
	  | rg -v 'payload\[[0-2]\]\s*=' || true); \
	if [ -n "$$bad" ]; then \
	  echo "$$bad"; \
	  echo "FAIL: raw payload bit-decode outside src/internal/ast.h (use typed accessors)"; \
	  exit 1; \
	fi; \
	echo "  OK      check-payload-accessors (sole decoder = src/internal/ast.h)"

# Tests link against libwgsl.a; libunicode is added too even when unused
# (linker only pulls in what's referenced).
test: $(TEST_BINS) check-payload-accessors
	@echo
	@echo "  Running $(words $(TEST_BINS)) unit test(s):"
	@pass=0; fail=0; \
	 for t in $(TEST_BINS); do \
	    if $$t; then pass=$$((pass+1)); else fail=$$((fail+1)); fi; \
	 done; \
	 echo "  $$pass passed, $$fail failed"; \
	 [ $$fail -eq 0 ]

linux-compile: lib cli lsp
	@echo "  OK      linux-compile ($(UNAME_S), unicode=$(UNICODE_LIB))"

linux-test: test
	@echo "  OK      linux-test ($(UNAME_S), unicode=$(UNICODE_LIB))"

# Metal compile-test (NOT part of default `make test`).
# Darwin+metal: hard-fail on compile error. Darwin without metal: LOUD fail
# Linux: skip with a message when the Metal toolchain is unavailable.
test-msl-metal: cli
	@bash tools/test-msl-metal.sh

test-cli: $(CLI_BIN)
	@printf 'fn f() {}\n' | $(CLI_BIN) check --quiet -
	@set +e; \
	 printf 'fn f() { var x: i32 = true; }\n' | \
	    $(CLI_BIN) check --quiet - >/dev/null 2>&1; \
	 rc=$$?; [ $$rc -eq 1 ]
	@$(CLI_BIN) json examples/hello-triangle.wgsl >/dev/null
	@$(CLI_BIN) ast examples/hello-triangle.wgsl >/dev/null
	@printf '@compute @workgroup_size(1)\nfn main() {\n  let x = 1;\n  let y = 2;\n  _ = y;\n}\n' > $(BUILD_DIR)/cli-opt.wgsl
	@$(CLI_BIN) optimize --apply $(BUILD_DIR)/cli-opt.wgsl | $(CLI_BIN) check --quiet -
	@! $(CLI_BIN) optimize --apply $(BUILD_DIR)/cli-opt.wgsl | grep -q 'let x'
	@echo "  CLI     smoke passed"

# LSP stdio Content-Length wire coverage (server methods over real framing).
test-lsp-wire: lsp
	@bash tools/test-lsp-wire.sh

# Python ctypes smoke against the shared library.
test-python-abi: lib-shared gen-abi
	@WGSL_LIB=$(SHARED) python3 tests/09-public/test_python_abi.py

# rustc-compile gate for the generated FFI stub (no link against libwgsl).
# SKIP when rustc is absent so plain `make test` stays dependency-light.
test-rust-abi: gen-abi
	@if command -v rustc >/dev/null 2>&1; then \
	  mkdir -p $(BUILD_DIR); \
	  rustc --crate-type lib -o $(BUILD_DIR)/libwgsl_abi_rs.rlib bindings/rust/abi.rs; \
	  echo "  rustc   bindings/rust/abi.rs OK"; \
	else \
	  echo "SKIP  test-rust-abi: rustc not found"; \
	fi

jni: gen-abi $(JNI_LIB) $(JNI_CLASSES_STAMP)

$(JNI_CLASSES_STAMP): $(JNI_JAVA_SRCS)
	@mkdir -p $(JNI_CLASSES_DIR)
	@$(JAVAC) -d $(JNI_CLASSES_DIR) $(JNI_JAVA_SRCS)
	@touch $@
	@echo "  JAVAC   $(JNI_DIR)"

$(JNI_LIB): $(JNI_SRC) $(JNI_HEADER) $(LIB) $(UNICODE_LIB)
	@mkdir -p $(@D)
	@java_home="$${JAVA_HOME:-}"; \
	if [ -z "$$java_home" ]; then \
	  java_home=$$($(JAVAC) -J-XshowSettings:properties -version 2>&1 | \
	    awk -F= '/java.home =/ {gsub(/^[ \t]+|[ \t]+$$/, "", $$2); print $$2; exit}'); \
	fi; \
	if [ ! -f "$$java_home/include/jni.h" ] && [ "$(UNAME_S)" = "Darwin" ] && \
	    [ -x /usr/libexec/java_home ]; then \
	  java_home=$$(/usr/libexec/java_home 2>/dev/null || printf '%s' "$$java_home"); \
	fi; \
	if [ ! -f "$$java_home/include/jni.h" ]; then \
	  echo "SKIP  jni: jni.h not found (set JAVA_HOME to a full JDK)"; \
	  exit 1; \
	fi; \
	$(CC) $(CFLAGS) $(JNI_LDFLAGS) \
	  -I"$$java_home/include" -I"$$java_home/include/$(JNI_OS_INC)" \
	  $(INCLUDES) -o $@ $< \
	  $(LIB) $(UNICODE_LIB) $(MATH_LIBS) $(PTHREAD_LIBS)
	@echo "  JNI     $@"

test-jni: gen-abi
	@if ! command -v $(JAVAC) >/dev/null 2>&1 || \
	    ! command -v $(JAVA) >/dev/null 2>&1; then \
	  echo "SKIP  test-jni: JDK not found"; \
	  exit 0; \
	fi
	@java_home="$${JAVA_HOME:-}"; \
	if [ -z "$$java_home" ]; then \
	  java_home=$$($(JAVAC) -J-XshowSettings:properties -version 2>&1 | \
	    awk -F= '/java.home =/ {gsub(/^[ \t]+|[ \t]+$$/, "", $$2); print $$2; exit}'); \
	fi; \
	if [ ! -f "$$java_home/include/jni.h" ] && [ "$(UNAME_S)" = "Darwin" ] && \
	    [ -x /usr/libexec/java_home ]; then \
	  java_home=$$(/usr/libexec/java_home 2>/dev/null || printf '%s' "$$java_home"); \
	fi; \
	if [ ! -f "$$java_home/include/jni.h" ]; then \
	  echo "SKIP  test-jni: jni.h not found (set JAVA_HOME to a full JDK)"; \
	  exit 0; \
	fi; \
	$(MAKE) jni JAVA_HOME="$$java_home"
	@$(JAVA) -Djava.library.path=$(JNI_BUILD_DIR) -cp $(JNI_CLASSES_DIR) WGSLTest

# Corpus tests — walk `examples/shaders/` and the bench fixtures.  Run
# separately because they're slow and gate real-world shader fidelity
# rather than language-rule conformance.
test-corpus: $(CORPUS_TEST_BINS)
	@echo
	@echo "  Running $(words $(CORPUS_TEST_BINS)) corpus test(s):"
	@pass=0; fail=0; \
	 for t in $(CORPUS_TEST_BINS); do \
	    if $$t; then pass=$$((pass+1)); else fail=$$((fail+1)); fi; \
	 done; \
	 echo "  $$pass passed, $$fail failed"; \
	 [ $$fail -eq 0 ]

# Accept/reject differential vs naga/tint (SKIP if tools absent).
test-diff: cli
	@bash tools/test-diff.sh

# Execution differential: interp vs golden (+ optional WGSL_EXEC_REF).
test-exec-diff: cli
	@bash tools/test-exec-diff.sh

# Execution differential against the in-tree wgpu reference runner.
test-exec-diff-ref: cli
	@WGSL_EXEC_REF=tools/wgpu-ref-runner.sh bash tools/test-exec-diff.sh

# Disk golden corpus: walk tests/15-golden/cases/*.wgsl + *.expect.json.
test-golden: $(GOLDEN_WALK_BIN)
	@echo
	@echo "  Running golden corpus walk ($(GOLDEN_WALK_BIN)):"
	@$(GOLDEN_WALK_BIN)

# CI gate surface: unit + corpus + diff (diff non-strict).
test-all: test test-cli test-lsp-wire test-corpus test-diff test-exec-diff

# Optional libFuzzer harness (SKIP without fuzzer-capable clang).
FUZZ_BIN = $(BUILD_DIR)/fuzz_check
fuzz: $(LIB)
	@if $(CC) -fsanitize=fuzzer -x c - -o /dev/null 2>/dev/null <<< 'int LLVMFuzzerTestOneInput(const unsigned char*d,unsigned long n){(void)d;(void)n;return 0;}' ; then \
	  $(CC) $(CFLAGS) -fsanitize=fuzzer,address -fno-omit-frame-pointer \
	    $(INCLUDES) -o $(FUZZ_BIN) tests/13-fuzz/fuzz_check.c \
	    $(LIB) $(UNICODE_LIB) $(MATH_LIBS) $(PTHREAD_LIBS); \
	  echo "  LD      $(FUZZ_BIN)"; \
	  echo "  Run short: $(FUZZ_BIN) -max_total_time=30 tests/13-fuzz/seeds"; \
	  echo "  Run long:  $(FUZZ_BIN) -max_total_time=3600 -rss_limit_mb=2048 tests/13-fuzz/seeds"; \
	else \
	  echo "SKIP  fuzz: toolchain lacks -fsanitize=fuzzer (use make fuzz-long)"; \
	fi

# Long deterministic mutational fuzz (no libFuzzer; always available).
# Override: FUZZ_ITERS=1000000 FUZZ_SEED=1 make fuzz-long
fuzz-long: $(LIB)
	@bash tools/fuzz_long.sh

# Optional external CTS / shader-tree import + crash-free walk.
import-cts:
	@bash tools/import-cts-wgsl.sh $(CTS_DIR)

test-cts-import: cli
	@bash tools/test-cts-import.sh

# Write .build/conformance.md.
dashboard:
	@bash tools/conformance_dashboard.sh

$(BUILD_DIR)/tests/%: tests/%.c $(LIB) $(UNICODE_LIB)
	@mkdir -p $(@D)
	@$(CC) $(CFLAGS) $(DEPFLAGS) -MF $@.d $(INCLUDES) -o $@ $< $(LIB) $(UNICODE_LIB) $(MATH_LIBS) $(PTHREAD_LIBS)
	@echo "  LD      $@"

# ThreadSanitizer build (`make tsan`).

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
	@$(CC) $(TSAN_CFLAGS) $(DEPFLAGS) $(INCLUDES) -c -o $@ $<
	@echo "  CC[ts]  $<"

$(TSAN_BUILD_DIR)/check/exprs/overloads.o: $(BUILTINS_GEN_H)
$(TSAN_BUILD_DIR)/resolver.o: $(BUILTINS_NAMES_H)

$(TSAN_LIB): $(TSAN_LIB_OBJS)
	@mkdir -p $(@D)
	@$(AR) rcs $@ $^
	@echo "  AR[ts]  $@"

$(TSAN_BUILD_DIR)/tests/%: tests/%.c $(TSAN_LIB) $(UNICODE_LIB)
	@mkdir -p $(@D)
	@$(CC) $(TSAN_CFLAGS) $(TSAN_LDFLAGS) $(DEPFLAGS) -MF $@.d $(INCLUDES) \
	    -o $@ $< $(TSAN_LIB) $(UNICODE_LIB) $(MATH_LIBS) $(PTHREAD_LIBS)
	@echo "  LD[ts]  $@"

# WASM build: `make wasm` produces wgsl_compiler.{js,wasm}.
#
# Drives Emscripten over the same source list as the native lib; links
# against the prebuilt `vendor/unicode/libunicode.wasm.a`.  The output
# is a self-contained ES module (`MODULARIZE=1`, `EXPORT_NAME=WGSL`)
# usable in Node, web workers, and the browser.

EMCC             ?= emcc
WASM_SIMD        ?= 0
ifeq ($(WASM_SIMD),1)
WASM_BUILD_DIR    = $(BUILD_DIR)/wasm-simd
EM_SIMD_FLAGS     = -msimd128
else
WASM_BUILD_DIR    = $(BUILD_DIR)/wasm
EM_SIMD_FLAGS     =
endif
WASM_BUNDLE       = $(WASM_BUILD_DIR)/wgsl_compiler.js
WASM_SIZE_REPORT  = $(WASM_BUILD_DIR)/size.txt
UNICODE_LIB_WASM  = vendor/unicode/libunicode.wasm.a

EM_CFLAGS = -Oz -flto -Wall -Wextra -Wno-unused-parameter \
            -std=c11 -fno-strict-aliasing -fvisibility=hidden \
            -DNDEBUG -DWGSL_NO_FS $(EM_SIMD_FLAGS)

# Public surface from include/wgsl.h, prefixed with `_` per emcc convention.
EM_EXPORTS = _wgsl_init,_wgsl_shutdown,\
_wgsl_spec_pin,_wgsl_unicode_version,\
_wgsl_check,_wgsl_check_n,_wgsl_free,\
_wgsl_ok,_wgsl_error,_wgsl_module_json,_wgsl_module_json_len,\
_wgsl_diagnostic_count,_wgsl_diagnostic,\
_wgsl_lex,_wgsl_semantic_tokens,_wgsl_lex_free,\
_wgsl_hover_at,_wgsl_definition_at,\
_wgsl_hover_at_into,_wgsl_definition_at_into,\
_wgsl_document_symbols,_wgsl_outline_free,\
_wgsl_references_at,_wgsl_references_free,\
_wgsl_prepare_rename,\
_wgsl_folding_ranges,_wgsl_folding_free,\
_wgsl_completions_at,_wgsl_completions_free,\
_wgsl_signature_help_at,_wgsl_signature_help_at_into,\
_wgsl_code_actions,_wgsl_code_actions_free,\
_wgsl_session_new,_wgsl_session_destroy,\
_wgsl_session_check,_wgsl_session_check_n,\
_wgsl_session_info,_wgsl_session_changed_decl,\
_wgsl_format,_wgsl_format_n,_wgsl_format_range,_wgsl_format_range_n,_wgsl_format_result,\
_wgsl_project_open_from_string,_wgsl_project_close,_wgsl_project_match,\
_wgsl_check_with_preamble,\
_wgsl_interp,_wgsl_debug_oracle,_wgsl_free_string,\
_wgsl_ml_analyze_json_src,\
_wgsl_optimize_json_src,_wgsl_optimize_apply,\
_wgsl_to_msl,\
_wgsl_abi_layout,\
_malloc,_free

EM_RUNTIME_METHODS = cwrap,ccall,UTF8ToString,stringToUTF8,\
lengthBytesUTF8,getValue,setValue,HEAP8,HEAPU8,HEAPU32

EM_LDFLAGS = \
    -Oz -flto --closure 1 \
    -s WASM=1 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME=WGSL \
    -s EXPORT_ES6=0 \
    -s ENVIRONMENT=node,web,worker \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INITIAL_MEMORY=16MB \
    -s STACK_SIZE=2MB \
    -s ASSERTIONS=0 \
    -s FILESYSTEM=0 \
    -s EXPORTED_FUNCTIONS='[$(EM_EXPORTS)]' \
    -s EXPORTED_RUNTIME_METHODS='[$(EM_RUNTIME_METHODS)]' \
    $(EM_SIMD_FLAGS) \
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
	@$(EMCC) $(EM_CFLAGS) $(DEPFLAGS) $(INCLUDES) -c -o $@ $<
	@echo "  EMCC    $<"

$(WASM_BUILD_DIR)/check/exprs/overloads.o: $(BUILTINS_GEN_H)
$(WASM_BUILD_DIR)/resolver.o: $(BUILTINS_NAMES_H)

# Emit a small size report for release review.  Informational by default;
# add thresholds in CI once product budgets exist.
wasm-size: $(WASM_BUNDLE)
	@mkdir -p $(WASM_BUILD_DIR)
	@js_bytes=$$(wc -c < "$(WASM_BUNDLE)" | tr -d ' '); \
	  wasm_file="$(WASM_BUNDLE:.js=.wasm)"; \
	  wasm_bytes=0; \
	  if [ -f "$$wasm_file" ]; then wasm_bytes=$$(wc -c < "$$wasm_file" | tr -d ' '); fi; \
	  { printf 'js_bytes=%s\n' "$$js_bytes"; \
	    printf 'wasm_bytes=%s\n' "$$wasm_bytes"; \
	    printf 'simd128=%s\n' "$(WASM_SIMD)"; } | tee "$(WASM_SIZE_REPORT)"

# Smoke-test the bundle in Node.  Verifies the public C/JS API survives
# the Emscripten round-trip and product JSON surfaces answer correctly.
NODE ?= node
wasm-test: $(WASM_BUNDLE) wasm-size
	@WGSL_WASM_JS=$(WASM_BUNDLE) $(NODE) tests/10-wasm/smoke.js
	@WGSL_WASM_JS=$(WASM_BUNDLE) $(NODE) tests/10-wasm/interp_smoke.js

# Build/test the same JS surface with wasm SIMD128 enabled.
wasm-simd:
	@$(MAKE) wasm-test WASM_SIMD=1

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "Targets:"
	@echo "  make            — build $(LIB_NAME)"
	@echo "  make cli        — build .build/$(WGSL_BIN) command-line tool"
	@echo "  make lsp        — build .build/wgsl_lsp (stdio JSON-RPC language server)"
	@echo "  make test       — build + run unit tests under tests/*"
	@echo "  make linux-compile — Linux build gate (lib + CLI + LSP)"
	@echo "  make linux-test — Linux unit-test gate"
	@echo "  make test-cli   — build + smoke-test the CLI"
	@echo "  make test-lsp-wire — LSP stdio Content-Length wire coverage"
	@echo "  make test-corpus — corpus / bench harness (SKIP if no examples/shaders)"
	@echo "  make test-diff  — accept/reject vs naga/tint (SKIP if tools absent)"
	@echo "  make check-payload-accessors — ban raw payload bit-decode outside ast.h"
	@echo "  make test-all   — test + test-cli + test-corpus + test-diff"
	@echo "  make example-embedder — build + run the C API example"
	@echo "  make tsan       — build + run ThreadSanitizer tests (tests/01-tsan/*)"
	@echo "  make wasm       — build wgsl_compiler.{js,wasm} via Emscripten"
	@echo "  make wasm-size  — write .build/wasm*/size.txt"
	@echo "  make wasm-test  — run Node smoke test against the wasm bundle"
	@echo "  make wasm-simd  — build + test the SIMD128 wasm bundle"
	@echo "  make test-diff       — verdict diff vs naga/tint (SKIP if absent)"
	@echo "  make test-exec-diff  — execution diff: interp vs golden buffers"
	@echo "  make test-exec-diff-ref — execution diff vs in-tree wgpu reference runner"
	@echo "  make test-golden     — disk golden corpus (cases/*.wgsl walk)"
	@echo "  make fuzz            — optional libFuzzer harness (SKIP if unsupported)"
	@echo "  make fuzz-long       — long mutational fuzz (FUZZ_ITERS, no libFuzzer)"
	@echo "  make import-cts CTS_DIR=path — copy *.wgsl into .build/cts-import"
	@echo "  make test-cts-import — crash-free walk of imported CTS shaders"
	@echo "  make gen-builtins — regenerate check/builtins.gen.h (+ names)"
	@echo "  make gen-abi    — regenerate Python/Rust ABI bindings"
	@echo "  make lib-shared — build shared lib for ctypes/dlopen"
	@echo "  make jni       — build Java JNI binding library/classes"
	@echo "  make test-jni  — smoke-test JNI bindings (SKIP if no JDK)"
	@echo "  make test-python-abi — smoke-test Python ABI bindings"
	@echo "  make test-rust-abi — rustc-compile bindings/rust/abi.rs (SKIP if no rustc)"
	@echo "  make test-msl-metal — xcrun metal compile-test (Darwin; not in make test)"
	@echo "  make dashboard  — write .build/conformance.md"
	@echo "  make clean      — remove $(BUILD_DIR)"

-include $(LIB_OBJS:.o=.d)
-include $(UNICODE_SRC_OBJS:.o=.d)
-include $(CLI_OBJ:.o=.d)
-include $(LSP_OBJS:.o=.d)
-include $(EMBEDDER_EXAMPLE:=.d)
-include $(TSAN_LIB_OBJS:.o=.d)
-include $(WASM_LIB_OBJS:.o=.d)
-include $(TEST_BINS:=.d)
-include $(CORPUS_TEST_BINS:=.d)
-include $(TSAN_TEST_BINS:=.d)
