# Makefile — local convenience build.
#   make test        build+run the diffusion core test (no ggml needed)
#   make fetch-ggml  vendor ggml into third_party/
#   make runtime     build the full runtime (requires third_party/ggml)
#   make clean

CXX      ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Isrc

THIRD   := third_party
GGML    := $(THIRD)/ggml
GGML_LIB:= $(GGML)/build/src/libggml.a

.PHONY: test fetch-ggml runtime clean

test:
	$(CXX) $(CXXFLAGS) -o /tmp/test_diffusion tests/test_diffusion.cpp
	/tmp/test_diffusion

fetch-ggml:
	mkdir -p $(THIRD)
	@test -d $(GGML) || git clone --depth 1 --branch b4227 https://github.com/ggml-org/ggml.git $(GGML)
	@echo "ggml vendored at $(GGML); build it with the CMake flow or point TF_RUNTIME_GGML_SOURCE at it"

runtime:
	@test -d $(GGML) || (echo "run 'make fetch-ggml' first" && exit 1)
	@test -f $(GGML_LIB) || (echo "build $(GGML) first (cmake -S $(GGML) -B $(GGML)/build)" && exit 1)
	$(CXX) $(CXXFLAGS) -I$(GGML) -I$(GGML)/include \
		src/main.cpp src/engine/model.cpp src/tokenizer/tokenizer.cpp \
		$(GGML_LIB) -lpthread -o runtime

clean:
	rm -f runtime /tmp/test_diffusion
