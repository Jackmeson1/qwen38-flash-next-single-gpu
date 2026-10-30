#!/usr/bin/env bash
# llama.cpp b10666 on Unsloth's UD-Q4_K_XL GGUF (the same files Strata uses).
#   ./llamacpp.sh install | serve
set -euo pipefail
GGUF_DIR=${GGUF_DIR:-$PWD/models/ud-q4_k_xl}

if [ "${1:-}" = install ]; then
  git clone https://github.com/ggml-org/llama.cpp.git && (cd llama.cpp && git checkout b10666 &&
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF &&
    cmake --build build --target llama-server -j 24)
  hf download unsloth/Qwen3.8-Flash-Next-GGUF --revision 38bb39ee97821de2c9009abb7e93950eec396e66 \
      --include 'UD-Q4_K_XL/*' --local-dir models/unsloth
  mkdir -p "$GGUF_DIR" && mv models/unsloth/UD-Q4_K_XL/*.gguf "$GGUF_DIR/"
  exit 0
fi

exec llama.cpp/build/bin/llama-server -m "$GGUF_DIR/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf" \
    --host 127.0.0.1 --port 8090 --alias qwen --jinja \
    -c 65536 --kv-unified --parallel 16 \
    -ngl 999 -ot per_layer_token_embd=CPU -ub 2048 \
    --load-mode none --tensor-read-lazy off
# --load-mode none --tensor-read-lazy off: without them the n-gram table is read lazily from disk while decoding
# (about 60 instead of 94 tokens/s here). llama-server closes the connection after each streamed answer, so clients
# should not reuse pooled connections.
