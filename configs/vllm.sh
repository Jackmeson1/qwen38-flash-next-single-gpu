#!/usr/bin/env bash
# vLLM 0.32.0 serving Qwen3.8-Flash-Next at 4 bits on one 96 GB card.
# Weights: Intel's AutoRound W4A16 (experts INT4, group 128; attention, linear attention, routers, shared experts,
# MTP and head in BF16), in the copy that stores the n-gram table in FP8 (124.75 GB downloaded).
#   ./vllm.sh install | draft-off | draft-on | many-users
set -euo pipefail
MODEL_DIR=${MODEL_DIR:-$PWD/models/w4a16}
MTP_DRAFT_DIR=${MTP_DRAFT_DIR:-$PWD/models/mtp-draft}

if [ "${1:-}" = install ]; then
  uv venv venv-vllm --python 3.12
  uv pip install --python venv-vllm/bin/python "vllm==0.32.0" --torch-backend=auto
  hf download albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE --revision c6af9fee7ad2cb8ef44b534a99b8c1b81fb0f6da \
      --local-dir "$MODEL_DIR" --exclude 'runtime/*' --exclude 'docs/*'
  exit 0
fi

export PATH=$PWD/venv-vllm/bin:$PATH      # FlashInfer builds kernels on first use and needs the venv's ninja
export VLLM_ENGINE_READY_TIMEOUT_S=3600   # loading 116 GiB plus kernel builds can outlast the 600 s default

SEQS=64                                   # the default 1024 exceeds the 379 linear-attention state blocks
[ "${1:-}" = many-users ] && SEQS=256
ARGS=(--served-model-name qwen --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.92
      --language-model-only --kv-cache-dtype auto --reasoning-parser qwen3
      --engram-config '{"cpu_offload": true}'   # n-gram table in pinned host memory (vLLM's default for this model)
      --max-num-seqs "$SEQS")
# The benchmark also passed --no-enable-prefix-caching so no prompt could be served from cache.

if [ "${1:-draft-off}" = draft-on ]; then
  # Build the draft folder first with make_mtp_draft.sh: this checkpoint has no MTP weights in its main files.
  ARGS+=(--speculative-config "{\"method\": \"mtp\", \"model\": \"$MTP_DRAFT_DIR\", \"num_speculative_tokens\": 3}")
fi
exec vllm serve "$MODEL_DIR" "${ARGS[@]}"
