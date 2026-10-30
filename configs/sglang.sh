#!/usr/bin/env bash
# SGLang 0.5.22 on the same AutoRound W4A16 files as vLLM.
# SGLang's default path for these weights (GPTQ-Marlin MoE) runs out of card memory while repacking the experts:
# a reference cycle in its Flash-Next model code keeps the old copies alive (sgl-project/sglang#38900, unmerged).
# The workaround below routes the experts to SGLang's Triton WNA16 kernel. That kernel has no tuned config for this
# card ("Using default MoE kernel config"), which is why SGLang ran 2.6x slower than vLLM for one user and 3.3x slower at 32 users here.
#   ./sglang.sh install | serve
set -euo pipefail
MODEL_DIR=${MODEL_DIR:-$PWD/models/w4a16}
MODEL_DIR_SGLANG=${MODEL_DIR_SGLANG:-$PWD/models/w4a16-sglang}

if [ "${1:-}" = install ]; then
  uv venv venv-sgl --python 3.12
  uv pip install --python venv-sgl/bin/python "sglang[all]==0.5.22" --prerelease=allow   # pins a pre-release dependency
  # Same weights, own config.json with quantization_config.sglang_backend = "gptq" (Triton MoE, no Marlin repack).
  mkdir -p "$MODEL_DIR_SGLANG"
  for f in "$MODEL_DIR"/*; do b=$(basename "$f"); [ "$b" = config.json ] || ln -sf "$f" "$MODEL_DIR_SGLANG/$b"; done
  python3 -c "import json,sys; c=json.load(open(sys.argv[1])); c['quantization_config']['sglang_backend']='gptq'; json.dump(c, open(sys.argv[2],'w'), indent=2)" \
      "$MODEL_DIR/config.json" "$MODEL_DIR_SGLANG/config.json"
  exit 0
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec venv-sgl/bin/python -m sglang.launch_server --model-path "$MODEL_DIR_SGLANG" --served-model-name qwen --port 8000 \
    --context-length 65536 --mem-fraction-static 0.93 --reasoning-parser qwen3 \
    --ple-offload-embedding \
    --disable-prefill-cuda-graph
# --disable-prefill-cuda-graph: prompt-side graph capture crashed with an illegal memory access on sm_120.
# The benchmark also passed --disable-radix-cache.
