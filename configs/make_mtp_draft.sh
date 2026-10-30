#!/usr/bin/env bash
# Build a small MTP draft folder for vLLM (and SGLang). The AutoRound W4A16 copy has no MTP tensors in its main files,
# so without this the draft layer has no weights and accepts almost nothing (acceptance length 1.00).
# Intel's original AutoRound checkpoint keeps the same model's MTP layer, in BF16, in one 5.2 GB file.
# Using a separate folder also stops vLLM from reading the whole 116 GiB checkpoint a second time for the draft.
set -euo pipefail
MODEL_DIR=${MODEL_DIR:-$PWD/models/w4a16}
MTP_DRAFT_DIR=${MTP_DRAFT_DIR:-$PWD/models/mtp-draft}
mkdir -p "$MTP_DRAFT_DIR"
hf download Intel/Qwen3.8-Flash-Next-W4A16-AutoRound model_extra_tensors.safetensors model.safetensors.index.json \
    --revision 4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897 --local-dir "$MTP_DRAFT_DIR/intel"
for f in chat_template.jinja generation_config.json preprocessor_config.json tokenizer.json tokenizer_config.json \
         video_preprocessor_config.json; do
  if [ -e "$MODEL_DIR/$f" ]; then ln -sf "$MODEL_DIR/$f" "$MTP_DRAFT_DIR/$f"; fi
done
cp "$MODEL_DIR/config.json" "$MTP_DRAFT_DIR/config.json"
ln -sf "$MTP_DRAFT_DIR/intel/model_extra_tensors.safetensors" "$MTP_DRAFT_DIR/model_extra_tensors.safetensors"
python3 "$(dirname "$0")/mtp_index.py" "$MTP_DRAFT_DIR"
