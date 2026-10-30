#!/usr/bin/env bash
# Strata 0.2.7.1 (commit 46c152a) on Unsloth's UD-Q4_K_XL GGUF, installed by Strata's own setup (release engine).
#   ./strata.sh install | one-user | one-user-no-draft | many-users
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

if [ "${1:-}" = install ]; then
  git clone https://github.com/Niko1221/Strata.git && cd Strata && git checkout 46c152a1310f77cd2f0ce2c7cd213962df14fe83
  # STRATA_UV=1: setup builds its venv with uv (needed where the system Python has no ensurepip).
  STRATA_UV=1 ./setup.sh --family unsloth --model UD-Q4_K_XL --context 65536 --vision no --mtp on --parallel 8 \
      --models-dir "$PWD/../models" --yes --no-start
  # Setup writes strata-unsloth-ud-q4_k_xl.json with KV streaming on (--kv-resident 32768). Derive the variants:
  python3 "$HERE/strata_variants.py" strata-unsloth-ud-q4_k_xl.json "$HERE/strata"
  exit 0
fi

cd Strata
case "${1:-one-user}" in
  one-user)          CFG=one-user.json;          SLOTS=1 ;;   # experts and KV cache on the card, draft layer on
  one-user-no-draft) CFG=one-user-no-draft.json; SLOTS=1 ;;
  many-users)        CFG=setup-default.json;     SLOTS=4 ;;   # setup's own config, the 4 slots it recommends here
esac
exec .venv/bin/python -m serve.server --engine strata --config "$HERE/strata/$CFG" \
    --host 127.0.0.1 --port 18080 --parallel "$SLOTS" --no-api-key --offline --no-fit-max-tokens
# Send "strata_checkpoint": false on requests nothing will continue. Otherwise finished long prompts are checkpointed
# and parked in a RAM cache, and once it fills new prompts stall (25 s at 8K, 60 to 90 s at 32K in this test).
