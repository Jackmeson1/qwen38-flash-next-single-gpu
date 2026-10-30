# Reference: setup, full results and fixes

The numbers behind the [README](../README.md), measured 2026-10-29 on one RTX PRO 6000 Blackwell Server Edition (96 GB). Every figure is computed from [results/](../results) by `bench/summarize.py`.

## Setup

| | |
| --- | --- |
| GPU | 1x NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB (95.6 GiB usable, sm_120), driver 580.82, CUDA 13.0 |
| Host | rented cloud machine, 48 vCPU, 176 GB RAM |
| Model | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): 125B-parameter MoE language model (512 experts, about 6B active per token), hybrid linear and full attention, one MTP layer, plus a 51.2B-parameter n-gram embedding table (47.7 GiB in FP8) |
| Context | 64K tokens on every engine |
| Requests | streaming chat completions, greedy, thinking off, each prompt a unique public-domain book passage ([data/](../data)) |

Every engine keeps the n-gram table in CPU memory. It is a lookup of a few rows per token, so it costs little speed
there and frees about 48 GiB of the card.

| Engine | Weights | On the card | Launch |
| --- | --- | --- | --- |
| vLLM 0.32.0 | [albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE): Intel's AutoRound W4A16 (experts INT4, group 128; attention, routers, shared experts and MTP in BF16) with the n-gram table in FP8, 124.75 GB downloaded | 68.6 GiB of weights | [configs/vllm.sh](../configs/vllm.sh) |
| SGLang 0.5.22 | same files | 69.0 GiB of weights | [configs/sglang.sh](../configs/sglang.sh) |
| Strata 0.2.7.1 (commit 46c152a) | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) UD-Q4_K_XL (4-bit with sensitive tensors kept higher), 111 GB | all 24,576 experts, 71.7 GiB | [configs/strata.sh](../configs/strata.sh) |
| llama.cpp b10666 | same GGUF as Strata | all layers | [configs/llamacpp.sh](../configs/llamacpp.sh) |

vLLM and SGLang load identical files, and so do Strata and llama.cpp. Across the two pairs the 4-bit formats differ,
which is why quality is measured too.

## One user

Strata and vLLM keep their decode speed as the prompt grows to 32K tokens; llama.cpp loses 38%. The draft layer adds
50 to 60% to vLLM and 20 to 25% to Strata. Decode speed alone hides prompt reading, which is where vLLM wins:

| Prompt length | 1K | 8K | 32K | Prompt speed at 32K |
| --- | --- | --- | --- | --- |
| **Time to first token** | | | | |
| vLLM, draft off / on | 0.09 / 0.10 s | 0.60 s | 2.41 / 2.49 s | 13,500 tokens/s |
| SGLang | 0.32 s | 1.39 s | 5.61 s | 5,800 tokens/s |
| Strata, 1 slot | 0.67 s | 1.68 s | 6.41 s | 5,100 tokens/s |
| llama.cpp | 0.42 s | 2.79 s | 12.38 s | 2,640 tokens/s |
| **Whole 512-token answer** | | | | |
| vLLM, draft on | 3.46 s | 4.12 s | 5.80 s | |
| Strata, 1 slot, draft on | 3.66 s | 4.88 s | 9.45 s | |
| vLLM, draft off | 5.33 s | 5.89 s | 7.71 s | |
| llama.cpp | 5.86 s | 9.10 s | 21.08 s | |

vLLM's draft layer accepted 2.74 tokens per step on average (per position 0.76, 0.56, 0.41). Decode with a draft layer
varies between blocks by up to 10% for vLLM and 12% for Strata; everything else stays within 1%. Strata with 4 slots
decoded a lone request at 197 tokens/s, above its 1-slot setting; the cause is not known.

## Many users

Each user sends a 1K-token prompt and gets a 512-token answer, then sends the next.

| Users | vLLM, draft off | vLLM, draft on | SGLang | llama.cpp | Strata, 4 slots |
| --- | --- | --- | --- | --- | --- |
| 1 | 96 (98) | 148 (151) | 38 (38) | 88 (95) | 151 (197) |
| 4 | 285 (74) | 404 (106) | 124 (33) | 170 (54) | 182 (53) |
| 8 | 457 (61) | 616 (80) | 168 (23) | 208 (32) | 179 (52) |
| 16 | 717 (49) | 745 (65) | 230 (16) | 226 (16) | |
| 32 | 1,031 (35) | | 311 (11) | | |
| 64 | 1,409 (24) | | | | |
| 256 | 1,429 (22) | | | | |

Total output tokens/s, per-user decode speed in brackets. All points: [results/summary/many_users.csv](../results/summary/many_users.csv).

- vLLM levels off at about 1,400 tokens/s from 64 users; past that requests queue (median first token 24 s at 128 users).
- vLLM's draft layer adds 35% at 8 users and 4% at 16: a busy card leaves little spare work for checking drafts. It
  also halves the KV cache (592,612 tokens without it, 254,234 with it).
- Strata's batching serves a few local users. With the 4 slots its setup recommends, the total stays near 180 tokens/s
  and the 8-user median first token is 11.8 s.

## Quality

| | GSM8K (250) | MMLU-Pro (300) |
| --- | --- | --- |
| vLLM, AutoRound W4A16 | 97.6% (95% CI 94.9 to 98.9) | 85.3% (80.9 to 88.9) |
| vLLM, AutoRound W4A16, draft on | 98.0% (95.4 to 99.1) | 83.7% (79.1 to 87.4) |
| Strata, UD-Q4_K_XL | 97.2% (94.3 to 98.6) | 82.7% (78.0 to 86.5) |

On the same MMLU-Pro questions vLLM was right where Strata was wrong 11 times and the reverse 3 times (exact McNemar
p = 0.057; [results/summary/quality_pairs.csv](../results/summary/quality_pairs.csv)). Fixed random samples, greedy,
thinking off, 4,096-token cap; an answer cut off at the cap counts as wrong (15 to 20 MMLU-Pro answers per run).

## Memory

| | Card, peak | Host RAM, peak | Time to ready |
| --- | --- | --- | --- |
| vLLM, draft off | 89.3 GiB | 72.8 GiB | 530 to 625 s |
| vLLM, draft on | 92.3 GiB | 73.0 GiB | 660 s |
| SGLang | 92.4 GiB | 83.5 GiB | 710 to 735 s |
| Strata, 1 slot | 79.8 GiB | 79.1 GiB | 625 to 770 s |
| Strata, 4 slots | 82.0 GiB | 83.5 GiB | 695 s |
| llama.cpp | 83.2 GiB | 43.1 GiB | 350 s |

The tightest fits are SGLang and vLLM with the draft layer, at 92.4 and 92.3 of 95.6 GiB. vLLM reserves 92% of the card
up front. Start-up time is mostly the checkpoint read from disk (about 0.25 GB/s here).

## Gotchas

Every engine needed at least one fix beyond its quick start. Error text is quoted so it can be searched.

| Engine | Symptom | Cause | Fix |
| --- | --- | --- | --- |
| vLLM | `max_num_seqs (1024) exceeds available Mamba cache blocks (379)` | The linear-attention state needs one block per sequence | `--max-num-seqs 64` (up to 256) |
| vLLM | `FileNotFoundError: ... 'ninja'` after about 9 minutes of loading | FlashInfer builds a kernel on first use and cannot find the venv's `ninja` | Put the venv's `bin` on `PATH` |
| vLLM | Draft layer accepts almost nothing (acceptance length 1.00) and decode drops to about 61 tokens/s (early run) | The W4A16 copy has no MTP weights in its main files | Point the draft at Intel's BF16 MTP file: [configs/make_mtp_draft.sh](../configs/make_mtp_draft.sh) |
| vLLM | Engine start times out | Large checkpoint read plus kernel builds outlast 600 s | `VLLM_ENGINE_READY_TIMEOUT_S=3600` |
| SGLang | Out of memory after loading, in `gptq_marlin_moe_repack` | A reference cycle keeps the pre-repack copies alive ([sglang#38900](https://github.com/sgl-project/sglang/pull/38900), unmerged) | `"sglang_backend": "gptq"` in `quantization_config` routes experts to the Triton kernel (`"triton"` is rejected: `Unsupported backend: triton`) |
| SGLang | `CUDA error: an illegal memory access` while capturing prompt CUDA graphs | Triton 4-bit kernel on sm_120 | `--disable-prefill-cuda-graph` |
| SGLang | Slow: `Using default MoE kernel config. Performance might be sub-optimal!` | No tuned Triton config for this expert shape on this card | Tune it with SGLang's MoE tuning script (not done here) |
| Strata | First token at 32K takes 59 to 93 s instead of 6.4 s; some 8K prompts take 25 s | Finished long prompts are checkpointed and parked in a RAM cache; once it fills, new prompts stall | `"strata_checkpoint": false` on requests nothing will continue |
| Strata | 8 slots slower than 1 even for one user (early run) | Each slot reserves KV for the full context, leaving little room to read prompts | Use the 4 slots setup recommends, or 1 for one user |
| Strata | `STRATA_MMQ_KQ=1` makes the first token 8 to 18 times slower | Unknown on this build | Leave it off |
| Strata | Engine exits at start with the draft removed: `needs --native SHARD1, --spec T (T >= 2)` | The native pack needs `--spec` even without `--mtp` | Remove only `--mtp` |
| llama.cpp | About 60 tokens/s instead of 94 (early run) | Default mmap reads the n-gram table lazily from disk while decoding | `--load-mode none --tensor-read-lazy off` |
| llama.cpp | Every second streamed request fails with a disconnect | llama-server closes the connection after each answer | Do not reuse pooled connections |

"Early run" figures come from stopped runs that are not in [results/](../results).

## Method

- One client for all engines ([bench/bench_client.py](../bench/bench_client.py)). Time to first token = first streamed
  text. Decode speed = (output tokens - 1) / time after the first token. Total throughput = output tokens / block wall
  time. Each table cell is the median over blocks of per-request medians.
- One user: 5 requests per block, 2 to 3 blocks per prompt length. Many users: 2 to 3 blocks of at least 90 s per point,
  after a warm-up at that concurrency.
- Prefix caching was off for vLLM and SGLang, Strata's checkpoints were off, and every prompt starts with unique text.
- Strata's one-user runs ran on a second machine of the same type. The same Strata config on both machines differed by
  8% in decode speed (172 vs 159 tokens/s at 1K) and not at all in time to first token.

Not tested: answers longer than 512 tokens, context past 64K, thinking mode, images, tool calls, multiple GPUs,
smaller cards, SGLang's draft layer and quality.

## Reproduce

```bash
# 1. Start one engine (each script has an `install` step first)
./configs/vllm.sh install && ./configs/vllm.sh draft-off

# 2. Unpack the frozen prompts and questions (or rebuild them with bench/make_prompts.py and bench/quality.py --prepare)
tar xJf data/prompts-and-questions.tar.xz

# 3. Run the suite against it, then rebuild the tables and figures
python bench/run_suite.py --url http://127.0.0.1:8000 --name my-vllm-run
python bench/summarize.py && python ../figures/make_figures.py
```

Add `--strata` for Strata (sends `strata_checkpoint: false`) and `--force-close` for llama.cpp. Per-request records
for every run in this report are in [results/raw](../results/raw); [results/README.md](../results/README.md) maps each folder
to a row above.
