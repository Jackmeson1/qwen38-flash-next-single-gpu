# Results

`summary/` holds the tables the README uses, rebuilt from `raw/` by `bench/summarize.py`.
`raw/<run>/` holds, per run: `run.json` (engine, weights, exact server command), `bench.jsonl` (one line per test
block: its summary plus per-request timings and token counts; generated text is left out) and, where measured,
`quality.json` (per-question answers parsed, correctness, token counts).

| Folder | README row |
| --- | --- |
| `vllm-draft-off` | vLLM, draft off: one user, 1 to 32 users |
| `vllm-draft-off-saturation-64-128` | vLLM, draft off: 64 and 128 users (`--max-num-seqs 256`, one block each) |
| `vllm-draft-off-saturation-256` | vLLM, draft off: 256 users (two blocks; a third was interrupted and left out) |
| `vllm-draft-off-quality` | vLLM quality, draft off |
| `vllm-draft-on` | vLLM, draft on: one user, 1 to 16 users, quality |
| `sglang-draft-off` | SGLang: one user, 1 and 2 users |
| `sglang-draft-off-users-4-32` | SGLang: 4 to 32 users |
| `strata-1slot-draft-on` | Strata, 1 slot, draft on: one user (machine B) |
| `strata-1slot-draft-on-machine-a` | the same config on machine A, for the machine-to-machine check |
| `strata-1slot-draft-off` | Strata, 1 slot, draft off: one user |
| `strata-1slot-draft-on-checkpoints-on` | Strata quality; also the one-user timings with `strata_checkpoint` left on (see Gotchas) |
| `strata-1slot-draft-on-mmq-kq` | Strata with `STRATA_MMQ_KQ=1` (see Gotchas) |
| `strata-4slots` | Strata, setup's default config with 4 slots: 1 to 8 users |
| `llamacpp` | llama.cpp: one user, 1 to 16 users |

Machine A ran vLLM, SGLang, llama.cpp and the Strata machine-a, MMQ and 4-slot runs; machine B ran the other Strata runs. Both
were the same type: one RTX PRO 6000 Blackwell Server Edition, 48 vCPU, 176 GB RAM.
