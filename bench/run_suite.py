"""Run the benchmark suite against an OpenAI-compatible server that is already running.

  python run_suite.py --url http://127.0.0.1:8000 --name vllm-draft-off --only one-user,many-users,quality
  python run_suite.py --url http://127.0.0.1:18080 --name strata-1slot-draft-on --strata --only one-user
  python run_suite.py --url http://127.0.0.1:8090 --name llamacpp --force-close --concs 1,2,4,8,16

Writes results/raw/<name>/bench.jsonl (and quality.jsonl). Prompts come from make_prompts.py (--prompts), questions
from quality.py --prepare (--qdata).

one-user:   1K, 8K and 32K-token prompts, 5 requests per block, 512-token answers
many-users: 1K prompts at each concurrency, at least --block-seconds and max(--req-min, --req-per-conc x C)
            requests per block, after 2 x C warm-up requests; concurrency order shuffled
quality:    250 GSM8K + 300 MMLU-Pro questions, greedy, thinking off, 4,096-token cap
"""
import argparse
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def bench(args, out, prompts, tag, **kw):
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    if args.strata:
        # Strata's field for one-off requests: no checkpoint, no kept session (long prompts stall without it).
        extra["strata_checkpoint"] = False
    cmd = [sys.executable, os.path.join(HERE, "bench_client.py"), "--url", args.url, "--model", args.model,
           "--prompts", os.path.join(args.prompts, prompts), "--out", out, "--tag", tag, "--extra", json.dumps(extra)]
    if args.force_close:
        cmd.append("--force-close")
    for k, v in kw.items():
        cmd += ["--" + k.replace("_", "-"), str(v)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(tag, (r.stdout or r.stderr).strip()[-300:], flush=True)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--name", required=True, help="results/raw/<name>/")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--only", default="one-user,many-users,quality")
    ap.add_argument("--concs", default="1,2,4,8,16,32")
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--block-seconds", type=float, default=90)
    ap.add_argument("--req-min", type=int, default=20)
    ap.add_argument("--req-per-conc", type=int, default=10)
    ap.add_argument("--q-conc", type=int, default=8)
    ap.add_argument("--prompts", default="prompts")
    ap.add_argument("--qdata", default="qdata")
    ap.add_argument("--results", default=os.path.join(HERE, "..", "results", "raw"))
    ap.add_argument("--strata", action="store_true", help="send strata_checkpoint=false")
    ap.add_argument("--force-close", action="store_true", help="new connection per request (llama-server)")
    args = ap.parse_args()

    out_dir = os.path.join(args.results, args.name)
    os.makedirs(out_dir, exist_ok=True)
    b = os.path.join(out_dir, "bench.jsonl")
    only = set(args.only.split(","))
    if "one-user" in only:
        for p in ("w1_1k", "w1_8k", "w1_32k"):
            for blk in range(1, args.blocks + 1):
                bench(args, b, p + ".jsonl", f"w1:{p}:b{blk}", concurrency=1, num_requests=5,
                      offset=(blk - 1) * 5, warmup=2 if blk == 1 else 0)
    if "many-users" in only:
        concs = [int(c) for c in args.concs.split(",")]
        random.Random(args.name).shuffle(concs)
        off = 0
        for c in concs:
            for blk in range(1, args.blocks + 1):
                n = max(args.req_min, args.req_per_conc * c)
                warm = 2 * c if blk == 1 else 0
                s = bench(args, b, "w2.jsonl", f"w2:c{c}:b{blk}", concurrency=c, num_requests=n,
                          min_duration=args.block_seconds, offset=off, warmup=warm)
                off += (s or {}).get("prompts_used", n) + warm
    if "quality" in only:
        subprocess.run([sys.executable, os.path.join(HERE, "quality.py"), "--url", args.url, "--model", args.model,
                        "--data", args.qdata, "--out", os.path.join(out_dir, "quality.jsonl"), "--tag", args.name,
                        "--concurrency", str(args.q_conc)])


if __name__ == "__main__":
    main()
