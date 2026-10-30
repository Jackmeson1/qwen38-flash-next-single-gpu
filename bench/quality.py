"""Answer-quality checks against a running OpenAI-compatible server: GSM8K and MMLU-Pro samples, thinking off, greedy.

  python quality.py --prepare qdata
  python quality.py --url http://127.0.0.1:8000 --model qwen --data qdata --out quality.jsonl --tag vllm [--concurrency 8]

Data files (made by --prepare): gsm8k.jsonl (250, seed 0) and mmlu_pro.jsonl (300, seed 0).
"""
import argparse
import asyncio
import json
import math
import os
import random
import re

import aiohttp

GSM8K = "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet"
MMLU_PRO = "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"
LETTERS = "ABCDEFGHIJ"


def prepare(out):
    import pandas as pd
    os.makedirs(out, exist_ok=True)
    g = pd.read_parquet(GSM8K)
    rows = random.Random(0).sample(range(len(g)), 250)
    with open(os.path.join(out, "gsm8k.jsonl"), "w", encoding="utf-8") as fh:
        for i in rows:
            ans = g.answer[i].split("####")[-1].strip().replace(",", "")
            prompt = (g.question[i] + "\n\nSolve the problem step by step. End with a line of the form "
                      "'Answer: <number>'.")
            fh.write(json.dumps({"id": f"gsm8k-{i}", "prompt": prompt, "gold": ans, "kind": "gsm8k"}) + "\n")
    m = pd.read_parquet(MMLU_PRO)
    rows = random.Random(0).sample(range(len(m)), 300)
    with open(os.path.join(out, "mmlu_pro.jsonl"), "w", encoding="utf-8") as fh:
        for i in rows:
            opts = list(m.options[i])
            body = "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(opts))
            prompt = (f"{m.question[i]}\n\n{body}\n\nThink it through briefly, then end with a line of the form "
                      "'Answer: <letter>'.")
            fh.write(json.dumps({"id": f"mmlu-{m.question_id[i]}", "prompt": prompt, "gold": m.answer[i],
                                 "kind": "mmlu_pro", "category": m.category[i]}) + "\n")


def extract(kind, text):
    tail = text.strip()
    if kind == "gsm8k":
        m = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", tail)
        nums = m or re.findall(r"-?[\d,]*\.?\d+", tail)
        if not nums:
            return None
        v = nums[-1].replace(",", "")
        try:
            f = float(v)
            return str(int(f)) if f == int(f) else str(f)
        except ValueError:
            return v
    m = re.findall(r"Answer:\s*\(?([A-J])\b", tail)
    return m[-1] if m else None


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - r) / d, 4), round((c + r) / d, 4))


async def ask(session, args, item):
    body = {"model": args.model, "messages": [{"role": "user", "content": item["prompt"]}],
            "max_tokens": args.max_tokens, "temperature": 0.0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(3):
        try:
            async with session.post(args.url + "/v1/chat/completions", json=body) as r:
                j = await r.json(content_type=None)
                msg = j["choices"][0]["message"]
                return msg.get("content") or "", j["choices"][0].get("finish_reason"), j.get("usage")
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
    return None, "error: " + err, None


async def run(args, items):
    sem = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as session:
        async def go(it):
            async with sem:
                text, fin, usage = await ask(session, args, it)
                pred = extract(it["kind"], text or "")
                return {**it, "text": text, "finish_reason": fin, "usage": usage, "pred": pred,
                        "correct": pred is not None and pred == it["gold"]}
        return await asyncio.gather(*(go(it) for it in items))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", help="write the data files to this folder and exit")
    ap.add_argument("--url")
    ap.add_argument("--model")
    ap.add_argument("--data")
    ap.add_argument("--out")
    ap.add_argument("--tag", default="")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()
    if args.prepare:
        prepare(args.prepare)
        return
    items = []
    for f in ("gsm8k.jsonl", "mmlu_pro.jsonl"):
        with open(os.path.join(args.data, f), encoding="utf-8") as fh:
            items += [json.loads(line) for line in fh]
    res = asyncio.run(run(args, items))
    summary = {"tag": args.tag}
    for kind in ("gsm8k", "mmlu_pro"):
        rs = [r for r in res if r["kind"] == kind]
        k = sum(r["correct"] for r in rs)
        summary[kind] = {"n": len(rs), "correct": k, "acc": round(k / len(rs), 4), "ci95": wilson(k, len(rs)),
                         "no_answer": sum(r["pred"] is None for r in rs),
                         "hit_cap": sum(r["finish_reason"] == "length" for r in rs)}
    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"summary": summary, "items": res}) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
