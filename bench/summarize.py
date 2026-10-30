"""Turn results/raw/<run>/ into the tables in results/summary/.

  python summarize.py [RAW_DIR] [OUT_DIR]

Speeds are computed from the per-request records, not from the rounded block summaries:
  first token = t_first - t_dispatch; decode = (output tokens - 1) / (t_end - t_first); answer = t_end - t_dispatch.
Each block's median is taken over its requests; each table cell is the median over blocks, with the block min and max.

one_user.csv    per run and prompt length: first token, decode speed, time to the whole answer, prompt speed
many_users.csv  per run and concurrency: total output tokens/s (block wall time), per-user decode, first token p50/p95
quality.csv     per run with a quality file: accuracy with 95% Wilson interval, answers cut off at the cap
(memory.csv and quality_pairs.csv are kept as written: peaks from the nvidia-smi and RAM traces, paired McNemar tests.)
"""
import csv
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def med(xs):
    xs = [x for x in xs if x is not None]
    return (round(statistics.median(xs), 3), round(min(xs), 3), round(max(xs), 3)) if xs else (None, None, None)


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def per_request(reqs):
    ok = [r for r in reqs if not r["error"] and r["t_first"] is not None and r["usage"]]
    ttft = [r["t_first"] - r["t_dispatch"] for r in ok]
    dec = [(r["usage"]["completion_tokens"] - 1) / (r["t_end"] - r["t_first"]) for r in ok
           if r["usage"]["completion_tokens"] > 1 and r["t_end"] > r["t_first"]]
    ans = [r["t_end"] - r["t_dispatch"] for r in ok]
    ptok = [r["usage"]["prompt_tokens"] for r in ok]
    out = sum(r["usage"]["completion_tokens"] for r in ok)
    return ok, ttft, dec, ans, ptok, out


def main():
    raw = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "raw")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "results", "summary")
    os.makedirs(dst, exist_ok=True)
    one, many, qual = [], [], []
    for d in sorted(glob.glob(os.path.join(raw, "*"))):
        run = os.path.basename(d)
        b = os.path.join(d, "bench.jsonl")
        if os.path.exists(b):
            groups = {}
            for line in open(b, encoding="utf-8"):
                r = json.loads(line)
                groups.setdefault(r["summary"]["tag"].rsplit(":b", 1)[0], []).append(r)
            for key, blocks in groups.items():
                kind = key.split(":")[0]
                stats = [per_request(r["requests"]) for r in blocks]
                errors = sum(len(r["requests"]) - len(s[0]) for r, s in zip(blocks, stats))
                if kind == "w1":
                    t = med([statistics.median(s[1]) for s in stats])
                    v = med([statistics.median(s[2]) for s in stats])
                    a = med([statistics.median(s[3]) for s in stats])
                    p = statistics.median([x for s in stats for x in s[4]])
                    one.append({"run": run, "prompt": key.split(":")[1].replace("w1_", ""), "blocks": len(blocks),
                                "requests": sum(len(s[0]) for s in stats), "errors": errors, "prompt_tokens": p,
                                "first_token_s": t[0], "first_token_min": t[1], "first_token_max": t[2],
                                "decode_tps": round(v[0], 1), "decode_min": round(v[1], 1), "decode_max": round(v[2], 1),
                                "answer_s": a[0], "prompt_tps": round(p / t[0])})
                elif kind == "w2":
                    tot = med([s[5] / r["summary"]["wall_s"] for r, s in zip(blocks, stats)])
                    pu = med([statistics.median(s[2]) for s in stats])
                    many.append({"run": run, "users": int(key.split(":c")[1]), "blocks": len(blocks),
                                 "requests": sum(len(s[0]) for s in stats), "errors": errors,
                                 "total_tps": round(tot[0], 1), "total_min": round(tot[1], 1), "total_max": round(tot[2], 1),
                                 "per_user_tps": round(pu[0], 1),
                                 "first_token_p50_s": med([pct(s[1], 0.5) for s in stats])[0],
                                 "first_token_p95_s": med([pct(s[1], 0.95) for s in stats])[0]})
        q, ql = os.path.join(d, "quality.json"), os.path.join(d, "quality.jsonl")
        if os.path.exists(q) or os.path.exists(ql):
            # quality.json = the trimmed copy kept in this repo; quality.jsonl = what quality.py writes (last line wins)
            s = json.load(open(q))["summary"] if os.path.exists(q) else json.loads(open(ql).readlines()[-1])["summary"]
            for task in ("gsm8k", "mmlu_pro"):
                k = s[task]
                qual.append({"run": run, "task": task, "n": k["n"], "correct": k["correct"], "accuracy": k["acc"],
                             "ci95_low": k["ci95"][0], "ci95_high": k["ci95"][1], "cut_off_at_cap": k["hit_cap"]})
    for name, rows in (("one_user.csv", one), ("many_users.csv", sorted(many, key=lambda r: (r["run"], r["users"]))),
                       ("quality.csv", qual)):
        with open(os.path.join(dst, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(name, len(rows))


if __name__ == "__main__":
    main()
