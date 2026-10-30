"""Streaming load client for any OpenAI-compatible /v1/chat/completions server.

One client for every engine, so the timing code is the same everywhere. Per request it records dispatch time, the
first non-empty delta (answer or reasoning text), every later delta time, the end time and the server's usage.

Closed loop (fixed concurrency), the default:
  python bench_client.py --url URL --model M --prompts p.jsonl --concurrency 8 --num-requests 80 --min-duration 90 \
      --warmup 8 --out run.jsonl --tag w2:c8:b1
Open loop (Poisson arrivals, latency counted from the scheduled arrival):
  ... --rate 4.0 --num-requests 200
Interference (one long request injected while the others decode):
  ... --inject long.jsonl:20   (the first prompt of long.jsonl is sent 20 s after the start)

Prompt file: JSONL with {"id", "messages", "max_tokens"}; prompts are used in file order from --offset, each once.
Sampling is explicit (greedy, no penalties) so no engine applies its own defaults.
"""
import argparse
import asyncio
import json
import random
import statistics
import time

import aiohttp

GREEDY = {"temperature": 0.0, "top_p": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
          "repetition_penalty": 1.0}


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 4)


async def one(session, args, prompt, t0, scheduled=None):
    body = {"model": args.model, "messages": prompt["messages"], "max_tokens": prompt.get("max_tokens", args.max_tokens),
            "stream": True, "stream_options": {"include_usage": True}, **args.sampling}
    body.update(args.extra)
    rec = {"id": prompt["id"], "t_sched": scheduled, "t_dispatch": time.perf_counter() - t0, "t_first": None,
           "deltas": [], "usage": None, "finish_reason": None, "text": "", "reasoning": "", "error": None,
           "first_kind": None}
    try:
        async with session.post(args.url + "/v1/chat/completions", json=body) as resp:
            if resp.status != 200:
                rec["error"] = f"HTTP {resp.status}: {(await resp.text())[:500]}"
                rec["t_end"] = time.perf_counter() - t0
                return rec
            buf = b""
            async for chunk in resp.content.iter_any():
                now = time.perf_counter() - t0
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        continue
                    ev = json.loads(data)
                    if ev.get("usage"):
                        rec["usage"] = ev["usage"]
                    for ch in ev.get("choices") or []:
                        d = ch.get("delta") or {}
                        content = d.get("content") or ""
                        reasoning = d.get("reasoning_content") or d.get("reasoning") or ""
                        if content or reasoning:
                            if rec["t_first"] is None:
                                rec["t_first"] = now
                                rec["first_kind"] = "content" if content else "reasoning"
                            rec["deltas"].append(round(now, 5))
                            rec["text"] += content
                            rec["reasoning"] += reasoning
                        if ch.get("finish_reason"):
                            rec["finish_reason"] = ch["finish_reason"]
    except Exception as e:  # noqa: BLE001 - a failed request is a result; no silent retry
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["t_end"] = time.perf_counter() - t0
    return rec


async def closed_loop(session, args, prompts, n, concurrency, t0, min_duration=0.0):
    it = iter(prompts)
    out = []
    state = {"started": 0}

    def next_prompt():
        elapsed = time.perf_counter() - t0
        if state["started"] >= n and elapsed >= min_duration:
            return None
        p = next(it, None)
        if p is not None:
            state["started"] += 1
        return p

    async def worker():
        while True:
            p = next_prompt()
            if p is None:
                return
            out.append(await one(session, args, p, t0))

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return out


async def open_loop(session, args, prompts, n, rate, t0):
    rng = random.Random(args.seed)
    t, tasks = 0.0, []
    for p in prompts[:n]:
        t += rng.expovariate(rate)
        delay = t - (time.perf_counter() - t0)
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(one(session, args, p, t0, scheduled=t)))
    return list(await asyncio.gather(*tasks))


async def main_async(args, measured, spare, inject):
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    # --force-close: a new connection per request (llama-server closes the connection after each streamed answer,
    # so a pooled connection fails on the next request with "Server disconnected").
    conn = aiohttp.TCPConnector(limit=0, force_close=args.force_close)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        if args.warmup:
            t0 = time.perf_counter()
            await closed_loop(session, args, spare, args.warmup, max(1, min(args.concurrency, args.warmup)), t0)
        t0 = time.perf_counter()
        extra = []
        if inject:
            async def late():
                await asyncio.sleep(inject[1])
                r = await one(session, args, inject[0], t0, scheduled=inject[1])
                r["injected"] = True
                extra.append(r)
            inj = asyncio.create_task(late())
        if args.rate:
            recs = await open_loop(session, args, measured, args.num_requests, args.rate, t0)
        else:
            recs = await closed_loop(session, args, measured, args.num_requests, args.concurrency, t0,
                                     args.min_duration)
        if inject:
            await inj
        return recs, extra


def summarize(recs, args):
    ok = [r for r in recs if not r["error"] and r["t_first"] is not None]
    wall = max(r["t_end"] for r in recs) - min(r["t_dispatch"] for r in recs)
    outs = [r["usage"]["completion_tokens"] for r in ok if r["usage"]]
    base = (lambda r: r["t_sched"]) if args.rate else (lambda r: r["t_dispatch"])
    ttft = [r["t_first"] - base(r) for r in ok]
    e2e = [r["t_end"] - base(r) for r in ok]
    tpot = []
    for r in ok:
        n = (r["usage"] or {}).get("completion_tokens", 0)
        if n > 1 and r["t_end"] > r["t_first"]:
            tpot.append((r["t_end"] - r["t_first"]) / (n - 1))
    gaps = [b - a for r in ok for a, b in zip(r["deltas"], r["deltas"][1:])]
    fins = {}
    for r in ok:
        fins[r["finish_reason"]] = fins.get(r["finish_reason"], 0) + 1
    return {
        "mode": "open" if args.rate else "closed", "rate": args.rate, "concurrency": args.concurrency,
        "requests": len(recs), "ok": len(ok), "errors": len(recs) - len(ok),
        "missing_usage": sum(1 for r in ok if not r["usage"]), "finish": fins, "wall_s": round(wall, 3),
        "output_tokens": sum(outs), "total_output_tps": round(sum(outs) / wall, 2) if wall else None,
        "out_len_p10": pct(outs, 0.1), "out_len_p50": pct(outs, 0.5), "out_len_p90": pct(outs, 0.9),
        "prompt_tokens_p50": pct([r["usage"]["prompt_tokens"] for r in ok if r["usage"]], 0.5),
        "ttft_p50_s": pct(ttft, 0.5), "ttft_p95_s": pct(ttft, 0.95), "e2e_p50_s": pct(e2e, 0.5),
        "e2e_p95_s": pct(e2e, 0.95), "tpot_p50_ms": pct([x * 1000 for x in tpot], 0.5),
        "tpot_p95_ms": pct([x * 1000 for x in tpot], 0.95),
        "per_user_tps_p50": round(1 / pct(tpot, 0.5), 2) if tpot else None,
        "delta_gap_max_s": round(max(gaps), 4) if gaps else None, "first_kinds": sorted({r["first_kind"] for r in ok}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--num-requests", type=int, default=16)
    ap.add_argument("--min-duration", type=float, default=0.0, help="closed loop: keep sending until this many seconds")
    ap.add_argument("--rate", type=float, default=0.0, help="open loop: Poisson arrivals per second")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=0, help="requests sent first and discarded (from the spare prompts)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--sampling", type=json.loads, default=GREEDY)
    ap.add_argument("--extra", type=json.loads, default={})
    ap.add_argument("--inject", default="", help="PROMPTFILE:SECONDS, one extra request sent late (interference test)")
    ap.add_argument("--request-timeout", type=float, default=1800)
    ap.add_argument("--force-close", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    with open(args.prompts, encoding="utf-8") as fh:
        prompts = [json.loads(line) for line in fh if line.strip()][args.offset:]
    # Measured requests take prompts from the front; warm-up takes them from the far end, so none repeats.
    if len(prompts) < args.num_requests + args.warmup:
        raise SystemExit("not enough prompts for the measured requests plus warm-up")
    spare = prompts[len(prompts) - args.warmup:] if args.warmup else []
    measured = prompts[:len(prompts) - args.warmup]
    inject = None
    if args.inject:
        path, sec = args.inject.rsplit(":", 1)
        with open(path, encoding="utf-8") as fh:
            inject = (json.loads(fh.readline()), float(sec))
    recs, extra = asyncio.run(main_async(args, measured, spare, inject))
    summary = summarize(recs, args)
    if extra:
        x = extra[0]
        summary["injected"] = {"ttft_s": round(x["t_first"] - x["t_sched"], 3) if x["t_first"] else None,
                               "prompt_tokens": (x["usage"] or {}).get("prompt_tokens"), "error": x["error"]}
        during = [b - a for r in recs if not r["error"] for a, b in zip(r["deltas"], r["deltas"][1:])
                  if x["t_dispatch"] <= b <= (x["t_first"] or x["t_end"])]
        summary["injected"]["others_max_gap_during_prefill_s"] = round(max(during), 4) if during else None
    used = len({r["id"] for r in recs})
    summary.update({"tag": args.tag, "prompts": args.prompts, "offset": args.offset, "prompts_used": used,
                    "sampling": args.sampling, "extra": args.extra, "unix_time": time.time()})
    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"summary": summary, "requests": recs + extra}) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
