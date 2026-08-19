#!/usr/bin/env python3
"""Sustained decode tps with N concurrent streams (the subagent metric).
Each stream: short prompt, long generation, streamed. Report per-stream tps.
Usage: decode_tps.py PORT [N_PAR] [MAX_TOKENS] [PROMPT_TOKENS] [MODEL] [TAG]"""
import json, sys, time, urllib.request, concurrent.futures, statistics

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8086
N_PAR = int(sys.argv[2]) if len(sys.argv) > 2 else 4
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 200
TARGET = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
MODEL = sys.argv[5] if len(sys.argv) > 5 else "qwen4b"
TAG = sys.argv[6] if len(sys.argv) > 6 else ""

base = "The quick brown fox jumps over the lazy dog near the old mill. " * 60
prompt = (base * 2)[: TARGET * 4]

def one(i):
    t0 = time.time()
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt + f"\nWrite a long detailed essay about river ecosystems, part {i}."}],
        "max_tokens": MAXTOK,
        "stream": True,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    ttf = None
    toks = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            d = line[5:].strip()
            if d == b"[DONE]":
                break
            try:
                j = json.loads(d)
            except Exception:
                continue
            ch = (j.get("choices") or [{}])[0].get("delta") or {}
            if ch.get("content") or ch.get("reasoning_content"):
                if ttf is None:
                    ttf = time.time() - t0
                toks += 1
    t1 = time.time() - t0  # relative, like ttf
    if not toks:
        return (i, 0.0, ttf)
    tps = (toks - 1) / max(1e-9, t1 - ttf)  # first token is "free" (prompt logits)
    return (i, tps, ttf, toks)

t_start = time.time()
with concurrent.futures.ThreadPoolExecutor(N_PAR) as ex:
    res = list(ex.map(one, range(N_PAR)))
wall = time.time() - t_start
ok = [(i, t) for i, t, _, _ in res if t > 0]
for i, tps, ttf, toks in sorted(res, key=lambda x: x[0]):
    print(f"  stream {i}: {tps:6.1f} t/s   (TTFT {ttf:.2f}s, {toks} tok)" if tps else f"  stream {i}: FAIL (ttf={ttf}, toks={toks})")
if ok:
    vals = [t for _, t in ok]
    agg = sum(v for v in vals)
    print(f"  min={min(vals):.1f}  mean={statistics.mean(vals):.1f}  agg={agg:.1f} t/s   wall={wall:.1f}s")
