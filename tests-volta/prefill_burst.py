#!/usr/bin/env python3
"""4x concurrent ~22K-token prefill burst; report per-request time-to-first-token.
Usage: prefill_burst.py PORT [N_PAR] [APPROX_PROMPT_TOKENS]"""
import json, sys, time, urllib.request, concurrent.futures

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8086
N_PAR = int(sys.argv[2]) if len(sys.argv) > 2 else 4
TARGET = int(sys.argv[3]) if len(sys.argv) > 3 else 22000
MODEL = sys.argv[4] if len(sys.argv) > 4 else "ling"
SUFFIX = sys.argv[5] if len(sys.argv) > 5 else ""

# ~4 chars/token on typical prose
base = "The quick brown fox jumps over the lazy dog near the old mill. " * 300
prompt = (base * 4)[: TARGET * 4]
if SUFFIX:
    prompt = SUFFIX + " " + prompt  # change first token -> no prefix-cache reuse

def one(i):
    t0 = time.time()
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt + f"\nReply with exactly: done{i}"}],
        "max_tokens": 16,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if line.startswith(b"data:") and b'"content"' in line and b'"": null' not in line:
                return (i, time.time() - t0)
    return (i, None)

t_start = time.time()
with concurrent.futures.ThreadPoolExecutor(N_PAR) as ex:
    res = list(ex.map(one, range(N_PAR)))
wall = time.time() - t_start
res.sort(key=lambda x: (x[1] is None, x[1]))
for i, ttft in res:
    print(f"  req {i}: TTFT {ttft:.2f}s" if ttft is not None else f"  req {i}: FAIL")
ok = [t for _, t in res if t is not None]
if ok:
    print(f"max TTFT = {max(ok):.2f}s   (staggered-baseline: ~37s+; parallel-target: ~16s)")
