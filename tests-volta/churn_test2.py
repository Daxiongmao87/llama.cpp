#!/usr/bin/env python3
"""Churn workload v2: 4 growing subagent convos + 1 fixed-size interloper, N rounds.
Unique prompt sizes per request -> metrics map by size, no task-ID correlation.
Sizes (approx tokens): sub r = 12000 (SYSTEM) + 2000 (TAIL) + 500*r filler
                       E   = 11000 (SYSTEM_E) + 1000 (E_TAIL), constant
Usage: churn_test2.py PORT N_ROUNDS [MODEL]"""
import json, sys, time, urllib.request, urllib.error, concurrent.futures

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8086
N_ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
MODEL = sys.argv[3] if len(sys.argv) > 3 else "qwen4b"

def mktext(seed, nchars):
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron ".split()
    out, i = [], 0
    while sum(len(w) + 1 for w in out) < nchars:
        out.append(words[(i + seed * 7) % len(words)])
        i += 1
    return " ".join(out)

def approx_tokens(s):
    return len(s) // 4

SYSTEM = mktext(0, 48000)        # ~12K tokens
SYSTEM_E = mktext(99, 44000)     # ~11K tokens, distinct from SYSTEM
TAIL = {n: mktext(ord(n), 8000) for n in "ABCD"}   # ~2K each
E_TAIL = mktext(123, 4000)       # ~1K

convos = {}
for n in "ABCD":
    convos[n] = {"hist": [{"role": "user", "content": SYSTEM + f"\n\n[WORKER {n}]\n" + TAIL[n]}]}
e_hist = [{"role": "user", "content": SYSTEM_E + "\n\n[MA]\n" + E_TAIL}]

METRICS = []

def fire(tag, messages, filler):
    msgs = messages + [{"role": "user", "content": filler}] if filler else list(messages)
    body = json.dumps({"model": MODEL, "messages": msgs, "max_tokens": 32,
                       "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            j = json.loads(r.read())
        lat = time.time() - t0
        usage = j.get("usage", {})
        pt = usage.get("prompt_tokens")
        ct = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        METRICS.append((tag, pt, ct))
        content = (j.get("choices") or [{}])[0].get("message", {}).get("content", "") or "ok"
        print(f"  {tag}: OK {lat:6.2f}s prompt={pt} cached={ct}", flush=True)
        return content
    except urllib.error.HTTPError as e:
        print(f"  {tag}: HTTP {e.code} {time.time()-t0:5.2f}s body={e.read().decode()[:200]}", flush=True)
        return "ok"
    except Exception as e:
        print(f"  {tag}: ERR {time.time()-t0:5.2f}s {e}", flush=True)
        return "ok"

# warmup
req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
    data=json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                     "max_tokens": 4, "temperature": 0, "stream": False}).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
with urllib.request.urlopen(req, timeout=300) as r:
    r.read()
print("WARMUP-DONE", flush=True)

for rd in range(1, N_ROUNDS + 1):
    filler = mktext(500 + rd, 1200 * rd)   # 300*rd tokens of new content
    def sub(n):
        c = convos[n]
        a = fire(f"r{rd} {n}", c["hist"], filler)
        c["hist"].append({"role": "assistant", "content": a})
        c["hist"].append({"role": "user", "content": filler})
    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        list(ex.map(sub, "ABCD"))
    fire(f"r{rd} E", e_hist, None)

print("\n=== METRICS ===")
zs = cs = ze = ce = 0
for rd in range(1, N_ROUNDS + 1):
    sub_m = [(t, p, c) for t, p, c in METRICS if t.startswith(f"r{rd} ") and not t.endswith(" E")]
    e_m = [(t, p, c) for t, p, c in METRICS if t.endswith(f"r{rd} E")]
    z = sum(p for _, p, _ in sub_m if p); c = sum((cc or 0) for _, _, cc in sub_m)
    ze_r = sum(p for _, p, _ in e_m if p); ce_r = sum((cc or 0) for _, _, cc in e_m)
    zs += z; cs += c; ze += ze_r; ce += ce_r
    print(f"r{rd}: sub prompt={z:>6} cached={c:>6} hit={100*c/z if z else 0:5.1f}% | E prompt={ze_r:>6} cached={ce_r:>6} hit={100*ce_r/ze_r if ze_r else 0:5.1f}%")
print(f"TOTAL: sub prompt={zs:>6} cached={cs:>6} hit={100*cs/zs if zs else 0:5.1f}% | E prompt={ze:>6} cached={ce:>6} hit={100*ce/ze if ze else 0:5.1f}%")
print("TOTAL-DONE", flush=True)
