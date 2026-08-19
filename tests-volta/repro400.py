#!/usr/bin/env python3
"""Repro: fire churn-style requests, capture 400 bodies."""
import json, sys, time, urllib.request, urllib.error

PORT = 8086
def mktext(seed, nchars):
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron ".split()
    out = []
    i = 0
    while sum(len(w) + 1 for w in out) < nchars:
        out.append(words[(i + seed * 7) % len(words)])
        i += 1
    return " ".join(out)

SYSTEM = mktext(0, 48000)

def fire(tag, messages):
    body = json.dumps({"model": "qwen4b", "messages": messages, "max_tokens": 32,
                       "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            j = json.loads(r.read())
        print(f"{tag}: OK {time.time()-t0:.1f}s usage={j.get('usage')}")
        return (j.get("choices") or [{}])[0].get("message", {}).get("content", "") or "ok"
    except urllib.error.HTTPError as e:
        print(f"{tag}: HTTP {e.code} after {time.time()-t0:.2f}s body={e.read().decode()[:300]}")
        return "ok"
    except Exception as e:
        print(f"{tag}: ERR {e}")
        return "ok"

u1 = SYSTEM + "\n\n[WORKER A]\n" + mktext(65, 8000)
hist = [{"role": "user", "content": u1}]
for i in range(4):
    a = fire(f"A-turn{i+1}", hist)
    hist.append({"role": "assistant", "content": a})
    hist.append({"role": "user", "content": "Continue with step " + str(i + 1) + ". " + mktext(100 + i, 600)})
    print(f"   history now: {len(hist)} msgs, ~{len(hist[-1]['content'])} chars last")
