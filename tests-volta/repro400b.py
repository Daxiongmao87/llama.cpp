#!/usr/bin/env python3
"""Test: does the server 400 on an identical repeated request?"""
import json, sys, time, urllib.request, urllib.error

PORT = 8086
def mktext(seed, nchars):
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron ".split()
    out, i = [], 0
    while sum(len(w) + 1 for w in out) < nchars:
        out.append(words[(i + seed * 7) % len(words)])
        i += 1
    return " ".join(out)

u1 = mktext(0, 48000) + "\n\n[WORKER A]\n" + mktext(65, 8000)
body = json.dumps({"model": "qwen4b", "messages": [{"role": "user", "content": u1}],
                   "max_tokens": 32, "temperature": 0, "stream": False}).encode()

def fire(tag, messages, model="qwen4b"):
    b = json.dumps({"model": model, "messages": messages, "max_tokens": 32,
                    "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=b,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            j = json.loads(r.read())
        print(f"{tag}: OK {time.time()-t0:.1f}s cached={j.get('usage',{}).get('prompt_tokens_details',{}).get('cached_tokens')}")
        return j
    except urllib.error.HTTPError as e:
        print(f"{tag}: HTTP {e.code} {time.time()-t0:.2f}s body={e.read().decode()[:300]}")
        return None

m1 = [{"role": "user", "content": u1}]
a = fire("A1 (u1)", m1)
m2 = m1 + [{"role": "assistant", "content": "ok"}]
b = fire("B1 (u1,a1)", m2)
c = fire("B2 (u1,a1) IDENTICAL RESEND", m2)
d = fire("B3 (u1,a1) 3rd SEND", m2)
e = fire("E1 (unknown model)", [{"role": "user", "content": "hi"}], model="nosuchmodel")
