#!/usr/bin/env python3
"""Subagent-churn workload: 4 subagent conversations + 1 interloper, N rounds, 4 slots.
Each conversation resends full history per turn (like OMP). Shared 12K-token system
prefix across all 5 -> measures prefix-reuse vs re-prefill under slot churn.
Usage: churn_test.py PORT N_ROUNDS [MODEL]"""
import json, sys, time, urllib.request, concurrent.futures

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8086
N_ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
MODEL = sys.argv[3] if len(sys.argv) > 3 else "qwen4b"

def mktext(seed, nchars):
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron ".split()
    out = []
    i = 0
    while sum(len(w) + 1 for w in out) < nchars:
        out.append(words[(i + seed * 7) % len(words)])
        i += 1
    return " ".join(out)

SYSTEM = mktext(0, 48000)          # ~12K tokens, shared by all conversations
TURNS = 8
convos = {}
for name in "ABCDE":
    tail = mktext(ord(name), 8000)  # ~2K tokens, unique per conversation
    convos[name] = {
        "name": name,
        "history": [{"role": "user", "content": SYSTEM + f"\n\n[WORKER {name}]\n" + tail}],
        "turn": 1,
    }

def one(name):
    c = convos[name]
    t0 = time.time()
    body = json.dumps({
        "model": MODEL,
        "messages": c["history"],
        "max_tokens": 32,
        "temperature": 0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            j = json.loads(r.read())
        c["history"].append({"role": "assistant",
                             "content": (j.get("choices") or [{}])[0].get("message", {}).get("content", "ok")})
        usage = j.get("usage", {})
        return (name, c["turn"], time.time() - t0, usage.get("prompt_tokens"), 0)
    except Exception as e:
        return (name, c["turn"], time.time() - t0, None, str(e)[:80])

# warmup (task 0): tiny request so round task-ids start at 1
wu = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                 "max_tokens": 4, "temperature": 0, "stream": False}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=wu,
                             headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
with urllib.request.urlopen(req, timeout=300) as r:
    r.read()
print("WARMUP-DONE", flush=True)

report = []
for rd in range(N_ROUNDS):
    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, "ABCD"))
    report.extend(res)
    e = one("E")
    report.append(e)
    for name, turn, lat, ptok, err in res + [e]:
        print(f"  r{rd+1} {name} t{turn}: {lat:7.2f}s prompt={ptok} err={err}", flush=True)

print(f"TOTAL-DONE {len(report)} requests", flush=True)
