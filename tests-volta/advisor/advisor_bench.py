#!/usr/bin/env python3
"""Advisor-role benchmark.

Uses the REAL advisor system prompt and tool schemas captured from an omp
request (advisor_fixture.json), so only the agent transcript is synthetic.

Scores the two things the role actually needs:
  restraint  -- stays silent when the spec says silent (precision)
  detection  -- fires when there is concrete technical risk (recall)
plus schema validity and, with --pad, behaviour at depth.

--pad N       pad the transcript to ~N tokens of filler agent turns
--needle P    where the real scenario sits in the padding: head|middle|tail
              (tail = most recent, the easiest; middle = classic long-ctx blind spot)

Usage: advisor_bench.py PORT MODEL [--pad 100000] [--needle middle] [--runs 1]
"""
import json, os, random, re, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from advisor_scenarios import SCENARIOS, CALIBRATION, STUBS  # noqa: E402

FIXTURE = json.load(open(os.path.join(HERE, "advisor_fixture.json")))
SEVERITIES = {"nit", "concern", "blocker"}


def filler(nchunks, seed):
    """Benign agent activity. Must contain nothing advisable, or it pollutes
    the SILENT cases -- so it is deliberately dull, correct, routine work."""
    r = random.Random(seed)
    verbs = ["Read", "Listed", "Searched", "Opened", "Checked"]
    files = ["utils/format.py", "api/routes.py", "db/models.py", "web/app.tsx",
             "tests/test_api.py", "scripts/deploy.sh", "lib/parse.go"]
    out = []
    for i in range(nchunks):
        f = r.choice(files)
        out.append({"role": "assistant", "content": [{"type": "text", "text":
            f"Thought: getting oriented before the change; nothing surprising so far.\n"
            f"{r.choice(verbs)} {f} ({r.randint(40,600)} lines). Imports and call sites "
            f"line up with what the task description said. Continuing to survey."}]})
        out.append({"role": "user", "content": [{"type": "text", "text":
            f"<system-reminder>step {i}: survey continues</system-reminder>"}]})
    return out


def _as_watched(msgs):
    """omp shape: the WATCHED agent's transcript arrives as `user` turns, with
    its thinking prefixed `_thinking:_`. `assistant`/`tool` roles belong to the
    advisor itself. Getting this backwards makes the model read the agent's work
    as its own prior output -- and stay correctly silent about it."""
    out = [{"role": "user", "content": [{"type": "text", "text":
        "<system-reminder>\nToday: 2026-08-19; current working directory: "
        "'/home/agent/projects/app'. Do not repeat this information.\n"
        "</system-reminder>"}]}]
    for m in msgs:
        txt = " ".join(p["text"] for p in m["content"])
        if m["role"] == "user":
            out.append({"role": "user", "content": [{"type": "text",
                        "text": f"[user] {txt}"}]})
        else:
            body = txt.replace("Thought: ", "_thinking:_ ", 1)
            if not body.startswith("_thinking:_"):
                body = "_thinking:_ " + body
            out.append({"role": "user", "content": [{"type": "text", "text": body}]})
    return out


def build(scenario, pad_tokens, needle):
    msgs = []
    if pad_tokens:
        # ~55 tokens per filler pair, measured empirically on this vocabulary
        n = max(0, pad_tokens // 55)
        pad = _as_watched(filler(n, seed=hash(scenario["name"]) & 0xffff))
        sc = _as_watched(scenario["msgs"])
        if needle == "head":
            msgs = sc + pad
        elif needle == "middle":
            h = len(pad) // 2
            msgs = pad[:h] + sc + pad[h:]
        else:  # tail -- scenario is the most recent turn
            msgs = pad + sc
    else:
        msgs = _as_watched(scenario["msgs"])
    return [{"role": "system", "content": FIXTURE["system"]}] + msgs


def ask(port, model, messages, timeout=1800):
    body = json.dumps({
        "model": model, "messages": messages, "tools": FIXTURE["tools"],
        "temperature": 0.0, "max_completion_tokens": 32768, "stream": False,
    }).encode()
    hdr = {"Content-Type": "application/json"}
    key = os.environ.get("LLAMA_API_KEY")
    if key:
        hdr["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers=hdr)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        j = json.loads(r.read())
    return j, time.time() - t0


def _req(url, data=None):
    hdr = {"Content-Type": "application/json"}
    key = os.environ.get("LLAMA_API_KEY")
    if key:
        hdr["Authorization"] = f"Bearer {key}"
    return urllib.request.Request(url, data=data, headers=hdr)


NOOP_RE = re.compile(r"nothing to (flag|advise|report)|no (issues?|concerns?|problems?)"
                     r"|looks? (fine|good|correct)|no advice|all good|no action needed", re.I)

TOOL_STUB = {
    "read":  "1  def round_half_up(x, places=2):\n2      # uses binary float; 2.675 is stored as 2.67499999...\n3      return float(Decimal(x).quantize(Decimal('0.01'), ROUND_HALF_UP))\n",
    "grep":  "batch.py:41:    if rid not in seen: seen.append(rid)\nbatch.py:38:    seen = []\n",
    "glob":  "build/\nbuild/artifacts/\nsrc/\ntests/\n",
}


def run_loop(port, model, messages, max_hops=4, scen=None):
    """Drive the tool loop like omp does. Without this, a model that reaches for
    read/grep/glob to verify never gets to advise, and scores as a false silence."""
    msgs = list(messages)
    total = 0.0
    used_tools = []
    for _ in range(max_hops):
        j, dt = ask(port, model, msgs)
        total += dt
        m = j["choices"][0]["message"]
        calls = m.get("tool_calls") or []
        verify = [c for c in calls
                  if c.get("function", {}).get("name") in TOOL_STUB]
        if not verify or any(c.get("function", {}).get("name") == "advise" for c in calls):
            return j, total, used_tools
        msgs.append({"role": "assistant", "tool_calls": calls,
                     "content": m.get("content") or ""})
        for c in verify:
            name = c["function"]["name"]
            used_tools.append(name)
            per = STUBS.get((scen or {}).get("name"), {})
            msgs.append({"role": "tool", "tool_call_id": c.get("id", name),
                         "content": per.get(name, TOOL_STUB[name])})
    return j, total, used_tools


def grade(j):
    """-> (fired, severity, schema_ok, note, other_tools, truncated)"""
    ch = j["choices"][0]
    msg = ch["message"]
    # a run that hit the token ceiling emitted no call for lack of room, NOT
    # out of restraint -- it must never be scored as a correct silence.
    trunc = ch.get("finish_reason") == "length"
    calls = msg.get("tool_calls") or []
    advise = [c for c in calls if c.get("function", {}).get("name") == "advise"]
    other = [c["function"]["name"] for c in calls
             if c.get("function", {}).get("name") != "advise"]
    if not advise:
        # some models emit the call as text instead of a tool_call
        txt = (msg.get("content") or "")
        if re.search(r'"?advise"?\s*[\(:]', txt):
            return True, None, False, txt[:80], other, trunc
        return False, None, True, "", other, trunc
    try:
        args = json.loads(advise[0]["function"]["arguments"])
    except Exception:
        return True, None, False, "", other, trunc
    sev = args.get("severity")
    ok = isinstance(args.get("note"), str) and bool(args.get("note")) and \
        (sev is None or sev in SEVERITIES)
    note = str(args.get("note", ""))
    if NOOP_RE.search(note):
        # semantically silence, but still delivered to the agent -> counts as
        # firing, tracked separately as the milder "no-op advice" failure.
        return True, sev, ok, "[NO-OP] " + note[:60], other, trunc
    return True, sev, ok, note[:70], other, trunc


def preflight(port, model, pad):
    """Fail loudly BEFORE burning a run. Each check exists because its absence
    already produced a plausible wrong number instead of an error."""
    try:
        with urllib.request.urlopen(_req(f"http://127.0.0.1:{port}/v1/models"), timeout=10) as r:
            served = [m["id"] for m in json.loads(r.read()).get("data", [])]
    except Exception as e:
        sys.exit(f"PREFLIGHT FAIL: no server on :{port} ({type(e).__name__} {str(e)[:80]})")
    if served and model not in served:
        sys.exit(f"PREFLIGHT FAIL: {model!r} not served on :{port}. have: {served}")
    probe = build(SCENARIOS[0], pad, "tail")
    try:
        j, _ = ask(port, model, probe, timeout=1800)
    except Exception as e:
        sys.exit(f"PREFLIGHT FAIL: probe rejected: {str(e)[:160]}")
    used = j.get("usage", {}).get("prompt_tokens", 0)
    fin = j["choices"][0].get("finish_reason")
    print(f"  preflight ok: {model} on :{port}, probe prompt={used} tok, finish={fin}", flush=True)
    if fin == "length":
        sys.exit("PREFLIGHT FAIL: probe hit the completion ceiling; scores would "
                 "count truncation as silence.")
    if pad and used < pad * 0.6:
        sys.exit(f"PREFLIGHT FAIL: asked ~{pad} tok padding, prompt was {used}. "
                 f"Filler sizing wrong; runs would not test depth.")


def main():
    port, model = sys.argv[1], sys.argv[2]
    pad = int(_arg("--pad", 0))
    needle = _arg("--needle", "tail")
    runs = int(_arg("--runs", 1))
    preflight(port, model, pad)
    # calibration gate: prove the harness can see a fire and a silence at all
    print("  -- calibration (harness sanity, not scored) --", flush=True)
    calfail = []
    for c in CALIBRATION:
        try:
            j, dt, hops = run_loop(port, model, build(c, 0, "tail"), scen=c)
            fired, sev, ok, note, other, trunc = grade(j)
        except Exception as e:
            calfail.append(f"{c['name']}: request failed {type(e).__name__}"); continue
        want_fire = c["expect"] is not None
        good = fired == want_fire
        print(f"     {'ok ' if good else 'FAIL'} {c['name']:20s} "
              f"want={'fire' if want_fire else 'silent':6s} got={'advise' if fired else 'silent'} {dt:5.1f}s")
        if not good and c["expect"] is not None:
            calfail.append(c["name"])          # only a missed FIRE means blind harness
        elif not good:
            print(f"        ^ model over-fires on a greeting; recorded, not a harness fault")
    if calfail:
        sys.exit(f"  CALIBRATION FAILED: {calfail}\n"
                 f"  Harness could not observe an unmissable FIRE -- it is blind. "
                 f"Scores would be meaningless. Fix the harness before scoring models.")
    print("  -- calibration passed, scoring --\n", flush=True)
    rows = []
    for s in SCENARIOS:
        for run in range(runs):
            msgs = build(s, pad, needle)
            try:
                j, dt, hops = run_loop(port, model, msgs, scen=s)
            except Exception as e:
                print(f"  {s['name']:26s} REQUEST FAILED {type(e).__name__}: {str(e)[:60]}")
                rows.append((s, False, None, False, 0.0, [], True)); continue
            fired, sev, ok, note, other, trunc = grade(j)
            other = list(dict.fromkeys(other + hops))
            ntok = j.get("usage", {}).get("prompt_tokens", 0)
            want = s["expect"]
            correct = (fired and want is not None) or (not fired and want is None)
            mark = "TRUNC" if trunc else ("ok " if correct else "MISS")
            print(f"  {mark} {s['name']:26s} want={str(want or 'SILENT'):7s} "
                  f"got={'advise:'+str(sev) if fired else 'silent':15s} "
                  f"schema={'y' if ok else 'N'} ctx={ntok:>6} {dt:5.1f}s"
                  f"{'  tools=' + ','.join(other) if other else ''}")
            if fired and note:
                print(f"       note: {note!r}")
            rows.append((s, fired, sev, ok, dt, other, trunc))
    _summary(model, rows, pad, needle)


def _summary(model, rows, pad, needle):
    ntrunc = sum(1 for r in rows if r[6])
    valid = [r for r in rows if not r[6]]
    fire_rows = [r for r in valid if r[0]["expect"]]
    sil_rows = [r for r in valid if not r[0]["expect"]]
    recall = sum(1 for r in fire_rows if r[1]) / max(len(fire_rows), 1)
    restraint = sum(1 for r in sil_rows if not r[1]) / max(len(sil_rows), 1)
    sev_ok = sum(1 for r in fire_rows if r[1] and r[2] == r[0]["expect"])
    schema = sum(1 for r in valid if r[3]) / max(len(valid), 1)
    lat = sum(r[4] for r in rows) / max(len(rows), 1)
    f1 = 0.0
    if recall + restraint:
        f1 = 2 * recall * restraint / (recall + restraint)
    print(f"\n  {model}  pad={pad} needle={needle}")
    print(f"    restraint (silent when it should be): {restraint*100:5.1f}%  "
          f"({sum(1 for r in sil_rows if not r[1])}/{len(sil_rows)})")
    print(f"    detection (fired when it should):     {recall*100:5.1f}%  "
          f"({sum(1 for r in fire_rows if r[1])}/{len(fire_rows)})")
    print(f"    severity exact:                       {sev_ok}/{len(fire_rows)}")
    print(f"    schema valid:                         {schema*100:5.1f}%")
    print(f"    balanced score (harmonic):            {f1*100:5.1f}%")
    print(f"    mean latency:                         {lat:5.1f}s")
    noop = sum(1 for r in valid if r[1] and isinstance(r[3], bool) is False)
    print(f"    (no-op advice is marked [NO-OP] per row above)")
    if ntrunc:
        print(f"    TRUNCATED (excluded, not scored):     {ntrunc}/{len(rows)}"
              f"  <- hit the 32768 ceiling; not restraint")


def _arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


if __name__ == "__main__":
    main()
