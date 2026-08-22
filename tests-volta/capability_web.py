#!/usr/bin/env python3
"""Capability benchmark: one fixed webpage build task, objectively graded.

Answers "how much more capable is qwen4b than ling", not "how fast is it" --
the question the TPS suite and subagent_fit.py do NOT answer.

The task is specified tightly enough that every requirement maps to a
machine-checkable assertion, so the score is not a vibe. Artifacts are kept on
disk for human/model review of the parts a regex cannot judge (layout sense,
naming, whether it actually looks like a dashboard).

Usage: capability_web.py PORT ALIAS OUTDIR [RUNS]
"""
import json, os, re, subprocess, sys, time
import urllib.request

TASK = """Build a single self-contained HTML file: a server status dashboard.

Requirements:
1. A header with the title "Fleet Status" and a live clock that updates every second.
2. A hardcoded JavaScript array of exactly 4 server objects, each with: name, status ("online" or "offline"), and cpu (a number 0-100). At least one must be offline.
3. A table or card list rendering all 4 servers from that array.
4. Each server row shows a progress bar whose filled width is set from its cpu value.
5. Three filter buttons - "All", "Online", "Offline" - that actually re-render the list to show only matching servers.
6. A "Toggle theme" button that switches the page between light and dark by toggling a class on the body or root element.
7. Everything inline in one file: no external stylesheets, scripts, fonts, or images. No CDN links.

Output the complete HTML file in a single ```html code block. No explanation before or after."""

CHECKS = [
    # (name, weight, predicate over (html, js))
    ("html_extracted",  1, lambda h, j: len(h) > 200),
    ("doctype_and_html",1, lambda h, j: re.search(r"<!DOCTYPE\s+html", h, re.I) and "</html>" in h.lower()),
    ("has_style",       1, lambda h, j: bool(re.search(r"<style[\s>]", h, re.I))),
    ("has_script",      1, lambda h, j: len(j) > 50),
    ("self_contained",  1, lambda h, j: not re.search(r'(src|href)\s*=\s*["\']https?://', h, re.I)),
    ("title_fleet",     1, lambda h, j: "fleet status" in h.lower()),
    ("live_clock",      1, lambda h, j: bool(re.search(r"setInterval\s*\(", j))),
    ("four_servers",    2, lambda h, j: _server_array_ok(j)),
    ("progress_width",  2, lambda h, j: bool(re.search(r"(width|--\w+)\s*[:=][^;\n]{0,40}(cpu|\$\{[^}]*cpu)", j + h, re.I))),
    ("three_filters",   2, lambda h, j: _filters_ok(h, j)),
    ("filters_wired",   2, lambda h, j: bool(re.search(r"addEventListener\s*\(\s*['\"]click|onclick\s*=", h + j, re.I))),
    ("theme_toggle",    2, lambda h, j: bool(re.search(r"classList\s*\.\s*toggle", j))),
    ("js_parses",       3, lambda h, j: _node_check(j)),
]


def _server_array_ok(js):
    """>=4 objects carrying name + status + cpu, and >=1 offline."""
    objs = re.findall(r"\{[^{}]*\}", js)
    hits = [o for o in objs
            if re.search(r"\bname\b", o) and re.search(r"\bstatus\b", o) and re.search(r"\bcpu\b", o)]
    return len(hits) >= 4 and any("offline" in o.lower() for o in hits)


def _filters_ok(html, js):
    blob = (html + js).lower()
    return all(w in blob for w in ("all", "online", "offline")) and \
        len(re.findall(r"<button", html, re.I)) >= 3


def _node_check(js):
    if not js.strip():
        return False
    p = "/tmp/_capcheck.js"
    open(p, "w").write(js)
    r = subprocess.run(["node", "--check", p], capture_output=True)
    return r.returncode == 0


def extract(text):
    """Return (html, concatenated_js). Prefers a fenced block, falls back to raw."""
    m = re.search(r"```(?:html)?\s*\n(.*?)```", text, re.S | re.I)
    html = m.group(1) if m else text
    if "<html" not in html.lower():
        m2 = re.search(r"(<!DOCTYPE.*?</html>)", text, re.S | re.I)
        html = m2.group(1) if m2 else html
    js = "\n".join(re.findall(r"<script[^>]*>(.*?)</script>", html, re.S | re.I))
    return html.strip(), js


MAX_TOKENS = 10000


def ask(port, run_idx):
    body = json.dumps({
        "messages": [{"role": "user", "content": TASK}],
        "temperature": 0.2,
        "seed": 1000 + run_idx,      # fixed per run index -> same seeds across models
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.loads(r.read())
    dt = time.time() - t0
    ch = j["choices"][0]
    msg = ch["message"]
    # reasoning models put the answer in content; keep any reasoning out of grading,
    # but return it so an empty content is diagnosable instead of a silent zero.
    text = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    usage = j.get("usage", {})
    return text, reasoning, dt, usage.get("completion_tokens", 0), ch.get("finish_reason", "?")


def main():
    port, alias, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    runs = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    os.makedirs(outdir, exist_ok=True)
    maxpts = sum(w for _, w, _ in CHECKS)
    totals, complete = [], []

    for i in range(runs):
        try:
            text, reasoning, dt, ntok, fin = ask(port, i)
        except Exception as e:
            print(f"{alias} run{i}: REQUEST FAILED: {e}")
            totals.append(0)
            continue
        html, js = extract(text)
        open(f"{outdir}/{alias}_run{i}.raw.txt", "w").write(text)
        open(f"{outdir}/{alias}_run{i}.html", "w").write(html)
        if reasoning:
            open(f"{outdir}/{alias}_run{i}.reasoning.txt", "w").write(reasoning)

        got, failed = 0, []
        for name, w, fn in CHECKS:
            try:
                ok = bool(fn(html, js))
            except Exception:
                ok = False
            if ok:
                got += w
            else:
                failed.append(name)
        totals.append(got)
        # A truncated run scores low for lack of room, not lack of skill -- keep it
        # out of the "completed" average rather than silently averaging it in.
        trunc = fin == "length"
        if not trunc:
            complete.append(got)
        flag = f"  [TRUNCATED @{MAX_TOKENS}]" if trunc else ""
        empty = "  [EMPTY content, %dB reasoning]" % len(reasoning) if not text.strip() else ""
        print(f"{alias} run{i}: {got}/{maxpts}  {dt:6.1f}s  {ntok:5d}tok  fin={fin:8s} "
              f"{len(html):6d}B html{flag}{empty}   failed: {', '.join(failed) or 'none'}")

    avg = sum(totals) / len(totals) if totals else 0
    cavg = sum(complete) / len(complete) if complete else 0
    ntrunc = len(totals) - len(complete)
    print(f"{alias} TOTAL: all-runs {avg:.1f}/{maxpts} over {runs} | "
          f"completed-only {cavg:.1f}/{maxpts} over {len(complete)} | "
          f"truncated {ntrunc}/{runs}  (scores: {totals})")
    with open(f"{outdir}/scores.txt", "a") as f:
        f.write(f"{alias}\tall {avg:.1f}/{maxpts}\tcompleted {cavg:.1f}/{maxpts} (n={len(complete)})"
                f"\ttrunc {ntrunc}/{runs}\t{totals}\n")


if __name__ == "__main__":
    main()
