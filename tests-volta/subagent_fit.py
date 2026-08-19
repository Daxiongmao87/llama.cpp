#!/usr/bin/env python3
"""Subagent-fit checker: parse an OMP `--mode json` event log from a 4-subagent
atomic-task run. Scores per-subagent completion (status, duration, word_count
from <task-result> blocks) plus overall wall time and final-answer format.

Usage: subagent_fit.py OMP_EVENTS.json
"""
import json, os, re, sys, collections

def iter_blocks(path):
    for line in open(path):
        try:
            j = json.loads(line)
        except Exception:
            continue
        if j.get("type") not in ("message_start", "toolResult"):
            continue
        c = j.get("message", {}).get("content")
        if isinstance(c, str):
            yield c
        elif isinstance(c, list):
            for x in c:
                if isinstance(x, dict) and isinstance(x.get("text"), str):
                    yield x["text"]

def main():
    path = sys.argv[1]
    full = "\n".join(iter_blocks(path))
    # each <task-result> block, deduped (delivery messages repeat)
    seen, results = set(), []
    for m in re.finditer(
        r'<task-result id="(\w+)" agent="task" status="(\w+)" duration="([\dms]+)">.*?<output>\s*(\{[^}]*\})\s*</output>',
        full, re.S):
        if m.group(1) in seen:
            continue
        seen.add(m.group(1))
        wc = re.search(r'word_count["\']?\s*:\s*(\d+)', m.group(4))
        results.append((m.group(1), m.group(2), m.group(3), int(wc.group(1)) if wc else None))
    ok_lines = len(re.findall(r"SUB\s+\d+:\s*OK", full))
    ts = None
    for line in open(path):
        try:
            j = json.loads(line)
        except Exception:
            continue
        if j.get("type") == "session":
            ts = j.get("timestamp")
    wall = None
    if ts:
        mt = os.path.getmtime(path)
        from datetime import datetime
        wall = round((mt - datetime.fromisoformat(ts).timestamp()) / 60, 1)
    done = sum(1 for _, s, _, _ in results if s == "completed")
    print(f"subagent       status      dur     words")
    for name, st, dur, wc in sorted(results, key=lambda r: r[2]):
        print(f"  {name:14s} {st:9s} {dur:5s}  {wc if wc is not None else '-'}")
    wcs = [w for _, s, _, w in results if s == "completed" and w]
    if wcs:
        print(f"word counts: {wcs}  (target ~400)")
    print(f"completed: {done}/{len(results) or 'unknown'}   'SUB n: OK' lines: {ok_lines}   wall: {wall} min")

if __name__ == "__main__":
    main()
