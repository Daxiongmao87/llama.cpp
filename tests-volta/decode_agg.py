#!/usr/bin/env python3
"""Aggregate decode tps from a llama-server -v log.
Pairs per-task 'eval time' lines; decode window = [end - D ms, end];
distributes tokens uniformly into 0.5s wall-clock buckets."""
import re, sys

path = sys.argv[1]
step = 0.5

def ts(line):
    m = re.match(r'(\d+)\.(\d+)\.(\d+)\.(\d+)', line)
    if not m:
        return None
    a, b, c, d = map(int, m.groups())
    return a * 60 + b + c / 1000.0 + d / 1e6

wins = []
for line in open(path, errors='replace'):
    m = re.search(r'print_timing: id\s+(\d+) \| task (\d+) \|\s+eval time =\s+([\d.]+) ms\s*/\s+(\d+) tokens', line)
    if m:
        t = ts(line)
        if t is not None:
            wins.append((t - float(m.group(3)) / 1000.0, t, int(m.group(4))))

tot_tok = sum(w[2] for w in wins)
tot_ms = sum((w[1] - w[0]) for w in wins) * 1000.0
per_stream = tot_tok / (tot_ms / 1000.0) if tot_ms else 0

tmax = max(w[1] for w in wins)
nb = int(tmax / step) + 2
buckets = [0.0] * nb
for t0, t1, tok in wins:
    n_b = max(1, int((t1 - t0) / step) + 1)
    per = tok / n_b
    for i in range(n_b):
        lo = t0 + i * step
        hi = min(lo + step, t1)
        if hi > lo:
            buckets[int(lo / step)] += per * (hi - lo) / step

active = [x / step for x in buckets if x > 1e-9]
print(f"decode windows: {len(wins)}, total tokens: {tot_tok}")
print(f"sum(decode time): {tot_ms/1000:.1f} s")
print(f"per-stream avg:      {per_stream:.1f} tok/s")
print(f"aggregate avg (over active {step}s buckets): {sum(active)/len(active):.1f} tok/s")
print(f"aggregate peak:      {max(active):.1f} tok/s   min: {min(active):.1f} tok/s")
