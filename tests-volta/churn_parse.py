#!/usr/bin/env python3
"""Parse a churn-test server log: per-task prompt totals vs re-prefilled tokens.
Task layout: 0=warmup; round r: A-D = 1+5r..4+5r, E = 5+5r."""
import re, sys

log = sys.argv[1]
n_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 8

total, proc = {}, {}
for line in open(log):
    m = re.search(r"task (\d+) \| new prompt, n_ctx_slot = \d+, n_keep = \d+, task\.n_tokens = (\d+)", line)
    if m:
        total[int(m.group(1))] = int(m.group(2))
        continue
    m = re.search(r"task (\d+) \| prompt eval time = +[\d.]+ ms / +(\d+) tokens", line)
    if m:
        proc[int(m.group(1))] = int(m.group(2))

def agg(tasks):
    z = sum(total.get(t, 0) for t in tasks)
    y = sum(proc.get(t, 0) for t in tasks)
    return z, y, (1 - y / z * 100) if z else 0.0

print(f"{'round':>5} | {'subA-D prompt':>14} {'re-pref':>8} {'hit%':>6} | {'E prompt':>9} {'re-pref':>8} {'hit%':>6}")
tz = ty = ez = ey = 0
for r in range(n_rounds):
    subs = list(range(1 + 5 * r, 5 + 5 * r))
    e = 5 + 5 * r
    z, y, h = agg(subs)
    z2, y2, h2 = agg([e])
    tz += z; ty += y; ez += z2; ey += y2
    print(f"{r+1:>5} | {z:>14} {y:>8} {h:>5.1f}% | {z2:>9} {y2:>8} {h2:>5.1f}%")
print(f"{'TOTAL':>5} | {tz:>14} {ty:>8} {(100-ty/tz*100 if tz else 0):>5.1f}% | {ez:>9} {ey:>8} {(100-ey/ez*100 if ez else 0):>5.1f}%")
missing = [t for t in list(range(1, 1 + 5 * n_rounds)) if t not in total or t not in proc]
if missing:
    print(f"WARNING: {len(missing)} tasks missing metrics: {missing[:10]}")
