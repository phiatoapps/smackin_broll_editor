# re-pick weak listening shots: best quiet window where the listener faces the speaker
import json, sys
D = sys.argv[1]
p = json.load(open(f"{D}/plan.json"))
words = json.load(open(f"{D}/words_all.json"))
cfg = json.load(open(f"{D}/cfg.json"))
toward = {"A": 1, "B": -1}
tl = {n: json.load(open(f"{D}/gaze_{n}.json")) for n in "AB"}
import re
phr = [tuple(map(float, re.match(r"\s*([\d.]+)-\s*([\d.]+)", ln).groups())) for ln in open(f"{D}/phrases.txt") if ln.strip()]
spk = [(w["start"], w["end"]) for w in words] + phr
speech = {n: [(a - 0.15, b + 0.15) for a, b in spk if cfg[n][0] - 1 <= a <= cfg[n][1] + 1] for n in "AB"}
used = {n: [] for n in "AB"}
for s in p["segments"]:
    for n, (st, r) in s["listen"].items():
        used[n].append((st, st + s["duration"] * r))
def score(n, a, b):
    ys = [e["yaw"] for e in tl[n] if a <= e["t"] <= b]
    if not ys or any(x < b and a < y for x, y in speech[n]): return -1
    return sum(1 for y in ys if y is not None and toward[n] * y >= 5) / len(ys)
t = 0
for i, s in enumerate(p["segments"]):
    for n, (st, r) in list(s["listen"].items()):
        d = s["duration"] * r
        ys = [e["yaw"] for e in tl[n] if st <= e["t"] <= st + d]
        if s.get("unison") or not ys or sum(1 for y in ys if y is not None and toward[n] * y >= 5) / len(ys) >= 0.8: continue
        best = None
        for e in tl[n]:
            a = e["t"]
            if a < cfg[n][0] + 3 or a + d > cfg[n][1] - 1: continue
            sc = score(n, a, a + d)
            overlap = any(x < a + d and a < y for x, y in used[n] if (x, y) != (st, st + d))
            key = (sc, not overlap)
            if sc >= 0 and (best is None or key > best[0]): best = (key, a)
        if best and best[0][0] > score(n, st, st + d):
            print(f"seg {i} t={t:.1f} {n}: {st:.1f} ({score(n, st, st+d):.0%}) -> {best[1]:.1f} ({best[0][0]:.0%}, fresh={best[0][1]})")
            s["listen"][n] = [best[1], r]; used[n].append((best[1], best[1] + d))
    t += s["duration"]
json.dump(p, open(f"{D}/plan.json", "w"), indent=1)
