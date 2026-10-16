"""Listener gaze check: python tools/gazecheck.py <dir with plan.json, gaze_A.json, gaze_B.json> [-v]

Every listening shot must show the listener facing the person talking (A on the
left faces right, B on the right faces left). Flags shots where the listener looks
away ("AWAY", never acceptable) or where the face isn't clearly turned ("weak").
Gaze timelines come from `python -m broll_editor.gaze raw lo hi x0 x1 out.json`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sys, json, numpy as np
from broll_editor import multiscreen as ms
D = sys.argv[1]
p = ms.MPlan.load(f"{D}/plan.json")
tl = {n: json.load(open(f"{D}/gaze_{n}.json")) for n in ("A", "B")}
toward = {"A": 1, "B": -1}   # A (left) faces right (+), B (right) faces left (-)
def yaws(n, a, b):
    return [e["yaw"] for e in tl[n] if a <= e["t"] <= b]
t = 0; bad = 0
for i, s in enumerate(p.segments):
    for n, (st, rate) in s.listen.items():
        ys = yaws(n, st, st + s.duration * rate)
        if not ys: continue
        good = sum(1 for y in ys if y is not None and toward[n] * y >= 5) / len(ys)
        # a single-sample flicker (0.2s head bob) isn't "looking away"; two in a row is
        aw = [y is not None and toward[n] * y <= -5 for y in ys]
        away = sum(1 for i in range(len(aw)) if aw[i] and ((i > 0 and aw[i - 1]) or (i + 1 < len(aw) and aw[i + 1]))) / len(ys)
        flag = "AWAY" if away > 0.05 else ("weak" if good < 0.8 else "ok")
        bad += flag != "ok"
        if flag != "ok" or "-v" in sys.argv:
            print(f"out {t:5.1f}-{t + s.duration:5.1f} listener {n} src {st:.1f}: facing {good:.0%}, away {away:.0%}  {flag}")
    t += s.duration
print("segments needing a fix:", bad)
