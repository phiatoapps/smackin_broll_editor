"""python tools/build_cfg.py VIDEO_DIR  (reads VIDEO_DIR/cfg.json, writes plan.json)

cfg.json: {"A": [lo, hi, center], "B": [lo, hi, center],
           "labels": {"A": "2009", "B": "2027"},            (optional)
           "label_changes": {"B": [[line, "NEW LABEL"]]},   (optional: label text from that line on)
           "ends": {"text prefix": raw_end},                 (optional, soft last words)
           "band": [lo, hi],    (optional, default [0.30, 0.92]: where both clones may
                                 appear; narrow it when they never cross the middle)
           "picks": [[line, "A"|"B", start, end, zoom, text], ...]}
A is the clone standing on the left, B the one on the right.
A unison pick (both clones say the line together) is
    [line, "U", a_start, a_end, zoom, text, b_start, b_end]
and plays A's take with B's take of the same line beside it, audio mixed.
"""
import sys, json, os, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from broll_editor import multiscreen as ms, gaze
from broll_editor.captions import Word
from broll_editor.multiscreen import _speech_intervals

D = sys.argv[1]
cfg = json.load(open(f"{D}/cfg.json"))
words = json.load(open(f"{D}/words_all.json"))
wobj = [Word(w["text"], w["start"], w["end"]) for w in words]
P = f"{D}/proxy.mp4"
env = np.load(f"{D}/env.npy") if os.path.exists(f"{D}/env.npy") else ms.audio_envelope(P)
np.save(f"{D}/env.npy", env)
labels = cfg.get("labels", {})
chars = [ms.Character("A", *cfg["A"], labels.get("A", ""), 0.20), ms.Character("B", *cfg["B"], labels.get("B", ""), 0.20)]
takes, unison = [], {}
for k, (l, ch, a, b, z, t, *other) in enumerate(cfg["picks"]):
    if ch == "U":
        # both voices start together: each take starts just before its own speech onset
        ra, _ = ms.refine(env, a, b, back=0.3, fwd=0.4)
        oa, _ = ms.refine(env, *other, back=0.3, fwd=0.4)

        def voice_end(lo, hi):  # last speech-loud moment of the line (no trailing room tone)
            loud = np.where(env[int(lo * 100):int(hi * 100)] > 0.03)[0]
            return lo + (loud[-1] + 1) / 100 if len(loud) else hi
        d = max(voice_end(ra, b) - ra, voice_end(oa, other[1]) - oa) + 0.10
        unison[k] = round(oa - 0.04, 3)
        takes.append(ms.MTake(l, "A", ra - 0.04, ra - 0.04 + d, z, t))
        continue
    # never cut into a word: if the take ends (or starts) inside one word, grow it
    # once to cover that word - never chained on into the next words
    a0, b0 = a, b
    inside = [w for w in words if w["start"] < b0 - 0.02 < w["end"] - 0.04 and w["start"] >= a0 - 0.02
              and w["end"] - w["start"] < 1.2 and w["end"] - b0 <= 0.6]
    if inside:
        b = max(w["end"] for w in inside) + 0.05
        print(f"  extended end of '{t[:30]}' {b0} -> {b:.2f} to finish '{inside[-1]['text']}'")
    first = [w for w in words if w["start"] < a0 - 0.02 and w["end"] > a0 + 0.05 and w["end"] - w["start"] < 1.0
             and a0 - w["start"] <= 0.6]
    if first:
        a = min(w["start"] for w in first)
        print(f"  extended start of '{t[:30]}' {a0} -> {a:.2f} to include '{first[0]['text']}'")
    seg_env = env[int(a * 100):int(b * 100)]
    if len(seg_env) and (seg_env > 0.04).mean() < 0.15:
        print(f"  WARNING: take '{t[:30]}' ({a}-{b}) is mostly silent - check the timing")
    ra, rb = ms.refine(env, a, b, back=0.3, fwd=0.4)
    # padding and sound-following may never reach into the next/previous word
    nxt = [w["start"] for w in words if w["start"] >= b - 0.02 and w["end"] - w["start"] < 1.2]
    prv = [w["end"] for w in words if w["end"] <= a + 0.02 and w["end"] - w["start"] < 1.2]
    hi = min(nxt) - 0.01 if nxt else 1e9
    lo = max(prv) + 0.01 if prv else -1e9
    takes.append(ms.MTake(l, ch, max(ra - 0.04, lo, 0), min(rb + 0.06, hi), z, t))
plan = ms.schedule_multi(takes, wobj, chars, [cfg.get("band", [0.30, 0.92])])
for k, b_start in unison.items():
    plan.segments[k].unison = True
    plan.segments[k].listen["B"] = [b_start, 1.0]
plan.counter_y = 0.55
plan.label_changes = cfg.get("label_changes", {})
plan.gains = ms.match_gains(P, chars)
print(os.path.basename(D), ms.tighten(plan, env, thr=0.015, words=words))
# hand-checked word endings the envelope misses (a soft final word): {"text prefix": raw end}
for prefix, end in cfg.get("ends", {}).items():
    seg = [s for s in plan.segments if s.text.startswith(prefix)][-1]
    if end > seg.start + seg.duration:
        print(f"  held '{prefix}' to {end} (+{end - seg.start - seg.duration:.2f}s)")
        seg.duration = round(end - seg.start, 3)
ms.auto_framing(plan)
# listeners face the speaker
for n, side, toward in (("A", (0.0, 0.55), "right"), ("B", (0.45, 1.0), "left")):
    gp = f"{D}/gaze_{n}.json"
    if not os.path.exists(gp):
        lo, hi = cfg[n][0], cfg[n][1]
        json.dump(gaze.scan(P, lo, hi, *side), open(gp, "w"))
    tl = json.load(open(gp))
    rng = []
    for thr in (6.0, 4.0, 2.0):
        rng = gaze.listening_ranges(tl, toward, _speech_intervals(wobj, cfg[n][0], cfg[n][1], margin=0.15), min_yaw=thr)
        need = max([s.duration for s in plan.segments if n in s.listen] or [0])
        if sum(b - a for a, b in rng) > 1.3 * sum(s.duration for s in plan.segments if n in s.listen) and any(b - a >= need for a, b in rng):
            break
    notes = gaze.assign_listening(plan, n, rng) if rng else ["NO GOOD LISTENING FOOTAGE"]
    print(f"  {n} listening ranges (yaw>={thr}): {len(rng)}, total {sum(b-a for a,b in rng):.1f}s  {notes}")
plan.save(f"{D}/plan.json")
t = 0
for i, s in enumerate(plan.segments):
    print(f"  {i:2d} {t:5.1f} {s.char} {s.duration:4.2f} z={int(s.zoom)} {s.text[:40]}"); t += s.duration
print(f"  TOTAL {t:.1f}s")
