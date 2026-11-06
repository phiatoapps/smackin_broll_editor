"""python tools/build_cfg.py VIDEO_DIR  (reads VIDEO_DIR/cfg.json, writes plan.json)

cfg.json: {"A": [lo, hi, center], "B": [lo, hi, center],
           "labels": {"A": "2009", "B": "2027"},            (optional)
           "label_changes": {"B": [[line, "NEW LABEL"]]},   (optional: label text from that line on)
           "ends": {"text prefix": raw_end},                 (optional, soft last words)
           "starts": {"text prefix": raw_start},             (optional, trim quiet lead-in)
           "empty": [lo, hi], "depth": ["B", "A"], "counter_y": 0.55,   (optional, see README)
           "listen": {"text prefix": {"A": raw_start}},     (optional: hand-picked listening footage)
           "framing": {"text prefix": [zoom, cx, cy] | null},   (optional: hand-picked camera framing)
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
names = [n for n in ("A", "B", "C") if n in cfg]
chars = [ms.Character(n, *cfg[n], labels.get(n, ""), 0.20) for n in names]
for c in chars:  # "placed" mode: {"A": {"src": [x0, x1], "dx": -0.15}}
    c.src = cfg.get("place", {}).get(c.name, {}).get("src")
    c.dx = cfg.get("place", {}).get(c.name, {}).get("dx", 0.0)
    c.cut = cfg.get("place", {}).get(c.name, {}).get("cut")
    c.empty = cfg.get("place", {}).get(c.name, {}).get("empty")


def voice_end(lo, hi):  # last speech-loud moment of the line (no trailing room tone)
    loud = np.where(env[int(lo * 100):int(hi * 100)] > 0.03)[0]
    return lo + (loud[-1] + 1) / 100 if len(loud) else hi


takes, unison, solo = [], {}, set()
for k, (l, ch, a, b, z, t, *other) in enumerate(cfg["picks"]):
    if ch == "U":
        # everyone starts together: each take starts just before its own speech onset;
        # `other` holds the other characters' takes of the line, in order: b0, b1, c0, c1
        ra, _ = ms.refine(env, a, b, back=0.3, fwd=0.4)
        d = voice_end(ra, b) - ra
        unison[k] = {}
        for n, (o0, o1) in zip(names[1:], zip(other[::2], other[1::2])):
            oa, _ = ms.refine(env, o0, o1, back=0.3, fwd=0.4)
            d = max(d, voice_end(oa, o1) - oa)
            unison[k][n] = round(oa - 0.04, 3)
        takes.append(ms.MTake(l, "A", ra - 0.04, ra - 0.04 + d + 0.10, z, t))
        continue
    if ch.endswith("!"):  # solo shot: this character's take alone, full frame
        ch = ch[:-1]
        solo.add(t)
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
plan = ms.schedule_multi(takes, wobj, chars, cfg.get("bands", [cfg.get("band", [0.30, 0.92])]))
for k, starts in unison.items():
    plan.segments[k].unison = True
    for n, st in starts.items():
        plan.segments[k].listen[n] = [st, 1.0]
plan.counter_y = cfg.get("counter_y", 0.55)
plan.label_changes = cfg.get("label_changes", {})
plan.empty = cfg.get("empty")      # [lo, hi] seconds of empty room, if the clip has some
plan.depth = cfg.get("depth")      # ["B", "A"]: back to front (stand-behind format)
plan.mode = cfg.get("mode", "mirror")
plan.seg_bodies = cfg.get("seg_bodies", False)
plan.split_x = cfg.get("split_x", 0.5)  # "stand_behind": front clone cut out over the other take
plan.gains = ms.match_gains(P, chars)
print(os.path.basename(D), ms.tighten(plan, env, thr=0.015, words=words))
# hand-checked line starts: {"text prefix": raw_start} trims quiet lead-in off the first
# piece of that line (never later than its first word)
for prefix, st in cfg.get("starts", {}).items():
    seg = [s for s in plan.segments if s.text.startswith(prefix)][0]
    if seg.start < st < seg.start + seg.duration:
        print(f"  trimmed start of '{prefix}' by {st - seg.start:.2f}s")
        seg.duration = round(seg.duration - (st - seg.start), 3)
        seg.start = st
# hand-checked word endings the envelope misses (a soft final word): {"text prefix": raw end}
for prefix, end in cfg.get("ends", {}).items():
    seg = [s for s in plan.segments if s.text.startswith(prefix)][-1]
    if end > seg.start + seg.duration:
        print(f"  held '{prefix}' to {end} (+{end - seg.start - seg.duration:.2f}s)")
        seg.duration = round(end - seg.start, 3)
ms.auto_framing(plan)
# listeners face the speaker
for n, side, toward in (("A", (0.0, 0.55), "right"), ("B", (0.45, 1.0), "left")):
    if n not in names or len(names) > 2:
        continue  # three-character intros keep the listening footage chosen by schedule_multi
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
for sg in plan.segments:
    if sg.text in solo:
        sg.solo, sg.framing = True, None
# hand-picked framing: {"text prefix": [zoom, center_x, center_y]} or null for wide
for prefix, fr in cfg.get("framing", {}).items():
    for sg in [x for x in plan.segments if x.text.startswith(prefix)]:
        sg.framing = fr
# hand-picked listening footage: {"text prefix": {"A": raw_start}} (pieces of a split
# line continue from where the previous piece's footage ended)
for prefix, who in cfg.get("listen", {}).items():
    for n, st in who.items():
        st, rate = (st if isinstance(st, list) else [st, 1.0])   # raw_start or [raw_start, rate]
        for sg in [x for x in plan.segments if x.text.startswith(prefix)]:
            sg.listen[n] = [round(st, 3), rate]
            st += sg.duration * rate
if plan.mode == "split":
    for sg in plan.segments:
        if not sg.solo:   # full-res, full-speed: no zooms on the split, listeners at 1x
            sg.framing, sg.pull_out = None, None
            for n, (st, rate) in list(sg.listen.items()):
                sg.listen[n] = [st, 1.0]
plan.save(f"{D}/plan.json")
t = 0
for i, s in enumerate(plan.segments):
    print(f"  {i:2d} {t:5.1f} {s.char} {s.duration:4.2f} z={int(s.zoom)} {s.text[:40]}"); t += s.duration
print(f"  TOTAL {t:.1f}s")
