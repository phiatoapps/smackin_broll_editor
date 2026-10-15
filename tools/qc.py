"""Frame QC for a finished edit:
    python tools/qc.py final.mp4 [report.json] [--plan plan.json --raw raw.mp4 --words words.json]

Prints dead-air pauses, black/frozen frames and see-through stats, and writes a
grading sheet (final_grade.png: one large frame per second) to look over and
score every shot out of 10. Anything under 8 gets fixed and re-rendered.

With --plan/--raw/--words it also checks every cut for clipped words: the source
audio right past each cut must be quiet. Flags are listed with the words around
the cut so each one can be confirmed (a word cut short) or cleared (a breath,
room noise, or a retake word that is meant to be cut).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from broll_editor.multiscreen import audio_envelope  # noqa: E402

args = sys.argv[1:]
opts = {}
for key in ("--plan", "--raw", "--words"):
    if key in args:
        k = args.index(key)
        opts[key] = args[k + 1]
        del args[k:k + 2]
sys.argv = [sys.argv[0]] + args
video = Path(sys.argv[1])
env = audio_envelope(video)
loud = env > 0.012
i, gaps = int(np.argmax(loud)), []
while i < len(loud):
    if not loud[i]:
        j = i
        while j < len(loud) and not loud[j]:
            j += 1
        if j < len(loud) and (j - i) >= 15:
            gaps.append((round(i / 100, 2), round((j - i) / 100, 2)))
        i = j
    else:
        i += 1
print(f"pauses >= 0.15s: {len(gaps)} {gaps}")
log = subprocess.run(["ffmpeg", "-v", "info", "-i", str(video), "-vf",
                      "blackdetect=d=0.05:pix_th=0.1,freezedetect=n=0.001:d=0.4", "-an", "-f", "null", "-"],
                     capture_output=True, text=True).stderr
bad = [l.split("] ")[-1] for l in log.splitlines() if "black_start" in l or "freeze_start" in l]
print(f"black/frozen: {len(bad)} {bad[:10]}")
if len(sys.argv) > 2:
    r = json.loads(Path(sys.argv[2]).read_text())
    for n in [k[6:] for k in r[0] if k.startswith("ghost_")]:
        print(f"{n}: see-through px max {max(x['ghost_' + n] for x in r)}, covered max {max(x['covered_' + n] for x in r):.2f}")
if opts.get("--plan") and opts.get("--raw"):
    from broll_editor.multiscreen import MPlan

    plan = MPlan.load(opts["--plan"])
    renv = audio_envelope(opts["--raw"])
    words = json.loads(Path(opts["--words"]).read_text()) if opts.get("--words") else []
    soft = max(0.007, float(np.percentile(renv, 50)) * 1.6)
    t, flags = 0.0, []
    for i, seg in enumerate(plan.segments):
        a, b = seg.start, seg.start + seg.duration
        for side, x, lo, hi in (("end", b, b, b + 0.08), ("start", a, a - 0.08, a)):
            lvl = float(renv[int(lo * 100):int(hi * 100)].max())
            if lvl > soft:
                near = " ".join(w["text"] for w in words if w["end"] > x - 0.3 and w["start"] < x + 0.3)
                flags.append(f"  out {t + (seg.duration if side == 'end' else 0):5.2f}s {side:5s} level {lvl:.3f} | {near}")
        t += seg.duration
    print(f"cuts with sound right past them (check for clipped words): {len(flags)}")
    print("\n".join(flags))
sheet = video.with_name(video.stem + "_grade.png")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf",
                "fps=1,scale=270:-1,drawtext=text='%{eif\\:t\\:d}':x=4:y=4:fontsize=22:fontcolor=yellow,tile=8x6",
                "-frames:v", "1", str(sheet)], check=True)
print(f"grading sheet: {sheet}")
