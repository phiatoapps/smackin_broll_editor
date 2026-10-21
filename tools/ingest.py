"""python tools/ingest.py RAW.MOV VIDEO_DIR

Everything a split-screen edit needs from one raw phone clip, into VIDEO_DIR:
  proxy.mp4        1080x1920 30fps working copy
  words_all.json   word timestamps (whisper medium, whole clip)
  transcript.txt   the same, one line per phrase
  phrases.txt      every stretch of speech found in the audio, transcribed on its
                   own - catches lines the whole-clip pass drops (it often does)
  sides.txt        where the person stands in each half of the clip (motion centroid)
Then pick the takes from phrases.txt + transcript.txt into cfg.json (see build_cfg.py).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from broll_editor import multiscreen as ms  # noqa: E402
from broll_editor import splitscreen as sp  # noqa: E402

raw, D = Path(sys.argv[1]), Path(sys.argv[2])
D.mkdir(parents=True, exist_ok=True)
P = D / "proxy.mp4"
if not P.exists():
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-map", "0:v:0", "-map", "0:a:0",
                    "-vf", "scale=1080:1920,fps=30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "14",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", str(P)], check=True)
print("proxy done", flush=True)

from faster_whisper import WhisperModel  # noqa: E402

pcm = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(P), "-vn", "-f", "s16le", "-ac", "1",
                      "-ar", "16000", "-"], capture_output=True).stdout
audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
model = WhisperModel("medium", device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
if not (D / "words_all.json").exists():
    segs, _ = model.transcribe(audio, language="en", vad_filter=False, word_timestamps=True,
                               condition_on_previous_text=False)
    ws = [{"text": w.word.strip(), "start": float(w.start), "end": float(w.end)}
          for s in segs if s.no_speech_prob < 0.6 for w in s.words if w.word.strip()]
    (D / "words_all.json").write_text(json.dumps(ws, indent=1))
    out, line, last = [f"dur {len(audio) / 16000:.1f}"], [], None
    for w in ws:
        if last is not None and w["start"] - last > 0.5:
            out.append(" ".join(line))
            line = []
        line.append(f"[{w['start']:.2f}]{w['text']}")
        last = w["end"]
    out.append(" ".join(line))
    (D / "transcript.txt").write_text("\n".join(out) + "\n")
print("transcript done", flush=True)

# every speech stretch from the audio envelope, transcribed on its own
env = ms.audio_envelope(P)
np.save(D / "env.npy", env)
on = env > 0.03
runs, s = [], None
for i in range(len(on)):
    if on[i] and s is None:
        s = i
    if not on[i] and s is not None:
        if i - s >= 8:
            runs.append([s / 100, i / 100])
        s = None
phr = []
for a, b in runs:
    if phr and a - phr[-1][1] < 0.6:
        phr[-1][1] = b
    else:
        phr.append([a, b])
pad = np.zeros(8000, np.float32)
lines = []
for a, b in phr:
    if b - a < 0.25:
        continue
    x = np.concatenate([pad, audio[int(max(a - 0.15, 0) * 16000):int((b + 0.15) * 16000)], pad])
    sg, _ = model.transcribe(x, language="en", vad_filter=False, beam_size=1, condition_on_previous_text=False)
    lines.append(f"{a:7.2f}-{b:7.2f} ({b - a:4.2f}) {' '.join(q.text.strip() for q in sg)}")
(D / "phrases.txt").write_text("\n".join(lines) + "\n")
print("phrases done", flush=True)

# where the person stands in each half
m = sp._motion(sp.Path(str(P)))
W, n = m.shape[1], len(m)
out = []
for lo, hi in ((0, n // 2), (n // 2, n)):
    pr = m[lo:hi].sum(0)
    pr = pr / max(pr.max(), 1e-6)
    out.append(f"{lo / 10:.0f}-{hi / 10:.0f}s: centroid {(pr * np.arange(W)).sum() / pr.sum() / W:.2f} "
               f"extent {np.where(pr > 0.05)[0].min() / W:.2f}..{np.where(pr > 0.05)[0].max() / W:.2f}")
(D / "sides.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
print("INGEST DONE", flush=True)
