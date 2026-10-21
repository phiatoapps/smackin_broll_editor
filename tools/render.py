"""python tools/render.py VIDEO_DIR [--draft]

Renders VIDEO_DIR/plan.json (from build_cfg.py) and runs every check, writing
VIDEO_DIR/qc.txt:
  --draft  540x960, fast encode -> draft.mp4 (about a minute): check audio, labels,
           framing and pauses here, fix the cfg, repeat
  default  1080x1920 final quality -> final.mp4 (about 4 minutes), plus the grading
           sheet (final_grade.png) and the gaze check
Both use every CPU core. Ends with a readback: the edit's audio transcribed, to
compare against the script (catches clipped or leaked words).
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from broll_editor import multiscreen as ms  # noqa: E402

D = Path(sys.argv[1])
draft = "--draft" in sys.argv
name = "draft" if draft else "final"
out = D / f"{name}.mp4"
plan = ms.MPlan.load(D / "plan.json")
size = dict(width=540, height=960, crf=23, preset="veryfast") if draft else dict(crf=18, preset="medium")
ms.render_multi(D / "proxy.mp4", plan, out, report_path=D / f"{name}_report.json", **size)

py = [sys.executable, "-W", "ignore"]
lines = []
qc = subprocess.run(py + [str(ROOT / "tools/qc.py"), str(out), str(D / f"{name}_report.json"), "--plan", str(D / "plan.json"),
                          "--raw", str(D / "proxy.mp4"), "--words", str(D / "words_all.json")],
                    capture_output=True, text=True)
lines += [ln for ln in qc.stdout.splitlines() if not re.match(r"^  out.*level 0\.0[0-2]", ln)]  # faint: room tone
if not draft:
    gz = subprocess.run(py + [str(ROOT / "tools/gazecheck.py"), str(D)], capture_output=True, text=True,
                        env={**os.environ, "PYTHONPATH": str(ROOT)})
    lines += [ln for ln in gz.stdout.splitlines()]

from faster_whisper import WhisperModel  # noqa: E402
import numpy as np  # noqa: E402

raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(out), "-vn", "-f", "s16le", "-ac", "1",
                      "-ar", "16000", "-"], capture_output=True).stdout
audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
segs, _ = WhisperModel("medium", device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4).transcribe(
    audio, language="en", vad_filter=False, beam_size=1, condition_on_previous_text=False)
lines += ["readback: " + " ".join(s.text.strip() for s in segs)]
(D / "qc.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
