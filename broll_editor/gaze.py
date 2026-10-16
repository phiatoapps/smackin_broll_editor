"""Where is each person looking? Head-turn (yaw) timeline per take.

Listening shots must show the listener facing the person who is talking (never
looking the other way). This scans a take with MediaPipe's face landmarker and
returns, per sampled time, the head yaw in degrees: positive = turned toward the
right side of the frame, negative = toward the left, plus eye direction.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

MODEL = Path(__file__).parent / "models" / "face_landmarker.task"


def _landmarker():
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(MODEL)),
        output_facial_transformation_matrixes=True, num_faces=1)
    return vision.FaceLandmarker.create_from_options(opts)


def scan(raw: str | Path, lo: float, hi: float, x0: float = 0.0, x1: float = 1.0,
         fps: float = 5.0, width: int = 1080, height: int = 1920) -> list[dict]:
    """Yaw timeline for the person standing in [x0, x1] of the frame during raw[lo:hi]."""
    import mediapipe as mp

    cw = int((x1 - x0) * width) // 2 * 2
    cx = int(x0 * width)
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{lo:.3f}", "-t", f"{hi - lo:.3f}", "-i", str(raw),
         "-vf", f"fps={fps},scale={width}:{height},crop={cw}:{height // 2}:{cx}:0",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    h = height // 2
    frames = np.frombuffer(out, np.uint8).reshape(-1, h, cw, 3)
    lm = _landmarker()
    res = []
    for i, f in enumerate(frames):
        r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f)))
        t = round(lo + i / fps, 2)
        if not r.face_landmarks:
            res.append({"t": t, "yaw": None, "eyes": None})
            continue
        m = np.array(r.facial_transformation_matrixes[0])
        yaw = float(np.degrees(np.arctan2(m[0, 2], m[2, 2])))
        p = r.face_landmarks[0]
        # iris position inside each eye, -1 (one corner) .. +1 (other), image-x direction
        def iris(c, a, b):
            return (p[c].x - (p[a].x + p[b].x) / 2) / max(abs(p[b].x - p[a].x) / 2, 1e-6)
        eyes = float((iris(468, 33, 133) + iris(473, 362, 263)) / 2) if len(p) > 473 else None
        res.append({"t": t, "yaw": round(yaw, 1), "eyes": None if eyes is None else round(eyes, 2)})
    lm.close()
    return res


def facing(timeline: list[dict], toward: str, min_yaw: float = 4.0) -> list[tuple[float, float]]:
    """Time ranges where the face is turned toward `toward` ('left'/'right' of frame)."""
    sign = 1 if toward == "right" else -1
    ok = [(e["t"], e["yaw"] is not None and sign * e["yaw"] >= min_yaw) for e in timeline]
    ranges, start = [], None
    for t, g in ok:
        if g and start is None:
            start = t
        elif not g and start is not None:
            ranges.append((start, t)); start = None
    if start is not None:
        ranges.append((start, ok[-1][0]))
    return ranges


if __name__ == "__main__":
    import sys
    raw, lo, hi, x0, x1, outp = sys.argv[1], *map(float, sys.argv[2:6]), sys.argv[6]
    tl = scan(raw, lo, hi, x0, x1)
    Path(outp).write_text(json.dumps(tl))
    ys = [e["yaw"] for e in tl if e["yaw"] is not None]
    print(f"{len(tl)} samples, face found {len(ys)}, yaw range {min(ys):.0f}..{max(ys):.0f}")


def listening_ranges(timeline: list[dict], toward: str, speech: list[tuple[float, float]],
                     min_yaw: float = 6.0, min_len: float = 0.6, max_gap: float = 0.45) -> list[tuple[float, float]]:
    """Stretches where this person is quiet AND facing `toward` (the other person)."""
    sign = 1 if toward == "right" else -1
    good = [e["t"] for e in timeline if e["yaw"] is not None and sign * e["yaw"] >= min_yaw
            and not any(a <= e["t"] <= b for a, b in speech)]
    out, s, prev = [], None, None
    for t in good:
        if s is None:
            s = t
        elif t - prev > max_gap:
            out.append((s, prev)); s = t
        prev = t
    if s is not None:
        out.append((s, prev))
    return [(a, b) for a, b in out if b - a >= min_len]


def assign_listening(plan, char: str, ranges: list[tuple[float, float]], min_rate: float = 0.8) -> list[str]:
    """Re-pick `char`'s listening footage from `ranges` only. In order of preference:
    any unused stretch long enough (longest shots placed first, tightest fit); an
    unused stretch at most 20% short, played slightly slower; and only when all of
    that fails, reuse the longest stretch. Returns notes on compromises."""
    free = [list(r) for r in ranges]
    notes = []
    segs = [s for s in plan.segments if char in s.listen]
    # Longest shots first, each into the tightest unused stretch it fits (best-fit
    # decreasing), so short shots can't eat the only stretches long shots fit in.
    for seg in sorted(segs, key=lambda s: -s.duration):
        need = seg.duration
        pick = None
        cands = [f for f in free if f[1] - f[0] >= need]
        if cands:
            pick = min(cands, key=lambda f: f[1] - f[0]); st, rate = pick[0], 1.0
        else:
            cands = [f for f in free if f[1] - f[0] >= need * min_rate]
            if cands:
                pick = max(cands, key=lambda f: f[1] - f[0]); st = pick[0]
                rate = round((pick[1] - pick[0]) / need, 3)
                notes.append(f"slowed to {rate:.2f}x for '{seg.text[:25]}'")
        if pick is not None:
            seg.listen[char] = [round(st, 3), min(rate, 1.0)]
            pick[0] = st + need * min(rate, 1.0) + 0.05
            continue
        a, b = max(ranges, key=lambda r: r[1] - r[0])
        if b - a >= need:
            off = (b - a - need) / 2
            seg.listen[char] = [round(a + off, 3), 1.0]
            notes.append(f"reused {a + off:.1f}s for '{seg.text[:25]}'")
        else:
            seg.listen[char] = [round(a, 3), round(max(0.6, (b - a) / need), 3)]
            notes.append(f"reused+slowed {a:.1f}s for '{seg.text[:25]}'")
    return notes
