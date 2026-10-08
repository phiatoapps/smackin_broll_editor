"""Three-or-more-character version of the split screen, for people standing close together.

Same idea as `splitscreen`: one locked-off raw clip in which you play each character
in turn, at a different spot in the frame. Here each character has a time range of
the raw clip (their take) and an order from left to right.

Straight vertical seams don't work when neighbours overlap, so every frame gets
curved seams instead:

* An empty-room plate (per-pixel median over all takes) serves as a reference.
  Wherever a take differs from it, that take has someone (or a prop) there, so
  take-specific props are always kept whole rather than flickering in and out.
* Each row of the seam goes where it hides the least of either person, with the
  current speaker weighted up, so when two people overlap the speaker stays whole.
* Seams are kept smooth from row to row and frame to frame, then feathered.
* All takes are colour-matched to the first one so the background is seamless.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .captions import Word
from .splitscreen import _label_png, _sample, _speech_intervals
from .stacker import _run_ffmpeg

FPS = 30
_DS = 8  # analysis downscale factor for seam finding


@dataclass
class Character:
    name: str
    lo: float          # take range in the raw clip
    hi: float
    center: float      # where they stand, fraction of frame width (for labels / zoom)
    label: str = ""


@dataclass
class MTake:
    line: int
    char: str
    start: float
    end: float
    zoom: bool = False
    text: str = ""


@dataclass
class MSegment:
    line: int
    char: str
    start: float
    duration: float
    listen: dict[str, list[float]] = field(default_factory=dict)  # name -> [start, rate]
    zoom: bool = False
    text: str = ""


@dataclass
class MPlan:
    chars: list[Character]                      # left to right
    bands: list[list[float]]                    # allowed seam range between neighbours
    segments: list[MSegment] = field(default_factory=list)
    gains: dict[str, list[float]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "MPlan":
        d = json.loads(Path(path).read_text())
        d["chars"] = [Character(**c) for c in d["chars"]]
        d["segments"] = [MSegment(**s) for s in d["segments"]]
        return cls(**d)

    def char(self, name: str) -> Character:
        return next(c for c in self.chars if c.name == name)


# ---------------------------------------------------------------- timing


def audio_envelope(raw: str | Path, hop: float = 0.01) -> np.ndarray:
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(raw), "-vn", "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
        check=True, capture_output=True,
    ).stdout
    a = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
    n = int(16000 * hop)
    k = len(a) // n
    return np.sqrt((a[: k * n].reshape(k, n) ** 2).mean(axis=1))


def refine(env: np.ndarray, start: float, end: float, hop: float = 0.01,
           back: float = 0.8, fwd: float = 0.6) -> tuple[float, float]:
    """Snap a line's rough [start, end] to where the voice actually starts and stops.

    Word timestamps can be off by a word (Whisper sometimes hangs the first word of a
    line on the end of the previous one), so look a little outside them in the audio.
    """
    floor = np.percentile(env, 20)
    thr = max(0.008, floor * 3)
    loud = env > thr

    def idx(t: float) -> int:
        return min(max(int(round(t / hop)), 0), len(env) - 1)

    gap = int(0.15 / hop)
    s = idx(start)
    lo = idx(start - back)
    quiet = 0
    i = s
    while i > lo:
        quiet = 0 if loud[i - 1] else quiet + 1
        if quiet >= gap:
            break
        i -= 1
    s_out = (i + quiet) * hop if quiet >= gap else start
    e = idx(end)
    hi = idx(end + fwd)
    quiet = 0
    j = e
    while j < hi:
        quiet = 0 if loud[j] else quiet + 1
        if quiet >= gap:
            break
        j += 1
    e_out = (j - quiet + 1) * hop if quiet >= gap else end
    return round(min(s_out, start), 3), round(max(e_out, end), 3)


def schedule_multi(takes: list[MTake], words: list[Word], chars: list[Character],
                   bands: list[list[float]], missing: list[str] | None = None) -> MPlan:
    """Lay takes back to back; every non-speaker gets listening footage from their own take."""
    speech = {c.name: _speech_intervals(words, c.lo, c.hi) for c in chars}
    durs = [t.end - t.start for t in takes]
    cursor = {}
    for c in chars:
        own = [k for k, t in enumerate(takes) if t.char == c.name]
        cursor[c.name] = max(c.lo, takes[own[0]].start - sum(durs[:own[0]]) - 0.2) if own else c.lo
    plan = MPlan(chars, bands, missing=list(missing or []))
    for t, dur in zip(takes, durs):
        listen = {}
        for c in chars:
            if c.name == t.char:
                continue
            cur = cursor[c.name]
            for a, b in speech[c.name]:
                if a <= cur < b:
                    cur = b
            prev_end = max([b for a, b in speech[c.name] if b <= cur + 1e-3] + [c.lo])
            next_start = min([a for a, b in speech[c.name] if a >= cur - 1e-3] + [c.hi])
            start, rate = cur, 1.0
            if next_start - start < dur:
                start = max(prev_end, next_start - dur)
                avail = next_start - start
                if 0.3 < avail < dur:
                    rate = max(0.5, avail / dur)
            cursor[c.name] = start + dur * rate
            listen[c.name] = [round(start, 3), round(rate, 4)]
        cursor[t.char] = t.end
        plan.segments.append(MSegment(t.line, t.char, round(t.start, 3), round(dur, 3), listen, t.zoom, t.text))
    return plan


# ---------------------------------------------------------------- colour


def match_gains(raw: str | Path, chars: list[Character]) -> dict[str, list[float]]:
    """RGB gains per take so every take's static background matches the first take's."""
    raw = Path(raw)
    frames = {c.name: _sample(raw, c.lo, c.hi - c.lo, 0.5) for c in chars}
    med = {k: np.median(v, axis=0) for k, v in frames.items()}
    still = {k: v.std(axis=0).max(-1) < 6 for k, v in frames.items()}
    ref = chars[0].name
    gains = {ref: [1.0, 1.0, 1.0]}
    for c in chars[1:]:
        a, b = med[ref], med[c.name]
        la, lb = a.mean(-1), b.mean(-1)
        ok = still[ref] & still[c.name] & (la > 20) & (lb > 20) & (la < 235) & (lb < 235)
        ratio = la / np.maximum(lb, 1)
        if ok.sum() < 200:
            gains[c.name] = [1.0, 1.0, 1.0]
            continue
        r0 = np.median(ratio[ok])
        ok &= np.abs(ratio - r0) < 0.12 * r0
        gains[c.name] = [round(float(np.clip(np.median(a[..., k][ok] / np.maximum(b[..., k][ok], 1)), 0.7, 1.4)), 4)
                         for k in range(3)]
    return gains


def background_plate(raw: str | Path, chars: list[Character], gains: dict[str, list[float]]
                     ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Empty-room estimate at analysis resolution: per-pixel median over all takes.

    Each spot in the frame has a person (or a take-specific prop) in it for less than
    half the clip, so the median is the bare background - a fair reference for
    deciding what in each take is "someone/something" vs. scenery.

    Also returns, per take, a map of its props: spots that differ from the plate
    but barely change during the take. Props sit on the counter in front of the
    people, so the seams treat them as in front (never cut, never hidden by a person).
    """
    raw = Path(raw)
    takes = {c.name: _sample(raw, c.lo, c.hi - c.lo, 1) * np.array(gains.get(c.name, [1, 1, 1]), np.float32)
             for c in chars}
    plate = np.median(np.concatenate(list(takes.values())), axis=0).astype(np.float32)
    props = {}
    for name, f in takes.items():
        differs = np.abs(np.median(f, axis=0) - plate).sum(-1) > 60
        steady = f.std(axis=0).max(-1) < 12
        m = (differs & steady).astype(np.float32)
        # grow a little so the whole object (edges included) counts
        for axis in (0, 1):
            m = np.maximum.reduce([np.roll(m, k, axis=axis) for k in (-2, -1, 0, 1, 2)])
        props[name] = m
    return plate, props


# ---------------------------------------------------------------- seams


def _row_seam(cost: np.ndarray, prev: np.ndarray | None, lam_t: float, step: int = 2) -> np.ndarray:
    """Minimal-cost top-to-bottom path through cost (H, W), moving <= step columns per row."""
    H, W = cost.shape
    if prev is not None:
        cost = cost + lam_t * np.abs(np.arange(W)[None, :] - prev[:, None])
    acc = cost.copy()
    back = np.zeros((H, W), np.int16)
    big = np.float32(1e18)
    for y in range(1, H):
        p = acc[y - 1]
        cands = np.full((2 * step + 1, W), big, np.float32)
        for k, d in enumerate(range(-step, step + 1)):
            if d < 0:
                cands[k, -d:] = p[:d] + abs(d) * 0.02
            elif d > 0:
                cands[k, :-d] = p[d:] + d * 0.02
            else:
                cands[k] = p
        k = cands.argmin(axis=0)
        acc[y] += cands[k, np.arange(W)]
        back[y] = k - step
    path = np.zeros(H, np.int32)
    path[-1] = int(acc[-1].argmin())
    for y in range(H - 1, 0, -1):
        path[y - 1] = path[y] + back[y, path[y]]
    return path


class SeamTracker:
    """Curved seam between a left and a right take, tracked frame to frame."""

    def __init__(self, band: tuple[float, float], width: int):
        self.lo = int(band[0] * width)
        self.hi = int(band[1] * width)
        self.default = (self.lo + self.hi) / 2
        self.prev: np.ndarray | None = None

    def update(self, left: np.ndarray, right: np.ndarray, ref: np.ndarray,
               w_left: float, w_right: float, props_left: np.ndarray | None = None,
               props_right: np.ndarray | None = None, prop_weight: float = 8.0) -> np.ndarray:
        """Seam x per analysis row (analysis-resolution columns)."""
        sl = slice(self.lo, self.hi)
        fg_l = np.maximum(np.abs(left[:, sl] - ref[:, sl]).sum(-1) - 40, 0)
        fg_r = np.maximum(np.abs(right[:, sl] - ref[:, sl]).sum(-1) - 40, 0)
        if props_left is not None:
            fg_l = fg_l * (1 + prop_weight * props_left[:, sl] / max(w_left, 1))
        if props_right is not None:
            fg_r = fg_r * (1 + prop_weight * props_right[:, sl] / max(w_right, 1))
        diff = np.maximum(np.abs(left[:, sl] - right[:, sl]).sum(-1) - 30, 0)
        # Seam at column x: left take shows [0, x), right take shows [x, W).
        hide_r = np.concatenate([np.zeros((fg_r.shape[0], 1)), np.cumsum(fg_r, axis=1)], axis=1)[:, :-1]
        cl = np.cumsum(fg_l, axis=1)
        hide_l = cl[:, -1:] - np.concatenate([np.zeros((fg_l.shape[0], 1)), cl], axis=1)[:, :-1]
        cost = w_right * hide_r + w_left * hide_l + 2.0 * diff
        cost = cost / 255.0
        W = cost.shape[1]
        cost += 0.002 * np.abs(np.arange(W) + self.lo - self.default)[None, :]
        prev = None if self.prev is None else self.prev - self.lo
        path = _row_seam(cost.astype(np.float32), prev, lam_t=0.08) + self.lo
        if self.prev is not None:
            path = 0.6 * path + 0.4 * self.prev  # temporal smoothing
        self.prev = path.astype(np.float32)
        return self.prev


def _ramp(seam_rows: np.ndarray, W: int, H: int, feather: float) -> np.ndarray:
    """(H, W, 1) alpha: 0 left of the seam, 1 right of it, linear over `feather` px."""
    ys = np.linspace(0, len(seam_rows) - 1, H)
    xs = np.interp(ys, np.arange(len(seam_rows)), seam_rows) * _DS
    k = 41
    xs = np.convolve(np.pad(xs, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    x = np.arange(W, dtype=np.float32)[None, :]
    a = np.clip((x - xs[:, None]) / feather + 0.5, 0, 1)
    return a[..., None].astype(np.float32)


# ---------------------------------------------------------------- rendering


class _Reader:
    def __init__(self, raw: Path, start: float, n: int, W: int, H: int, rate: float = 1.0):
        vf = f"setpts=(PTS-STARTPTS)/{rate:.5f},fps={FPS},scale={W}:{H}" if rate != 1 else f"fps={FPS},scale={W}:{H}"
        self.p = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(raw), "-t", f"{n / FPS * rate + 0.5:.3f}",
             "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,  # it's closed early on purpose
        )
        self.size = W * H * 3
        self.shape = (H, W, 3)
        self.last: np.ndarray | None = None

    def read(self) -> np.ndarray:
        buf = self.p.stdout.read(self.size)
        if len(buf) == self.size:
            self.last = np.frombuffer(buf, np.uint8).reshape(self.shape)
        return self.last

    def close(self) -> None:
        self.p.stdout.close()
        self.p.wait()


def render_multi(raw: str | Path, plan: MPlan, output: str | Path, *, width: int = 1080, height: int = 1920,
                 feather: float = 40, zoom: float = 1.6, label_y: float = 0.39, label_size: int = 52,
                 speaker_weight: float = 3.0, crf: int = 20, preset: str = "medium",
                 on_progress: Callable[[float], None] | None = None) -> Path:
    raw, output = Path(raw), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    W, H = width, height
    names = [c.name for c in plan.chars]
    gains = {n: np.array(plan.gains.get(n, [1, 1, 1]), np.float32) for n in names}
    total = sum(s.duration for s in plan.segments)
    plate, props = background_plate(raw, plan.chars, plan.gains)
    if plate.shape[:2] != (H // _DS, W // _DS):
        raise ValueError(f"render size must be {plate.shape[1] * _DS}x{plate.shape[0] * _DS} for seam analysis")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        labels = {}
        for c in plan.chars:
            if c.label:
                from PIL import Image

                png = tmp / f"label_{c.name}.png"
                _label_png(png, c.label, label_size)
                img = np.asarray(Image.open(png).convert("RGBA"), np.float32)
                labels[c.name] = img

        video = tmp / "video.mp4"
        enc = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
             "-pix_fmt", "yuv420p", str(video)],
            stdin=subprocess.PIPE,
        )
        done = 0
        nframes_total = sum(round(s.duration * FPS) for s in plan.segments)
        for seg in plan.segments:
            n = max(1, round(seg.duration * FPS))
            spk = plan.char(seg.char)
            if seg.zoom:
                r = _Reader(raw, seg.start, n, W, H)
                zw, zh = int(W / zoom), int(H / zoom)
                x0 = int(min(max(spk.center * W - zw / 2, 0), W - zw))
                y0 = int(min(max(H * 0.2 - zh / 2, 0), H - zh))
                for _ in range(n):
                    f = r.read().astype(np.float32) * gains[spk.name]
                    crop = np.clip(f[y0:y0 + zh, x0:x0 + zw], 0, 255).astype(np.uint8)
                    from PIL import Image

                    out = np.asarray(Image.fromarray(crop).resize((W, H), Image.LANCZOS))
                    enc.stdin.write(out.tobytes())
                    done += 1
                r.close()
            else:
                readers = {}
                for c in plan.chars:
                    if c.name == seg.char:
                        readers[c.name] = _Reader(raw, seg.start, n, W, H)
                    else:
                        st, rate = seg.listen[c.name]
                        readers[c.name] = _Reader(raw, st, n, W, H, rate)
                trackers = [SeamTracker(tuple(b), W // _DS) for b in plan.bands]
                for _ in range(n):
                    fr = {k: np.clip(r.read().astype(np.float32) * gains[k], 0, 255) for k, r in readers.items()}
                    small = {k: v[::_DS, ::_DS] for k, v in fr.items()}
                    out = fr[names[0]]
                    for i, tr in enumerate(trackers):
                        left, right = names[i], names[i + 1]
                        wl = speaker_weight if left == seg.char else 1.0
                        wr = speaker_weight if right == seg.char else 1.0
                        seam = tr.update(small[left], small[right], plate, wl, wr, props[left], props[right])
                        a = _ramp(seam, W, H, feather)
                        out = out + a * (fr[right] - out)
                    for c in plan.chars:
                        if c.name in labels:
                            lab = labels[c.name]
                            lh, lw = lab.shape[:2]
                            x0 = int(c.center * W - lw / 2)
                            y0 = int(label_y * H - lh / 2)
                            x0 = min(max(x0, 0), W - lw)
                            al = lab[..., 3:4] / 255
                            out[y0:y0 + lh, x0:x0 + lw] = out[y0:y0 + lh, x0:x0 + lw] * (1 - al) + lab[..., :3] * al
                    enc.stdin.write(np.clip(out, 0, 255).astype(np.uint8).tobytes())
                    done += 1
                for r in readers.values():
                    r.close()
            if on_progress:
                on_progress(0.9 * done / nframes_total)
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError("video encode failed")

        # Audio: each segment's speaker audio, exactly as long as its video.
        parts, filters = [], []
        for k, seg in enumerate(plan.segments):
            d = round(seg.duration * FPS) / FPS
            fade = min(0.012, d / 4)
            filters.append(
                f"[0:a]atrim={seg.start:.4f}:{seg.start + d:.4f},asetpts=PTS-STARTPTS,"
                f"apad=whole_dur={d:.4f},atrim=0:{d:.4f},afade=t=in:d={fade},"
                f"afade=t=out:st={d - fade:.4f}:d={fade}[a{k}]"
            )
            parts.append(f"[a{k}]")
        filters.append("".join(parts) + f"concat=n={len(parts)}:v=0:a=1[a]")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-i", str(raw), "-i", str(video),
               "-filter_complex", ";".join(filters), "-map", "1:v", "-map", "[a]",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
               "-progress", "pipe:1", "-nostats", str(output)]
        _run_ffmpeg(cmd, total, None)
    if on_progress:
        on_progress(1.0)
    return output
