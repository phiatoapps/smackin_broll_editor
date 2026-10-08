"""Split-screen "talking to yourself" edit from one raw clip.

The raw clip is one locked-off shot: in the first half you play one character on
one side of the frame, in the second half you play the other character on the
other side. Given the script, this module:

1. finds where the raw clip switches halves and which side came first
   (from where the motion is),
2. finds the seam between the two positions (the column with the least motion
   from either take), so the mask doesn't cut through anyone,
3. transcribes the raw clip and finds every script line in its speaker's half,
4. lays the lines out back to back in script order: the speaker's side plays
   that line, the other side plays the other take's "listening" footage,
5. masks the two takes together with a soft vertical seam and renders 9:16.

Script format, one line per line of dialogue:

    Toilet Brush: I'm the toilet brush.
    Lysol: That's disgusting.
    [zoom] Toilet Brush: I live in a puddle of my own water.

`[zoom]` (or `*`) at the start of a line punches in full-frame on the speaker.
Lines without a "Name:" prefix continue the previous speaker's line; if the
script has no names at all, lines alternate between two speakers.
"""

from __future__ import annotations

import difflib
import json
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .captions import FONTS_DIR, Word
from .stacker import _even, _run_ffmpeg, probe

SIDES = ("left", "right")


@dataclass
class Line:
    speaker: str
    text: str
    zoom: bool = False


@dataclass
class SplitOptions:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    # Seconds of the raw clip where the second half starts (None = auto-detect).
    switch_time: float | None = None
    # Side of the frame you are on in the first half (None = auto-detect).
    first_side: str | None = None
    # Seam position as a fraction of the frame width (None = auto-detect).
    seam: float | None = None
    # Width of the soft edge on the seam, fraction of frame width.
    feather: float = 0.10
    # Extra time kept before/after each line's words.
    pad_before: float = 0.08
    pad_after: float = 0.15
    # Punch-in amount for [zoom] lines.
    zoom: float = 1.6
    # Optional on-screen names over each character; "" = none.
    left_label: str = ""
    right_label: str = ""
    label_y: float = 0.39
    label_size: int = 52
    whisper_model: str = "small"
    language: str | None = "en"
    crf: int = 18
    preset: str = "medium"

    def validate(self) -> None:
        if self.first_side not in (None, *SIDES):
            raise ValueError("first_side must be 'left' or 'right'")
        if self.seam is not None and not 0.1 <= self.seam <= 0.9:
            raise ValueError("seam must be between 0.1 and 0.9")
        if not 0 <= self.feather <= 0.3:
            raise ValueError("feather must be between 0 and 0.3")
        if self.zoom < 1:
            raise ValueError("zoom must be at least 1")


# ---------------------------------------------------------------- script

_ZOOM_RE = re.compile(r"^\s*(\[zoom\]|\*)\s*", re.I)
_SPEAKER_RE = re.compile(r"^\s*([^:\n]{1,40}?)\s*:\s*(.+)$")


def parse_script(text: str) -> list[Line]:
    raw = [ln for ln in text.splitlines() if ln.strip()]
    named = sum(bool(_SPEAKER_RE.match(_ZOOM_RE.sub("", ln))) for ln in raw)
    lines: list[Line] = []
    for i, ln in enumerate(raw):
        zoom = bool(_ZOOM_RE.match(ln))
        ln = _ZOOM_RE.sub("", ln)
        m = _SPEAKER_RE.match(ln)
        if named == 0:
            lines.append(Line("A" if i % 2 == 0 else "B", ln.strip(), zoom))
        elif m:
            lines.append(Line(m.group(1).strip(), m.group(2).strip(), zoom))
        elif lines:
            lines[-1].text += " " + ln.strip()
            lines[-1].zoom |= zoom
        else:
            raise ValueError(f"First script line has no 'Name:' prefix: {ln!r}")
    speakers = list(dict.fromkeys(l.speaker for l in lines))
    if len(speakers) != 2:
        raise ValueError(f"Script needs exactly 2 speakers, found {len(speakers)}: {speakers}")
    return lines


def _tokens(text: str) -> list[str]:
    text = text.lower().replace("’", "'").replace("-", " ")
    return re.findall(r"[a-z0-9']+", text)


# ---------------------------------------------------------------- motion analysis

_AW, _AH, _AFPS = 72, 128, 10  # analysis frame size / rate


def _motion(raw: Path) -> np.ndarray:
    """(frames, columns) per-column motion energy of a small grayscale copy."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(raw), "-vf",
         f"fps={_AFPS},scale={_AW}:{_AH},format=gray", "-f", "rawvideo", "-"],
        check=True, capture_output=True,
    ).stdout
    frames = np.frombuffer(out, np.uint8).reshape(-1, _AH, _AW).astype(np.float32)
    diff = np.abs(np.diff(frames, axis=0))
    diff = np.maximum(diff - 6, 0)  # ignore sensor noise / compression shimmer
    return diff.sum(axis=1)


def analyze_layout(raw: Path, opts: SplitOptions) -> tuple[float, str, float]:
    """Return (switch_time, first_side, seam) filling in whatever opts leaves as None."""
    need = opts.switch_time is None or opts.first_side is None or opts.seam is None
    if not need:
        return opts.switch_time, opts.first_side, opts.seam
    m = _motion(raw)
    n = len(m)
    cols = np.arange(_AW)
    left = m[:, cols < _AW // 2].sum(axis=1)
    right = m[:, cols >= _AW // 2].sum(axis=1)

    switch, first = opts.switch_time, opts.first_side
    if switch is None:
        # Changepoint: before t one side moves more, after t the other side does.
        c = np.concatenate([[0], np.cumsum(left - right)])
        idx = np.arange(int(n * 0.15), int(n * 0.85))
        score = c[idx] - (c[-1] - c[idx])  # large = left first, small = right first
        if first is None:
            first = "left" if score.max() >= -score.min() else "right"
        switch = idx[np.argmax(score) if first == "left" else np.argmin(score)] / _AFPS
    k = min(max(1, int(round(switch * _AFPS))), n - 1)
    if first is None:
        before = left[:k].mean() - right[:k].mean()
        after = left[k:].mean() - right[k:].mean()
        first = "left" if before >= after else "right"

    seam = opts.seam
    if seam is None:
        a = m[:k].sum(axis=0)  # motion of first take per column
        b = m[k:].sum(axis=0)
        a, b = a / (a.sum() or 1), b / (b.sum() or 1)
        if first == "right":
            a, b = b, a  # a = left character's take
        # Seam at x: left character's motion right of x + right character's motion left of x.
        cost = [a[x:].sum() + b[:x].sum() for x in range(_AW + 1)]
        lo, hi = int(_AW * 0.2), int(_AW * 0.8)
        x = lo + int(np.argmin(cost[lo:hi + 1]))
        seam = x / _AW
    return float(switch), first, float(seam)


# ---------------------------------------------------------------- alignment


@dataclass
class Match:
    line: int
    start: float
    end: float
    score: float


def _candidates(line_tok: list[str], word_tok: list[str], thr: float) -> list[tuple[int, int, float]]:
    k = len(line_tok)
    if k == 0:
        return []
    found = []
    for size in sorted({max(1, k + d) for d in (-2, -1, 0, 1, 2)}):
        for i in range(0, max(1, len(word_tok) - size + 1)):
            window = word_tok[i:i + size]
            if not window:
                continue
            r = difflib.SequenceMatcher(None, line_tok, window, autojunk=False).ratio()
            if r >= thr:
                found.append((i, i + size - 1, r))
    # Keep the best window around each spot (non-max suppression on overlap).
    found.sort(key=lambda c: -c[2])
    kept: list[tuple[int, int, float]] = []
    for c in found:
        if all(c[1] < o[0] or c[0] > o[1] for o in kept):
            kept.append(c)
    return sorted(kept)


def align(lines: list[Line], words: list[Word], first_speaker: str, switch: float,
          thr: float = 0.55) -> list[Match | None]:
    """Find each script line in its speaker's half of the transcript.

    Lines are kept in script order within each half; when a line was said more
    than once (a retake), the later take wins.
    """
    halves = {
        first_speaker: [w for w in words if w.start < switch],
        "_second": [w for w in words if w.start >= switch],
    }
    result: list[Match | None] = [None] * len(lines)
    for key, hw in halves.items():
        idxs = [i for i, l in enumerate(lines) if (l.speaker == first_speaker) == (key == first_speaker)]
        # Flatten to tokens, remembering which word each token came from.
        wt, owner = [], []
        for wi, w in enumerate(hw):
            for t in _tokens(w.text):
                wt.append(t)
                owner.append(wi)
        cands = [_candidates(_tokens(lines[i].text), wt, thr) for i in idxs]
        # DP over lines: at most one candidate per line, strictly increasing in time,
        # maximizing total match score; ties go to later windows (retakes).
        frontier: dict[int, tuple[float, float, list]] = {-1: (0.0, 0.0, [])}
        for li, cs in enumerate(cands):
            new = {}
            for last, (sc, tb, path) in frontier.items():
                # skip this line
                cand_state = (sc, tb, path + [None])
                if last not in new or cand_state[:2] > new[last][:2]:
                    new[last] = cand_state
                for c in cs:
                    if c[0] > last:
                        st = (sc + c[2], tb + c[0] * 1e-6, path + [c])
                        if c[1] not in new or st[:2] > new[c[1]][:2]:
                            new[c[1]] = st
            frontier = new
        _, _, path = max(frontier.values(), key=lambda s: s[:2])
        for li, c in zip(idxs, path):
            if c is not None:
                result[li] = Match(li, hw[owner[c[0]]].start, hw[owner[c[1]]].end, c[2])
    return result


# ---------------------------------------------------------------- edit plan


@dataclass
class Segment:
    line: int
    speaker_side: str
    start: float            # speaker footage/audio start in raw clip
    duration: float
    listen_start: float     # other side's footage start in raw clip
    listen_rate: float = 1.0  # <1 = listening footage slowed to fill a short gap
    zoom: bool = False
    text: str = ""
    seam: float | None = None  # this segment's seam (None = the plan's seam)


@dataclass
class Plan:
    switch_time: float
    first_side: str
    seam: float
    sides: dict[str, str]   # speaker name -> side
    segments: list[Segment] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    # RGB gains applied to second-half footage so both takes' lighting matches.
    second_gain: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Plan":
        d = json.loads(Path(path).read_text())
        d["segments"] = [Segment(**s) for s in d["segments"]]
        return cls(**d)


def _first_speaker(lines: list[Line], words: list[Word], switch: float) -> str:
    """The speaker whose lines match the first half's transcript better."""
    first = _tokens(" ".join(w.text for w in words if w.start < switch))
    speakers = list(dict.fromkeys(l.speaker for l in lines))

    def overlap(sp: str) -> float:
        toks = _tokens(" ".join(l.text for l in lines if l.speaker == sp))
        return difflib.SequenceMatcher(None, toks, first, autojunk=False).find_longest_match(
            0, len(toks), 0, len(first)).size + sum(t in set(first) for t in toks) / (len(toks) or 1)

    return max(speakers, key=overlap)


@dataclass
class Take:
    """One piece of footage for the edit: the speaker's side plays raw[start:end]."""
    line: int
    side: str
    start: float
    end: float
    zoom: bool = False
    text: str = ""


def _speech_intervals(words: list[Word], lo: float, hi: float, margin: float = 0.15) -> list[tuple[float, float]]:
    """Merged time ranges in [lo, hi) where anything was said (retakes and ad-libs included)."""
    out: list[list[float]] = []
    for w in words:
        if lo <= w.start < hi:
            a, b = w.start - margin, w.end + margin
            if out and a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
    return [(a, b) for a, b in out]


def schedule(takes: list[Take], words: list[Word], duration: float, switch: float,
             first_side: str, seam: float, sides: dict[str, str], missing: list[str] | None = None) -> Plan:
    """Lay takes back to back and pick each one's listening footage from the other take.

    Listening footage never overlaps anything said in that half (so no silent
    mouthing of a retake), runs continuously while the other character keeps
    talking, and the first listening shot ends just before that side's first line.
    """
    second_side = "right" if first_side == "left" else "left"
    halves = {first_side: (0.0, switch), second_side: (switch, duration)}
    speech = {s: _speech_intervals(words, *halves[s]) for s in SIDES}
    durs = [t.end - t.start for t in takes]

    cursor = {}
    for side in SIDES:
        lo, hi = halves[side]
        own = [k for k, t in enumerate(takes) if t.side == side]
        if own:
            lead = sum(durs[:own[0]])
            cursor[side] = max(lo, takes[own[0]].start - lead - 0.2)
        else:
            cursor[side] = lo

    plan = Plan(switch, first_side, seam, sides, missing=list(missing or []))
    for t, dur in zip(takes, durs):
        listener = second_side if t.side == first_side else first_side
        lo, hi = halves[listener]
        c = cursor[listener]
        for a, b in speech[listener]:  # step out of speech
            if a <= c < b:
                c = b
        prev_end = max([b for a, b in speech[listener] if b <= c + 1e-3] + [lo])
        next_start = min([a for a, b in speech[listener] if a >= c - 1e-3] + [hi])
        start, rate = c, 1.0
        if next_start - start < dur:
            start = max(prev_end, next_start - dur)
            avail = next_start - start
            if 0.3 < avail < dur:
                rate = max(0.5, avail / dur)  # slow idle footage up to 2x rather than show lips moving
        cursor[listener] = start + dur * rate
        cursor[t.side] = t.end
        plan.segments.append(Segment(t.line, t.side, t.start, dur, start, rate, t.zoom, t.text))
    return plan


def build_plan(lines: list[Line], words: list[Word], duration: float,
               switch: float, first_side: str, seam: float, opts: SplitOptions) -> Plan:
    first_speaker = _first_speaker(lines, words, switch)
    other = next(l.speaker for l in lines if l.speaker != first_speaker)
    second_side = "right" if first_side == "left" else "left"
    sides = {first_speaker: first_side, other: second_side}
    matches = align(lines, words, first_speaker, switch)
    halves = {first_side: (0.0, switch), second_side: (switch, duration)}

    def padded(side: str, m: Match) -> tuple[float, float]:
        """Pad a line's words, without running into neighbouring speech."""
        lo, hi = halves[side]
        hw = [w for w in words if lo <= w.start < hi]
        before = [w.end for w in hw if w.end <= m.start - 0.01]
        after = [w.start for w in hw if w.start >= m.end + 0.01]
        a = max(m.start - opts.pad_before, (before[-1] + 0.02) if before else lo, lo)
        b = min(m.end + opts.pad_after, (after[0] - 0.02) if after else hi, hi)
        return a, max(b, a + 0.2)

    takes, missing = [], []
    for i, (line, m) in enumerate(zip(lines, matches)):
        if m is None:
            missing.append(f"{line.speaker}: {line.text}")
            continue
        side = sides[line.speaker]
        a, b = padded(side, m)
        takes.append(Take(i, side, a, b, line.zoom, line.text))
    return schedule(takes, words, duration, switch, first_side, seam, sides, missing)


# ---------------------------------------------------------------- lighting match and per-segment seams

_SW, _SH = 135, 240  # sampling size for color/seam analysis


def _sample(raw: Path, start: float, dur: float, fps: float) -> np.ndarray:
    """(n, _SH, _SW, 3) float RGB frames from raw[start:start+dur]."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, start):.3f}", "-t", f"{dur:.3f}", "-i", str(raw),
         "-vf", f"fps={fps},scale={_SW}:{_SH}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, _SH, _SW, 3).astype(np.float32)


def match_lighting(raw: str | Path, plan: Plan, duration: float) -> list[float]:
    """RGB gains that make second-half footage match the first half's exposure / white balance.

    Compares the two halves' median frames on pixels that are static background in
    both (no one moving there, and both halves agree on what's there).
    """
    raw = Path(raw)
    a = _sample(raw, 0, plan.switch_time, 1)
    b = _sample(raw, plan.switch_time, duration - plan.switch_time, 1)
    ma, mb = np.median(a, axis=0), np.median(b, axis=0)
    still = (a.std(axis=0).max(-1) < 6) & (b.std(axis=0).max(-1) < 6)
    la, lb = ma.mean(-1), mb.mean(-1)
    ratio = la / np.maximum(lb, 1)
    ok = still & (la > 20) & (lb > 20) & (la < 235) & (lb < 235)
    if ok.sum() < 200:
        return [1.0, 1.0, 1.0]
    r0 = np.median(ratio[ok])
    ok &= np.abs(ratio - r0) < 0.12 * r0  # drop pixels where a person sat still in one half
    gains = [float(np.median(ma[..., c][ok] / np.maximum(mb[..., c][ok], 1))) for c in range(3)]
    return [round(min(max(g, 0.7), 1.4), 4) for g in gains]


def choose_seams(raw: str | Path, plan: Plan, opts: SplitOptions, reach: float = 0.18) -> None:
    """Give each segment its own seam: the column where the two takes look most alike.

    A person (or an arm reaching across) makes the takes differ, so the seam slides
    away from them. Uses the worst frame in the segment, over the whole soft-edge
    width, so a quick lean still counts. Stays within `reach` of the plan's seam.
    """
    raw = Path(raw)
    gain = np.array(plan.second_gain, np.float32)
    fw = max(3, int(round(opts.feather * _SW)))
    x = np.arange(_SW) / _SW
    for seg in plan.segments:
        if seg.zoom:
            continue
        a = _sample(raw, seg.start, seg.duration, 6)
        b = _sample(raw, seg.listen_start, seg.duration * seg.listen_rate, 6 / seg.listen_rate)
        n = min(len(a), len(b))
        if n == 0:
            continue
        a, b = a[:n], b[:n]
        if seg.start >= plan.switch_time:
            a = a * gain
        if seg.listen_start >= plan.switch_time:
            b = b * gain
        diff = np.abs(a - b).sum(-1)                          # (n, H, W)
        diff = np.maximum(diff - 24, 0)                       # ignore noise / leftover lighting drift
        col = diff.mean(axis=1)                               # (n, W)
        kernel = np.ones(fw * 2 + 1) / (fw * 2 + 1)           # cover the feathered band + margin
        col = np.stack([np.convolve(c, kernel, mode="same") for c in col])
        cost = np.percentile(col, 95, axis=0)
        cost = cost / (cost.max() or 1) + 0.15 * np.abs(x - plan.seam) / reach
        cost[np.abs(x - plan.seam) > reach] = np.inf
        seg.seam = round(float(x[int(np.argmin(cost))]), 4)


# ---------------------------------------------------------------- rendering


def _mask_pgm(path: Path, w: int, h: int, seam: float, feather: float) -> None:
    """Grayscale mask: black = left take, white = right take, soft ramp at the seam."""
    x = np.arange(w, dtype=np.float32)
    fw = max(1.0, feather * w)
    row = np.clip((x - (seam * w - fw / 2)) / fw, 0, 1) * 255
    img = np.tile(row.astype(np.uint8), (h, 1))
    path.write_bytes(f"P5 {w} {h} 255\n".encode() + img.tobytes())


def _label_png(path: Path, text: str, size: int) -> None:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(FONTS_DIR / "Montserrat-ExtraBold.ttf"), size)
    lines = text.split("\\n") if "\\n" in text else text.split("|")
    tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    boxes = [tmp.textbbox((0, 0), ln, font=font) for ln in lines]
    lh = int(size * 1.1)
    tw = max(b[2] - b[0] for b in boxes)
    padx, pady = int(size * 0.35), int(size * 0.2)
    W, H = tw + 2 * padx, lh * len(lines) + 2 * pady
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, W - 1, H - 1), radius=int(size * 0.3), fill=(255, 255, 255, 255))
    for j, (ln, b) in enumerate(zip(lines, boxes)):
        d.text(((W - (b[2] - b[0])) / 2 - b[0], pady + j * lh + (lh - size) / 2 - b[1] * 0.5),
               ln, font=font, fill=(0, 0, 0, 255))
    img.save(path)


def _gain_filter(plan: Plan, t: float) -> str:
    """Color correction for footage starting at raw time t (only second-half footage is adjusted)."""
    r, g, b = plan.second_gain
    if t < plan.switch_time or (r, g, b) == (1.0, 1.0, 1.0):
        return ""
    return f",colorchannelmixer=rr={r:.4f}:gg={g:.4f}:bb={b:.4f}"


def _render_segment(raw: Path, seg: Segment, plan: Plan, opts: SplitOptions, out: Path,
                    mask: Path, labels: dict[str, Path]) -> None:
    W, H, fps = _even(opts.width), _even(opts.height), opts.fps
    _mask_pgm(mask, W, H, plan.seam if seg.seam is None else seg.seam, opts.feather)
    frames = max(1, round(seg.duration * fps))
    dur = frames / fps
    fit = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,fps={fps}"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-v", "error",
           "-ss", f"{seg.start:.3f}", "-t", f"{dur + 0.5:.3f}", "-i", str(raw)]
    filters = []
    if seg.zoom:
        # Punch in on the speaker: crop around the middle of their side, upper part of frame.
        zw, zh = W / opts.zoom, H / opts.zoom
        cx = plan.seam * W / 2 if seg.speaker_side == "left" else (1 + plan.seam) * W / 2
        x = min(max(cx - zw / 2, 0), W - zw)
        y = min(max(H * 0.36 - zh / 2, 0), H - zh)
        filters.append(f"[0:v]{fit}{_gain_filter(plan, seg.start)},"
                       f"crop={zw:.0f}:{zh:.0f}:{x:.0f}:{y:.0f},scale={W}:{H},setsar=1[v0]")
    else:
        cmd += ["-ss", f"{seg.listen_start:.3f}", "-t", f"{dur * seg.listen_rate + 0.5:.3f}", "-i", str(raw),
                "-loop", "1", "-i", str(mask)]
        slow = f",setpts=PTS/{seg.listen_rate:.4f}" if seg.listen_rate != 1 else ""
        filters.append(f"[1:v]setpts=PTS-STARTPTS{slow},{fit}{_gain_filter(plan, seg.listen_start)}[lst]")
        filters.append(f"[0:v]setpts=PTS-STARTPTS,{fit}{_gain_filter(plan, seg.start)}[spk]")
        lft = "[spk]" if seg.speaker_side == "left" else "[lst]"
        rgt = "[lst]" if seg.speaker_side == "left" else "[spk]"
        filters.append(f"[2:v]scale={W}:{H},format=gray[m]")
        filters.append(f"{rgt}[m]alphamerge[ra]")
        filters.append(f"{lft}[ra]overlay=shortest=1[v0]")
        last = "v0"
        n_in = 3
        for side, png in labels.items():
            cmd += ["-i", str(png)]
            cx = plan.seam * W / 2 if side == "left" else (1 + plan.seam) * W / 2
            filters.append(f"[{last}][{n_in}:v]overlay=x={cx:.0f}-w/2:y={opts.label_y * H:.0f}-h/2[v{n_in}]")
            last = f"v{n_in}"
            n_in += 1
        if last != "v0":
            filters.append(f"[{last}]null[v0]")
    fade = min(0.015, dur / 4)
    filters.append(
        f"[0:a]asetpts=PTS-STARTPTS,atrim=0:{dur:.4f},apad=whole_dur={dur:.4f},"
        f"afade=t=in:d={fade},afade=t=out:st={dur - fade:.4f}:d={fade},aresample=48000[a]"
    )
    cmd += ["-filter_complex", ";".join(filters), "-map", "[v0]", "-map", "[a]",
            "-frames:v", str(frames), "-c:v", "libx264", "-preset", opts.preset, "-crf", str(opts.crf),
            "-pix_fmt", "yuv420p", "-r", str(fps), "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on line {seg.line + 1}:\n{r.stderr[-3000:]}")


def render_plan(raw: str | Path, plan: Plan, output: str | Path, opts: SplitOptions,
                on_progress: Callable[[float], None] | None = None) -> Path:
    raw, output = Path(raw), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    info = probe(raw)
    if not info.has_audio:
        raise ValueError("raw clip has no audio track")
    if not plan.segments:
        raise ValueError("no script lines were found in the raw clip")
    W, H = _even(opts.width), _even(opts.height)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mask = tmp / "mask.pgm"
        labels = {}
        for side, text in (("left", opts.left_label), ("right", opts.right_label)):
            if text:
                labels[side] = tmp / f"label_{side}.png"
                _label_png(labels[side], text, opts.label_size)
        parts = []
        total = sum(s.duration for s in plan.segments)
        done = 0.0
        for k, seg in enumerate(plan.segments):
            part = tmp / f"seg{k:04d}.mkv"
            _render_segment(raw, seg, plan, opts, part, mask, labels)
            parts.append(part)
            done += seg.duration
            if on_progress:
                on_progress(0.95 * done / total)
        listing = tmp / "list.txt"
        listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
        cmd = ["ffmpeg", "-y", "-hide_banner", "-f", "concat", "-safe", "0", "-i", str(listing),
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
               "-progress", "pipe:1", "-nostats", str(output)]
        _run_ffmpeg(cmd, total, None)
    if on_progress:
        on_progress(1.0)
    return output


def make_split_screen(
    raw: str | Path,
    script: str,
    output: str | Path,
    opts: SplitOptions | None = None,
    transcript: str | Path | None = None,
    plan_path: str | Path | None = None,
    on_status: Callable[[str], None] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> tuple[Path, Plan]:
    """Full pipeline. `transcript` / `plan_path` are reused if they exist, else saved there."""
    from . import captions

    opts = opts or SplitOptions()
    opts.validate()
    raw = Path(raw)
    if plan_path and Path(plan_path).is_file():
        plan = Plan.load(plan_path)
    else:
        lines = parse_script(script)
        info = probe(raw)
        if on_status:
            on_status("analyzing")
        switch, first_side, seam = analyze_layout(raw, opts)
        if transcript and Path(transcript).is_file():
            words = captions.load_transcript(transcript)
        else:
            if on_status:
                on_status("transcribing")
            words = captions.transcribe(raw, opts.whisper_model, opts.language)
            if transcript:
                captions.save_transcript(words, transcript)
        plan = build_plan(lines, words, info.duration, switch, first_side, seam, opts)
        if on_status:
            on_status("matching lighting and seams")
        plan.second_gain = match_lighting(raw, plan, info.duration)
        choose_seams(raw, plan, opts)
        if plan_path:
            plan.save(plan_path)
    if on_status:
        on_status("rendering")
    return render_plan(raw, plan, output, opts, on_progress), plan


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from .sources import fetch

    p = argparse.ArgumentParser(
        prog="python -m broll_editor.splitscreen",
        description="Turn one raw clip (you on one side, then the other) plus its script into a split-screen dialogue.",
    )
    p.add_argument("raw", help="raw clip: file path or Google Drive link")
    p.add_argument("script", help="script .txt file, one 'Name: line' per line")
    p.add_argument("-o", "--output", default="output/split.mp4")
    p.add_argument("--size", default="1080x1920")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--switch", type=float, default=None, help="seconds where the second half starts (default: auto)")
    p.add_argument("--first-side", choices=SIDES, default=None, help="your side in the first half (default: auto)")
    p.add_argument("--seam", type=float, default=None, help="mask seam, fraction of width (default: auto)")
    p.add_argument("--feather", type=float, default=0.10, help="soft edge width, fraction of width")
    p.add_argument("--pad-before", type=float, default=0.08)
    p.add_argument("--pad-after", type=float, default=0.15)
    p.add_argument("--zoom", type=float, default=1.6, help="punch-in for [zoom] lines")
    p.add_argument("--left-label", default="", help="name shown over the left character ('|' = new line)")
    p.add_argument("--right-label", default="")
    p.add_argument("--label-y", type=float, default=0.39)
    p.add_argument("--label-size", type=int, default=52)
    p.add_argument("--transcript", metavar="JSON", help="Whisper words: reused if it exists, else saved here")
    p.add_argument("--plan", metavar="JSON", help="edit plan: reused if it exists (edit it to tweak cuts), else saved")
    p.add_argument("--whisper-model", default="small")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--preset", default="medium")
    args = p.parse_args(argv)
    try:
        width, height = (int(v) for v in args.size.lower().split("x"))
    except ValueError:
        p.error("--size must look like 1080x1920")
    opts = SplitOptions(
        width=width, height=height, fps=args.fps, switch_time=args.switch, first_side=args.first_side,
        seam=args.seam, feather=args.feather, pad_before=args.pad_before, pad_after=args.pad_after,
        zoom=args.zoom, left_label=args.left_label, right_label=args.right_label, label_y=args.label_y,
        label_size=args.label_size, whisper_model=args.whisper_model, crf=args.crf, preset=args.preset,
    )

    def progress(frac: float) -> None:
        sys.stderr.write(f"\rRendering... {frac * 100:5.1f}%")
        sys.stderr.flush()

    try:
        with tempfile.TemporaryDirectory() as tmp:
            raw = fetch(args.raw, tmp, "raw")
            script = Path(args.script).read_text()
            out, plan = make_split_screen(
                raw, script, args.output, opts, transcript=args.transcript, plan_path=args.plan,
                on_status=lambda s: sys.stderr.write(f"{s.capitalize()}...\n"), on_progress=progress,
            )
    except Exception as exc:
        sys.stderr.write(f"\nError: {exc}\n")
        return 1
    sides = ", ".join(f"{name} = {side}" for name, side in plan.sides.items())
    sys.stderr.write(f"\nSwitch at {plan.switch_time:.1f}s, seam at {plan.seam:.0%}, {sides}\n")
    for m in plan.missing:
        sys.stderr.write(f"Not found in the raw clip, skipped: {m}\n")
    sys.stderr.write(f"Done: {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
