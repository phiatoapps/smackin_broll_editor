"""Three-or-more-character version of the split screen, for people standing close together.

Same idea as `splitscreen`: one locked-off raw clip in which you play each character
in turn, at a different spot in the frame. Here each character has a time range of
the raw clip (their take) and an order from left to right.

Seams between neighbours can't work when they overlap (a seam through someone's
shoulder blends it with another take's empty background: a see-through "ghost").
So every frame is layered instead:

* An empty-room plate (each spot taken from takes whose character stands far from
  it) is the reference. Wherever
  a take differs from it, that take has someone (or a prop) there: that's their
  silhouette, cleaned up and hole-filled.
* Background (including that take's props) comes from whichever take "owns" that
  part of the room, so props never pop in and out between cuts; people go on top
  in a fixed depth order. A person can only be covered by another person's body,
  never by background or a stray patch of another take.
* Every rendered frame is checked for see-through person pixels (written to a
  report), so problem spots can be found without watching the whole video.
* All takes are colour-matched to the first one so the background is seamless.
"""

from __future__ import annotations

import json
import os
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
_DS = 4  # analysis downscale factor for person masks


@dataclass
class Character:
    name: str
    lo: float          # take range in the raw clip
    hi: float
    center: float      # where they stand, fraction of frame width (for labels / zoom)
    label: str = ""
    face_y: float = 0.32  # face height, fraction of frame height (for tight framings)


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
    # "Camera" framing applied to the finished composite: [zoom, center_x, center_y]
    # (fractions of the frame). None = full frame.
    framing: list[float] | None = None
    # Animated pull-out at the start of the line: [start_zoom, seconds]. Used on the
    # opening line so the hook lands on everyone in frame.
    pull_out: list[float] | None = None
    # Unison line: everyone speaks at once. Listeners' footage (seg.listen) is their own
    # take of the same line, and their audio is mixed in with the speaker's.
    unison: bool = False


@dataclass
class MPlan:
    chars: list[Character]                      # left to right
    bands: list[list[float]]                    # allowed seam range between neighbours
    segments: list[MSegment] = field(default_factory=list)
    gains: dict[str, list[float]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    # Height of the counter/table top as a fraction of frame height. Things that stay
    # put during a take and stand on it are props (in front of people); things that
    # stay put above it without touching it (a moved blanket, a picture) are scenery
    # behind the people. None = treat everything that stays put as a prop.
    counter_y: float | None = None
    # Label changes mid-video: {name: [[line, new label], ...]} - from that script
    # line on, the character's label reads the new text (e.g. "PRO" pops on).
    label_changes: dict[str, list] = field(default_factory=dict)
    # Seconds of the raw clip showing the empty room (nobody in frame), if any: the most
    # reliable background, needed when the characters' spots overlap (stand-behind).
    empty: list[float] | None = None
    # Layer order, back to front (stand-behind: the one behind first). None = automatic.
    depth: list[str] | None = None
    # "mirror": clones side by side, layered by comparing against the empty room.
    # "stand_behind": the last character in `depth` is always in front, cut out with a
    # person-segmentation model; everyone else is simply the background footage.
    mode: str = "mirror"

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


def _sound_edge(env: np.ndarray, i: int, soft: float, direction: int, limit: int, hop: float,
                quiet: float = 0.03, loud: float = 0.015, hard_limit: int | None = None,
                tail: float = 0.12) -> int:
    """From frame i (edge of a loud run) find where to cut, never inside a word:
    1. walk while the sound is still speech-loud (> `loud`), up to `limit` (and on to
       `hard_limit` if the speech simply keeps going);
    2. then follow a soft ending (s/f/t tails, breathy onsets: > `soft`) for at most
       `tail` seconds, stopping at the first `quiet` stretch, else at the quietest point.
    Bounding step 2 keeps faint room noise from adding dead air."""
    n = len(env)
    lim = hard_limit if hard_limit is not None else limit
    k = i
    gap = int(round(0.06 / hop))  # a dip this short inside speech is mid-word ("dea-ls")
    while (k < lim if direction > 0 else k > lim) and 0 <= k < n:
        if env[k] > loud:
            k += direction
            continue
        ahead = env[k:k + gap + 1] if direction > 0 else env[max(0, k - gap):k + 1]
        if (ahead > loud).any():
            k += direction
            continue
        break
    need = max(1, int(round(quiet / hop)))
    end = k + direction * int(round(tail / hop))
    run, best, best_v = 0, k, np.inf
    j = k
    while (j < end if direction > 0 else j > end) and (j < lim if direction > 0 else j > lim) and 0 <= j < n:
        v = env[j]
        if v < best_v:
            best, best_v = j, v
        if v < soft:
            run += 1
            if run >= need:
                return j - direction * (need - 1)
        else:
            run = 0
        j += direction
    return best if best_v < np.inf else k


def tighten(plan: MPlan, env: np.ndarray, *, max_gap: float = 0.08, keep: float = 0.035, edge: float = 0.03,
            thr: float | None = None, soft: float | None = None, reach: float = 0.45, hop: float = 0.01,
            min_sound: float = 0.08, min_piece: float = 0.2, words: list | None = None) -> dict:
    """Cut dead air: every silence in the speaker's audio longer than `max_gap` is cut
    down to `keep` on each side, including at the start/end of each line (trimmed to
    `edge`). A line with a pause in it becomes several pieces back to back - the
    speaker jump-cuts, while everyone listening keeps playing continuously. Cuts
    follow soft word endings out to real quiet, so no word gets clipped.
    """
    if thr is None:
        thr = max(0.008, float(np.percentile(env, 20)) * 3)
    if soft is None:
        soft = max(0.007, float(np.percentile(env, 50)) * 1.6)  # soft endings, just above room noise
    loud = env > thr
    # bridge tiny dips inside words (plosives, s-sounds)
    k = int(round(0.05 / hop))
    loud = np.convolve(loud.astype(np.float32), np.ones(k), mode="same") > 0
    new, removed = [], 0.0
    for seg in plan.segments:
        if seg.unison:  # both voices must stay in sync: no jump cuts inside the unison line
            new.append(seg)
            continue
        a, b = int(round(seg.start / hop)), int(round((seg.start + seg.duration) / hop))
        # Words just outside this line (a retake, an ad-lib) are a hard wall: line
        # edges may grow to catch soft endings, but never into the next word.
        wall_lo, wall_hi = 0, len(env)
        if words:
            s0, s1 = seg.start, seg.start + seg.duration
            for w in words:
                ws_, we_ = (w.start, w.end) if hasattr(w, "start") else (w["start"], w["end"])
                if we_ <= s0 + 0.02:
                    wall_lo = max(wall_lo, int(round((we_ + 0.01) / hop)))
                if ws_ >= s1 - 0.02:
                    wall_hi = min(wall_hi, int(round((ws_ - 0.01) / hop)))
        m = loud[a:b]
        if not m.any():
            new.append(seg)
            continue
        # speech runs, merged across gaps <= max_gap
        runs, i = [], 0
        while i < len(m):
            if m[i]:
                j = i
                while j < len(m) and m[j]:
                    j += 1
                if runs and (i - runs[-1][1]) * hop <= max_gap:
                    runs[-1][1] = j
                else:
                    runs.append([i, j])
                i = j
            else:
                i += 1
        # drop clicks/breaths (too short to be a word); keep a short word with its
        # neighbour rather than making a flicker-length jump cut
        # Words peak well above a breath, and stay up for a syllable (a mouth click or
        # tap spikes loud for only a couple of hundredths of a second).
        runs = [r for r in runs if (r[1] - r[0]) * hop >= min_sound
                and env[a + r[0]:a + r[1]].max() >= 2.5 * thr
                and (env[a + r[0]:a + r[1]] > 2.5 * thr).sum() * hop >= 0.09] or runs
        merged = []
        for r in runs:
            if merged and ((merged[-1][1] - merged[-1][0]) * hop < min_piece or (r[1] - r[0]) * hop < min_piece):
                merged[-1][1] = r[1]
            else:
                merged.append(list(r))
        runs = merged
        # Cut points: follow each sound outward past the loud threshold while it stays
        # above the soft threshold (s/f/t/th endings, breathy onsets), then cut in the
        # quiet right after it. If speech runs straight into other speech (no quiet in
        # reach), cut at the quietest point in the gap instead. Never inside a word.
        pieces = []
        for idx, (i, j) in enumerate(runs):
            lead = edge if idx == 0 else keep
            tail = edge if idx == len(runs) - 1 else keep
            lo_lim = a - int(reach / hop) if idx == 0 else a + (runs[idx - 1][1] if idx else 0)
            hi_lim = b + int(reach / hop) if idx == len(runs) - 1 else a + runs[idx + 1][0]
            if idx == 0:
                lo_lim = max(lo_lim, min(wall_lo, a))
            if idx == len(runs) - 1:
                hi_lim = min(hi_lim, max(wall_hi, b))
            hard_lo = max(a - int(1.0 / hop), min(wall_lo, a)) if idx == 0 else lo_lim
            hard_hi = min(b + int(1.0 / hop), max(wall_hi, b)) if idx == len(runs) - 1 else hi_lim
            ps = _sound_edge(env, a + i, soft, -1, lo_lim, hop, loud=thr, hard_limit=hard_lo) * hop - lead
            pe = _sound_edge(env, a + j, soft, +1, hi_lim, hop, loud=thr, hard_limit=hard_hi) * hop + tail
            ps, pe = max(ps, hard_lo * hop), min(pe, hard_hi * hop)
            if pieces and ps <= pieces[-1][1] + 0.02:
                pieces[-1] = (pieces[-1][0], pe)  # nothing worth cutting between them
            else:
                pieces.append((ps, pe))
        out_t = 0.0
        for idx, (ps, pe) in enumerate(pieces):
            d = pe - ps
            listen = {n: [round(st + out_t * rate, 3), rate] for n, (st, rate) in seg.listen.items()}
            new.append(MSegment(seg.line, seg.char, round(ps, 3), round(d, 3), listen, seg.zoom, seg.text,
                                seg.framing, seg.pull_out if idx == 0 else None))
            out_t += d
        removed += seg.duration - out_t
    before = sum(s.duration for s in plan.segments)
    plan.segments = new
    return {"before": round(before, 2), "after": round(before - removed, 2), "removed": round(removed, 2)}


# ---------------------------------------------------------------- framing


def auto_framing(plan: MPlan, hold: tuple[float, ...] = (3.0, 2.5, 3.5, 2.5), medium: float = 1.22,
                 tight: float = 1.45, skip_zoom: bool = True, min_hold: float = 1.2) -> None:
    """Jump-zoom "camera" edits to keep attention, like the reference edits.

    Run after tighten(). Hard cuts between framings at line changes or at the jump
    cuts tighten() made inside a line (hiding them), each held for a few seconds
    (cycling through `hold`): wide (everyone), tight on whoever is talking, medium
    on the speaker with their neighbour(s) still in frame. Never the same framing
    twice in a row. Lines flagged `zoom` keep their own speaker-only punch-in.
    """
    order = ["wide", "tight", "medium", "tight", "wide", "medium"]
    k = 0
    style = order[0]
    since = 0.0
    hk = 0
    open_wide(plan)
    for i, seg in enumerate(plan.segments):
        if i == 0:
            since += seg.duration  # the opening is always wide (see open_wide)
            continue
        if seg.zoom and skip_zoom:
            seg.framing = None
            since = 1e9  # the punch-in was itself a cut: change framing after it
            continue
        # Change framing once the hold is up - or early at a jump cut inside a line
        # (a new framing hides the jump), as long as the framing has had a moment.
        jump = i > 0 and plan.segments[i - 1].line == seg.line and plan.segments[i - 1].char == seg.char
        if i > 0 and (since >= hold[hk % len(hold)] or (jump and since >= min_hold)):
            k += 1
            hk += 1
            style = order[k % len(order)]
            since = 0.0
        seg.framing = _style_framing(plan, seg, style, medium, tight)
        since += seg.duration


def open_wide(plan: MPlan, start_zoom: float = 1.15, seconds: float = 0.4) -> None:
    """The hook: the video opens on everyone in frame, via a quick pull-out."""
    if plan.segments:
        first = plan.segments[0]
        first.zoom = False
        first.framing = None
        first.pull_out = [start_zoom, seconds]


def set_framing(plan: MPlan, styles: list[str | None], medium: float = 1.22, tight: float = 1.4) -> None:
    """Explicit per-segment framing: "wide", "medium" or "tight" (on that line's speaker).
    The opening line is always wide with a quick pull-out, whatever `styles` says."""
    for seg, style in zip(plan.segments, styles):
        seg.framing = _style_framing(plan, seg, style, medium, tight)
    open_wide(plan)


def _style_framing(plan: MPlan, seg: MSegment, style: str | None, medium: float, tight: float
                   ) -> list[float] | None:
    """Framing for a style, avoiding half-cut neighbours (graded as bad frames):
    with two people, medium keeps both fully in frame and tight pushes in far enough
    that the other person mostly drops out of frame."""
    spk = plan.char(seg.char)
    others = sorted((c for c in plan.chars if c.name != seg.char), key=lambda c: abs(c.center - spk.center))
    if style == "tight":
        z = tight
        if len(plan.chars) == 2:
            z = max(tight, 1.6)
        return [z, spk.center, spk.face_y + 0.06]
    if style == "medium":
        if len(plan.chars) == 2:
            return [min(medium, 1.12), 0.5, spk.face_y + 0.14]
        return [medium, (spk.center * 2 + others[0].center) / 3, spk.face_y + 0.12]
    return None


def _frame_crop(framing: list[float] | None, W: int, H: int) -> tuple[float, int, int, int, int]:
    if not framing or framing[0] <= 1.0:
        return 1.0, 0, 0, W, H
    z, cx, cy = framing
    cw, ch = int(round(W / z)), int(round(H / z))
    x0 = int(min(max(cx * W - cw / 2, 0), W - cw))
    y0 = int(min(max(cy * H - ch / 2, 0), H - ch))
    return W / cw, x0, y0, cw, ch


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


def _sample_at(raw: Path, start: float, dur: float, fps: float, w: int, h: int) -> np.ndarray:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, start):.3f}", "-t", f"{dur:.3f}", "-i", str(raw),
         "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, h, w, 3).astype(np.float32)


def background_plate(raw: str | Path, chars: list[Character], gains: dict[str, list[float]],
                     size: tuple[int, int], counter_y: float | None = None, reach: float = 0.22,
                     empty: list[float] | None = None) -> tuple[np.ndarray, dict, dict, dict, dict]:
    """Empty-room estimate at analysis resolution - the reference for deciding what in
    each take is "someone/something" vs. scenery.

    Also returns, per take, masks of what stays put but differs from the plate:
    props (standing on the counter, in front of people, layered on top) and scenery
    changes (e.g. a blanket moved between takes; behind people, never part of anyone's
    silhouette).
    """
    import cv2

    raw = Path(raw)
    w, h = size
    takes = {c.name: _sample_at(raw, c.lo, c.hi - c.lo, 1, w, h) * np.array(gains.get(c.name, [1, 1, 1]), np.float32)
             for c in chars}
    if empty:
        # an empty-room stretch joins as a pseudo-take standing infinitely far away, so
        # it is "bare room" everywhere and wins the consensus wherever it agrees
        takes["_empty"] = _sample_at(raw, empty[0], empty[1] - empty[0], 4, w, h)
        chars = list(chars) + [Character("_empty", *empty, 1e3)]
    medians = {name: np.median(f, axis=0).astype(np.float32) for name, f in takes.items()}
    # Each spot comes only from takes whose character stands well away from it (so it
    # is bare room in those takes), skipping any take that has a prop standing there.
    # Where no far take is free, the farthest take without a prop there is used. (A
    # plain median over all takes fails where two people's spots overlap.)
    xs = (np.arange(w) + 0.5) / w
    names = [c.name for c in chars]
    centers = np.array([c.center for c in chars])
    stack = np.stack([medians[n] for n in names])            # (k, h, w, 3)

    def build(blocked: np.ndarray) -> np.ndarray:
        out = np.zeros((h, w, 3), np.float32)
        for x0 in range(w):
            dist = np.abs(xs[x0] - centers)
            order = np.argsort(-dist)
            col = stack[:, :, x0]                              # (k, h, 3)
            ok = ~blocked[:, :, x0]                            # (k, h)
            far = (dist > reach)[:, None] & ok
            vals = np.where(far[..., None], col, np.nan)
            with np.errstate(all="ignore"):
                import warnings

                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(vals, axis=0) if far.any() else np.full((h, 3), np.nan)
            need = np.isnan(med[:, 0])
            for k in order:                                    # farthest free take
                fill = need & ok[k]
                med[fill] = col[k][fill]
                need &= ~fill
            med[need] = np.median(col, axis=0)[need]
            out[:, x0] = med
        return out

    # Start from consensus: at each spot, the take whose usual picture agrees with the
    # most other takes (ties go to the take whose character stands farthest away).
    agree = np.zeros((len(names), h, w), np.float32)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                agree[i] += np.abs(stack[i] - stack[j]).sum(-1) < 60
    dist_all = np.abs(xs[None, :] - centers[:, None])           # (k, w)
    agree += 0.5 * dist_all[:, None, :] / max(float(dist_all.max()), 1e-6)
    pick = agree.argmax(axis=0)
    plate = np.take_along_axis(stack, pick[None, ..., None].repeat(3, -1), axis=0)[0]
    for _ in range(2):  # props found against the plate are excluded from the next plate
        # (a take whose usual picture differs from the room there: a prop, steady or
        # handled during the take - e.g. a bottle that's picked up later)
        blocked = np.stack([np.abs(stack[i] - plate).sum(-1) > 60 for i in range(len(names))])
        blocked = np.stack([cv2.dilate(b.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) for b in blocked])
        plate = build(blocked)
    # Lighting can change across the room between takes (a lamp on in one take only),
    # which a single colour gain can't fix. Per take, measure plate/take on bare room far
    # from that take's character and spread it smoothly over the frame.
    if empty:
        chars = chars[:-1]
        del takes["_empty"]
    light = {}
    for c in chars:
        med = medians[c.name]
        valid = (np.abs(xs - c.center) > reach)[None, :] & (np.abs(med - plate).sum(-1) < 90)
        valid &= (med.mean(-1) > 15)
        wv = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), 60) + 1e-4
        g = np.ones((h, w, 3), np.float32)
        for k in range(3):
            ratio = np.where(valid, plate[..., k] / np.maximum(med[..., k], 1), 0).astype(np.float32)
            g[..., k] = cv2.GaussianBlur(ratio, (0, 0), 60) / wv
        # fade to no correction where little bare room was seen, and keep it gentle
        g = 1 + (g - 1) * np.clip(wv / 0.3, 0, 1)[..., None]
        light[c.name] = np.clip(g, 0.88, 1.12)
        takes[c.name] = takes[c.name] * light[c.name]
        medians[c.name] = med * light[c.name]
    props, scenery = {}, {}
    for name, f in takes.items():
        differs = np.abs(medians[name] - plate).sum(-1) > 60
        steady = f.std(axis=0).max(-1) < 12
        m = (differs & steady).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.dilate(m, np.ones((5, 5), np.uint8))
        prop = np.zeros((h, w), bool)
        back = np.zeros((h, w), bool)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            bottom = (stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]) / h
            if counter_y is None or bottom >= counter_y - 0.01:
                prop |= lab == i
            else:
                back |= lab == i
        props[name], scenery[name] = prop, back
    return plate, props, scenery, medians, light


# ---------------------------------------------------------------- layered compositing

_KERN_CLOSE = None


def _person_mask(frame: np.ndarray, plate: np.ndarray, zone: np.ndarray, thr: float = 45,
                 close: int = 11, grow: int = 0, grow_thr: float = 20) -> np.ndarray:
    """Solid silhouette of whatever in this take differs from the empty room, inside its zone.
    `close` bridges gaps where clothing matches what's behind it (bigger = more solid);
    `grow` > 0 also takes faint differences (> grow_thr) within that many pixels of the
    silhouette - a navy shirt in front of a black cabinet differs only a little."""
    import cv2

    d = np.abs(frame - plate).sum(-1)
    m = ((d > thr) & zone).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if grow and m.any():
        # grow only from the person's main body, never from stray specks of lighting change
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        body = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        near = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))).astype(bool)
        m = (m.astype(bool) | (near & (d > grow_thr) & zone)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    # fill holes (a hoodie the same colour as the wall behind it still counts as hoodie)
    inv = (1 - m).astype(np.uint8)
    h, w = m.shape
    ff = inv.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for x in range(0, w, 8):
        for y in (0, h - 1):
            if ff[y, x]:
                cv2.floodFill(ff, mask, (x, y), 0)
    for y in range(0, h, 8):
        for x in (0, w - 1):
            if ff[y, x]:
                cv2.floodFill(ff, mask, (x, y), 0)
    m = m | ff
    # drop specks
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 120
    return keep[lab]


def _row_path(cost: np.ndarray, step: int = 1) -> np.ndarray:
    """Cheapest top-to-bottom path through cost (H, W), moving <= step columns per row."""
    H, W = cost.shape
    acc = cost.copy()
    back = np.zeros((H, W), np.int8)
    for y in range(1, H):
        prev = acc[y - 1]
        best = prev.copy()
        arg = np.zeros(W, np.int8)
        for d in range(1, step + 1):
            left = np.full(W, np.inf, np.float32)
            left[d:] = prev[:-d]
            right = np.full(W, np.inf, np.float32)
            right[:-d] = prev[d:]
            m = left < best
            best[m], arg[m] = left[m], -d
            m = right < best
            best[m], arg[m] = right[m], d
        acc[y] += best
        back[y] = arg
    path = np.zeros(H, np.int32)
    path[-1] = int(acc[-1].argmin())
    for y in range(H - 1, 0, -1):
        path[y - 1] = path[y] + back[y, path[y]]
    return path


class LayerCompositor:
    """Per-frame layering of the takes: background by zone, people on top in a fixed
    depth order, props on top of everyone. A person's pixels can only ever be covered
    by another person's body or a prop - never by another take's background - so
    there is no see-through ghosting.
    """

    def __init__(self, plan: MPlan, plate: np.ndarray, props: dict[str, np.ndarray],
                 scenery: dict[str, np.ndarray] | None = None, medians: dict[str, np.ndarray] | None = None,
                 depth: list[str] | None = None, dilate: int = 2, soften: float = 1.0, reach: float = 12,
                 seam_blur: float = 10.0):
        import cv2

        self.cv2 = cv2
        self.plan = plan
        self.plate = plate
        self.props = props
        self.scenery = scenery or {}
        self.medians = medians or {}
        h, w = plate.shape[:2]
        self.h, self.w = h, w
        names = [c.name for c in plan.chars]
        self.names = names
        # Zones: where each character can possibly be (from the seam bands).
        xs = np.arange(w) / w
        self.zone = {}
        for i, c in enumerate(plan.chars):
            lo = plan.bands[i - 1][0] if i > 0 else 0.0
            hi = plan.bands[i][1] if i < len(names) - 1 else 1.0
            self.zone[c.name] = np.broadcast_to(((xs >= lo) & (xs <= hi))[None, :], (h, w))
        # Background owner: each boundary between neighbours runs top to bottom along
        # the path where the two takes' backgrounds look most alike (so it goes around
        # anything that changed between takes, like a moved blanket, instead of
        # through it), near halfway between them.
        centers = [c.center for c in plan.chars]
        cut_cols = []
        for i in range(len(names) - 1):
            mid = (centers[i] + centers[i + 1]) / 2
            lo, hi = int((mid - 0.12) * w), int((mid + 0.12) * w)
            if medians:
                cost = np.abs(medians[names[i]][:, lo:hi] - medians[names[i + 1]][:, lo:hi]).sum(-1) / 255
            else:
                cost = np.zeros((h, hi - lo), np.float32)
            cost += 0.01 * np.abs(np.arange(hi - lo) + lo - mid * w)[None, :] / w * 100
            cut_cols.append(_row_path(cost.astype(np.float32)) + lo)
        self.base = np.zeros((h, w), np.int32)
        cols = np.arange(w)[None, :]
        for i, path in enumerate(cut_cols):
            self.base[cols >= path[:, None]] = i + 1
        # Background (and each take's props) come only from the take that owns that part
        # of the room - so props never pop in and out between cuts.
        # Soft enough (~40px at full size) that small lighting differences between
        # takes fade across instead of showing as a line down the middle.
        self.base_soft = [cv2.GaussianBlur((self.base == i).astype(np.float32), (0, 0), seam_blur)
                          for i in range(len(names))]
        # Band around each background boundary, for per-frame exposure matching.
        cols = np.arange(w)[None, :]
        self.band = np.zeros((h, w), bool)
        for path in cut_cols:
            self.band |= np.abs(cols - path[:, None]) < int(0.10 * w)
        # Default depth: the middle character at the back, the outer ones in front.
        self.explicit_depth = bool(depth)
        self.depth = depth or sorted(names, key=lambda n: -abs(plan.char(n).center - 0.5))[::-1]
        self.kd = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        self.soften = soften
        self.reach = reach
        self.prev: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self.prev = {}

    def exposure(self, small: dict[str, np.ndarray], state: dict) -> dict[str, np.ndarray]:
        """Per-frame RGB gain per take so plain room near the seams matches the empty-room
        plate - cancels auto-exposure / white-balance drift that shows as a middle line.
        Smoothed over time; `state` carries it between frames."""
        gains = {}
        for n in self.names:
            f = small[n]
            plain = self.band & (np.abs(f - self.plate).sum(-1) < 40) & (f.mean(-1) > 20)
            if plain.sum() < 300:
                g = np.ones(3, np.float32)
            else:
                g = np.median(self.plate[plain] / np.maximum(f[plain], 1), axis=0).astype(np.float32)
                g = np.clip(g, 0.9, 1.1)
            prev = state.get(n)
            gains[n] = g if prev is None else 0.85 * prev + 0.15 * g
            state[n] = gains[n]
        return gains

    def seam_delta(self, small: dict[str, np.ndarray]) -> float:
        """Colour mismatch between neighbouring takes' plain room inside the seam band."""
        worst = 0.0
        for i in range(len(self.names) - 1):
            a, b = small[self.names[i]], small[self.names[i + 1]]
            plain = self.band & (np.abs(a - self.plate).sum(-1) < 40) & (np.abs(b - self.plate).sum(-1) < 40)
            if plain.sum() > 300:
                worst = max(worst, float(np.abs(a[plain].mean(0) - b[plain].mean(0)).max()))
        return worst

    def alphas(self, small: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
        cv2 = self.cv2
        body = {}
        for n in self.names:
            # Props and scenery changes stay put for the whole take; they aren't
            # anyone's body (otherwise hole-filling glues them onto the person).
            # A scenery change only counts while it is actually what the take shows there;
            # when the person moves in front of it, that is their body (a very still head
            # can otherwise read as "scenery" and get painted over with wall).
            scen = self.scenery.get(n, False)
            if n in self.medians and isinstance(scen, np.ndarray) and scen.any():
                scen = scen & (np.abs(small[n] - self.medians[n]).sum(-1) < 40)
            exclude = self.props[n] | scen
            # Stand-behind: whoever is in front gets a much more solid outline - a dark
            # shirt against a dark cabinet must never let the clone behind show through.
            front = self.explicit_depth and n == self.depth[-1]
            b = _person_mask(small[n], self.plate, self.zone[n] & ~exclude, close=31 if front else 11,
                             grow=12 if front else 0)
            body[n] = b
        # People on top of the background, in depth order (later = in front).
        person = np.full((self.h, self.w), -1, np.int32)
        strong = {n: np.abs(small[n] - self.plate).sum(-1) > 45 for n in self.names}
        masks = {}
        for n in self.depth:
            m = body[n].copy()
            if n in self.prev:
                m |= self.prev[n] & cv2.dilate(m.astype(np.uint8), self.kd).astype(bool)  # steadier edges
            self.prev[n] = body[n]
            masks[n] = m
            person[m] = self.names.index(n)
        # Where outlines overlap, a take that really shows something there (or whose
        # person surely covers it: deep inside their outline) beats one whose outline
        # merely got smoothed over plain room; between two real claims, front wins.
        core_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        for n in self.depth:
            core = cv2.erode(masks[n].astype(np.uint8), core_k).astype(bool)  # deep inside: surely them
            person[masks[n] & (strong[n] | core)] = self.names.index(n)
        # Around each person, anything that looks like plain empty room in their take
        # (soft edges, hair, a sleeve the same colour as the couch) comes from their own
        # take - the one take guaranteed to have those edge pixels right. Further out,
        # the fixed background owner takes over, so scenery never shifts around.
        # If two takes compete for such a pixel, the one where something faint is
        # actually there (differs more from the empty room) wins; if both look exactly
        # like the room, the nearer person wins.
        free = person < 0
        best = np.full((self.h, self.w), -np.inf, np.float32)
        near = np.full((self.h, self.w), -1, np.int32)
        for i, n in enumerate(self.names):
            if not body[n].any():
                continue
            dist = cv2.distanceTransform((~body[n]).astype(np.uint8), cv2.DIST_L2, 3)
            diff = np.abs(small[n] - self.plate).sum(-1)
            score = np.where(diff > 15, diff, 0) - dist * 0.5
            ok = free & (diff < 45) & (dist < self.reach) & (score > best)
            best[ok] = score[ok]
            near[ok] = i
        person[near >= 0] = near[near >= 0]
        anyone = cv2.GaussianBlur((person >= 0).astype(np.float32), (0, 0), self.soften)
        al = {}
        for i, n in enumerate(self.names):
            p = cv2.GaussianBlur((person == i).astype(np.float32), (0, 0), self.soften)
            al[n] = p + (1 - anyone) * self.base_soft[i]
        tot = sum(al.values())
        al = {n: a / np.maximum(tot, 1e-6) for n, a in al.items()}
        # Self-check: person pixels that end up partly see-through, and how much of
        # each person is covered by someone else.
        stats = {}
        for i, n in enumerate(self.names):
            b = body[n]
            if not b.any():
                stats[n] = (0, 0.0)
                continue
            others = np.zeros_like(b)
            for m in self.names:
                if m != n:
                    others |= body[m]
            # The few pixels right at an edge where one person passes in front of
            # another are anti-aliasing, not see-through; only count the rest.
            near_other = cv2.dilate(others.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            ghost = b & (al[n] > 0.15) & (al[n] < 0.85) & ~near_other
            covered = (b & (al[n] < 0.5)).sum() / b.sum()
            stats[n] = (int(ghost.sum()), float(covered))
        return al, stats


# ---------------------------------------------------------------- stand-behind matte


class _FrontMatte:
    """Soft cutout of the person nearest the camera (stand-behind format): MediaPipe's
    selfie segmenter, edges refined with a guided filter against the frame itself (so the
    edge follows hair and shirt, not the model's coarse 256px grid), steadied over time."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        model = Path(__file__).resolve().parent / "models" / "selfie_segmenter.tflite"
        self.mp = mp
        self.seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(model)), output_confidence_masks=True))
        self.prev: np.ndarray | None = None

    def reset(self) -> None:
        self.prev = None

    def __call__(self, frame: np.ndarray, plate: np.ndarray | None = None) -> np.ndarray:
        import cv2

        H, W = frame.shape[:2]
        img = np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8))
        r = self.seg.segment(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=img))
        conf = np.squeeze(r.confidence_masks[0].numpy_view()).astype(np.float32)
        conf = cv2.resize(conf, (W, H), interpolation=cv2.INTER_LINEAR)
        hard = np.clip((conf - 0.35) / 0.3, 0, 1)                      # decisive, no grey fog
        if plate is not None:
            # what the model leaves out right next to the person but clearly isn't empty
            # room (a cup they hold, the shirt behind it) is theirs too
            ph, pw = plate.shape[:2]
            small = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA).astype(np.float32)
            diff = (np.abs(small - plate).sum(-1) > 60).astype(np.uint8)
            diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            near = cv2.dilate((cv2.resize(hard, (pw, ph)) > 0.5).astype(np.uint8),
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
            extra = cv2.resize((diff & near).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            hard = np.maximum(hard, extra)
        hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        guide = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        a = cv2.ximgproc.guidedFilter(guide, cv2.resize(hard, (W // 2, H // 2)), 8, 1e-3)
        a = np.clip(cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR), 0, 1)
        a = np.clip((a - 0.15) / 0.7, 0, 1)                             # firm core, short soft edge
        if self.prev is not None:
            a = 0.6 * a + 0.4 * self.prev                               # no edge shimmer frame to frame
        self.prev = a
        return a


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
                 zoom: float = 1.6, label_y: float = 0.39, label_size: int = 52,
                 depth: list[str] | None = None, crf: int = 20, preset: str = "medium",
                 report_path: str | Path | None = None, workers: int | None = None,
                 on_progress: Callable[[float], None] | None = None) -> Path:
    """workers: processes rendering chunks of segments in parallel (default: one per CPU)."""
    raw, output = Path(raw), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    W, H = width, height
    names = [c.name for c in plan.chars]
    gains = {n: np.array(plan.gains.get(n, [1, 1, 1]), np.float32) for n in names}
    total = sum(s.duration for s in plan.segments)
    import cv2

    plate, props, scenery, medians, light = background_plate(raw, plan.chars, plan.gains, (W // _DS, H // _DS),
                                                             plan.counter_y, empty=plan.empty)
    comp = LayerCompositor(plan, plate, props, scenery, medians, depth=depth or plan.depth)
    # full-resolution lighting correction per take (global gain x smooth local field)
    light_full = {n: cv2.resize(light[n], (W, H), interpolation=cv2.INTER_LINEAR) * gains[n] for n in names}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        from PIL import Image

        def label_img(text: str, key: str) -> np.ndarray:
            png = tmp / f"label_{key}.png"
            _label_png(png, text, max(12, round(label_size * W / 1080)))  # same look at draft size
            return np.asarray(Image.open(png).convert("RGBA"), np.float32)

        base_labels = {c.name: label_img(c.label, c.name) for c in plan.chars if c.label}
        changes = {n: [(int(ln), label_img(t, f"{n}_{k}")) for k, (ln, t) in enumerate(sorted(ch, key=lambda x: x[0]))]
                   for n, ch in plan.label_changes.items()}

        def labels_for(line: int) -> dict[str, np.ndarray]:
            out = dict(base_labels)
            for n, ch in changes.items():
                for ln, img in ch:
                    if line >= ln:
                        out[n] = img
            return out

        # Segments are independent (every cut resets the compositor's state), so the
        # video is rendered in contiguous chunks of segments, one process per chunk,
        # and the chunks are joined without re-encoding.
        nframes = [max(1, round(s.duration * FPS)) for s in plan.segments]
        nframes_total = sum(nframes)
        workers = max(1, min(workers or os.cpu_count() or 1, len(plan.segments)))
        chunks, cur, acc = [], [], 0
        for i, n in enumerate(nframes):
            cur.append(i)
            acc += n
            if acc >= nframes_total * (len(chunks) + 1) / workers and len(chunks) < workers - 1:
                chunks.append(cur)
                cur = []
        if cur:
            chunks.append(cur)

        def encode(seg_ids: list[int], path: Path, rep_path: Path | None) -> list[dict]:
            matte = _FrontMatte() if plan.mode == "stand_behind" else None  # per process
            front = (plan.depth or names)[-1]
            done = sum(nframes[:seg_ids[0]])
            report: list[dict] = []
            exp_state: dict = {}
            enc = subprocess.Popen(
                ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                 "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                 "-pix_fmt", "yuv420p", str(path)],
                stdin=subprocess.PIPE,
            )
            for si in seg_ids:
                seg = plan.segments[si]
                n = nframes[si]
                spk = plan.char(seg.char)
                if seg.zoom:
                    r = _Reader(raw, seg.start, n, W, H)
                    zw, zh = int(W / zoom), int(H / zoom)
                    x0 = int(min(max(spk.center * W - zw / 2, 0), W - zw))
                    y0 = int(min(max(H * 0.2 - zh / 2, 0), H - zh))
                    for _ in range(n):
                        f = r.read().astype(np.float32) * light_full[spk.name]
                        crop = np.clip(f[y0:y0 + zh, x0:x0 + zw], 0, 255).astype(np.uint8)
                        from PIL import Image

                        out = np.asarray(Image.fromarray(crop).resize((W, H), Image.LANCZOS))
                        enc.stdin.write(out.tobytes())
                        done += 1
                    r.close()
                    continue
                readers = {}
                for c in plan.chars:
                    if c.name == seg.char:
                        readers[c.name] = _Reader(raw, seg.start, n, W, H)
                    else:
                        st, rate = seg.listen[c.name]
                        readers[c.name] = _Reader(raw, st, n, W, H, rate)
                comp.reset()
                if matte is not None:
                    matte.reset()
                exp_state.clear()  # new footage at a cut: match its exposure fresh, no lag
                labels = labels_for(seg.line)
                for fi in range(n):
                    fr = {k: np.clip(r.read().astype(np.float32) * light_full[k], 0, 255) for k, r in readers.items()}
                    small = {k: v[::_DS, ::_DS] for k, v in fr.items()}
                    eg = comp.exposure(small, exp_state)
                    fr = {k: np.clip(v * eg[k], 0, 255) for k, v in fr.items()}
                    small = {k: v[::_DS, ::_DS] for k, v in fr.items()}
                    if matte is not None:
                        # stand-behind: front person over the other take's whole frame
                        a = matte(fr[front], comp.plate)[..., None]
                        back = next(k for k in names if k != front)
                        out = a * fr[front] + (1 - a) * fr[back]
                        st, sd = {k: (0, 0.0) for k in names}, 0.0
                    else:
                        al, st = comp.alphas(small)
                        sd = comp.seam_delta(small)
                        out = np.zeros_like(fr[names[0]])
                        for k in names:
                            a = cv2.resize(al[k], (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
                            out += a * fr[k]
                    report.append({"t": round(done / FPS, 2), "line": seg.line, "speaker": seg.char,
                                   **{f"ghost_{k}": v[0] for k, v in st.items()},
                                   **{f"covered_{k}": round(v[1], 3) for k, v in st.items()},
                                   "seam_delta": round(sd, 1)})
                    # camera framing: crop the composite and scale back up (labels stay
                    # the same size and follow the people they belong to)
                    framing = seg.framing
                    if seg.pull_out and fi < seg.pull_out[1] * FPS:
                        u = fi / (seg.pull_out[1] * FPS)
                        u = 1 - (1 - u) ** 3  # ease out: fast start, settles on the wide shot
                        framing = [seg.pull_out[0] + (1 - seg.pull_out[0]) * u, 0.5, 0.42]
                    z, fx0, fy0, cw, ch = _frame_crop(framing, W, H)
                    if z != 1.0:
                        out = cv2.resize(out[fy0:fy0 + ch, fx0:fx0 + cw], (W, H), interpolation=cv2.INTER_CUBIC)
                    for c in plan.chars:
                        if c.name in labels:
                            lab = labels[c.name]
                            lh, lw = lab.shape[:2]
                            cx = (c.center * W - fx0) * z
                            cy = (label_y * H - fy0) * z
                            if not (0.04 * W < cx < 0.96 * W and lh < cy < H - lh):
                                continue  # their spot is out of frame in this framing
                            x0 = int(min(max(cx - lw / 2, 0), W - lw))
                            y0 = int(cy - lh / 2)
                            la = lab[..., 3:4] / 255
                            out[y0:y0 + lh, x0:x0 + lw] = out[y0:y0 + lh, x0:x0 + lw] * (1 - la) + lab[..., :3] * la
                    enc.stdin.write(np.clip(out, 0, 255).astype(np.uint8).tobytes())
                    done += 1
                for r in readers.values():
                    r.close()
                if on_progress and len(chunks) == 1:
                    on_progress(0.9 * done / nframes_total)
            enc.stdin.close()
            if enc.wait() != 0:
                raise RuntimeError("video encode failed")
            if rep_path:
                rep_path.write_text(json.dumps(report))
            return report

        video = tmp / "video.mp4"
        report: list[dict] = []
        if len(chunks) == 1:
            report = encode(chunks[0], video, None)
        else:
            import multiprocessing as mp

            ctx = mp.get_context("fork")  # children share the plate/compositor, nothing pickled
            parts = [(ids, tmp / f"part{i}.mp4", tmp / f"part{i}.json") for i, ids in enumerate(chunks)]
            procs = [ctx.Process(target=encode, args=p) for p in parts]
            for pr in procs:
                pr.start()
            for pr in procs:
                pr.join()
            if any(pr.exitcode != 0 for pr in procs):
                raise RuntimeError("video encode failed")
            (tmp / "parts.txt").write_text("".join(f"file '{p[1]}'\n" for p in parts))
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(tmp / "parts.txt"),
                            "-c", "copy", str(video)], check=True)
            for p in parts:
                report += json.loads(p[2].read_text())
            if on_progress:
                on_progress(0.9)

        # Audio: each segment's speaker audio, exactly as long as its video. Sliced
        # by sample in numpy (an ffmpeg graph mixing far-apart trims of one input
        # pulled the wrong stretch of audio into unison lines).
        sr = 48000
        pcm = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(raw), "-vn", "-ac", "2",
                              "-ar", str(sr), "-f", "f32le", "-"], capture_output=True, check=True).stdout
        src = np.frombuffer(pcm, np.float32).reshape(-1, 2)
        pieces, frames_out = [], 0
        for seg in plan.segments:
            frames_out += round(seg.duration * FPS)
            n = round(frames_out / FPS * sr) - sum(len(p) for p in pieces)
            fade = min(int(0.012 * sr), n // 4)
            ramp = np.ones(n, np.float32)
            if fade:
                ramp[:fade] = np.linspace(0, 1, fade)
                ramp[n - fade:] = np.linspace(1, 0, fade)
            starts = [seg.start] + ([st for st, _ in seg.listen.values()] if seg.unison else [])
            mix = np.zeros((n, 2), np.float32)
            for st in starts:
                cut = src[int(round(st * sr)):int(round(st * sr)) + n]
                mix[:len(cut)] += cut
            pieces.append(mix * ramp[:, None] / len(starts) ** 0.5)
        wav = tmp / "audio.f32"
        np.concatenate(pieces).astype(np.float32).tofile(wav)
        cmd = ["ffmpeg", "-y", "-hide_banner", "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", str(wav),
               "-i", str(video), "-map", "1:v", "-map", "0:a",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
               "-progress", "pipe:1", "-nostats", str(output)]
        _run_ffmpeg(cmd, total, None)
    if report_path:
        Path(report_path).write_text(json.dumps(report))
    if on_progress:
        on_progress(1.0)
    return output
