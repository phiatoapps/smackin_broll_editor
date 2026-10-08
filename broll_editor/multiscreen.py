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
    # Height of the counter/table top as a fraction of frame height. Things that stay
    # put during a take and stand on it are props (in front of people); things that
    # stay put above it without touching it (a moved blanket, a picture) are scenery
    # behind the people. None = treat everything that stays put as a prop.
    counter_y: float | None = None

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


def _sample_at(raw: Path, start: float, dur: float, fps: float, w: int, h: int) -> np.ndarray:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, start):.3f}", "-t", f"{dur:.3f}", "-i", str(raw),
         "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, h, w, 3).astype(np.float32)


def background_plate(raw: str | Path, chars: list[Character], gains: dict[str, list[float]],
                     size: tuple[int, int], counter_y: float | None = None, reach: float = 0.22
                     ) -> tuple[np.ndarray, dict, dict, dict, dict]:
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
    light = {}
    for c in chars:
        med = medians[c.name]
        valid = (np.abs(xs - c.center) > reach)[None, :] & (np.abs(med - plate).sum(-1) < 90)
        valid &= (med.mean(-1) > 15)
        wv = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), 25) + 1e-4
        g = np.ones((h, w, 3), np.float32)
        for k in range(3):
            ratio = np.where(valid, plate[..., k] / np.maximum(med[..., k], 1), 0).astype(np.float32)
            g[..., k] = cv2.GaussianBlur(ratio, (0, 0), 25) / wv
        g[wv < 0.02] = 1.0
        light[c.name] = np.clip(g, 0.6, 1.6)
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


def _person_mask(frame: np.ndarray, plate: np.ndarray, zone: np.ndarray, thr: float = 45) -> np.ndarray:
    """Solid silhouette of whatever in this take differs from the empty room, inside its zone."""
    import cv2

    d = np.abs(frame - plate).sum(-1)
    m = ((d > thr) & zone).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
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
                 depth: list[str] | None = None, dilate: int = 2, soften: float = 1.0, reach: float = 12):
        import cv2

        self.cv2 = cv2
        self.plan = plan
        self.plate = plate
        self.props = props
        self.scenery = scenery or {}
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
        self.base_soft = [cv2.GaussianBlur((self.base == i).astype(np.float32), (0, 0), 2)
                          for i in range(len(names))]
        # Default depth: the middle character at the back, the outer ones in front.
        self.depth = depth or sorted(names, key=lambda n: -abs(plan.char(n).center - 0.5))[::-1]
        self.kd = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        self.soften = soften
        self.reach = reach
        self.prev: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self.prev = {}

    def alphas(self, small: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
        cv2 = self.cv2
        body = {}
        for n in self.names:
            # Props and scenery changes stay put for the whole take; they aren't
            # anyone's body (otherwise hole-filling glues them onto the person).
            exclude = self.props[n] | self.scenery.get(n, False)
            b = _person_mask(small[n], self.plate, self.zone[n] & ~exclude)
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
                 report_path: str | Path | None = None,
                 on_progress: Callable[[float], None] | None = None) -> Path:
    raw, output = Path(raw), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    W, H = width, height
    names = [c.name for c in plan.chars]
    gains = {n: np.array(plan.gains.get(n, [1, 1, 1]), np.float32) for n in names}
    total = sum(s.duration for s in plan.segments)
    import cv2

    plate, props, scenery, medians, light = background_plate(raw, plan.chars, plan.gains, (W // _DS, H // _DS),
                                                             plan.counter_y)
    comp = LayerCompositor(plan, plate, props, scenery, medians, depth=depth)
    # full-resolution lighting correction per take (global gain x smooth local field)
    light_full = {n: cv2.resize(light[n], (W, H), interpolation=cv2.INTER_LINEAR) * gains[n] for n in names}
    report: list[dict] = []

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
                    f = r.read().astype(np.float32) * light_full[spk.name]
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
                comp.reset()
                for fi in range(n):
                    fr = {k: np.clip(r.read().astype(np.float32) * light_full[k], 0, 255) for k, r in readers.items()}
                    small = {k: v[::_DS, ::_DS] for k, v in fr.items()}
                    al, st = comp.alphas(small)
                    out = np.zeros_like(fr[names[0]])
                    for k in names:
                        a = cv2.resize(al[k], (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
                        out += a * fr[k]
                    report.append({"t": round(done / FPS, 2), "line": seg.line, "speaker": seg.char,
                                   **{f"ghost_{k}": v[0] for k, v in st.items()},
                                   **{f"covered_{k}": round(v[1], 3) for k, v in st.items()}})
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
    if report_path:
        Path(report_path).write_text(json.dumps(report))
    if on_progress:
        on_progress(1.0)
    return output
