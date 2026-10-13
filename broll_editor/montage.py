"""B-roll montage: cut several b-roll shots to the talking head's timeline.

An edit list (JSON) says which shot plays when:

    {
      "talking_head": "Talkinghead.mp4",
      "broll_dir": "clips/",               # optional, relative sources resolve here
      "shots": [
        {"at": 0.0, "source": "IMG_5431.MOV", "start": 3.0, "focus_y": 0.1,
         "zoom": [1.0, 1.25], "flash": true},
        {"at": 1.6, "source": "IMG_5434.MOV", "start": 5.5}
      ]
    }

Each shot runs from its "at" (seconds on the talking head) until the next
shot's "at"; the last runs to the end. Every shot gets a slow push-in or
pull-out ("zoom": [from, to]) so reused footage never looks identical, and an
optional white "flash" on the cut to punch up a beat.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .stacker import StackOptions, _panel_heights, probe, stack_videos

# Alternated when a shot doesn't set its own zoom, so consecutive cuts feel different.
DEFAULT_ZOOMS = ([1.0, 1.15], [1.18, 1.03])


@dataclass
class Shot:
    source: Path
    start: float
    duration: float
    focus_x: float = 0.5
    focus_y: float = 0.5
    zoom: list[float] = field(default_factory=lambda: [1.0, 1.15])
    flash: bool = False
    # Playback speed: 0.5 = half-speed slow motion (smoothest from 60/120fps footage).
    speed: float = 1.0
    saturation: float = 1.15
    contrast: float = 1.05


def _shot_filter(shot: Shot, w: int, h: int, fps: int) -> str:
    frames = max(1, round(shot.duration * fps))
    z0, z1 = shot.zoom
    # Cover-crop at 2x the panel size first: zoompan moves in whole pixels, so
    # working oversampled keeps slow zooms smooth.
    ow, oh = w * 2, h * 2
    chain = [
        f"setpts=(PTS-STARTPTS)/{shot.speed}",
        f"fps={fps}",
        f"scale={ow}:{oh}:force_original_aspect_ratio=increase",
        f"crop={ow}:{oh}:(iw-{ow})*{shot.focus_x}:(ih-{oh})*{shot.focus_y}",
        # Ease-out zoom from z0 to z1 across the shot, centred.
        f"zoompan=z='{z0}+({z1}-{z0})*(1-pow(1-on/{frames},2))'"
        f":x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d=1:s={w}x{h}:fps={fps}",
        f"eq=saturation={shot.saturation}:contrast={shot.contrast}",
        "setsar=1",
    ]
    if shot.flash:
        chain.append(f"fade=t=in:st=0:d=0.2:color=white")
    return ",".join(chain)


def render_shot(shot: Shot, out: Path, w: int, h: int, fps: int) -> Path:
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{shot.start:.3f}", "-t", f"{shot.duration * shot.speed + 0.1:.3f}", "-i", str(shot.source),
        "-vf", _shot_filter(shot, w, h, fps),
        "-frames:v", str(max(1, round(shot.duration * fps))),
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "14", "-pix_fmt", "yuv420p",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Rendering shot from {shot.source.name} failed:\n{result.stderr[-2000:]}")
    return out


def load_edit(path: str | Path) -> tuple[Path, list[dict], dict]:
    path = Path(path)
    edit = json.loads(path.read_text())
    base = path.parent
    broll_dir = base / edit.get("broll_dir", ".")
    head = Path(edit["talking_head"])
    if not head.is_absolute():
        head = broll_dir / head
    shots = sorted(edit["shots"], key=lambda s: s["at"])
    for s in shots:
        src = Path(s["source"])
        s["source"] = src if src.is_absolute() else broll_dir / src
    return head, shots, edit.get("options", {})


def build_shots(raw: list[dict], total: float) -> list[Shot]:
    shots = []
    for i, s in enumerate(raw):
        end = raw[i + 1]["at"] if i + 1 < len(raw) else total
        duration = end - s["at"]
        if duration <= 0:
            continue
        src_len = probe(s["source"]).duration
        start = s.get("start", 0.0)
        speed = s.get("speed", 1.0)
        if speed <= 0:
            raise ValueError(f"Shot at {s['at']}s has a non-positive speed")
        if start + duration * speed > src_len:
            raise ValueError(
                f"Shot at {s['at']}s needs {duration * speed:.1f}s of {Path(s['source']).name} from "
                f"{start}s, but the clip is only {src_len:.1f}s long"
            )
        shots.append(Shot(
            source=Path(s["source"]), start=start, duration=duration,
            focus_x=s.get("focus_x", 0.5), focus_y=s.get("focus_y", 0.5),
            zoom=s.get("zoom", DEFAULT_ZOOMS[i % 2]), flash=s.get("flash", False), speed=speed,
            saturation=s.get("saturation", 1.15), contrast=s.get("contrast", 1.05),
        ))
    return shots


def render_montage(shots: list[Shot], out: Path, w: int, h: int, fps: int, workers: int = 3) -> Path:
    """Render every shot at panel size and join them into one silent b-roll track."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parts = [tmp / f"shot{i:03d}.mp4" for i in range(len(shots))]
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(lambda a: render_shot(a[0], a[1], w, h, fps), zip(shots, parts)))
        listing = tmp / "list.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
             "-c", "copy", str(out)],
            check=True,
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="broll_editor.montage",
        description="Cut multiple b-roll shots under a talking head from an edit list, then stack them.",
    )
    p.add_argument("edit", help="edit list JSON")
    p.add_argument("-o", "--output", default="output/montage.mp4")
    p.add_argument("--broll-only", action="store_true", help="only render the bottom b-roll track")
    args = p.parse_args(argv)

    head, raw, extra = load_edit(args.edit)
    opts = StackOptions(**{k: v for k, v in extra.items() if k != "transcript"})
    opts.validate()
    w, _, bottom_h = _panel_heights(opts)
    total = probe(head).duration - opts.top_start
    if opts.max_duration:
        total = min(total, opts.max_duration)

    shots = build_shots(raw, total)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sys.stderr.write(f"Rendering {len(shots)} b-roll shots...\n")
    with tempfile.TemporaryDirectory() as tmp:
        broll = Path(tmp) / "broll.mp4"
        render_montage(shots, broll if not args.broll_only else out, w, bottom_h, opts.fps)
        if args.broll_only:
            sys.stderr.write(f"Done: {out}\n")
            return 0
        # The b-roll track is already panel-sized; stack it under the talking head.
        opts.bottom_fit, opts.bottom_focus_x, opts.bottom_focus_y = "cover", 0.5, 0.5
        transcript = extra.get("transcript")
        if transcript and not Path(transcript).is_absolute():
            transcript = Path(args.edit).parent / transcript

        def progress(frac: float) -> None:
            sys.stderr.write(f"\rStacking... {frac * 100:5.1f}%")

        stack_videos(head, broll, out, opts, on_progress=progress, transcript=transcript)
    sys.stderr.write(f"\nDone: {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
