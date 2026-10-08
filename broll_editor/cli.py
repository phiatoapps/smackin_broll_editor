"""Command line: python -m broll_editor TOP BOTTOM -o out.mp4"""

from __future__ import annotations

import argparse
import sys
import tempfile

from .sources import fetch
from .stacker import AUDIO_MODES, DURATION_MODES, FIT_MODES, StackOptions, stack_videos


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="broll_editor",
        description="Stack a talking-head clip (top) over a b-roll clip (bottom) as a 9:16 video.",
    )
    p.add_argument("top", help="talking-head clip: file path or Google Drive link")
    p.add_argument("bottom", help="b-roll clip: file path or Google Drive link")
    p.add_argument("-o", "--output", default="output/stacked.mp4")
    p.add_argument("--size", default="1080x1920", help="output WIDTHxHEIGHT (default 1080x1920)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--split", type=float, default=0.5, help="fraction of height for the top clip (default 0.5)")
    p.add_argument("--top-fit", choices=FIT_MODES, default="cover")
    p.add_argument("--bottom-fit", choices=FIT_MODES, default="cover")
    p.add_argument("--top-focus", default="0.5,0.5", metavar="X,Y", help="crop focus 0-1 for the top clip")
    p.add_argument("--bottom-focus", default="0.5,0.5", metavar="X,Y", help="crop focus 0-1 for the b-roll")
    p.add_argument("--top-start", type=float, default=0.0, help="seconds to skip in the talking head")
    p.add_argument("--bottom-start", type=float, default=0.0, help="seconds to skip in the b-roll")
    p.add_argument("--duration", choices=DURATION_MODES, default="top", help="which clip sets the length")
    p.add_argument("--max-duration", type=float, default=None, help="cap output length (seconds)")
    p.add_argument("--audio", choices=AUDIO_MODES, default="top")
    p.add_argument("--bottom-volume", type=float, default=0.15, help="b-roll volume when --audio mix")
    p.add_argument("--divider", type=int, default=0, help="divider line thickness in px (0 = none)")
    p.add_argument("--divider-color", default="white")
    p.add_argument("--crf", type=int, default=20, help="quality: lower = better/bigger (default 20)")
    p.add_argument("--preset", default="medium", help="x264 preset (ultrafast..veryslow)")
    args = p.parse_args(argv)

    try:
        width, height = (int(v) for v in args.size.lower().split("x"))
        tfx, tfy = (float(v) for v in args.top_focus.split(","))
        bfx, bfy = (float(v) for v in args.bottom_focus.split(","))
    except ValueError:
        p.error("--size must look like 1080x1920 and focus values like 0.5,0.4")

    opts = StackOptions(
        width=width, height=height, fps=args.fps, split=args.split,
        top_fit=args.top_fit, bottom_fit=args.bottom_fit,
        top_focus_x=tfx, top_focus_y=tfy, bottom_focus_x=bfx, bottom_focus_y=bfy,
        top_start=args.top_start, bottom_start=args.bottom_start,
        duration_mode=args.duration, max_duration=args.max_duration,
        audio_mode=args.audio, bottom_volume=args.bottom_volume,
        divider_px=args.divider, divider_color=args.divider_color,
        crf=args.crf, preset=args.preset,
    )

    def progress(frac: float) -> None:
        sys.stderr.write(f"\rRendering... {frac * 100:5.1f}%")
        sys.stderr.flush()

    try:
        with tempfile.TemporaryDirectory() as tmp:
            top = fetch(args.top, tmp, "top")
            bottom = fetch(args.bottom, tmp, "bottom")
            out = stack_videos(top, bottom, args.output, opts, on_progress=progress)
    except Exception as exc:  # surface a readable message instead of a traceback
        sys.stderr.write(f"\nError: {exc}\n")
        return 1
    sys.stderr.write(f"\nDone: {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
