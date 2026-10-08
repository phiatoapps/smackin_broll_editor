"""Core rendering: build and run the ffmpeg command that stacks two clips.

Layout (default 1080x1920, 9:16 vertical):

    +-----------------+
    |  talking head   |  <- top clip, fills `split` of the height
    +-----------------+  <- optional divider line
    |     b-roll      |  <- bottom clip, fills the rest
    +-----------------+

Each clip is scaled to fill its panel. "cover" crops the overflow (with an
adjustable focus point so you can keep a face in frame); "blur" fits the whole
clip inside the panel over a blurred, zoomed copy of itself.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

FIT_MODES = ("cover", "blur")
AUDIO_MODES = ("top", "mix", "bottom", "none")
DURATION_MODES = ("top", "bottom", "shortest")


@dataclass
class StackOptions:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    # Fraction of the frame height given to the top (talking head) panel.
    split: float = 0.5

    top_fit: str = "cover"
    bottom_fit: str = "cover"
    # Crop focus for "cover": 0 = left/top edge, 0.5 = center, 1 = right/bottom.
    top_focus_x: float = 0.5
    top_focus_y: float = 0.5
    bottom_focus_x: float = 0.5
    bottom_focus_y: float = 0.5

    # Seconds to skip at the start of each clip.
    top_start: float = 0.0
    bottom_start: float = 0.0

    # "top": output is as long as the talking head, b-roll loops if shorter.
    duration_mode: str = "top"
    # Hard cap on output length in seconds (None = no cap).
    max_duration: float | None = None

    # "top": talking-head audio only, "mix": talking head + quiet b-roll,
    # "bottom": b-roll audio only, "none": silent.
    audio_mode: str = "top"
    bottom_volume: float = 0.15

    divider_px: int = 0
    divider_color: str = "white"

    crf: int = 20
    preset: str = "medium"

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if not 0.1 <= self.split <= 0.9:
            raise ValueError("split must be between 0.1 and 0.9")
        if self.top_fit not in FIT_MODES or self.bottom_fit not in FIT_MODES:
            raise ValueError(f"fit must be one of {FIT_MODES}")
        if self.audio_mode not in AUDIO_MODES:
            raise ValueError(f"audio_mode must be one of {AUDIO_MODES}")
        if self.duration_mode not in DURATION_MODES:
            raise ValueError(f"duration_mode must be one of {DURATION_MODES}")
        for name in ("top_focus_x", "top_focus_y", "bottom_focus_x", "bottom_focus_y"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.top_start < 0 or self.bottom_start < 0:
            raise ValueError("start offsets cannot be negative")


@dataclass
class MediaInfo:
    duration: float
    width: int
    height: int
    has_audio: bool


def _require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found on PATH. Install ffmpeg first.")


def probe(path: str | Path) -> MediaInfo:
    _require_ffmpeg()
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    data = json.loads(out)
    video = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"{path} has no video stream")
    has_audio = any(s.get("codec_type") == "audio" for s in data["streams"])
    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    width, height = int(video["width"]), int(video["height"])
    # Phone footage often stores rotation as metadata; swap dims so layout math is right.
    rotation = 0
    for side in video.get("side_data_list", []):
        if "rotation" in side:
            rotation = int(side["rotation"])
    if abs(rotation) in (90, 270):
        width, height = height, width
    return MediaInfo(duration=duration, width=width, height=height, has_audio=has_audio)


def _even(n: float) -> int:
    return max(2, int(round(n / 2)) * 2)


def _panel_filter(src: str, out: str, w: int, h: int, fit: str, fx: float, fy: float, fps: int) -> str:
    """Filter chain that turns input `src` into a w x h panel labelled `out`."""
    if fit == "cover":
        return (
            f"[{src}]scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h}:(iw-{w})*{fx}:(ih-{h})*{fy},setsar=1,fps={fps}[{out}]"
        )
    # blur: blurred cover-scaled background with the full clip centered on top
    return (
        f"[{src}]split=2[{out}_a][{out}_b];"
        f"[{out}_a]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},"
        f"boxblur=20:2,eq=brightness=-0.08[{out}_bg];"
        f"[{out}_b]scale={w}:{h}:force_original_aspect_ratio=decrease[{out}_fg];"
        f"[{out}_bg][{out}_fg]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={fps}[{out}]"
    )


def build_command(top: str | Path, bottom: str | Path, output: str | Path, opts: StackOptions) -> tuple[list[str], float]:
    """Return (ffmpeg argv, output duration in seconds)."""
    opts.validate()
    top_info, bottom_info = probe(top), probe(bottom)

    top_len = max(0.0, top_info.duration - opts.top_start)
    bottom_len = max(0.0, bottom_info.duration - opts.bottom_start)
    if top_len <= 0:
        raise ValueError("top_start is past the end of the talking-head clip")
    if bottom_len <= 0:
        raise ValueError("bottom_start is past the end of the b-roll clip")

    if opts.duration_mode == "top":
        duration = top_len
    elif opts.duration_mode == "bottom":
        duration = bottom_len
    else:
        duration = min(top_len, bottom_len)
    if opts.max_duration:
        duration = min(duration, opts.max_duration)

    w = _even(opts.width)
    top_h = _even(opts.height * opts.split)
    bottom_h = _even(opts.height) - top_h

    cmd = ["ffmpeg", "-y", "-hide_banner"]
    # Loop whichever clip is shorter than the output so neither panel freezes.
    if top_len < duration - 0.05:
        cmd += ["-stream_loop", "-1"]
    cmd += ["-ss", f"{opts.top_start:.3f}", "-i", str(top)]
    if bottom_len < duration - 0.05:
        cmd += ["-stream_loop", "-1"]
    cmd += ["-ss", f"{opts.bottom_start:.3f}", "-i", str(bottom)]

    filters = [
        _panel_filter("0:v", "top", w, top_h, opts.top_fit, opts.top_focus_x, opts.top_focus_y, opts.fps),
        _panel_filter("1:v", "bot", w, bottom_h, opts.bottom_fit, opts.bottom_focus_x, opts.bottom_focus_y, opts.fps),
    ]
    stacked = "[top][bot]vstack=inputs=2"
    if opts.divider_px > 0:
        t = opts.divider_px
        stacked += f",drawbox=x=0:y={top_h - t // 2}:w=iw:h={t}:color={opts.divider_color}:t=fill"
    filters.append(stacked + ",format=yuv420p[v]")

    audio_label = None
    if opts.audio_mode == "top" and top_info.has_audio:
        audio_label = "0:a"
    elif opts.audio_mode == "bottom" and bottom_info.has_audio:
        audio_label = "1:a"
    elif opts.audio_mode == "mix":
        if top_info.has_audio and bottom_info.has_audio:
            filters.append(
                f"[1:a]volume={opts.bottom_volume}[ba];"
                "[0:a][ba]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]"
            )
            audio_label = "[a]"
        elif top_info.has_audio:
            audio_label = "0:a"
        elif bottom_info.has_audio:
            filters.append(f"[1:a]volume={opts.bottom_volume}[a]")
            audio_label = "[a]"

    cmd += ["-filter_complex", ";".join(filters), "-map", "[v]"]
    if audio_label:
        cmd += ["-map", audio_label, "-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    else:
        cmd += ["-an"]
    cmd += [
        "-c:v", "libx264", "-preset", opts.preset, "-crf", str(opts.crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-t", f"{duration:.3f}",
        "-progress", "pipe:1", "-nostats",
        str(output),
    ]
    return cmd, duration


def stack_videos(
    top: str | Path,
    bottom: str | Path,
    output: str | Path,
    opts: StackOptions | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> Path:
    """Render `top` over `bottom` into `output`. `on_progress` gets 0.0-1.0."""
    opts = opts or StackOptions()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd, duration = build_command(top, bottom, output, opts)

    # stderr goes to a temp file so a chatty ffmpeg can't fill a pipe and stall.
    with tempfile.TemporaryFile(mode="w+") as errlog:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errlog, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            if on_progress and line.startswith("out_time_us="):
                try:
                    done = int(line.split("=", 1)[1]) / 1_000_000
                except ValueError:
                    continue
                on_progress(min(1.0, max(0.0, done / duration)) if duration else 0.0)
        code = proc.wait()
        errlog.seek(0)
        stderr = errlog.read()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed:\n{stderr[-3000:]}")
    if on_progress:
        on_progress(1.0)
    return output
