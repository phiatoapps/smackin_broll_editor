"""Auto-captions: transcribe speech with Whisper and render word-by-word ASS subtitles.

Captions are shown a few words at a time, centered on the seam between the two
panels, with the word being spoken highlighted (the TikTok / CapCut look).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

FONTS_DIR = Path(__file__).parent / "fonts"
FONT_NAME = "Montserrat ExtraBold"


@dataclass
class Word:
    text: str
    start: float
    end: float


def transcribe(media: str | Path, model_size: str = "small", language: str | None = None) -> list[Word]:
    """Word-level transcript of `media`'s audio track."""
    import numpy as np
    from faster_whisper import WhisperModel  # heavy import, only when captioning

    # Decode with ffmpeg ourselves: works for any container ffmpeg can read.
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(media), "-vn", "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
        check=True, capture_output=True,
    ).stdout
    audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio, word_timestamps=True, language=language, vad_filter=True)
    words = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append(Word(text, float(w.start), float(w.end)))
    return words


def save_transcript(words: list[Word], path: str | Path) -> None:
    Path(path).write_text(json.dumps([asdict(w) for w in words], indent=1))


def load_transcript(path: str | Path) -> list[Word]:
    return [Word(**w) for w in json.loads(Path(path).read_text())]


def shift_words(words: list[Word], offset: float, duration: float) -> list[Word]:
    """Re-time words for a clip trimmed to start at `offset` and last `duration`."""
    out = []
    for w in words:
        start, end = w.start - offset, min(w.end - offset, duration)
        if end > 0 and start < duration:
            out.append(Word(w.text, max(0.0, start), end))
    return out


def group_words(words: list[Word], max_words: int = 3, max_gap: float = 0.6) -> list[list[Word]]:
    """Split words into short caption chunks, breaking at punctuation and pauses."""
    chunks: list[list[Word]] = []
    current: list[Word] = []
    for w in words:
        if current and (len(current) >= max_words or w.start - current[-1].end > max_gap):
            chunks.append(current)
            current = []
        current.append(w)
        if w.text[-1] in ".?!,;:":
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def _ass_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _ass_color(color: str) -> str:
    """'#RRGGBB' -> ASS '&H00BBGGRR'."""
    c = color.lstrip("#")
    if len(c) != 6:
        raise ValueError(f"color must look like #RRGGBB, got {color!r}")
    return f"&H00{c[4:6]}{c[2:4]}{c[0:2]}".upper()


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def build_ass(
    words: list[Word],
    width: int,
    height: int,
    center_y: int,
    *,
    max_words: int = 3,
    font_size: int = 84,
    color: str = "#FFFFFF",
    highlight: str = "#FFE135",
    uppercase: bool = True,
    hold: float = 0.4,
) -> str:
    """ASS subtitle script with one caption chunk at a time, centered at (width/2, center_y)."""
    base, hi = _ass_color(color), _ass_color(highlight)
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Cap,{FONT_NAME},{font_size},{base},{base},&H00000000,&H80000000,"
        f"0,0,0,0,100,100,0,0,1,{max(2, font_size // 12)},{max(1, font_size // 24)},5,60,60,0,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    pos = f"{{\\pos({width // 2},{center_y})}}"
    events = []
    chunks = group_words(words, max_words)
    for i, chunk in enumerate(chunks):
        next_start = chunks[i + 1][0].start if i + 1 < len(chunks) else None
        chunk_end = chunk[-1].end + hold
        if next_start is not None:
            chunk_end = min(chunk_end, next_start)
        # Trailing commas/periods look cluttered on-screen; keep ? and !.
        labels = [_escape((w.text.upper() if uppercase else w.text).rstrip(".,;:") or w.text) for w in chunk]
        # One event per word so the spoken word lights up; the last one holds to chunk_end.
        for j, w in enumerate(chunk):
            start = chunk[0].start if j == 0 else w.start
            end = chunk[j + 1].start if j + 1 < len(chunk) else chunk_end
            if end <= start:
                continue
            text = " ".join(
                f"{{\\c{hi}}}{t}{{\\c{base}}}" if k == j else t for k, t in enumerate(labels)
            )
            events.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Cap,,0,0,0,,{pos}{text}")
    return header + "\n".join(events) + "\n"
