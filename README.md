# Smackin B-roll Editor

Stack two clips into one vertical (9:16, 1080×1920) video:

```
+----------------+
|  talking head  |   top clip
+----------------+
|     b-roll     |   bottom clip
+----------------+
```

The react / podcast-clip format you see on TikTok, Reels and Shorts.

## Setup

Needs **Python 3.10+** and **ffmpeg**.

```bash
# macOS:   brew install ffmpeg
# Windows: winget install ffmpeg
# Linux:   sudo apt install ffmpeg
pip install -r requirements.txt
```

## Web app (easiest)

```bash
python -m broll_editor.web
```

Open <http://localhost:5000>, drop in (or paste Google Drive links for) the
talking-head and b-roll clips, adjust the layout in the live preview, then
click **Render video** and download the MP4.

## Command line

```bash
python -m broll_editor talking_head.mp4 broll.mp4 -o output/final.mp4

# Google Drive links work too (file must be shared "Anyone with the link"):
python -m broll_editor "https://drive.google.com/file/d/AAA/view" \
                       "https://drive.google.com/file/d/BBB/view" -o output/final.mp4
```

Useful options:

| Option | What it does | Default |
| --- | --- | --- |
| `--split 0.45` | Fraction of the height the talking head gets | `0.5` |
| `--top-focus X,Y` / `--bottom-focus X,Y` | Where to crop from, 0–1 (e.g. `0.5,0.3` keeps a face near the top) | `0.5,0.5` |
| `--top-fit blur` / `--bottom-fit blur` | Show the whole clip over a blurred background instead of cropping | `cover` |
| `--top-start` / `--bottom-start` | Seconds to skip at the start of each clip | `0` |
| `--duration top\|bottom\|shortest` | Which clip sets the length (the other loops if shorter) | `top` |
| `--max-duration 60` | Cap the output length in seconds | none |
| `--audio top\|mix\|bottom\|none` | Audio source; `mix` adds b-roll audio under the voice | `top` |
| `--bottom-volume 0.15` | B-roll volume for `--audio mix` | `0.15` |
| `--divider 6 --divider-color white` | Line between the two panels | off |
| `--size 1080x1920` | Output resolution | `1080x1920` |
| `--crf 20 --preset medium` | Quality / speed trade-off | |

## Use from Python

```python
from broll_editor import StackOptions, stack_videos

stack_videos("head.mp4", "broll.mp4", "out.mp4",
             StackOptions(split=0.45, audio_mode="mix", divider_px=6))
```
