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
| `--captions` | Auto-caption the talking head (Whisper), centered on the divider, spoken word highlighted | off |
| `--transcript words.json` | Save the caption words here; if the file exists it's reused, so you can edit it to fix typos and re-render | |
| `--caption-words 3 --caption-size 84` | Words shown at once / text size | |
| `--caption-color "#FFFFFF" --caption-highlight "#FFE135"` | Caption colors | |
| `--no-caption-caps` | Normal casing instead of ALL CAPS | |
| `--size 1080x1920` | Output resolution | `1080x1920` |
| `--crf 20 --preset medium` | Quality / speed trade-off | |

### Captions

```bash
python -m broll_editor head.mp4 broll.mp4 -o final.mp4 --divider 8 --captions --transcript words.json
```

The first run downloads a Whisper speech model (~500 MB for `small`) and
writes `words.json`. Fix any misheard words in that file and run the same
command again; it reuses your edited words instead of re-transcribing.

Captions use Montserrat ExtraBold (bundled, SIL Open Font License, see `broll_editor/fonts/OFL.txt`).

## Split-screen dialogue (one person, two characters)

Film one locked-off shot: first half you play one character on one side of the
frame, second half you play the other character on the other side. Give it the
script and it builds the "talking to yourself" split screen:

```bash
python -m broll_editor.splitscreen raw.mp4 script.txt -o output/final.mp4 \
    --transcript words.json --plan plan.json
```

`script.txt`, one line of dialogue per line:

```
Toilet Brush: I'm the toilet brush.
Lysol: That's disgusting.
[zoom] Toilet Brush: I live in a puddle of my own water.
```

What it does automatically:

- finds where the raw clip switches halves and which side you started on (from where the motion is),
- puts the mask seam in the gap between your two positions, with a soft edge so it's invisible,
- transcribes the clip (Whisper) and finds each script line in its speaker's half; if you said a line more than once, the last take wins,
- plays the lines back to back in script order: the speaker's side plays the line, the other side plays your "listening" footage from the other take,
- `[zoom]` lines punch in full-frame on the speaker.

Useful options: `--switch 41.5`, `--first-side left`, `--seam 0.5` override the
auto-detection; `--pad-before` / `--pad-after` control how tight the cuts are;
`--left-label "Detailed|Car" --right-label "Clean|Car"` add name tags. Lines it
couldn't find are listed at the end. `plan.json` holds every cut; edit it and
re-run the same command to tweak timing without re-transcribing.

## Use from Python

```python
from broll_editor import StackOptions, stack_videos

stack_videos("head.mp4", "broll.mp4", "out.mp4",
             StackOptions(split=0.45, audio_mode="mix", divider_px=6))
```
