# Clone edit playbook (one video per session)

You are editing ONE raw phone clip into a finished 1080x1920 split-screen "clone"
video. One locked-off shot: the person plays clone A in one stretch of the clip and
clone B in another, standing at different spots. The tools in this repo composite
them into one frame. Read README.md for the tools; this file is the job.

## The rules (the user's, non-negotiable)

1. **Hook:** open on both clones in frame (the tools do a quick pull-out to the wide shot).
2. **Unison opening (Clone Argument format):** both clones say the first line together:
   a `"U"` pick with both takes of that line.
3. **Labels** over each clone for the whole video, from the script doc. The product
   ("your product", 🅱️ in the doc) always gets 2027; the old way gets the other label.
   Match the label to whichever side that clone actually stands on.
4. **No dead air.** Not even 0.3s. QC must show no pauses over ~0.16s.
5. **Never cut off a word** and never leak the first sound of the next word.
6. **Listeners face the speaker.** Never looking the other way (the gaze tools handle it; verify).
7. **No see-through people, no ghosting, no visible middle line.**
8. **Every frame grades 8/10 or better.** Look at the grading sheet yourself and fix anything lower.
9. **Follow the script's order and lines,** using what was actually filmed (ad-libs that
   fit are fine). Respect the doc's warnings: no invented claims, no prices in the
   Clone Argument format. Note anything borderline for the user rather than guessing.

## Workflow (about 45 minutes)

```bash
pip install -q -r requirements.txt; (apt-get install -y -q ffmpeg libegl1 libgles2 >/dev/null 2>&1 || true)
mkdir -p work/raw && cd work/raw && gdown <FILE_ID> && cd ../..
python tools/ingest.py work/raw/<file>.MOV work/V          # proxy, transcript, phrases, sides (~15 min)
```
1. Read `work/V/phrases.txt` (every speech stretch, transcribed on its own; more
   reliable than transcript.txt, which drops lines) and `sides.txt` (which clone stands where).
   Match the clip to its script in the doc. Ignore "Thanks for watching!" style lines:
   whisper invents them in silence.
2. Write `work/V/cfg.json` (format in tools/build_cfg.py): `A` = the clone on the LEFT,
   `B` = the clone on the RIGHT, `[lo, hi, center]` = the seconds of the clip that clone's
   takes span and where they stand (0..1). Picks in script order, zoom always `false`,
   the first pick `"U"`. Use the best take of each line (usually the last clean one).
   If both clones stay on their own half (sides.txt extents don't overlap), set
   `"band": [0.40, 0.62]`: it stops one take's lighting drift from smearing over the other clone.
3. `python tools/build_cfg.py work/V` and read its output: "extended end ... to finish
   'X'" must not swallow the next word; a "mostly silent" warning means the timing is wrong.
4. `python tools/render.py work/V --draft` (~2 min). Read `qc.txt`:
   - `pauses` must be empty or ≤0.16s
   - the **readback** line must read as the script: no doubled, missing, or clipped words,
     and no stray first word of the next sentence. Fix the pick times (check the
     audio envelope in `env.npy`, 100 values per second) and redo steps 3-4.
   - for a soft last word the envelope misses, use `"ends": {"text prefix": raw_end}`.
5. `python tools/render.py work/V` (final, ~5 min). Open `final_grade.png` and grade every
   frame: faces whole, labels on, nothing smeared or see-through, listeners facing the
   speaker. Fix and re-render anything under 8.
6. Deliver:
   ```bash
   mkdir -p deliver && bash tools/compress.sh work/V/final.mp4 deliver/<Name>_Clones.mp4
   cp work/V/cfg.json deliver/<Name>_cfg.json; cp work/V/qc.txt deliver/<Name>_qc.txt
   cp work/V/final_grade.png deliver/<Name>_grade.png
   git checkout -b <your delivery branch> && git add -f deliver && git commit -m "<Name> clone edit" && git push -u origin HEAD
   ```
   Then report: the branch, length, labels, and any script/claim notes for the user.

## Gotchas already learned

- `zoom: true` on a pick makes an old-style punch-in that hides labels. Never use it;
  framing (wide/medium/tight jump zooms) is automatic.
- Whisper stretches word ends over silence; trust the envelope (`env.npy`) for exact cut points.
- A pick's end landing inside a word gets extended to finish it, which can pull in the
  start of the next word ("expensive, but"). End the pick just before the next word.
- Takes are usually filmed twice; the second is normally cleaner.
