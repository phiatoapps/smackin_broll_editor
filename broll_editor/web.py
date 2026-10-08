"""Local web UI: python -m broll_editor.web, then open http://localhost:5000"""

from __future__ import annotations

import os
import threading
import traceback
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from .sources import fetch
from .stacker import StackOptions, stack_videos

WORK_DIR = Path(os.environ.get("BROLL_WORK_DIR", "workspace")).resolve()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024**3  # 4 GB uploads

# job_id -> {"status": queued|downloading|rendering|done|error, "progress": float, "error": str}
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def _update(job_id: str, **fields) -> None:
    with jobs_lock:
        jobs[job_id].update(fields)


def _float(form, key: str, default: float) -> float:
    value = form.get(key, "")
    return float(value) if value not in ("", None) else default


def _options_from_form(form) -> StackOptions:
    max_dur = form.get("max_duration", "").strip()
    return StackOptions(
        split=_float(form, "split", 50) / 100,
        top_fit=form.get("top_fit", "cover"),
        bottom_fit=form.get("bottom_fit", "cover"),
        top_focus_x=_float(form, "top_focus_x", 50) / 100,
        top_focus_y=_float(form, "top_focus_y", 50) / 100,
        bottom_focus_x=_float(form, "bottom_focus_x", 50) / 100,
        bottom_focus_y=_float(form, "bottom_focus_y", 50) / 100,
        top_start=_float(form, "top_start", 0),
        bottom_start=_float(form, "bottom_start", 0),
        duration_mode=form.get("duration_mode", "top"),
        max_duration=float(max_dur) if max_dur else None,
        audio_mode=form.get("audio_mode", "top"),
        bottom_volume=_float(form, "bottom_volume", 15) / 100,
        divider_px=int(_float(form, "divider_px", 0)),
        divider_color=form.get("divider_color", "#ffffff").replace("#", "0x"),
        captions=form.get("captions") == "on",
        caption_words=int(_float(form, "caption_words", 3)),
        caption_size=int(_float(form, "caption_size", 84)),
        caption_color=form.get("caption_color", "#FFFFFF"),
        caption_highlight=form.get("caption_highlight", "#FFE135"),
        caption_uppercase=form.get("caption_uppercase") == "on",
        preset=form.get("preset", "medium"),
    )


def _run_job(job_id: str, job_dir: Path, top_src: str, bottom_src: str, opts: StackOptions) -> None:
    try:
        _update(job_id, status="downloading")
        top = fetch(top_src, job_dir / "top", "top")
        bottom = fetch(bottom_src, job_dir / "bottom", "bottom")
        stack_videos(top, bottom, job_dir / "stacked.mp4", opts,
                     on_progress=lambda f: _update(job_id, progress=f),
                     on_status=lambda stage: _update(job_id, status=stage))
        _update(job_id, status="done", progress=1.0)
    except Exception as exc:
        traceback.print_exc()
        _update(job_id, status="error", error=str(exc))


def _source(kind: str, job_dir: Path) -> str:
    """Uploaded file wins over a pasted link."""
    upload = request.files.get(f"{kind}_file")
    if upload and upload.filename:
        dest = job_dir / kind
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / secure_filename(upload.filename)
        upload.save(path)
        return str(path)
    link = request.form.get(f"{kind}_url", "").strip()
    if not link.startswith(("http://", "https://")):
        abort(400, f"Provide a {kind} video file or a Google Drive / http link.")
    return link


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/render")
def render():
    job_id = uuid.uuid4().hex[:12]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    top_src = _source("top", job_dir)
    bottom_src = _source("bottom", job_dir)
    try:
        opts = _options_from_form(request.form)
        opts.validate()
    except ValueError as exc:
        abort(400, str(exc))
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "progress": 0.0, "error": None}
    threading.Thread(target=_run_job, args=(job_id, job_dir, top_src, bottom_src, opts), daemon=True).start()
    return jsonify(job_id=job_id)


@app.get("/status/<job_id>")
def status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            abort(404)
        return jsonify(job)


@app.get("/video/<job_id>")
def video(job_id: str):
    with jobs_lock:
        if jobs.get(job_id, {}).get("status") != "done":
            abort(404)
    download = request.args.get("download") == "1"
    return send_file(WORK_DIR / job_id / "stacked.mp4", mimetype="video/mp4",
                     as_attachment=download, download_name=f"stacked_{job_id}.mp4")


@app.errorhandler(400)
def bad_request(err):
    return jsonify(error=err.description), 400


def main() -> None:
    port = int(os.environ.get("PORT", "5000"))
    print(f"B-roll editor running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()
