"""Resolve a clip source: a local path, a Google Drive share link, or any http(s) URL."""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

_DRIVE_ID_PATTERNS = (
    r"drive\.google\.com/file/d/([\w-]+)",
    r"drive\.google\.com/open\?id=([\w-]+)",
    r"drive\.google\.com/uc\?(?:.*&)?id=([\w-]+)",
    r"docs\.google\.com/.*/d/([\w-]+)",
)


def drive_file_id(url: str) -> str | None:
    for pattern in _DRIVE_ID_PATTERNS:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def is_drive_folder(url: str) -> bool:
    return "drive.google.com" in url and "/folders/" in url


def fetch(source: str, dest_dir: str | Path, name: str) -> Path:
    """Return a local path for `source`, downloading it into `dest_dir` if needed.

    Drive files must be shared as "Anyone with the link can view".
    """
    source = source.strip()
    if not source.startswith(("http://", "https://")):
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"No such file: {path}")
        return path

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if is_drive_folder(source):
        raise ValueError(
            "That's a Drive folder link. Open the folder, right-click the video, "
            "choose Share > Copy link, and paste that file link instead."
        )

    file_id = drive_file_id(source)
    if file_id:
        import gdown  # imported lazily so local-only use doesn't need it

        out = gdown.download(id=file_id, output=str(dest_dir) + "/", quiet=True)
        if not out:
            raise RuntimeError(
                "Couldn't download from Google Drive. Make sure the file is shared as "
                "'Anyone with the link can view'."
            )
        return Path(out)

    suffix = Path(source.split("?")[0]).suffix or ".mp4"
    path = dest_dir / f"{name}{suffix}"
    urllib.request.urlretrieve(source, path)
    return path
