import os
import re
import time
import shutil
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

INCOMING = Path("/incoming")
ORGANIZED = Path("/organized")

STABLE_SECONDS = int(os.getenv("STABLE_SECONDS", "15"))
UNKNOWN_ARTIST = os.getenv("UNKNOWN_ARTIST", "Unknown Artist")
COPY_MODE = os.getenv("COPY_MODE", "0") == "1"

SKIP_EXTS = {".part", ".tmp", ".ytdl", ".download", ".crdownload"}
MEDIA_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}

# Pattern: "Artist - Title" (common)
ARTIST_TITLE_RE = re.compile(r"^\s*(?P<artist>[^-–—]+?)\s*[-–—]\s*(?P<title>.+?)\s*$")


def sanitize(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)
    name = re.sub(r"\s+", " ", name).rstrip(" .")
    return name or UNKNOWN_ARTIST


def is_stable(path: Path) -> bool:
    """Consider a file done if size hasn't changed for STABLE_SECONDS."""
    last_size = -1
    stable_for = 0
    while stable_for < STABLE_SECONDS:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        if size == last_size:
            stable_for += 1
        else:
            stable_for = 0
            last_size = size
        time.sleep(1)
    return True


def infer_artist_title(path: Path) -> tuple[str, str]:
    stem = path.stem

    m = ARTIST_TITLE_RE.match(stem)
    if m:
        return sanitize(m.group("artist")), sanitize(m.group("title"))

    # Fallback: first subfolder under Incoming
    try:
        rel_parent = path.parent.relative_to(INCOMING)
        artist = sanitize(rel_parent.parts[0]) if rel_parent.parts else UNKNOWN_ARTIST
    except ValueError:
        artist = UNKNOWN_ARTIST

    return artist, sanitize(stem)


def resolve_collision(dst: Path) -> Path:
    if not dst.exists():
        return dst
    base, ext, parent = dst.stem, dst.suffix, dst.parent
    for i in range(2, 9999):
        c = parent / f"{base} ({i}){ext}"
        if not c.exists():
            return c
    raise RuntimeError(f"Too many collisions for {dst}")


def move_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if COPY_MODE:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def process_path(path: Path) -> None:
    """
    Process a single file if it looks like a finished media file inside INCOMING.
    Safe to call repeatedly (will skip non-matching files).
    """
    if path.is_dir():
        return

    ext = path.suffix.lower()
    if ext in SKIP_EXTS or ext not in MEDIA_EXTS:
        return

    try:
        path.relative_to(INCOMING)
    except ValueError:
        return

    # Wait until file stops changing
    if not is_stable(path):
        return

    artist, title = infer_artist_title(path)
    dst = ORGANIZED / artist / f"{title}{ext}"
    dst = resolve_collision(dst)

    try:
        move_or_copy(path, dst)
        print(f"[OK] {path} -> {dst}", flush=True)
    except Exception as e:
        print(f"[ERR] Failed to process {path}: {e}", flush=True)


def initial_sweep() -> None:
    """
    Process any existing media files already present in INCOMING at startup.
    """
    print(f"Initial sweep of {INCOMING} ...", flush=True)

    # Sort for stable/pleasant logs (folders first, then files)
    candidates = sorted(INCOMING.rglob("*"), key=lambda p: (p.is_dir(), str(p)))

    count = 0
    for p in candidates:
        if p.is_file():
            before = p.exists()
            process_path(p)
            # If we moved it, it won't exist here anymore
            if before and not p.exists():
                count += 1

    print(f"Initial sweep complete. Moved/copied {count} file(s).", flush=True)


class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            process_path(Path(event.src_path))

    def on_moved(self, event):
        if not event.is_directory:
            process_path(Path(event.dest_path))

    def on_modified(self, event):
        if not event.is_directory:
            # noisy but safe; stability check prevents half-files
            process_path(Path(event.src_path))


def main():
    INCOMING.mkdir(parents=True, exist_ok=True)
    ORGANIZED.mkdir(parents=True, exist_ok=True)

    # NEW: handle pre-existing files
    initial_sweep()

    # Then keep watching for new ones
    observer = Observer()
    observer.schedule(Handler(), str(INCOMING), recursive=True)
    observer.start()
    print(
        f"Watching {INCOMING} -> {ORGANIZED} (stable={STABLE_SECONDS}s, copy={COPY_MODE})",
        flush=True,
    )

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
