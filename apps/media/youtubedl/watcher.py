import os, re, time, shutil
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
ARTIST_TITLE_RE = re.compile(r"^\s*(?P<artist>[^-–—]+?)\s*[-–—]\s*(?P<title>.+?)\s*$")

def sanitize(name: str) -> str:
  name = name.strip()
  name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)
  name = re.sub(r"\s+", " ", name).rstrip(" .")
  return name or UNKNOWN_ARTIST

def is_stable(path: Path) -> bool:
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

def infer_artist_title(path: Path):
  stem = path.stem
  m = ARTIST_TITLE_RE.match(stem)
  if m:
    return sanitize(m.group("artist")), sanitize(m.group("title"))
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

def move_or_copy(src: Path, dst: Path):
  dst.parent.mkdir(parents=True, exist_ok=True)
  if COPY_MODE:
    shutil.copy2(src, dst)
  else:
    shutil.move(str(src), str(dst))

class Handler(FileSystemEventHandler):
  def _maybe(self, p: Path):
    ext = p.suffix.lower()
    if ext in SKIP_EXTS or ext not in MEDIA_EXTS:
      return
    try:
      p.relative_to(INCOMING)
    except ValueError:
      return
    if not is_stable(p):
      return
    artist, title = infer_artist_title(p)
    dst = resolve_collision(ORGANIZED / artist / f"{title}{ext}")
    try:
      move_or_copy(p, dst)
      print(f"[OK] {p} -> {dst}", flush=True)
    except Exception as e:
      print(f"[ERR] {p}: {e}", flush=True)

  def on_created(self, e):
    if not e.is_directory: self._maybe(Path(e.src_path))
  def on_moved(self, e):
    if not e.is_directory: self._maybe(Path(e.dest_path))
  def on_modified(self, e):
    if not e.is_directory: self._maybe(Path(e.src_path))

def main():
  INCOMING.mkdir(parents=True, exist_ok=True)
  ORGANIZED.mkdir(parents=True, exist_ok=True)
  obs = Observer()
  obs.schedule(Handler(), str(INCOMING), recursive=True)
  obs.start()
  print(f"Watching {INCOMING} -> {ORGANIZED} stable={STABLE_SECONDS}s", flush=True)
  try:
    while True: time.sleep(5)
  except KeyboardInterrupt:
    obs.stop()
  obs.join()

if __name__ == "__main__":
  main()