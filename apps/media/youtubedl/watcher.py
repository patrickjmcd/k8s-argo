import os
import re
import time
import json
import shutil
import hashlib
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ---------------------------
# Paths / config
# ---------------------------
INCOMING = Path(os.getenv("INCOMING_DIR", "/incoming"))
ORGANIZED = Path(os.getenv("ORGANIZED_DIR", "/organized"))
NEEDS_REVIEW = Path(os.getenv("NEEDS_REVIEW_DIR", str(ORGANIZED / "Needs Review")))

STABLE_SECONDS = int(os.getenv("STABLE_SECONDS", "15"))
UNKNOWN_ARTIST = os.getenv("UNKNOWN_ARTIST", "Unknown Artist")
COPY_MODE = os.getenv("COPY_MODE", "0") == "1"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
USE_OPENAI = os.getenv("USE_OPENAI", "1") == "1"

# Confidence thresholds
CONF_AUTO = float(os.getenv("CONF_AUTO", "0.85"))
CONF_REVIEW = float(os.getenv("CONF_REVIEW", "0.70"))

# Cache
CACHE_DB = Path(os.getenv("CACHE_DB", "/cache/artist_cache.sqlite3"))

SKIP_EXTS = {".part", ".tmp", ".ytdl", ".download", ".crdownload"}
MEDIA_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
THUMB_EXTS = {".webp", ".jpg", ".jpeg", ".png"}

DEFAULT_PUBLISHERS = {
    "mtv", "npr music", "tiny desk", "kexp", "austin city limits", "bbc radio",
    "vevo", "the tonight show starring jimmy fallon", "the late late show",
    "jimmy kimmel live", "siriusxm", "pitchfork", "colors", "triple j", "mix"
}
PUBLISHER_CHANNELS = {
    s.strip().lower() for s in os.getenv("PUBLISHER_CHANNELS", "").split(",") if s.strip()
} or DEFAULT_PUBLISHERS

try:
    CHANNEL_ARTIST_OVERRIDES = json.loads(os.getenv("CHANNEL_ARTIST_OVERRIDES", "{}"))
    if not isinstance(CHANNEL_ARTIST_OVERRIDES, dict):
        CHANNEL_ARTIST_OVERRIDES = {}
except Exception:
    CHANNEL_ARTIST_OVERRIDES = {}

ARTIST_TITLE_SPLIT_RE = re.compile(r"^\s*(?P<artist>.+?)\s*[-–—:|]\s*(?P<title>.+?)\s*$")

PERFORMANCE_RE = re.compile(
    r"^\s*['\"“”‘’]?(?P<song>.+?)['\"“”‘’]?\s+(?P<artist>[A-Za-z0-9 &.+/'’“-]+?)\s+"
    r"(performance|perform|performs|performed|plays|playing)\b",
    re.IGNORECASE,
)

FEAT_RE = re.compile(
    r"^\s*(?P<artist>[A-Za-z0-9 &.+/'’“-]+?)\s+(ft\.?|feat\.?|featuring)\b",
    re.IGNORECASE,
)

# ---------------------------
# OpenAI client (lazy import)
# ---------------------------
_openai_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai python package not installed. Add `pip install openai`") from e
    _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client

SYSTEM_PROMPT = (
    "You are classifying music videos.\n"
    "Identify the PRIMARY performing artist.\n"
    "Ignore publishers (MTV, NPR, Tiny Desk, KEXP, etc).\n"
    "If multiple artists appear, choose the main act.\n"
    "Return ONLY valid JSON with keys: artist, confidence.\n"
    "confidence is 0.0-1.0.\n"
)

# ---------------------------
# Cache helpers
# ---------------------------
def init_cache():
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(CACHE_DB)) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS artist_cache (
              key TEXT PRIMARY KEY,
              artist TEXT NOT NULL,
              confidence REAL NOT NULL,
              model TEXT,
              created_at INTEGER NOT NULL
            )
            """
        )
        con.commit()

def cache_get(key: str) -> Optional[Tuple[str, float]]:
    with sqlite3.connect(str(CACHE_DB)) as con:
        row = con.execute(
            "SELECT artist, confidence FROM artist_cache WHERE key=?",
            (key,),
        ).fetchone()
        if row:
            return row[0], float(row[1])
    return None

def cache_put(key: str, artist: str, confidence: float, model: str):
    with sqlite3.connect(str(CACHE_DB)) as con:
        con.execute(
            "INSERT OR REPLACE INTO artist_cache(key, artist, confidence, model, created_at) VALUES(?,?,?,?,?)",
            (key, artist, confidence, model, int(time.time())),
        )
        con.commit()

# ---------------------------
# Utility
# ---------------------------
def sanitize(name: str) -> str:
    name = (name or "").strip()
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

def find_info_json(media_path: Path) -> Optional[Path]:
    p = media_path.with_suffix(media_path.suffix + ".info.json")
    return p if p.exists() else None

def load_info(info_path: Optional[Path]) -> Optional[dict]:
    if not info_path or not info_path.exists():
        return None
    try:
        with info_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def title_from_info(info: dict, media_path: Path) -> str:
    for key in ("track", "title", "fulltitle"):
        v = info.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return media_path.stem

def channel_from_info(info: dict) -> Optional[str]:
    for key in ("channel", "uploader", "creator"):
        v = info.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def best_music_artist(info: dict) -> Optional[str]:
    for key in ("artist", "album_artist"):
        v = info.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def artist_from_title_heuristics(title: str, channel: Optional[str]) -> Optional[str]:
    t = (title or "").strip()
    m = ARTIST_TITLE_SPLIT_RE.match(t)
    if m:
        artist = m.group("artist").strip()
        if channel and artist.lower() == channel.strip().lower():
            return None
        return artist

    m = PERFORMANCE_RE.match(t)
    if m:
        return m.group("artist").strip()

    m = FEAT_RE.match(t)
    if m:
        return m.group("artist").strip()

    return None

def choose_artist_rules(info: Optional[dict], media_path: Path) -> Tuple[str, float, str]:
    """
    Returns: (artist, confidence, source)
    confidence here is "rule-confidence", not OpenAI.
    """
    if not info:
        m = ARTIST_TITLE_SPLIT_RE.match(media_path.stem)
        if m:
            return sanitize(m.group("artist")), 0.75, "filename-split"
        return UNKNOWN_ARTIST, 0.30, "no-info"

    channel = channel_from_info(info)
    channel_lc = channel.lower().strip() if channel else ""

    if channel and channel in CHANNEL_ARTIST_OVERRIDES:
        return sanitize(CHANNEL_ARTIST_OVERRIDES[channel]), 0.95, "channel-override"

    a = best_music_artist(info)
    if a:
        return sanitize(a), 0.95, "music-metadata"

    title = title_from_info(info, media_path)

    if channel_lc in PUBLISHER_CHANNELS:
        inferred = artist_from_title_heuristics(title, channel)
        if inferred:
            return sanitize(inferred), 0.85, "title-heuristic(publisher)"

        tags = info.get("tags")
        if isinstance(tags, list):
            for tag in tags:
                if not isinstance(tag, str):
                    continue
                tl = tag.strip().lower()
                if tl and tl != channel_lc and tl not in PUBLISHER_CHANNELS and len(tag.strip()) >= 3:
                    return sanitize(tag), 0.65, "tag-fallback(publisher)"

        return UNKNOWN_ARTIST, 0.20, "publisher-unknown"

    if channel:
        return sanitize(channel), 0.70, "channel-as-artist"

    return UNKNOWN_ARTIST, 0.20, "unknown"

# ---------------------------
# OpenAI inference
# ---------------------------
def build_metadata_text(info: dict, media_path: Path) -> str:
    """
    Keep it compact and avoid sending URLs / giant fields.
    """
    title = title_from_info(info, media_path)
    desc = info.get("description") if isinstance(info.get("description"), str) else ""
    tags = info.get("tags") if isinstance(info.get("tags"), list) else []
    playlist = info.get("playlist_title") or info.get("playlist") or ""

    # trim description to keep token costs sane
    desc = desc.strip()
    if len(desc) > 2000:
        desc = desc[:2000] + "…"

    # limit tags
    tags = [t for t in tags if isinstance(t, str)]
    tags = tags[:40]

    channel = channel_from_info(info) or ""

    return (
        f"TITLE:\n{title}\n\n"
        f"CHANNEL/UPLOADER:\n{channel}\n\n"
        f"PLAYLIST:\n{playlist}\n\n"
        f"TAGS:\n{', '.join(tags)}\n\n"
        f"DESCRIPTION:\n{desc}\n"
    )

def cache_key(info: dict, media_path: Path) -> str:
    """
    Stable key for caching. Prefer YouTube id when present.
    """
    vid = info.get("id") if isinstance(info.get("id"), str) else ""
    title = info.get("title") if isinstance(info.get("title"), str) else media_path.stem
    uploader = info.get("uploader") if isinstance(info.get("uploader"), str) else ""
    base = f"{vid}|{title}|{uploader}"
    return hashlib.sha256(base.encode("utf-8", errors="ignore")).hexdigest()

def infer_artist_openai(info: dict, media_path: Path) -> Tuple[str, float]:
    txt = build_metadata_text(info, media_path)

    client = get_openai_client()
    user_prompt = (
        "Metadata:\n"
        f"{txt}\n\n"
        "Return format:\n"
        '{"artist":"<name>","confidence":0.0}\n'
    )

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_output_tokens=120,
    )

    out = (resp.output_text or "").strip()
    result = json.loads(out)

    artist = sanitize(str(result.get("artist", UNKNOWN_ARTIST)))
    conf = float(result.get("confidence", 0.75))
    if conf < 0.0:
        conf = 0.0
    if conf > 1.0:
        conf = 1.0
    return artist, conf

# ---------------------------
# Moving / bundling
# ---------------------------
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

def bundle_paths(media_path: Path) -> list[Path]:
    bundle = [media_path]
    info = find_info_json(media_path)
    if info and info.exists():
        bundle.append(info)
    for ext in THUMB_EXTS:
        p = media_path.with_suffix(ext)
        if p.exists():
            bundle.append(p)
    return bundle

def decide_artist(info: Optional[dict], media_path: Path) -> Tuple[str, float, str]:
    """
    Decide artist using:
    - cache
    - rules
    - OpenAI fallback
    """
    if not info:
        artist, conf, src = choose_artist_rules(info, media_path)
        return artist, conf, src

    # cache
    key = cache_key(info, media_path)
    cached = cache_get(key)
    if cached:
        return cached[0], cached[1], "cache"

    # rules first
    artist_r, conf_r, src_r = choose_artist_rules(info, media_path)

    # If rules are confident enough, accept and cache
    if conf_r >= CONF_AUTO or not USE_OPENAI or not OPENAI_API_KEY:
        cache_put(key, artist_r, conf_r, f"rules:{src_r}")
        return artist_r, conf_r, f"rules:{src_r}"

    # If rules are weak, ask OpenAI
    try:
        artist_ai, conf_ai = infer_artist_openai(info, media_path)
        cache_put(key, artist_ai, conf_ai, f"openai:{OPENAI_MODEL}")
        return artist_ai, conf_ai, f"openai:{OPENAI_MODEL}"
    except Exception as e:
        # fall back to rules if OpenAI fails
        cache_put(key, artist_r, conf_r, f"rules-fallback:{src_r} ({e})")
        return artist_r, conf_r, f"rules-fallback:{src_r}"

def process_media(media_path: Path) -> None:
    if not media_path.exists() or media_path.is_dir():
        return
    ext = media_path.suffix.lower()
    if ext in SKIP_EXTS or ext not in MEDIA_EXTS:
        return
    try:
        media_path.relative_to(INCOMING)
    except ValueError:
        return
    if not is_stable(media_path):
        return

    info_path = find_info_json(media_path)
    info = load_info(info_path)

    title = sanitize(title_from_info(info, media_path) if info else media_path.stem)
    artist, conf, src = decide_artist(info, media_path)

    # confidence gating
    if conf < CONF_REVIEW:
        target_root = NEEDS_REVIEW / artist
    else:
        target_root = ORGANIZED / artist

    dst_media = resolve_collision(target_root / f"{title}{ext}")
    dst_dir = dst_media.parent
    dst_stem = dst_media.stem

    b = bundle_paths(media_path)

    try:
        move_or_copy(media_path, dst_media)
        for p in b:
            if p == media_path:
                continue
            if not p.exists():
                continue
            if p.name.endswith(".info.json"):
                dst = dst_dir / f"{dst_stem}.info.json"
            else:
                dst = dst_dir / f"{dst_stem}{p.suffix.lower()}"
            dst = resolve_collision(dst)
            move_or_copy(p, dst)

        print(f"[OK] artist={artist} conf={conf:.2f} src={src} -> {dst_media}", flush=True)
    except Exception as e:
        print(f"[ERR] Failed {media_path}: {e}", flush=True)

def initial_sweep() -> None:
    print(f"Initial sweep of {INCOMING} ...", flush=True)
    n = 0
    for p in sorted(INCOMING.rglob("*"), key=lambda x: str(x)):
        if p.is_file() and p.suffix.lower() in MEDIA_EXTS:
            before = p.exists()
            process_media(p)
            if before and not p.exists():
                n += 1
    print(f"Initial sweep complete. Processed {n} media file(s).", flush=True)

class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            p = Path(event.src_path)
            if p.suffix.lower() in MEDIA_EXTS:
                process_media(p)

    def on_moved(self, event):
        if not event.is_directory:
            p = Path(event.dest_path)
            if p.suffix.lower() in MEDIA_EXTS:
                process_media(p)

    def on_modified(self, event):
        if not event.is_directory:
            p = Path(event.src_path)
            if p.suffix.lower() in MEDIA_EXTS:
                process_media(p)

def main():
    INCOMING.mkdir(parents=True, exist_ok=True)
    ORGANIZED.mkdir(parents=True, exist_ok=True)
    NEEDS_REVIEW.mkdir(parents=True, exist_ok=True)

    init_cache()
    initial_sweep()

    observer = Observer()
    observer.schedule(Handler(), str(INCOMING), recursive=True)
    observer.start()
    print(
        f"Watching {INCOMING} -> {ORGANIZED} (review->{NEEDS_REVIEW}) "
        f"stable={STABLE_SECONDS}s openai={USE_OPENAI and bool(OPENAI_API_KEY)} model={OPENAI_MODEL}",
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
