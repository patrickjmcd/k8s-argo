#!/usr/bin/env python3
"""
watcher.py

Watches an INCOMING directory for youtube-dl / yt-dlp style downloads, then:
- Infers PRIMARY ARTIST + SONG TITLE (rules first, OpenAI fallback)
- Detects "full set" / concert-ish items
- Moves (or copies) the media + sidecar files into ORGANIZED/<Artist>/
- Writes a CLEANED + ENRICHED .info.json next to the moved media
- Generates an embedding (trimmed metadata) and stores it in SQLite using sqlite-vec
- Stores metadata in SQLite so you can join semantic results -> file paths
- Cleans up orphaned sidecars (.info.json, thumbs, subs) and empty folders under INCOMING

Requires:
  pip install watchdog openai sqlite-vec
"""

import os
import re
import time
import json
import shutil
import hashlib
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, List

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ============================================================
# Configuration (env vars)
# ============================================================
INCOMING = Path(os.getenv("INCOMING_DIR", "/incoming"))
ORGANIZED = Path(os.getenv("ORGANIZED_DIR", "/organized"))
NEEDS_REVIEW = Path(os.getenv("NEEDS_REVIEW_DIR", str(ORGANIZED / "Needs Review")))

STABLE_SECONDS = int(os.getenv("STABLE_SECONDS", "15"))
COPY_MODE = os.getenv("COPY_MODE", "0") == "1"

UNKNOWN_ARTIST = os.getenv("UNKNOWN_ARTIST", "Unknown Artist")
UNKNOWN_TITLE = os.getenv("UNKNOWN_TITLE", "Unknown Title")

MEDIA_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
SKIP_EXTS = {".part", ".tmp", ".ytdl", ".download", ".crdownload"}
THUMB_EXTS = {".webp", ".jpg", ".jpeg", ".png"}
SUB_EXTS = {".vtt", ".srt", ".ass", ".lrc"}

CONF_AUTO = float(os.getenv("CONF_AUTO", "0.85"))
CONF_REVIEW = float(os.getenv("CONF_REVIEW", "0.70"))

CACHE_DB = Path(os.getenv("CACHE_DB", "/cache/media.sqlite3"))

# OpenAI inference
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
USE_OPENAI = (os.getenv("USE_OPENAI", "1") == "1") and bool(OPENAI_API_KEY)

# Embeddings (SQLite-only via sqlite-vec)
ENABLE_EMBEDDINGS = (os.getenv("ENABLE_EMBEDDINGS", "1") == "1") and bool(OPENAI_API_KEY)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
# sqlite-vec requires a fixed dimension in the table schema.
EMBEDDING_DIMS = int(os.getenv("EMBEDDING_DIMS", "1536"))

# TRIMMED embedding content:
EMBED_TEXT_MAX_DESC = int(os.getenv("EMBED_TEXT_MAX_DESC", "1200"))
EMBED_TEXT_MAX_TAGS = int(os.getenv("EMBED_TEXT_MAX_TAGS", "30"))

# JSON cleaning
JSON_TRIM_DESCRIPTION_TO = int(os.getenv("JSON_TRIM_DESCRIPTION_TO", "2000"))

# Orphan cleanup
CLEAN_ORPHANS = os.getenv("CLEAN_ORPHANS", "1") == "1"
ORPHAN_GRACE_SECONDS = int(os.getenv("ORPHAN_GRACE_SECONDS", "3600"))  # default 1 hour
# Any of these may be deleted if orphaned (no matching media in same folder)
ORPHAN_SUFFIXES = {".info.json"} | THUMB_EXTS | SUB_EXTS

# ============================================================
# Regex + heuristics
# ============================================================
ARTIST_TITLE_RE = re.compile(r"^\s*(?P<artist>.+?)\s*[-–—:|]\s*(?P<title>.+?)\s*$")

PERFORMANCE_RE = re.compile(
    r"^\s*['\"“”‘’]?(?P<song>.+?)['\"“”‘’]?\s+(?P<artist>[A-Za-z0-9 &.+/'’“”\-]+?)\s+"
    r"(performance|perform|performs|performed|plays|playing)\b",
    re.IGNORECASE,
)

FEAT_RE = re.compile(
    r"^\s*(?P<artist>[A-Za-z0-9 &.+/'’“”\-]+?)\s+(ft\.?|feat\.?|featuring)\b",
    re.IGNORECASE,
)

FULL_SET_HINTS = (
    "full concert", "full set", "full show", "complete", "entire", "livestream",
    "festival", "glastonbury", "lollapalooza", "outside lands", "acl",
    "live at", "tiny desk concert", "session", "interview", "performance + interview"
)

# channels that are commonly publishers, not artists
PUBLISHERS = {
    "mtv", "npr music", "tiny desk", "kexp", "bbc", "vevo",
    "the tonight show", "jimmy kimmel", "late late show",
}

# ============================================================
# OpenAI client (lazy)
# ============================================================
_openai_client = None

SYSTEM_PROMPT_ARTIST_TITLE = (
    "You classify music performance videos.\n"
    "Given metadata (title/channel/tags/description), infer:\n"
    "1) primary_artist: the primary performing artist (ignore publishers like MTV/NPR/TV shows)\n"
    "2) song_title: the song title if present; otherwise a short best title for the performance\n"
    "3) is_full_set: true if it's a full concert/set/interview rather than one song\n"
    "Return ONLY valid JSON:\n"
    "{"
    "\"primary_artist\":\"...\","
    "\"artist_confidence\":0.0,"
    "\"song_title\":\"...\","
    "\"title_confidence\":0.0,"
    "\"is_full_set\":false"
    "}\n"
)

def get_openai():
    global _openai_client
    if _openai_client:
        return _openai_client
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    from openai import OpenAI
    _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client

# ============================================================
# sqlite-vec helpers (lazy import)
# ============================================================
def _load_vec(conn: sqlite3.Connection) -> None:
    import sqlite_vec  # type: ignore
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

# ============================================================
# DB / cache
# ============================================================
def init_db():
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_DB)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS media_cache (
        key TEXT PRIMARY KEY,
        path TEXT,
        artist TEXT,
        artist_conf REAL,
        title TEXT,
        title_conf REAL,
        is_full_set INTEGER,
        source TEXT,
        created INTEGER
    )
    """)

    if ENABLE_EMBEDDINGS:
        _load_vec(conn)
        conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS media_vec USING vec0(
            key TEXT PRIMARY KEY,
            embedding FLOAT[{EMBEDDING_DIMS}] distance_metric=cosine
        )
        """)

    conn.commit()
    conn.close()

def stable_key(info: dict, media: Path) -> str:
    base = f"{info.get('id','')}|{info.get('title') or info.get('fulltitle') or media.stem}|{info.get('uploader') or info.get('channel') or ''}"
    return hashlib.sha256(base.encode("utf-8", errors="ignore")).hexdigest()

def cache_get(key: str):
    conn = sqlite3.connect(CACHE_DB)
    row = conn.execute(
        "SELECT artist, artist_conf, title, title_conf, is_full_set, source, path FROM media_cache WHERE key=?",
        (key,),
    ).fetchone()
    conn.close()
    return row

def cache_put(key: str, path: str, artist: str, artist_conf: float, title: str, title_conf: float, is_full_set: bool, source: str):
    conn = sqlite3.connect(CACHE_DB)
    conn.execute(
        "INSERT OR REPLACE INTO media_cache VALUES (?,?,?,?,?,?,?,?,?)",
        (key, path, artist, float(artist_conf), title, float(title_conf), 1 if is_full_set else 0, source, int(time.time())),
    )
    conn.commit()
    conn.close()

def vec_upsert(key: str, vec: List[float]) -> None:
    if len(vec) != EMBEDDING_DIMS:
        raise ValueError(f"Embedding dims mismatch: got {len(vec)} expected {EMBEDDING_DIMS}. Set EMBEDDING_DIMS or change model.")
    conn = sqlite3.connect(CACHE_DB)
    _load_vec(conn)
    conn.execute(
        "INSERT OR REPLACE INTO media_vec(key, embedding) VALUES (?, ?)",
        (key, json.dumps(vec)),
    )
    conn.commit()
    conn.close()

def vec_exists(key: str) -> bool:
    if not ENABLE_EMBEDDINGS:
        return False
    conn = sqlite3.connect(CACHE_DB)
    _load_vec(conn)
    row = conn.execute("SELECT 1 FROM media_vec WHERE key=? LIMIT 1", (key,)).fetchone()
    conn.close()
    return row is not None

# ============================================================
# Utilities
# ============================================================
def sanitize(s: str, fallback: str) -> str:
    s = (s or "").strip()
    s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", s)
    s = re.sub(r"\s+", " ", s).rstrip(" .")
    return s or fallback

def is_stable(path: Path) -> bool:
    last = -1
    stable = 0
    while stable < STABLE_SECONDS:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        if size == last:
            stable += 1
        else:
            stable = 0
            last = size
        time.sleep(1)
    return True

def load_info(p: Optional[Path]) -> Optional[dict]:
    if not p:
        return None
    try:
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def channel_from_info(info: dict) -> str:
    for k in ("channel", "uploader", "creator"):
        v = info.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def title_from_info(info: dict, media: Path) -> str:
    for k in ("track", "title", "fulltitle"):
        v = info.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return media.stem

def looks_like_full_set(title: str, desc: str) -> bool:
    t = (title or "").lower()
    d = (desc or "").lower()
    return any(h in t for h in FULL_SET_HINTS) or any(h in d for h in FULL_SET_HINTS)

# ============================================================
# Incoming cleanup helpers (empty dirs + orphans)
# ============================================================
def _safe_rmdir(path: Path) -> bool:
    try:
        path.rmdir()
        return True
    except OSError:
        return False

def _is_only_eadir(dirpath: Path) -> bool:
    try:
        entries = list(dirpath.iterdir())
    except FileNotFoundError:
        return False

    if not entries:
        return True

    if len(entries) == 1 and entries[0].is_dir() and entries[0].name == "@eaDir":
        eadir = entries[0]
        # prune empties inside @eaDir
        prune_empty_dirs(eadir, stop_at=eadir)
        _safe_rmdir(eadir)
        try:
            return len(list(dirpath.iterdir())) == 0
        except FileNotFoundError:
            return True

    return False

def prune_empty_dirs(start_dir: Path, stop_at: Path) -> None:
    stop_at = stop_at.resolve()
    cur = start_dir.resolve()

    try:
        cur.relative_to(stop_at)
    except ValueError:
        return

    while True:
        if cur == stop_at:
            break
        if not cur.exists() or not cur.is_dir():
            break

        if _is_only_eadir(cur):
            _safe_rmdir(cur)
        else:
            if not _safe_rmdir(cur):
                break

        cur = cur.parent
        try:
            cur.relative_to(stop_at)
        except ValueError:
            break

def _age_seconds(p: Path) -> float:
    try:
        return time.time() - p.stat().st_mtime
    except FileNotFoundError:
        return 0.0

def _candidate_media_paths_for_sidecar(sidecar: Path) -> List[Path]:
    """
    Handles:
      Foo.mp4.info.json -> Foo.mp4
      Foo.info.json     -> Foo.mp4/Foo.mkv/... (try common media exts)
      Foo.webp          -> Foo.mp4/Foo.mkv/...
      Foo.mp4.webp      -> Foo.mp4
    """
    name = sidecar.name

    # *.info.json special cases
    if name.endswith(".info.json"):
        base = name[:-len(".info.json")]  # Foo.mp4 OR Foo
        direct = sidecar.with_name(base)
        candidates: List[Path] = []
        if direct.suffix.lower() in MEDIA_EXTS:
            candidates.append(direct)
            return candidates
        # base had no media extension -> try Foo.mp4 etc
        for ext in MEDIA_EXTS:
            candidates.append(sidecar.with_name(base + ext))
        return candidates

    # Thumbs/subs:
    stem = sidecar.stem  # strips only last suffix
    # If stem ends with a media ext, that's the direct pair
    for ext in MEDIA_EXTS:
        if stem.lower().endswith(ext):
            return [sidecar.with_name(stem)]

    # Otherwise try adding common media exts
    return [sidecar.with_name(stem + ext) for ext in MEDIA_EXTS]

def _has_any_candidate_media(sidecar: Path) -> bool:
    for c in _candidate_media_paths_for_sidecar(sidecar):
        if c.exists() and c.is_file():
            return True
    return False

def cleanup_orphan_sidecars(root: Path) -> int:
    """
    Remove orphan sidecars under root:
      - sidecar must be older than ORPHAN_GRACE_SECONDS
      - and no paired media exists in the same directory
    Returns number of files removed.
    """
    if not CLEAN_ORPHANS:
        return 0

    removed = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue

        # Detect sidecars:
        if p.name.endswith(".info.json"):
            suffix_match = ".info.json"
        else:
            suffix_match = p.suffix.lower()

        if suffix_match not in ORPHAN_SUFFIXES:
            continue

        # Don't delete fresh files (possibly mid-download)
        if _age_seconds(p) < ORPHAN_GRACE_SECONDS:
            continue

        if not _has_any_candidate_media(p):
            try:
                p.unlink()
                removed += 1
            except Exception:
                pass

    return removed

# ============================================================
# Heuristic inference (artist + title)
# ============================================================
def heuristic_infer(info: Optional[dict], media: Path) -> Tuple[str, float, str, float, bool, str]:
    """
    Returns (artist, artist_conf, song_title, title_conf, is_full_set, source)
    """
    if not info:
        m = ARTIST_TITLE_RE.match(media.stem)
        if m:
            artist = sanitize(m.group("artist"), UNKNOWN_ARTIST)
            song = sanitize(m.group("title"), UNKNOWN_TITLE)
            return artist, 0.75, song, 0.70, looks_like_full_set(song, ""), "filename-split"
        return UNKNOWN_ARTIST, 0.30, sanitize(media.stem, UNKNOWN_TITLE), 0.30, False, "no-info"

    title = title_from_info(info, media)
    desc = info.get("description") if isinstance(info.get("description"), str) else ""
    channel = channel_from_info(info)
    channel_lc = channel.lower().strip()
    publisherish = (channel_lc in PUBLISHERS)

    # explicit music metadata (rare)
    for k in ("artist", "album_artist"):
        v = info.get(k)
        if isinstance(v, str) and v.strip():
            artist = sanitize(v, UNKNOWN_ARTIST)
            song = sanitize(info.get("track") if isinstance(info.get("track"), str) else title, UNKNOWN_TITLE)
            return artist, 0.95, song, 0.85, looks_like_full_set(title, desc), "music-metadata"

    # "Artist - Song"
    m = ARTIST_TITLE_RE.match(title)
    if m:
        artist = sanitize(m.group("artist"), UNKNOWN_ARTIST)
        song = sanitize(m.group("title"), UNKNOWN_TITLE)
        if publisherish and artist.lower() == channel_lc:
            return UNKNOWN_ARTIST, 0.35, song, 0.75, looks_like_full_set(title, desc), "title-split(publisher-bad-artist)"
        return artist, 0.85, song, 0.80, looks_like_full_set(title, desc), "title-split"

    # "'Song' Artist performance ..."
    m = PERFORMANCE_RE.match(title)
    if m:
        song = sanitize(m.group("song"), UNKNOWN_TITLE)
        artist = sanitize(m.group("artist"), UNKNOWN_ARTIST)
        return artist, 0.85, song, 0.80, looks_like_full_set(title, desc), "performance-pattern"

    # "Artist feat ..."
    m = FEAT_RE.match(title)
    if m:
        artist = sanitize(m.group("artist"), UNKNOWN_ARTIST)
        song = sanitize(title, UNKNOWN_TITLE)
        return artist, 0.75, song, 0.55, looks_like_full_set(title, desc), "feat-pattern"

    # publisher channels: pick an artist-like tag (weak)
    if publisherish:
        tags = info.get("tags") if isinstance(info.get("tags"), list) else []
        tags = [t for t in tags if isinstance(t, str) and t.strip()]
        for t in tags:
            tl = t.strip().lower()
            if tl and tl != channel_lc and tl not in PUBLISHERS and len(t.strip()) >= 3:
                return sanitize(t, UNKNOWN_ARTIST), 0.60, sanitize(title, UNKNOWN_TITLE), 0.60, looks_like_full_set(title, desc), "tags-fallback(publisher)"
        return UNKNOWN_ARTIST, 0.25, sanitize(title, UNKNOWN_TITLE), 0.55, looks_like_full_set(title, desc), "publisher-unknown"

    # non-publisher: channel often equals artist
    if channel:
        return sanitize(channel, UNKNOWN_ARTIST), 0.70, sanitize(title, UNKNOWN_TITLE), 0.60, looks_like_full_set(title, desc), "channel-as-artist"

    return UNKNOWN_ARTIST, 0.25, sanitize(title, UNKNOWN_TITLE), 0.40, looks_like_full_set(title, desc), "unknown"

# ============================================================
# OpenAI inference (artist + title + full_set)
# ============================================================
def openai_infer(info: dict, media: Path) -> Tuple[str, float, str, float, bool]:
    title = title_from_info(info, media)
    channel = channel_from_info(info)
    tags = info.get("tags") if isinstance(info.get("tags"), list) else []
    tags = [t for t in tags if isinstance(t, str)][:40]
    desc = info.get("description") if isinstance(info.get("description"), str) else ""
    desc = desc.strip()
    if len(desc) > 2000:
        desc = desc[:2000] + "…"

    payload = (
        f"TITLE:\n{title}\n\n"
        f"CHANNEL/UPLOADER:\n{channel}\n\n"
        f"TAGS:\n{', '.join(tags)}\n\n"
        f"DESCRIPTION:\n{desc}\n"
    )

    resp = get_openai().responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT_ARTIST_TITLE},
            {"role": "user", "content": payload},
        ],
        temperature=0,
        max_output_tokens=220,
    )

    data = json.loads((resp.output_text or "").strip())

    artist = sanitize(str(data.get("primary_artist", UNKNOWN_ARTIST)), UNKNOWN_ARTIST)
    artist_conf = float(data.get("artist_confidence", 0.75))

    song = sanitize(str(data.get("song_title", title)), UNKNOWN_TITLE)
    title_conf = float(data.get("title_confidence", 0.75))

    is_full_set = bool(data.get("is_full_set", False))

    artist_conf = min(max(artist_conf, 0.0), 1.0)
    title_conf = min(max(title_conf, 0.0), 1.0)
    return artist, artist_conf, song, title_conf, is_full_set

# ============================================================
# Embeddings (TRIMMED text)
# ============================================================
def build_embedding_text(info: Optional[dict], media: Path, artist: str, song: str, is_full_set: bool) -> str:
    if not info:
        return (
            f"primary_artist: {artist}\n"
            f"song_title: {song}\n"
            f"is_full_set: {is_full_set}\n"
            f"filename: {media.name}\n"
        )

    original_title = title_from_info(info, media)
    channel = channel_from_info(info)

    tags = info.get("tags") if isinstance(info.get("tags"), list) else []
    tags = [t.strip() for t in tags if isinstance(t, str) and t.strip()][:EMBED_TEXT_MAX_TAGS]
    tags_str = ", ".join(tags)

    desc = info.get("description") if isinstance(info.get("description"), str) else ""
    desc = (desc or "").strip()
    if len(desc) > EMBED_TEXT_MAX_DESC:
        desc = desc[:EMBED_TEXT_MAX_DESC] + "…"

    return (
        f"primary_artist: {artist}\n"
        f"song_title: {song}\n"
        f"is_full_set: {is_full_set}\n"
        f"original_title: {original_title}\n"
        f"channel: {channel}\n"
        f"tags: {tags_str}\n"
        f"description: {desc}\n"
    )

def embed_text(text: str) -> List[float]:
    resp = get_openai().embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return resp.data[0].embedding

# ============================================================
# JSON cleaning / enrichment
# ============================================================
JSON_KEEP_KEYS = {
    "id", "title", "fulltitle", "track",
    "uploader", "uploader_id", "channel", "channel_id",
    "upload_date", "duration", "categories", "tags",
    "playlist", "playlist_id", "playlist_title", "playlist_uploader",
    "ext", "width", "height", "fps", "vcodec", "acodec",
    "webpage_url", "thumbnail",
}

def cleaned_info_json(
    info: dict,
    inferred_artist: str,
    artist_conf: float,
    inferred_title: str,
    title_conf: float,
    is_full_set: bool,
    source: str,
    embed_meta: Optional[dict],
) -> dict:
    out = {}
    for k in JSON_KEEP_KEYS:
        if k in info:
            out[k] = info[k]

    desc = info.get("description")
    if isinstance(desc, str) and desc.strip():
        desc = desc.strip()
        if len(desc) > JSON_TRIM_DESCRIPTION_TO:
            desc = desc[:JSON_TRIM_DESCRIPTION_TO] + "…"
        out["description"] = desc

    out["inferred_artist"] = inferred_artist
    out["inferred_artist_confidence"] = round(float(artist_conf), 4)

    out["inferred_title"] = inferred_title
    out["inferred_title_confidence"] = round(float(title_conf), 4)

    out["inferred_is_full_set"] = bool(is_full_set)
    out["inferred_source"] = source

    if embed_meta:
        out["embedding"] = embed_meta

    return out

def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)

# ============================================================
# File moving / bundling helpers
# ============================================================
def resolve_collision(dst: Path) -> Path:
    if not dst.exists():
        return dst
    base, ext, parent = dst.stem, dst.suffix, dst.parent
    for i in range(2, 10000):
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

def find_info_json(media_path: Path) -> Optional[Path]:
    # yt-dlp default is: Foo.mp4.info.json (i.e. media suffix + .info.json)
    p = media_path.with_suffix(media_path.suffix + ".info.json")
    if p.exists():
        return p
    # some setups produce: Foo.info.json
    p2 = media_path.with_suffix(".info.json")
    return p2 if p2.exists() else None

def bundle_paths(media_path: Path) -> List[Path]:
    """
    Return paths that belong to this media, located next to it.
    We handle:
      - Foo.mp4.info.json OR Foo.info.json
      - Foo.webp / Foo.jpg / ...
      - Foo.mp4.webp (rare)
      - subs: Foo.vtt, Foo.srt, etc or Foo.mp4.vtt
    """
    bundle = [media_path]

    info = find_info_json(media_path)
    if info and info.exists():
        bundle.append(info)

    # thumbs and subs can be either Foo.webp OR Foo.mp4.webp
    for ext in THUMB_EXTS | SUB_EXTS:
        p1 = media_path.with_suffix(ext)  # Foo.webp
        if p1.exists():
            bundle.append(p1)
        p2 = media_path.with_suffix(media_path.suffix + ext)  # Foo.mp4.webp
        if p2.exists():
            bundle.append(p2)

    return bundle

# ============================================================
# Decision logic (rules -> openai; caching; embeddings)
# ============================================================
def decide(info: Optional[dict], media_path: Path) -> Tuple[str, float, str, float, bool, str, str]:
    """
    Returns:
      artist, artist_conf, song, title_conf, is_full_set, source, key
    """
    if not info:
        artist, aconf, song, tconf, is_full_set, src = heuristic_infer(None, media_path)
        return artist, aconf, song, tconf, is_full_set, src, ""

    key = stable_key(info, media_path)
    cached = cache_get(key)
    if cached:
        artist, aconf, song, tconf, is_full_set, src, _path = cached
        return str(artist), float(aconf), str(song), float(tconf), bool(is_full_set), str(src), key

    artist, aconf, song, tconf, is_full_set, src = heuristic_infer(info, media_path)

    if USE_OPENAI and (aconf < CONF_AUTO or tconf < CONF_AUTO):
        try:
            oa_artist, oa_aconf, oa_song, oa_tconf, oa_full = openai_infer(info, media_path)
            artist, aconf, song, tconf, is_full_set = oa_artist, oa_aconf, oa_song, oa_tconf, oa_full
            src = f"openai:{OPENAI_MODEL}"
        except Exception as e:
            src = f"{src} (openai-failed: {e})"

    cache_put(key, "", artist, aconf, song, tconf, is_full_set, src)
    return artist, aconf, song, tconf, is_full_set, src, key

# ============================================================
# Main processing
# ============================================================
def process_media(media_path: Path) -> None:
    if not media_path.exists() or media_path.is_dir():
        return

    ext = media_path.suffix.lower()
    if ext not in MEDIA_EXTS or ext in SKIP_EXTS:
        return

    try:
        media_path.relative_to(INCOMING)
    except ValueError:
        return

    if not is_stable(media_path):
        return

    info_path = find_info_json(media_path)
    info = load_info(info_path)

    artist, aconf, song, tconf, is_full_set, src, key = decide(info, media_path)

    target_root = NEEDS_REVIEW if min(aconf, tconf) < CONF_REVIEW else ORGANIZED
    artist_dir = target_root / sanitize(artist, UNKNOWN_ARTIST)
    artist_dir.mkdir(parents=True, exist_ok=True)

    base_title = sanitize(song, UNKNOWN_TITLE)
    dst_media = resolve_collision(artist_dir / f"{base_title}{ext}")
    dst_dir = dst_media.parent
    dst_stem = dst_media.stem

    # Embedding (TRIMMED) before move
    embed_meta = None
    if ENABLE_EMBEDDINGS and info and key:
        if not vec_exists(key):
            try:
                emb_text = build_embedding_text(info, media_path, artist, song, is_full_set)
                vec = embed_text(emb_text)
                vec_upsert(key, vec)
            except Exception as e:
                print(f"[WARN] embeddings failed for {media_path.name}: {e}", flush=True)

        embed_meta = {"model": EMBEDDING_MODEL, "dims": EMBEDDING_DIMS, "cache_key": key}

    bundle = bundle_paths(media_path)

    try:
        # Move/copy the media first
        move_or_copy(media_path, dst_media)

        # Update cache row now that final path is known
        if key:
            cache_put(key, str(dst_media), artist, aconf, song, tconf, is_full_set, src)

        # Companions: rewrite cleaned json; move others
        for p in bundle:
            if p == media_path:
                continue
            if not p.exists():
                continue

            if p.name.endswith(".info.json"):
                dst_json = resolve_collision(dst_dir / f"{dst_stem}.info.json")
                original_info = load_info(p)

                if original_info:
                    cleaned = cleaned_info_json(
                        original_info,
                        inferred_artist=artist,
                        artist_conf=aconf,
                        inferred_title=song,
                        title_conf=tconf,
                        is_full_set=is_full_set,
                        source=src,
                        embed_meta=embed_meta,
                    )
                    write_json_atomic(dst_json, cleaned)

                    if not COPY_MODE:
                        try:
                            p.unlink(missing_ok=True)
                        except TypeError:
                            if p.exists():
                                p.unlink()
                else:
                    move_or_copy(p, dst_json)

            else:
                # Keep same suffix (normalized lower), but attach to dst_stem
                suffix = p.suffix.lower()
                dst = resolve_collision(dst_dir / f"{dst_stem}{suffix}")
                move_or_copy(p, dst)

        print(
            f"[OK] artist={artist} ({aconf:.2f}) title={song} ({tconf:.2f}) full_set={is_full_set} src={src} -> {dst_media}",
            flush=True,
        )

    except Exception as e:
        print(f"[ERR] Failed {media_path}: {e}", flush=True)
        return

    # Cleanup in Incoming (after success)
    try:
        if CLEAN_ORPHANS:
            cleanup_orphan_sidecars(media_path.parent)
        prune_empty_dirs(media_path.parent, stop_at=INCOMING)
    except Exception as e:
        print(f"[WARN] cleanup failed under {media_path.parent}: {e}", flush=True)

# ============================================================
# Watchdog
# ============================================================
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

def initial_sweep() -> None:
    print(f"Initial sweep of {INCOMING} ...", flush=True)
    processed = 0
    for p in sorted(INCOMING.rglob("*"), key=lambda x: str(x)):
        if p.is_file() and p.suffix.lower() in MEDIA_EXTS:
            before = p.exists()
            process_media(p)
            if before and not p.exists():
                processed += 1

    # One-time cleanup pass across incoming
    if CLEAN_ORPHANS:
        removed = cleanup_orphan_sidecars(INCOMING)
        print(f"Orphan sidecars removed: {removed}", flush=True)

    # Prune empty dirs deepest-first
    for d in sorted([x for x in INCOMING.rglob("*") if x.is_dir()], key=lambda x: len(x.parts), reverse=True):
        prune_empty_dirs(d, stop_at=INCOMING)

    print(f"Initial sweep complete. Processed {processed} media file(s).", flush=True)

def main():
    INCOMING.mkdir(parents=True, exist_ok=True)
    ORGANIZED.mkdir(parents=True, exist_ok=True)
    NEEDS_REVIEW.mkdir(parents=True, exist_ok=True)

    init_db()
    initial_sweep()

    observer = Observer()
    observer.schedule(Handler(), str(INCOMING), recursive=True)
    observer.start()

    print(
        f"Watching {INCOMING} -> {ORGANIZED} (review->{NEEDS_REVIEW}) "
        f"stable={STABLE_SECONDS}s copy={COPY_MODE} "
        f"openai={USE_OPENAI} model={OPENAI_MODEL} "
        f"embeddings={ENABLE_EMBEDDINGS} embed_model={EMBEDDING_MODEL} dims={EMBEDDING_DIMS} "
        f"orphans={CLEAN_ORPHANS} orphan_grace={ORPHAN_GRACE_SECONDS}s",
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
