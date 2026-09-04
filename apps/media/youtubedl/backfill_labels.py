#!/usr/bin/env python3
"""
backfill_labels.py

One-time backfill: classify + Plex-label existing items in the Youtube/Other
library that were organized before Plex-label support existed in watcher.py.

Run manually (not part of the container's normal entrypoint), e.g.:
  kubectl exec -n media deploy/youtubedl -c postprocessor -- python3 /app/backfill_labels.py
"""
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from watcher import (  # noqa: E402
    ORGANIZED,
    PLEX_PATH_PREFIX,
    PLEX_SECTION_ID,
    PLEX_TOKEN,
    PLEX_URL,
    _plex_request,
    classify_category,
    load_info,
    plex_set_label,
)


def local_info_json_for(plex_file: str) -> Optional[Path]:
    if not plex_file or not plex_file.startswith(PLEX_PATH_PREFIX):
        return None
    rel = plex_file[len(PLEX_PATH_PREFIX):].lstrip("/")
    media_path = ORGANIZED / rel
    candidate = media_path.with_suffix(media_path.suffix + ".info.json")
    return candidate if candidate.exists() else None


def main() -> None:
    assert PLEX_URL and PLEX_TOKEN and PLEX_SECTION_ID, "Plex env vars not set"

    start = 0
    page_size = 50
    labeled = 0
    skipped = 0

    while True:
        data = _plex_request(
            "GET",
            f"/library/sections/{PLEX_SECTION_ID}/all",
            {"X-Plex-Container-Start": str(start), "X-Plex-Container-Size": str(page_size)},
        )
        if not data:
            break
        videos = ET.fromstring(data).findall("Video")
        if not videos:
            break

        for v in videos:
            rk = v.get("ratingKey")
            plex_title = v.get("title") or ""

            if v.find("Label") is not None:
                skipped += 1
                continue

            meta = _plex_request("GET", f"/library/metadata/{rk}", {})
            if not meta:
                continue
            part = next(ET.fromstring(meta).iter("Part"), None)
            plex_file = part.get("file") if part is not None else None

            title, desc = plex_title, ""
            info_path = local_info_json_for(plex_file) if plex_file else None
            if info_path:
                info = load_info(info_path)
                if info:
                    title = info.get("title") or info.get("fulltitle") or plex_title
                    desc = info.get("description") or ""

            category = classify_category(title, desc)
            plex_set_label(rk, category)
            labeled += 1
            print(f"[OK] {plex_title!r} -> {category}", flush=True)
            time.sleep(0.2)  # be polite to Plex

        start += page_size

    print(f"Done. labeled={labeled} skipped(existing label)={skipped}", flush=True)


if __name__ == "__main__":
    main()
