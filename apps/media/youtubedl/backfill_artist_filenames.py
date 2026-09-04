#!/usr/bin/env python3
"""
backfill_artist_filenames.py

One-time backfill: rename existing organized files so the artist name is
always part of the filename, not just the parent folder — matches the
always-prefix-artist behavior in watcher.py's process_media().

Renames the media file plus any co-located sidecars sharing its exact
stem (.info.json, thumbs, subs), then triggers a targeted Plex refresh
of that artist folder so the library picks up the new path. Skips a
file if its name already starts with the artist folder name.

Dry-run by default — pass --apply to actually rename files.

Run manually, e.g.:
  kubectl exec -n media deploy/youtubedl -c postprocessor -- python3 /app/backfill_artist_filenames.py
  kubectl exec -n media deploy/youtubedl -c postprocessor -- python3 /app/backfill_artist_filenames.py --apply
"""
import sys

from watcher import (  # noqa: E402
    MEDIA_EXTS,
    NEEDS_REVIEW,
    ORGANIZED,
    PLEX_PATH_PREFIX,
    UNKNOWN_ARTIST,
    plex_refresh_path,
    resolve_collision,
    sanitize,
)


def rename_in_artist_dir(artist_dir, dry_run: bool) -> int:
    artist = artist_dir.name
    if artist == UNKNOWN_ARTIST:
        return 0

    renamed = 0
    touched_dir = False

    for media_path in sorted(artist_dir.iterdir()):
        if not media_path.is_file() or media_path.suffix.lower() not in MEDIA_EXTS:
            continue

        stem = media_path.stem
        if stem.lower().startswith(artist.lower()):
            continue

        new_stem = sanitize(f"{artist} - {stem}", stem)
        new_media = resolve_collision(artist_dir / f"{new_stem}{media_path.suffix}")
        new_stem = new_media.stem  # may differ if a collision suffix was added

        prefix = stem + "."
        sidecars = [
            f for f in artist_dir.iterdir()
            if f.is_file() and f != media_path and f.name.startswith(prefix)
        ]

        print(
            f"{'[DRY RUN]' if dry_run else '[RENAME]'} {media_path.name!r} -> {new_media.name!r}"
            + (f"  (+{len(sidecars)} sidecar(s))" if sidecars else ""),
            flush=True,
        )
        renamed += 1
        if dry_run:
            continue

        media_path.rename(new_media)
        for sidecar in sidecars:
            suffix_chain = sidecar.name[len(stem):]
            new_sidecar = artist_dir / f"{new_stem}{suffix_chain}"
            if new_sidecar.exists():
                continue
            sidecar.rename(new_sidecar)
        touched_dir = True

    if touched_dir and PLEX_PATH_PREFIX:
        rel = artist_dir.relative_to(ORGANIZED)
        plex_refresh_path(f"{PLEX_PATH_PREFIX}/{rel.as_posix()}")

    return renamed


def main() -> None:
    dry_run = "--apply" not in sys.argv[1:]
    if dry_run:
        print("Running in DRY RUN mode — pass --apply to actually rename files.\n", flush=True)

    total = 0
    for root in (ORGANIZED, NEEDS_REVIEW):
        if not root.exists():
            continue
        for artist_dir in sorted(root.iterdir()):
            if artist_dir.is_dir():
                total += rename_in_artist_dir(artist_dir, dry_run)

    verb = "would rename" if dry_run else "renamed"
    print(f"\nDone. {verb} {total} file(s).", flush=True)


if __name__ == "__main__":
    main()
