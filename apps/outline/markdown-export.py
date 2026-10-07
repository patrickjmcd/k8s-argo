"""Nightly Markdown export of the whole Outline workspace to the UNAS.

Asks Outline for a full export (outline-markdown, attachments and private
collections included), waits for it, downloads the zip into
/backup/outline-markdown/, checks it, deletes the export from Outline (so the
zips don't pile up in Garage), and keeps the newest KEEP copies.

Document content itself lives in Postgres (backed up to B2); this is a
human-readable copy that doesn't need Outline to read.
"""
import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile

API = "http://outline.default.svc.cluster.local/api"
TOKEN = os.environ["OUTLINE_API_TOKEN"]
DEST = "/backup/outline-markdown"
KEEP = int(os.environ.get("KEEP", "14"))


def call(op, body):
    req = urllib.request.Request(f"{API}/{op}", data=json.dumps(body).encode(), method="POST")
    req.add_header("Authorization", f"Bearer {TOKEN}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"{op} failed: HTTP {e.code} {e.read().decode(errors='replace')[:300]}")


def main():
    os.makedirs(DEST, exist_ok=True)
    op = call("collections.export_all", {
        "format": "outline-markdown", "includeAttachments": True, "includePrivate": True,
    })["data"]["fileOperation"]
    op_id = op["id"]
    print(f"export started: {op_id}")

    for _ in range(120):  # up to 20 minutes
        op = call("fileOperations.info", {"id": op_id})["data"]
        if op["state"] == "complete":
            break
        if op["state"] in ("error", "expired"):
            sys.exit(f"export {op_id} ended in state {op['state']}: {op.get('error')}")
        time.sleep(10)
    else:
        sys.exit(f"export {op_id} still {op['state']} after 20 minutes")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    final = os.path.join(DEST, f"outline-{stamp}.zip")
    tmp = final + ".part"
    # fileOperations.redirect answers with a redirect to a signed Garage URL;
    # urllib follows it (as a GET).
    req = urllib.request.Request(f"{API}/fileOperations.redirect",
                                 data=json.dumps({"id": op_id}).encode(), method="POST")
    req.add_header("Authorization", f"Bearer {TOKEN}")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)

    with zipfile.ZipFile(tmp) as z:
        bad = z.testzip()
        if bad:
            sys.exit(f"corrupt member in export: {bad}")
        names = z.namelist()
    md = sum(n.endswith(".md") for n in names)
    if md == 0:
        sys.exit(f"export contains no .md files ({len(names)} entries) -- refusing to keep it")
    os.replace(tmp, final)
    print(f"saved {final}: {os.path.getsize(final)} bytes, {md} markdown files, {len(names)} entries")

    call("fileOperations.delete", {"id": op_id})
    print("export removed from Outline")

    zips = sorted(f for f in os.listdir(DEST) if f.startswith("outline-") and f.endswith(".zip"))
    for old in zips[:-KEEP]:
        os.remove(os.path.join(DEST, old))
        print(f"pruned {old}")
    for f in os.listdir(DEST):
        if f.endswith(".part") and os.path.join(DEST, f) != tmp:
            os.remove(os.path.join(DEST, f))
    print(f"{min(len(zips), KEEP)} export(s) kept in {DEST}")


if __name__ == "__main__":
    main()
