"""Nightly dump of every CouchDB database to the UNAS.

Writes /backup/couchdb/<db>-<stamp>.ndjson.gz (one document per line, all
revisions' winning bodies as returned by _all_docs?include_docs=true), checks
it, and keeps the newest KEEP dumps per database.

The LiveSync data is end-to-end encrypted, so the dump is too: reading notes
from it needs the E2EE passphrase (1Password item `couchdb`). The Longhorn
volume itself is also backed up to B2.

Restore a database (into an empty or new db) with:
  zcat notes-YYYYMMDD-HHMM.ndjson.gz | jq -cs '{docs: ., new_edits: false}' |
    curl -u USER:PASS -H 'Content-Type: application/json' \
      -X POST https://obsidian-sync.pmcd.dev/notes/_bulk_docs -d @-
(create the db first with PUT /notes; split into batches if it is large).
"""
import base64
import datetime
import gzip
import json
import os
import sys
import urllib.parse
import urllib.request

URL = os.environ.get("COUCHDB_URL", "http://couchdb.default.svc")
AUTH = "Basic " + base64.b64encode(
    f"{os.environ['COUCHDB_USER']}:{os.environ['COUCHDB_PASSWORD']}".encode()).decode()
DEST = "/backup/couchdb"
KEEP = int(os.environ.get("KEEP", "14"))
PAGE = 500


def get(path):
    req = urllib.request.Request(URL + path, headers={"Authorization": AUTH, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def dump(db, stamp):
    final = os.path.join(DEST, f"{db}-{stamp}.ndjson.gz")
    tmp = final + ".part"
    q = urllib.parse.quote(db, safe="")
    count, startkey = 0, None
    with gzip.open(tmp, "wt", encoding="utf-8") as out:
        while True:
            params = {"include_docs": "true", "limit": str(PAGE + 1)}
            if startkey is not None:
                params["startkey"] = json.dumps(startkey)
            rows = get(f"/{q}/_all_docs?{urllib.parse.urlencode(params)}")["rows"]
            for row in rows[:PAGE]:
                out.write(json.dumps(row["doc"], separators=(",", ":")) + "\n")
                count += 1
            if len(rows) <= PAGE:
                break
            startkey = rows[PAGE]["id"]
    expected = get(f"/{q}")["doc_count"]
    # Deleted docs aren't in _all_docs; design docs are. Allow for writes
    # during the run, but refuse a dump that is clearly short.
    if count == 0 or count < expected * 0.9:
        os.remove(tmp)
        sys.exit(f"{db}: dumped {count} docs but the db reports {expected} -- refusing to keep it")
    os.replace(tmp, final)
    print(f"saved {final}: {os.path.getsize(final)} bytes, {count} docs (db reports {expected})")

    dumps = sorted(f for f in os.listdir(DEST) if f.startswith(f"{db}-") and f.endswith(".ndjson.gz")
                   and f[len(db) + 1:].count("-") == 1)
    for old in dumps[:-KEEP]:
        os.remove(os.path.join(DEST, old))
        print(f"pruned {old}")


def main():
    os.makedirs(DEST, exist_ok=True)
    for f in os.listdir(DEST):
        if f.endswith(".part"):
            os.remove(os.path.join(DEST, f))
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    dbs = [d for d in get("/_all_dbs") if not d.startswith("_")]
    if not dbs:
        sys.exit("no databases to back up")
    for db in dbs:
        dump(db, stamp)


if __name__ == "__main__":
    main()
