"""Idempotent Garage bootstrap, run by bootstrap-job.yaml after every sync.

1. Single-node layout: assign the node (zone dc1) and apply, if unassigned.
2. Buckets + keys from BUCKETS: create the bucket, import the key from its
   1Password-synced secret, grant read/write/owner, and set CORS so browsers
   on the listed origins can use presigned URLs.

Stdlib only (the Garage image is FROM scratch, so this runs in python:alpine).
"""
import datetime
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ADMIN = "http://garage.garage.svc.cluster.local:3903"
S3 = "http://garage.garage.svc.cluster.local:3900"
REGION = "garage"
TOKEN = os.environ["GARAGE_ADMIN_TOKEN"]
CAPACITY = 20 * 1000**3  # bytes; keep in step with the PVC in storage.yaml

# One entry per app. key_id/key_secret come from that app's 1Password item,
# passed in by bootstrap-job.yaml as <APP>_ACCESS_KEY_ID / <APP>_SECRET_ACCESS_KEY.
BUCKETS = [
    {
        "name": "kaneo",
        "key_name": "kaneo",
        "key_id": os.environ["KANEO_ACCESS_KEY_ID"],
        "key_secret": os.environ["KANEO_SECRET_ACCESS_KEY"],
        "cors_origins": ["https://kaneo.x.pmcd.io"],
    },
    {
        "name": "outline",
        "key_name": "outline",
        "key_id": os.environ["OUTLINE_ACCESS_KEY_ID"],
        "key_secret": os.environ["OUTLINE_SECRET_ACCESS_KEY"],
        "cors_origins": ["https://outline.x.pmcd.io"],
    },
]


def admin(method, op, body=None, query=None):
    url = f"{ADMIN}/v2/{op}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {TOKEN}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def wait_ready():
    for attempt in range(60):
        try:
            status, body = admin("GET", "GetClusterStatus")
            if status == 200 and body.get("nodes"):
                return body
        except OSError:
            pass
        print(f"waiting for garage admin API ({attempt + 1}/60)")
        time.sleep(5)
    sys.exit("garage admin API never became ready")


def ensure_layout(cluster):
    status, layout = admin("GET", "GetClusterLayout")
    assert status == 200, layout
    if layout["roles"]:
        print(f"layout v{layout['version']} already assigned ({len(layout['roles'])} role(s))")
        return
    node_id = cluster["nodes"][0]["id"]
    status, body = admin("POST", "UpdateClusterLayout", {
        "roles": [{"id": node_id, "zone": "dc1", "capacity": CAPACITY, "tags": []}],
    })
    assert status == 200, body
    status, body = admin("POST", "ApplyClusterLayout", {"version": layout["version"] + 1})
    assert status == 200, body
    print(f"layout applied: node {node_id[:16]}… in dc1, version {layout['version'] + 1}")


def ensure_bucket(name):
    status, body = admin("GET", "GetBucketInfo", query={"globalAlias": name})
    if status == 200:
        print(f"bucket {name} exists")
        return body["id"]
    status, body = admin("POST", "CreateBucket", {"globalAlias": name})
    assert status == 200, body
    print(f"bucket {name} created")
    return body["id"]


def ensure_key(key_id, secret, name):
    status, body = admin("GET", "GetKeyInfo", query={"id": key_id})
    if status == 200:
        print(f"key {name} exists")
        return
    status, body = admin("POST", "ImportKey",
                         {"accessKeyId": key_id, "secretAccessKey": secret, "name": name})
    assert status == 200, body
    print(f"key {name} imported")


def allow(bucket_id, key_id, name):
    status, body = admin("POST", "AllowBucketKey", {
        "bucketId": bucket_id, "accessKeyId": key_id,
        "permissions": {"read": True, "write": True, "owner": True},
    })
    assert status == 200, body
    print(f"key {name}: read/write/owner on bucket")


def _sign(key, msg):
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def s3_put_cors(bucket, key_id, secret, origins):
    rules = "".join(f"<AllowedOrigin>{o}</AllowedOrigin>" for o in origins)
    body = (
        '<CORSConfiguration><CORSRule>'
        f'{rules}'
        '<AllowedMethod>GET</AllowedMethod><AllowedMethod>PUT</AllowedMethod>'
        '<AllowedMethod>POST</AllowedMethod><AllowedMethod>HEAD</AllowedMethod>'
        '<AllowedHeader>*</AllowedHeader><ExposeHeader>ETag</ExposeHeader>'
        '<MaxAgeSeconds>3600</MaxAgeSeconds>'
        '</CORSRule></CORSConfiguration>'
    ).encode()
    host = urllib.parse.urlparse(S3).netloc
    now = datetime.datetime.now(datetime.timezone.utc)
    amz_date, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(body).hexdigest()
    canonical = "\n".join([
        "PUT", f"/{bucket}", "cors=",
        f"host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n",
        "host;x-amz-content-sha256;x-amz-date", payload_hash,
    ])
    scope = f"{day}/{REGION}/s3/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope,
                         hashlib.sha256(canonical.encode()).hexdigest()])
    k = _sign(("AWS4" + secret).encode(), day)
    for part in (REGION, "s3", "aws4_request"):
        k = _sign(k, part)
    signature = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()
    req = urllib.request.Request(f"{S3}/{bucket}?cors=", data=body, method="PUT")
    req.add_header("Host", host)
    req.add_header("x-amz-date", amz_date)
    req.add_header("x-amz-content-sha256", payload_hash)
    req.add_header("Authorization",
                   f"AWS4-HMAC-SHA256 Credential={key_id}/{scope}, "
                   f"SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature={signature}")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            print(f"bucket {bucket}: CORS set for {', '.join(origins)} (HTTP {r.status})")
    except urllib.error.HTTPError as e:
        sys.exit(f"PutBucketCors on {bucket} failed: HTTP {e.code} {e.read().decode(errors='replace')}")


def main():
    cluster = wait_ready()
    ensure_layout(cluster)
    for b in BUCKETS:
        bucket_id = ensure_bucket(b["name"])
        ensure_key(b["key_id"], b["key_secret"], b["key_name"])
        allow(bucket_id, b["key_id"], b["key_name"])
        if b.get("cors_origins"):
            # a freshly applied layout can take a moment before writes succeed
            for attempt in range(10):
                try:
                    s3_put_cors(b["name"], b["key_id"], b["key_secret"], b["cors_origins"])
                    break
                except SystemExit:
                    if attempt == 9:
                        raise
                    time.sleep(3)
    print("bootstrap complete")


if __name__ == "__main__":
    main()
