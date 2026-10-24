"""Mirror every repo in a GitHub org ($GITHUB_ORG) to the UNAS.

Each repo is kept as a bare mirror at $BACKUP_DIR/<org>/<repo>.git with every branch
and tag. New repos are cloned; existing ones are fetched. Nothing is ever
deleted: branches removed upstream and repos deleted or renamed on GitHub stay
in the backup. Rewritten branches are force-updated (UNAS snapshots cover that).

Restore: git clone $BACKUP_DIR/<org>/<repo>.git

If GITEA_TOKEN is set, it also makes sure every repo has a Gitea pull mirror
in a Gitea org of the same name (created private if missing). Gitea keeps
those in sync on its own mirror interval; this only adds new ones. Unlike the
UNAS copy, Gitea mirrors prune branches deleted upstream.
"""
import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

ORG = os.environ["GITHUB_ORG"]
TOKEN = os.environ["GITHUB_TOKEN"]
BACKUP_DIR = os.path.join(os.environ.get("BACKUP_DIR", "/backup"), ORG)
GITEA_URL = os.environ.get("GITEA_URL", "").rstrip("/")
GITEA_TOKEN = os.getenv("GITEA_TOKEN")

# Only branches and tags; a plain --mirror would also pull every refs/pull/*.
REFSPECS = ["+refs/heads/*:refs/heads/*", "+refs/tags/*:refs/tags/*"]

# Auth via an env-supplied header, so the token never lands in a repo's config
# on the UNAS or in the process's argv.
basic = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()
GIT_ENV = {
    **os.environ,
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_CONFIG_COUNT": "2",
    "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
    "GIT_CONFIG_VALUE_0": f"Authorization: Basic {basic}",
    # The SMB mount's ownership doesn't match the container user.
    "GIT_CONFIG_KEY_1": "safe.directory",
    "GIT_CONFIG_VALUE_1": "*",
}


def list_repos():
    repos, url = [], f"https://api.github.com/orgs/{ORG}/repos?type=all&per_page=100"
    while url:
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        with urllib.request.urlopen(req, timeout=60) as resp:
            repos += json.load(resp)
            url = None
            for part in resp.headers.get("Link", "").split(","):
                if 'rel="next"' in part:
                    url = part[part.index("<") + 1:part.index(">")]
    return repos


def git(*args, cwd=None):
    subprocess.run(["git", *args], cwd=cwd, env=GIT_ENV, check=True)


def mirror(repo):
    path = os.path.join(BACKUP_DIR, f"{repo['name']}.git")
    if not os.path.isdir(path):
        print(f"cloning {repo['full_name']}", flush=True)
        git("init", "--quiet", "--bare", path)
        # Not `git remote add`: that also adds a refs/remotes/origin/* refspec.
        git("config", "remote.origin.url", repo["clone_url"], cwd=path)
        for spec in REFSPECS:
            git("config", "--add", "remote.origin.fetch", spec, cwd=path)
    else:
        print(f"fetching {repo['full_name']}", flush=True)
        # Follow renames/transfers: GitHub redirects, but keep the URL current.
        git("remote", "set-url", "origin", repo["clone_url"], cwd=path)
    git("fetch", "--quiet", "--tags", "origin", cwd=path)
    if repo.get("default_branch"):
        git("symbolic-ref", "HEAD", f"refs/heads/{repo['default_branch']}", cwd=path)


def gitea(method, path, body=None):
    """Call the Gitea API; returns the HTTP status (404 is not an error)."""
    req = urllib.request.Request(
        f"{GITEA_URL}/api/v1{path}", method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"token {GITEA_TOKEN}", "Content-Type": "application/json"},
    )
    try:
        # Migrations clone synchronously, so give big repos time.
        with urllib.request.urlopen(req, timeout=900) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return 404
        raise RuntimeError(f"{method} {path}: {e.code} {e.read().decode(errors='replace')[:300]}")


def ensure_gitea_org():
    if gitea("GET", f"/orgs/{ORG}") == 404:
        print(f"creating Gitea org {ORG}", flush=True)
        gitea("POST", "/orgs", {"username": ORG, "visibility": "private"})


def gitea_mirror(repo):
    if gitea("GET", f"/repos/{ORG}/{repo['name']}") != 404:
        return
    print(f"creating Gitea mirror {ORG}/{repo['name']}", flush=True)
    gitea("POST", "/repos/migrate", {
        "service": "github",
        "clone_addr": repo["clone_url"],
        "auth_token": TOKEN,
        "mirror": True,
        "repo_owner": ORG,
        "repo_name": repo["name"],
        "private": repo["private"],
        "description": repo.get("description") or "",
    })


def main():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    repos = list_repos()
    print(f"{len(repos)} repos in {ORG}", flush=True)
    with open(os.path.join(BACKUP_DIR, "_repos.json"), "w") as f:
        json.dump(repos, f, indent=2)

    failed = []
    if GITEA_TOKEN:
        ensure_gitea_org()
    for repo in repos:
        if repo.get("size") == 0:
            print(f"skipping {repo['full_name']} (empty)", flush=True)
            continue
        try:
            mirror(repo)
        except subprocess.CalledProcessError as e:
            print(f"FAILED {repo['full_name']}: {e}", file=sys.stderr, flush=True)
            failed.append(repo["full_name"])
        if GITEA_TOKEN:
            try:
                gitea_mirror(repo)
            except (RuntimeError, urllib.error.URLError) as e:
                print(f"FAILED Gitea mirror {repo['full_name']}: {e}", file=sys.stderr, flush=True)
                failed.append(f"{repo['full_name']} (gitea)")

    if failed:
        print(f"{len(failed)} failed: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)
    print("done")


if __name__ == "__main__":
    main()
