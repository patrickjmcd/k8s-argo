#!/usr/bin/env python3
"""Regenerates the glance 'Services' / 'Media Services' monitor widgets in
apps/glance/values.yaml from HTTPRoutes across the repo.

An app opts in by setting, on its HTTPRoute (raw manifest):
  metadata:
    annotations:
      glance.pmcd.io/monitor: "true"
      glance.pmcd.io/monitor-title: "ArgoCD"      # optional, else derived from hostname
      glance.pmcd.io/monitor-group: "Services"    # optional, else apps/media/* -> "Media Services", else "Services"
      glance.pmcd.io/monitor-hostname: "a.b.c"    # optional, disambiguates multi-hostname routes

or, for apps using the homelab-app Helm chart, in their values.yaml (a plain
key the chart itself ignores):
  glance:
    monitor: true
    title: "N8N"       # optional
    group: "Services"  # optional
    hostname: "a.b.c"  # optional

Generated sites are spliced between marker comments in apps/glance/values.yaml:
  # BEGIN GLANCE-GENERATED:<group>
  ...
  # END GLANCE-GENERATED:<group>
Everything outside those markers (including static entries like Home
Assistant / PocketID, which have no in-repo HTTPRoute) is left untouched.
"""
import pathlib
import re
import sys

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GLANCE_VALUES = REPO_ROOT / "apps/glance/values.yaml"
ANNOTATION_PREFIX = "glance.pmcd.io/"


def title_from_hostname(hostname: str) -> str:
    label = hostname.split(".")[0]
    parts = re.split(r"[-_]", label)
    return " ".join(p.capitalize() for p in parts if p)


def default_group(path: pathlib.Path) -> str:
    return "Media Services" if "/media/" in path.relative_to(REPO_ROOT).as_posix() else "Services"


def pick_hostname(hostnames, override):
    if override:
        if override not in hostnames:
            raise ValueError(f"monitor-hostname override {override!r} not in hostnames {hostnames!r}")
        return override
    return hostnames[0]


def collect_from_httproutes():
    entries = []
    for path in REPO_ROOT.glob("**/*.yaml"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        parts = rel.split("/")
        if "charts" in parts or "templates" in parts or parts[0] in (".claude", ".git"):
            continue  # Helm templates / Go-templated YAML, not rendered manifests
        try:
            text = path.read_text()
        except OSError:
            continue
        if "kind: HTTPRoute" not in text:
            continue
        try:
            docs = list(yaml.safe_load_all(text))
        except yaml.YAMLError as e:
            print(f"WARNING: skipping {rel}, not parseable YAML: {e}", file=sys.stderr)
            continue
        for doc in docs:
            if not doc or doc.get("kind") != "HTTPRoute":
                continue
            ann = (doc.get("metadata") or {}).get("annotations") or {}
            if ann.get(ANNOTATION_PREFIX + "monitor") != "true":
                continue
            hostnames = (doc.get("spec") or {}).get("hostnames") or []
            if not hostnames:
                print(f"ERROR: {rel} opts into glance monitoring but has no spec.hostnames", file=sys.stderr)
                sys.exit(1)
            hostname = pick_hostname(hostnames, ann.get(ANNOTATION_PREFIX + "monitor-hostname"))
            title = ann.get(ANNOTATION_PREFIX + "monitor-title") or title_from_hostname(hostname)
            group = ann.get(ANNOTATION_PREFIX + "monitor-group") or default_group(path)
            entries.append((group, title, f"https://{hostname}", rel))
    return entries


def collect_from_helm_values():
    entries = []
    for path in REPO_ROOT.glob("apps/**/values.yaml"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            doc = yaml.safe_load(path.read_text())
        except yaml.YAMLError as e:
            print(f"ERROR: failed to parse {rel}: {e}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(doc, dict):
            continue
        glance_cfg = doc.get("glance") or {}
        if not glance_cfg.get("monitor"):
            continue
        httproute = doc.get("httproute") or {}
        if not httproute.get("enabled"):
            print(f"ERROR: {rel} sets glance.monitor but httproute.enabled is not true", file=sys.stderr)
            sys.exit(1)
        hostnames = httproute.get("hostnames") or []
        if not hostnames:
            print(f"ERROR: {rel} sets glance.monitor but has no httproute.hostnames", file=sys.stderr)
            sys.exit(1)
        hostname = pick_hostname(hostnames, glance_cfg.get("hostname"))
        title = glance_cfg.get("title") or title_from_hostname(hostname)
        group = glance_cfg.get("group") or default_group(path)
        entries.append((group, title, f"https://{hostname}", rel))
    return entries


def render_sites(sites, indent):
    pad = " " * indent
    lines = []
    for title, url in sorted(sites):
        lines.append(f"{pad}- title: {title}")
        lines.append(f"{pad}  url: {url}")
    return lines


def main():
    all_entries = collect_from_httproutes() + collect_from_helm_values()

    seen_urls = {}
    by_group = {}
    for group, title, url, src in all_entries:
        if url in seen_urls:
            print(f"ERROR: {url} opted into glance monitoring from both {seen_urls[url]} and {src}", file=sys.stderr)
            sys.exit(1)
        seen_urls[url] = src
        by_group.setdefault(group, []).append((title, url))

    text = GLANCE_VALUES.read_text()
    lines = text.split("\n")
    out = []
    skipping = False
    for line in lines:
        stripped = line.strip()
        if not skipping and stripped.startswith("# BEGIN GLANCE-GENERATED:"):
            group = stripped.split(":", 1)[1].strip()
            indent = len(line) - len(line.lstrip(" "))
            out.append(line)
            out.extend(render_sites(by_group.pop(group, []), indent))
            skipping = True
            continue
        if skipping and stripped.startswith("# END GLANCE-GENERATED:"):
            skipping = False
            out.append(line)
            continue
        if skipping:
            continue
        out.append(line)

    if by_group:
        print(
            f"ERROR: no '# BEGIN GLANCE-GENERATED:<group>' marker in {GLANCE_VALUES.relative_to(REPO_ROOT)} "
            f"for group(s): {', '.join(sorted(by_group))}",
            file=sys.stderr,
        )
        sys.exit(1)

    new_text = "\n".join(out)
    if new_text != text:
        GLANCE_VALUES.write_text(new_text)
        print("glance monitor list updated")
    else:
        print("glance monitor list unchanged")


if __name__ == "__main__":
    main()
