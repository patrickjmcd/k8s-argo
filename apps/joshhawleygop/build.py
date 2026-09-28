#!/usr/bin/env python3
"""Build joshhawleygop.com from markdown.

    pip install -r apps/joshhawleygop/requirements.txt
    python3 apps/joshhawleygop/build.py

Reads content/pages/*.md, content/news/*.md and content/partials/*.md, writes
the flat site into site/ (served by nginx, see nginx/default.conf for the URL
mapping) and regenerates kustomization.yaml's file list. CI runs this on every
push to main that touches the sources, so adding a news post is just adding
one markdown file to content/news/.

Front matter is simple "key: value" lines between --- markers. Lists
(related) are comma-separated.
"""

import html
import json
import re
import sys
from datetime import date
from pathlib import Path

import markdown

SITE_URL = "https://joshhawleygop.com"
SITE_NAME = "The Hawley Record"
ROOT = Path(__file__).resolve().parent
CONTENT = ROOT / "content"
OUT = ROOT / "site"
SRC = ROOT / "src"
# ConfigMaps cap out at 1 MiB; leave headroom for the object's metadata.
MAX_SITE_BYTES = 900 * 1024

HAWLEY = {
    "@type": "Person",
    "name": "Josh Hawley",
    "sameAs": [
        "https://en.wikipedia.org/wiki/Josh_Hawley",
        "https://www.hawley.senate.gov/",
        "https://bioguide.congress.gov/search/bio/H001089",
    ],
}
ORG = {"@type": "Organization", "name": SITE_NAME, "url": SITE_URL + "/"}

NAV = [
    ("/#record", "The record"),
    ("/timeline", "Timeline"),
    ("/by-the-numbers", "By the numbers"),
    ("/faq", "FAQ"),
    ("/news", "News"),
]

warnings = []


def warn(msg):
    warnings.append(msg)


# --- parsing -----------------------------------------------------------------

def parse(path):
    text = path.read_text(encoding="utf-8")
    meta = {}
    if text.startswith("---\n"):
        head, text = text[4:].split("\n---\n", 1)
        for line in head.splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
    meta["related"] = [s.strip() for s in meta.get("related", "").split(",") if s.strip()]
    for key in ("h1", "lede"):
        if meta.get(key):
            meta[key] = inline_md(meta[key])
    return meta, text


def render_md(text):
    out = markdown.markdown(
        text,
        extensions=["attr_list", "sane_lists", "smarty"],
        output_format="html",
    )
    # attr_list can't class a whole list, so style the list after "## Sources".
    return re.sub(r"(<h2[^>]*>Sources</h2>\s*)<ul>", r'\1<ul class="sources">', out)


def inline_md(text):
    """Front-matter strings (h1, lede) get the same smart quotes as body text."""
    return re.sub(r"^<p>(.*)</p>$", r"\1", render_md(text), flags=re.S)


def external_rel(markup):
    """rel="noopener" on every outbound link; internal links untouched."""
    def fix(m):
        tag = m.group(0)
        if "rel=" in tag:
            return tag
        return tag[:-1] + ' rel="noopener">'
    return re.sub(r'<a href="https?://(?!joshhawleygop\.com)[^"]*"[^>]*>', fix, markup)


def text_only(markup):
    return html.unescape(re.sub(r"<[^>]+>", "", markup)).strip()


def check_meta(slug, meta):
    title, desc = meta.get("title", ""), meta.get("description", "")
    if not title or len(title) > 60:
        warn(f"{slug}: title is {len(title)} chars (want 1-60): {title!r}")
    if not 150 <= len(desc) <= 160:
        warn(f"{slug}: description is {len(desc)} chars (want 150-160)")


# --- templates ---------------------------------------------------------------

def e(s):
    return html.escape(s, quote=True)


def jsonld(obj):
    # "</" can't appear inside a <script> element.
    return '<script type="application/ld+json">' + json.dumps(obj, ensure_ascii=False).replace("</", "<\\/") + "</script>"


def breadcrumb_ld(trail):
    return {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": [
            {"@type": "ListItem", "position": i + 1, "name": name, "item": SITE_URL + href}
            for i, (href, name) in enumerate(trail)
        ],
    }


def article_ld(meta, path):
    return {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": meta["h1_text"][:110],
        "description": meta["description"],
        "datePublished": meta["published"],
        "dateModified": meta.get("modified", meta["published"]),
        "author": ORG,
        "publisher": ORG,
        "mainEntityOfPage": SITE_URL + path,
        "image": SITE_URL + "/og.png",
        "about": HAWLEY,
        "inLanguage": "en-US",
    }


def page(*, path, meta, hero, body, css, schemas=(), trail=None, body_class=""):
    canonical = SITE_URL + path
    title = meta["title"]
    desc = meta["description"]
    og_type = "article" if meta.get("published") else "website"
    current = ' aria-current="page"'
    nav = "".join(
        f'<a href="{href}"{current if href == path else ""}>{e(label)}</a>'
        for href, label in NAV
    )
    crumbs = ""
    if trail:
        items = "".join(
            f'<li><a href="{href}">{e(name)}</a></li>' if i < len(trail) - 1 else f'<li aria-current="page">{e(name)}</li>'
            for i, (href, name) in enumerate(trail)
        )
        crumbs = f'<nav class="crumbs" aria-label="Breadcrumb"><ol>{items}</ol></nav>'
        schemas = list(schemas) + [breadcrumb_ld(trail)]
    article_meta = ""
    if meta.get("published"):
        article_meta = (
            f'<meta property="article:published_time" content="{meta["published"]}">'
            f'<meta property="article:modified_time" content="{meta.get("modified", meta["published"])}">'
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title>
<meta name="description" content="{e(desc)}">
<link rel="canonical" href="{canonical}">
<meta name="robots" content="index, follow, max-image-preview:large">
<meta name="theme-color" content="#14213d">
<meta property="og:site_name" content="{SITE_NAME}">
<meta property="og:type" content="{og_type}">
<meta property="og:title" content="{e(title)}">
<meta property="og:description" content="{e(desc)}">
<meta property="og:url" content="{canonical}">
<meta property="og:image" content="{SITE_URL}/og.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="The Hawley Record: Josh Hawley's record, with receipts">
<meta property="og:locale" content="en_US">
{article_meta}
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{e(title)}">
<meta name="twitter:description" content="{e(desc)}">
<meta name="twitter:image" content="{SITE_URL}/og.png">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="alternate" type="application/rss+xml" title="{SITE_NAME} news" href="{SITE_URL}/feed.xml">
<style>{css}</style>
{"".join(jsonld(s) for s in schemas)}
</head>
<body class="{body_class}">
<div class="disclaimer-bar">Independent citizen commentary. Not affiliated with Josh Hawley, his office, his campaign, or the Republican Party. <a href="/about">About this site</a></div>
<nav class="site-nav" aria-label="Site"><div class="wrap"><a class="brand" href="/">{SITE_NAME}</a><div class="links">{nav}</div></div></nav>
<header class="hero"><div class="wrap">{hero}</div></header>
{crumbs}
<main><div class="wrap">
{body}
</div></main>
{PARTS["cta"]}
<footer class="site-footer"><div class="wrap">
<nav aria-label="All topics">{FOOTER_LINKS}</nav>
<p><strong>About this site.</strong> Independent political commentary by a private Missouri citizen. Not affiliated with, authorized by, or paid for by Josh Hawley, his Senate office, any campaign or candidate committee, or the Republican Party.</p>
<p>Every factual claim links to its source. The insults are opinion, and they're earned. Spot a factual error? <a href="/about#corrections">Corrections policy</a>.</p>
</div></footer>
</body>
</html>
"""


def minify_css(css):
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = re.sub(r"\s+", " ", css)
    css = re.sub(r"\s*([{}:;,>])\s*", r"\1", css)
    return css.replace(";}", "}").strip()


def minify_html(markup):
    # Whitespace between tags only; no <pre>/<textarea> on this site.
    return re.sub(r">\s+<", "><", markup).strip() + "\n"


# --- build -------------------------------------------------------------------

def main():
    global PARTS, FOOTER_LINKS
    css = minify_css((SRC / "style.css").read_text(encoding="utf-8"))
    PARTS = {p.stem: external_rel(render_md(p.read_text(encoding="utf-8"))) for p in (CONTENT / "partials").glob("*.md")}
    PARTS["cta"] = f'<section class="act" aria-label="Take action"><div class="wrap">{PARTS["cta"]}</div></section>'

    pages = {}
    for p in sorted((CONTENT / "pages").glob("*.md")):
        meta, text = parse(p)
        meta["slug"] = p.stem
        meta["path"] = "/" if p.stem == "home" else f"/{p.stem}"
        meta["h1_text"] = text_only(meta.get("h1", ""))
        meta["_md"] = text
        pages[p.stem] = meta

    posts = []
    for p in sorted((CONTENT / "news").glob("*.md")):
        meta, text = parse(p)
        m = re.match(r"(\d{4}-\d{2}-\d{2})-(.+)", p.stem)
        if not m:
            sys.exit(f"{p.name}: news files must be named YYYY-MM-DD-slug.md")
        meta.setdefault("published", m.group(1))
        meta["slug"] = m.group(2)
        meta["path"] = f"/news/{m.group(2)}"
        meta["h1_text"] = text_only(meta.get("h1", meta.get("title", "")))
        meta["_md"] = text
        posts.append(meta)
    posts.sort(key=lambda m: (m["published"], m["slug"]), reverse=True)

    topics = sorted((m for m in pages.values() if m.get("type") == "topic"), key=lambda m: int(m["order"]))
    FOOTER_LINKS = "".join(
        f'<a href="{m["path"]}">{e(m["anchor"])}</a>' for m in topics
    ) + "".join(
        f'<a href="{href}">{label}</a>'
        for href, label in [("/timeline", "Hawley timeline"), ("/by-the-numbers", "Hawley by the numbers"), ("/faq", "Josh Hawley FAQ"), ("/news", "Latest news"), ("/about", "About &amp; corrections")]
    )

    def link_for(slug):
        if slug not in pages:
            sys.exit(f"related: unknown page {slug!r}")
        return pages[slug]

    def related_block(slugs):
        if not slugs:
            return ""
        items = "".join(f'<li><a href="{link_for(s)["path"]}">{e(link_for(s).get("anchor") or link_for(s)["h1_text"])}</a></li>' for s in slugs)
        return f'<aside class="related"><h2>Keep reading</h2><ul>{items}<li><a href="/">Back to the full Josh Hawley record</a></li></ul></aside>'

    def post_cards(items):
        return '<ul class="cards">' + "".join(
            f'<li class="card"><p class="date"><time datetime="{m["published"]}">{fmt_date(m["published"])}</time></p>'
            f'<h3><a href="{m["path"]}">{e(m["h1_text"])}</a></h3><p>{e(m["description"])}</p></li>'
            for m in items
        ) + "</ul>"

    OUT.mkdir(exist_ok=True)
    for old in OUT.iterdir():
        old.unlink()
    written = {}
    sitemap = []

    def emit(name, content):
        data = content.encode("utf-8") if isinstance(content, str) else content
        (OUT / name).write_bytes(data)
        written[name] = len(data)

    for slug, meta in pages.items():
        check_meta(slug, meta)
        body_html = external_rel(render_md(meta["_md"]))
        kind = meta.get("type", "page")
        schemas = []
        trail = [("/", "Home"), (meta["path"], meta.get("crumb", meta["h1_text"]))]
        kicker = f'<p class="kicker"><a href="/">{SITE_NAME}</a></p>'
        hero = f'{kicker}<h1>{meta["h1"]}</h1>'
        if meta.get("lede"):
            hero += f'<p class="lede">{meta["lede"]}</p>'
        body_class = meta.get("bodyclass", "")

        if kind == "home":
            trail = None
            cards = '<ul class="cards">' + "".join(
                f'<li class="card"><p class="num">{int(m["order"]):02d}</p>'
                f'<h3><a href="{m["path"]}">{e(m["anchor"])}</a></h3><p>{e(m["teaser"])}</p></li>'
                for m in topics
            ) + "</ul>"
            body_html = body_html.replace("<p>{{topics}}</p>", cards)
            body_html = body_html.replace("<p>{{latest}}</p>", post_cards(posts[:3]))
            schemas += [
                {"@context": "https://schema.org", "@type": "WebSite", "name": SITE_NAME, "url": SITE_URL + "/",
                 "description": meta["description"], "about": HAWLEY, "publisher": ORG, "inLanguage": "en-US"},
                {"@context": "https://schema.org", **ORG, "description": "Independent citizen commentary on Sen. Josh Hawley's record. Not affiliated with Josh Hawley or any party."},
            ]
            body_class = (body_class + " home").strip()
        elif kind == "topic":
            num = int(meta["order"])
            hero = f'<p class="kicker"><a href="/">{SITE_NAME}</a> &middot; {num:02d} of {len(topics)}</p><h1>{meta["h1"]}</h1>'
            if meta.get("lede"):
                hero += f'<p class="lede">{meta["lede"]}</p>'
            hero += f'<p class="meta-line">Updated <time datetime="{meta.get("modified", meta["published"])}">{fmt_date(meta.get("modified", meta["published"]))}</time></p>'
            schemas.append(article_ld(meta, meta["path"]))
            body_html += related_block(meta["related"])
        elif kind == "faq":
            qa = faq_pairs(meta["_md"])
            schemas.append({
                "@context": "https://schema.org", "@type": "FAQPage",
                "mainEntity": [
                    {"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": a}}
                    for q, a in qa
                ],
            })
            body_html += related_block(meta["related"])
        elif kind == "news":
            body_html = body_html.replace("<p>{{posts}}</p>", post_cards(posts))
            schemas.append({
                "@context": "https://schema.org", "@type": "CollectionPage", "name": meta["h1_text"],
                "url": SITE_URL + meta["path"], "about": HAWLEY, "publisher": ORG,
                "hasPart": [{"@type": "Article", "headline": m["h1_text"], "url": SITE_URL + m["path"], "datePublished": m["published"]} for m in posts],
            })
        else:
            if meta.get("published"):
                schemas.append(article_ld(meta, meta["path"]))
            body_html += related_block(meta["related"])

        if slug == "404":
            trail = None
            meta_robots = page(path="/404", meta=meta, hero=hero, body=body_html, css=css, body_class=body_class)
            meta_robots = meta_robots.replace('content="index, follow, max-image-preview:large"', 'content="noindex"')
            meta_robots = re.sub(r'<link rel="canonical"[^>]*>\n', "", meta_robots)
            emit("404.html", minify_html(meta_robots))
            continue
        name = "index.html" if kind == "home" else f"{slug}.html"
        emit(name, minify_html(page(path=meta["path"], meta=meta, hero=hero, body=body_html, css=css, schemas=schemas, trail=trail, body_class=body_class)))
        if not meta.get("modified"):
            warn(f"{slug}: needs a modified: date (sitemap lastmod)")
        sitemap.append((meta["path"], meta.get("modified") or meta.get("published") or BUILD_DATE))

    for meta in posts:
        check_meta("news/" + meta["slug"], meta)
        body_html = external_rel(render_md(meta["_md"]))
        body_html += related_block(meta["related"])
        hero = (
            f'<p class="kicker"><a href="/">{SITE_NAME}</a> &middot; <a href="/news">News</a></p>'
            f'<h1>{meta.get("h1", e(meta["title"]))}</h1>'
            + (f'<p class="lede">{meta["lede"]}</p>' if meta.get("lede") else "")
            + f'<p class="meta-line">Published <time datetime="{meta["published"]}">{fmt_date(meta["published"])}</time></p>'
        )
        trail = [("/", "Home"), ("/news", "News"), (meta["path"], meta.get("crumb", meta["h1_text"]))]
        emit(f"post--{meta['slug']}.html", minify_html(page(
            path=meta["path"], meta=meta, hero=hero, body=body_html, css=css,
            schemas=[article_ld(meta, meta["path"])], trail=trail,
        )))
        sitemap.append((meta["path"], meta.get("modified", meta["published"])))

    order = {"/": 0}
    sitemap.sort(key=lambda t: (order.get(t[0], 1), t[0]))
    emit("sitemap.xml", '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + "".join(
        f"  <url><loc>{SITE_URL}{p}</loc><lastmod>{d}</lastmod></url>\n" for p, d in sitemap
    ) + "</urlset>\n")
    emit("robots.txt", f"User-agent: *\nAllow: /\n\nSitemap: {SITE_URL}/sitemap.xml\n")
    emit("feed.xml", rss(posts))
    for asset in ("favicon.svg", "og.png"):
        emit(asset, (SRC / asset).read_bytes())

    total = sum(written.values())
    if total > MAX_SITE_BYTES:
        sys.exit(f"site/ is {total} bytes, over the {MAX_SITE_BYTES} ConfigMap budget. Time to move the site into an image.")
    write_kustomization(sorted(written))
    for w in warnings:
        print("WARNING:", w, file=sys.stderr)
    print(f"built {len(written)} files, {total / 1024:.0f} KiB")
    return 1 if warnings and "--strict" in sys.argv else 0


def faq_pairs(md_text):
    pairs = []
    for chunk in re.split(r"^## ", md_text, flags=re.M)[1:]:
        q, _, a = chunk.partition("\n")
        a_html = render_md(a.strip())
        # Google allows a small set of tags in FAQ answers; links and paragraphs are fine.
        a_html = re.sub(r"<(?!/?(a|p|ul|ol|li|strong|em|br)\b)[^>]+>", "", a_html)
        pairs.append((text_only(render_md(q.strip())), a_html.strip()))
    return pairs


def fmt_date(iso):
    d = date.fromisoformat(iso)
    return f"{d:%B} {d.day}, {d.year}"


def rss(posts):
    items = "".join(
        f"<item><title>{e(m['h1_text'])}</title><link>{SITE_URL}{m['path']}</link><guid>{SITE_URL}{m['path']}</guid>"
        f"<pubDate>{date.fromisoformat(m['published']):%a, %d %b %Y} 12:00:00 +0000</pubDate><description>{e(m['description'])}</description></item>"
        for m in posts
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<rss version="2.0"><channel>'
        f"<title>{SITE_NAME}: News</title><link>{SITE_URL}/news</link>"
        "<description>News and analysis on Sen. Josh Hawley's record.</description><language>en-us</language>"
        f"{items}</channel></rss>\n"
    )


def write_kustomization(files):
    listing = "\n".join(f"      - site/{f}" for f in files)
    (ROOT / "kustomization.yaml").write_text(f"""# GENERATED by build.py from content/ and src/. Do not edit the file list by
# hand; edit build.py's write_kustomization() for anything else.
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

# Site content and nginx config, mounted into the nginx container by
# values.yaml. Fixed names so the Helm release can reference them.
configMapGenerator:
  - name: joshhawleygop-site
    files:
{listing}
  - name: joshhawleygop-nginx
    files:
      - nginx/default.conf

generatorOptions:
  disableNameSuffixHash: true
""")


BUILD_DATE = date.today().isoformat()
PARTS = {}
FOOTER_LINKS = ""

if __name__ == "__main__":
    sys.exit(main())
