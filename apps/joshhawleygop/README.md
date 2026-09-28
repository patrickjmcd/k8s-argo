# joshhawleygop.com

Static site generated from markdown by `build.py`, served by nginx from a
ConfigMap. Owner-facing material (outreach kit, video scripts, checklist) is in
[`owner/`](owner/) and is not published.

## Add a news post (the whole workflow)

1. Create `content/news/YYYY-MM-DD-your-slug.md`. The date is the publish
   date; the URL becomes `/news/your-slug`.
2. Push to `main` (the GitHub web editor works fine from a phone).
3. The `Build joshhawleygop.com` workflow rebuilds `site/`, the sitemap, the
   RSS feed and `kustomization.yaml`, commits them, and ArgoCD deploys. New
   files appear on the live site within a few minutes, with no pod restart.

Template:

```markdown
---
title: Keyword-first title, 60 characters max
description: 150 to 160 characters, written like ad copy. Say what the reader gets and why it matters now.
h1: The headline on the page (can differ from the title)
crumb: Short breadcrumb label
lede: One-sentence standfirst shown under the headline.
modified: 2026-10-01
related: medicaid-vote, missouri-residency
---
Opening paragraph. Link to at least one topic page, e.g.
[how Hawley voted on Medicaid](/medicaid-vote).

## A subheading

Body text. Every factual claim gets a source.

## Sources

- [Outlet: headline](https://example.com/article)
```

`related` takes topic slugs: `missouri-residency`, `january-6`,
`medicaid-vote`, `pact-act-veterans`, `abortion-amendment-3`,
`ladder-climber`, `jackson-confirmation` (also `timeline`, `by-the-numbers`,
`faq`). The build warns if a title or description is out of range.

## Build locally

```bash
pip install -r apps/joshhawleygop/requirements.txt
python3 apps/joshhawleygop/build.py
```

## Layout

| Path | What |
|---|---|
| `content/pages/*.md` | Home, the 7 topic pages, FAQ, timeline, by-the-numbers, news index, about, 404 |
| `content/news/*.md` | Blog posts |
| `content/partials/cta.md` | The "register to vote" block on every page. **Update after Oct 7 and after Nov 3, 2026.** |
| `src/` | CSS, favicon, social share image (`og.png`, 1200x630) |
| `site/` | Generated. Don't edit by hand |
| `nginx/default.conf` | Clean URLs, redirects, JSON access logs. **Bump `nginx-conf-rev` in `values.yaml` after editing**, or the pod keeps the old config |

Page front matter keys: `type` (home, topic, faq, news, page), `order`
(topics), `anchor` (link text used when other pages link here), `teaser`
(homepage card), `published`/`modified` (ISO dates; drive Article schema and
the sitemap).

## Limits

The whole site ships as one ConfigMap, capped at 1 MiB; the build stops at
900 KiB. It's about 335 KiB now (roughly 12 KiB per post), so there's room
for 40+ more posts. Past that, move `site/` into a container image.
