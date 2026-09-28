# Owner checklist

Things only the site owner can do. Rough order of importance.

## This week

- [ ] **Confirm the deploy.** After ArgoCD syncs, check https://joshhawleygop.com/january-6
      loads and https://www.joshhawleygop.com/ redirects to the bare domain. If
      old pages still show, the pod didn't restart: check that
      `joshhawleygop/nginx-conf-rev` is `"3"` on the running Deployment.
- [ ] **Check DNS for the two new domains.** external-dns needs its Cloudflare
      API token to cover the `senatorjoshhawley.com` and `joshhawley4senate.com`
      zones. If the token is scoped to specific zones, add them in Cloudflare →
      My Profile → API Tokens, then confirm both domains redirect.
- [ ] **Google Search Console** (10 minutes, free): https://search.google.com/search-console
  1. Add a **Domain** property for `joshhawleygop.com`. Verification is a DNS
     TXT record; Search Console can add it to Cloudflare automatically.
  2. Sitemaps → submit `https://joshhawleygop.com/sitemap.xml`.
  3. URL Inspection → "Request indexing" for `/`, `/faq`, `/timeline` and the
     seven topic pages.
  4. Check back in 2–4 weeks: **Performance → Queries** shows exactly which
     searches the site appears for. It's the only way to see real ranking data.
- [ ] **Bing Webmaster Tools**: https://www.bing.com/webmasters. Use "Import
      from Google Search Console" (one click once GSC is verified). Bing also
      feeds DuckDuckGo and ChatGPT search.
- [ ] **Set up corrections@joshhawleygop.com.** The About page promises it.
      Cloudflare → Email → Email Routing (free) can forward it to your inbox.
- [ ] **Test the share cards.** Paste a URL into the
      [Facebook Sharing Debugger](https://developers.facebook.com/tools/debug/)
      and [LinkedIn Post Inspector](https://www.linkedin.com/post-inspector/)
      to confirm the image and title show.
- [ ] **Validate structured data** with Google's
      [Rich Results Test](https://search.google.com/test/rich-results) on a
      topic page (Article) and `/faq` (FAQPage).

## Cloudflare cache rules: review, but don't cache HTML

- [ ] **Leave HTML uncached at the edge** (the default: pages show
      `cf-cache-status: DYNAMIC`). If you add a "Cache Everything" rule, most
      page views would be answered by Cloudflare without reaching nginx, and
      the visitor dashboard would badly undercount. The site is already fast:
      Lighthouse mobile performance is 100 without edge caching.
- [ ] Caching `og.png` and `favicon.svg` at the edge is fine (nginx already
      sends a 7-day `Cache-Control`).
- [ ] Confirm **Always Use HTTPS** is on, and that **Bot Fight Mode**, if
      enabled, isn't challenging Googlebot (verified bots are allowed by
      default; check Security → Events if pages stop getting indexed).
- [ ] If you'd rather cache HTML for resilience, switch visitor stats to
      Cloudflare Web Analytics first, then add the cache rule.

## Keep it fresh (ongoing)

- [ ] **Post regularly.** Google rewards fresh content on political queries;
      aim for 2–4 posts a month. Adding a post is one markdown file in
      `content/news/` (see `../README.md`). Good hooks: every Hawley vote,
      bill or quote that contradicts his record; each Amendment 3 development.
- [ ] **After October 7, 2026:** edit `content/partials/cta.md`. The
      registration deadline has passed, so switch the call to action to "Vote
      November 3" with polling-place info.
- [ ] **After November 3, 2026:** update the CTA again, post the Amendment 3
      result, and update `/abortion-amendment-3` and the timeline.
- [ ] When you edit a page, bump its `modified:` date so the sitemap and
      Article schema show it's current.

## Link building and video

- [ ] Work through `outreach-kit.md`: a handful of personalized pitches per
      week beats a blast. Track who you contacted and who linked.
- [ ] Produce and publish the three videos in `youtube-scripts.md`. Mind the
      footage-rights notes at the top. Link each video from its topic page
      afterward (a new markdown line on the page).
- [ ] Share specific pages (not the homepage) where they answer a question:
      the FAQ and by-the-numbers pages are the easiest to share.

## Realistic expectations

- **"Josh Hawley"** is dominated by Wikipedia, senate.gov, and national news.
  The homepage is optimized for it, but page one takes strong backlinks and
  time. The realistic early wins are question queries ("where does josh
  hawley live", "did josh hawley vote to cut medicaid") and topic pages.
- **FAQ rich results:** since 2023 Google shows FAQ rich results mainly for
  government and health sites, so the FAQ markup is unlikely to produce
  expandable Q&As in Google. It still helps Bing and AI answer engines
  understand the page, and the questions themselves target real searches.
- New domains typically take weeks to a few months to rank. Watch Search
  Console's Queries report, not day-to-day traffic.

## Keep it legal

- It's a noncommercial criticism site: no ads, no donations, no selling. Keep
  it that way, and don't offer the Hawley-name domains for sale. Cybersquatting
  law (including the personal-name provision, 15 U.S.C. § 8131) targets
  registering someone's name to profit from it.
- Keep the non-affiliation disclaimer on every page (the template does this).
