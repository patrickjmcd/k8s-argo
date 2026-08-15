# Fantasy Football Draft MCP

Local [MCP](https://modelcontextprotocol.io) servers that let Claude help you
draft your fantasy football team — one for **Yahoo**, one for **ESPN**, plus a
**local mock-draft simulator** to rehearse — all sharing a single
platform-agnostic draft brain.

```
fantasy_core/     # the "brain": models, value-based drafting, tiers, roster needs, pick advice
mcp_yahoo/        # Yahoo Fantasy API (official OAuth2) -> MCP tools
mcp_espn/         # ESPN Fantasy API (unofficial, cookie auth) -> MCP tools
mcp_mock/         # local mock-draft simulator (snake + auction) -> MCP tools
tests/            # tests for the shared brain and simulator (no credentials needed)
```

Both **snake and auction** drafts are supported. In an auction the tools switch
to budget-aware advice — per-player dollar values, suggested max bids, and
budget/max-bid tracking — instead of snake pick order.

## Why two servers, one brain

Yahoo and ESPN could not be more different under the hood:

| | Yahoo | ESPN |
|---|---|---|
| API | Official, documented | Unofficial (reverse-engineered) |
| Auth | OAuth2 (consumer key/secret + token) | Browser cookies (`espn_s2`, `SWID`) |
| Projections | ADP + preseason ranks, no season points | Season point projections |

Because the auth models and data shapes are so different, each platform gets its
own thin server. But the valuable part — value-based drafting (VBD/VOR), tiers,
positional scarcity, roster construction, and "who should I pick right now" — is
identical regardless of platform, so it lives once in `fantasy_core` and both
servers call it. When a platform lacks projections (Yahoo), the brain degrades
gracefully to a rank/ADP-driven model.

During a live draft you're only ever in one league at a time, so you point Claude
at whichever server is drafting that night.

## Tools (both servers expose the same set)

- **`draft_status`** — pick number, whether you're on the clock, your current
  roster, and remaining starting-lineup needs.
- **`recommend_pick`** — ranked "who to draft right now," combining
  value-over-replacement, positional scarcity, and your roster holes. Strategies:
  `bpa` (best player available), `need`, or `balanced` (default).
- **`best_available`** — top undrafted players, optionally filtered by position.
- **`position_tiers`** — tier breaks at a position so you can see a talent cliff
  coming before it hits.

In an **auction** draft, `recommend_pick` returns targets with a par dollar value
and a suggested max bid, and `draft_status` reports your remaining budget, open
roster spots, and current max bid.

## Testing with a mock draft

**Heads up:** ESPN and Yahoo mock-draft *rooms* are ephemeral practice lobbies
with no API-queryable league id, so you can't point the Yahoo/ESPN servers at a
mock draft on their sites. Instead, this repo ships a **local simulator**
(`mcp_mock`) that runs a full draft against bots using the same recommendation
brain — the best way to rehearse and to sanity-check the advice before draft day.
It supports both snake and auction.

Add it like the other servers (no credentials needed):

```json
{
  "mcpServers": {
    "mock-draft": { "command": "/absolute/path/to/.venv/bin/mock-draft-mcp" }
  }
}
```

Its tools:

- **`start_mock_draft`** — `teams`, `my_slot`, `draft_format` (`snake`/`auction`),
  `budget`. Resets and starts a draft.
- **`draft_player`** (snake) — draft an available player by name; bots then pick
  up to your next turn.
- **`buy_player`** (auction) — win a player at a price (rejected if it exceeds your
  max bid); opponents then buy a wave of players.
- **`sim_opponents`** — advance bots to your next pick (snake) or run N opponent
  purchases (auction).
- Plus **`draft_status`**, **`recommend_pick`**, **`best_available`**,
  **`position_tiers`**, same as the live servers.

Then just talk to Claude: *"Start a 12-team auction mock, I'm team 5."* →
*"Who should I target and how much?"* → *"Buy WR1 for $40."* → *"What's my
budget now?"* Player names in the simulator are synthetic (`RB1`, `WR2`, …) since
it carries no real projections — it's for rehearsing mechanics and decisions.

## Want a real-data dry run?

Point the ESPN or Yahoo server at one of **your previous seasons** (set
`ESPN_SEASON` / `YAHOO_GAME_ID` to a past year you played). The API returns that
season's completed draft and rosters, which is a good way to validate parsing
against real data — though it won't feel "live."

## Setup

Requires Python 3.10+.

```bash
cd fantasy-football-draft
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"        # or ".[yahoo]" / ".[espn]" for just one
cp .env.example .env           # then fill it in (see below)
```

### Yahoo (one-time OAuth)

1. Register an app at <https://developer.yahoo.com/apps/> — application type
   *Installed Application*, with **Fantasy Sports: Read** permission. Copy the
   **Client ID (Consumer Key)** and **Client Secret**.
2. Put `YAHOO_CONSUMER_KEY`, `YAHOO_CONSUMER_SECRET`, `YAHOO_LEAGUE_ID`, and
   `YAHOO_TEAM_ID` in `.env`.
3. Run the one-time browser flow:
   ```bash
   python -m mcp_yahoo.auth
   ```
   Approve, paste the code back, and the token is saved to `.env`. The server
   refreshes it automatically after that — no more browser.

Your **league id** is the number in your league URL
(`.../f1/<league_id>`). Your **team id** is the number in your team URL.

### ESPN (cookies)

Public leagues need only `ESPN_LEAGUE_ID` and `ESPN_SEASON`. **Private** leagues
also need two cookies from a logged-in browser session:

1. Log into <https://espn.com> and open your league.
2. DevTools → Application → Cookies → `espn.com`. Copy the values of **`espn_s2`**
   and **`SWID`** (include SWID's surrounding `{...}` braces).
3. Put them in `.env` as `ESPN_S2` and `ESPN_SWID`, along with `ESPN_LEAGUE_ID`,
   `ESPN_SEASON`, and `ESPN_TEAM_ID`.

Cookies expire periodically — if ESPN calls start failing, re-copy them.

## Connecting to Claude

These run locally over stdio. Add them to your Claude client config, pointing at
this project's virtualenv Python and setting the env from your `.env` values.

**Claude Desktop** (`claude_desktop_config.json`) or **Claude Code**
(`.mcp.json` / `claude mcp add`):

```json
{
  "mcpServers": {
    "yahoo-draft": {
      "command": "/absolute/path/to/fantasy-football-draft/.venv/bin/yahoo-draft-mcp",
      "env": {
        "YAHOO_ENV_DIR": "/absolute/path/to/fantasy-football-draft",
        "YAHOO_LEAGUE_ID": "12345",
        "YAHOO_TEAM_ID": "4",
        "YAHOO_CONSUMER_KEY": "...",
        "YAHOO_CONSUMER_SECRET": "..."
      }
    },
    "espn-draft": {
      "command": "/absolute/path/to/fantasy-football-draft/.venv/bin/espn-draft-mcp",
      "env": {
        "ESPN_LEAGUE_ID": "67890",
        "ESPN_SEASON": "2026",
        "ESPN_TEAM_ID": "3",
        "ESPN_S2": "...",
        "ESPN_SWID": "{...}"
      }
    }
  }
}
```

`YAHOO_ENV_DIR` tells the Yahoo server where to find/refresh the OAuth token in
`.env`. Alternatively, put everything in `.env` and launch the servers from this
directory so they pick it up automatically.

Once connected, on draft night just ask Claude things like:

> *"What's my draft status?"*
> *"Recommend a pick — balanced strategy."*
> *"Best available RBs?"*
> *"Show me the WR tiers."*

## How the recommendations work

- **Value over replacement (VOR/VBD):** a player's value isn't their raw
  projection but how much they beat the *replacement-level* starter at their
  position (the last startable player across the league). This is what makes a
  workhorse RB worth more than a high-scoring QB in a 1-QB league.
- **Positional scarcity:** if the projected-points cliff at a position is steep
  and close, the pick gets more urgent.
- **Roster needs:** dedicated starting slots you still need to fill get a bonus;
  flex slots are handled as a shared pool so a single flex never reads as "need"
  at every eligible position; positions you're already deep at get a penalty.
- **Strategy** dials how heavily roster need weighs against raw value.
- **Auction values:** the league's total discretionary money (total budget minus
  a $1 minimum per roster spot) is spread across players in proportion to their
  value over replacement, giving each a par dollar value. Suggested max bids add a
  small premium for positions you still need to start and are always capped so you
  keep $1 for every remaining roster spot.

Yahoo lacks season projections, so there the brain uses preseason rank and ADP
for the same value/scarcity/tier/auction logic.

## Development

```bash
pip install -e ".[dev]"
pytest                 # tests the shared brain, no credentials required
```

The brain (`fantasy_core`) is pure Python with no platform dependencies, so it's
fully unit-testable offline. Platform clients normalize their payloads into
`fantasy_core.models` and everything downstream is shared.

## Notes & caveats

- **Run locally for the draft.** Lowest latency when you're on the clock, and
  your Yahoo tokens / ESPN cookies never leave your machine.
- **ESPN is unofficial** and can break when ESPN changes their site; the
  `espn-api` dependency is the community-maintained wrapper that tracks it.
- Not affiliated with or endorsed by Yahoo or ESPN.
```
