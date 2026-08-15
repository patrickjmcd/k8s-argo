"""Yahoo Fantasy Football client.

Yahoo has an official OAuth2 Fantasy Sports API. This wraps the ``yfpy`` library,
which handles the ugly nested-JSON parsing and OAuth token refresh for us.

Auth model: register an app at https://developer.yahoo.com/apps/ to get a
consumer key + secret, then run ``python -m mcp_yahoo.auth`` once to complete the
browser OAuth flow. The token is saved to the .env file and refreshed
automatically on every subsequent run, so the server itself is non-interactive.

Note: Yahoo's players collection exposes ADP and preseason ranks but not season
point projections, so the advisor runs in its rank/ADP fallback mode here.
"""

from __future__ import annotations

import os
from pathlib import Path

from yfpy.query import YahooFantasySportsQuery

from fantasy_core import DraftPick, DraftState, Player, Position, RosterSettings

_ENV_DIR = Path(os.environ.get("YAHOO_ENV_DIR", Path.cwd()))


def _query() -> YahooFantasySportsQuery:
    league_id = os.environ.get("YAHOO_LEAGUE_ID")
    if not league_id:
        raise RuntimeError("YAHOO_LEAGUE_ID is not set")
    return YahooFantasySportsQuery(
        league_id=league_id,
        game_code=os.environ.get("YAHOO_GAME_CODE", "nfl"),
        game_id=int(os.environ["YAHOO_GAME_ID"]) if os.environ.get("YAHOO_GAME_ID") else None,
        yahoo_consumer_key=os.environ.get("YAHOO_CONSUMER_KEY"),
        yahoo_consumer_secret=os.environ.get("YAHOO_CONSUMER_SECRET"),
        env_file_location=_ENV_DIR,
        save_token_data_to_env_file=True,
        browser_callback=False,
    )


def _team_num(team_key: str) -> str:
    """Extract the numeric team id from a Yahoo team_key like nfl.l.123.t.4."""
    return team_key.rsplit(".t.", 1)[-1] if team_key else ""


def _to_player(p) -> Player:
    name = getattr(getattr(p, "name", None), "full", None) or getattr(p, "name", "unknown")
    pos = Position.normalize(getattr(p, "display_position", "") or getattr(p, "primary_position", ""))

    adp = None
    da = getattr(p, "draft_analysis", None)
    if da is not None:
        raw = getattr(da, "average_pick", None)
        try:
            adp = float(raw) if raw not in (None, "", "-") else None
        except (TypeError, ValueError):
            adp = None

    pct = None
    po = getattr(p, "percent_owned", None)
    if po is not None:
        val = getattr(po, "value", po)
        try:
            pct = float(val)
        except (TypeError, ValueError):
            pct = None

    bye = None
    byes = getattr(p, "bye_weeks", None)
    if byes is not None:
        wk = getattr(byes, "week", None)
        try:
            bye = int(wk) if wk else None
        except (TypeError, ValueError):
            bye = None

    rank = _preseason_rank(p)

    return Player(
        id=str(getattr(p, "player_id", getattr(p, "player_key", name))),
        name=name,
        position=pos,
        nfl_team=getattr(p, "editorial_team_abbr", "") or "",
        adp=adp,
        percent_rostered=pct,
        bye_week=bye,
        overall_rank=rank,
        injury_status=(getattr(p, "status", "") or ""),
    )


def _preseason_rank(p) -> int | None:
    """Pull Yahoo's preseason (PS) overall rank if present, else season (S)."""
    ranks = getattr(p, "player_ranks", None) or []
    best = None
    for r in ranks:
        rtype = getattr(r, "rank_type", "")
        rval = getattr(r, "rank_value", None)
        try:
            rval = int(rval)
        except (TypeError, ValueError):
            continue
        if rtype in ("PS", "PS0"):  # preseason overall
            return rval
        if rtype == "S" and best is None:
            best = rval
    return best


def _roster_settings(query: YahooFantasySportsQuery) -> RosterSettings:
    settings = RosterSettings.standard()
    try:
        league_settings = query.get_league_settings()
    except Exception:
        return settings
    positions = getattr(league_settings, "roster_positions", None) or []
    slots: dict[Position, int] = {}
    flex = 0
    superflex = 0
    bench = 6
    for rp in positions:
        label = getattr(rp, "position", "")
        count = int(getattr(rp, "count", 0) or 0)
        if label in ("W/R/T", "W/R", "R/W/T", "FLEX"):
            flex += count
        elif label in ("Q/W/R/T", "SUPERFLEX", "SFLEX"):
            superflex += count
        elif label in ("BN", "BE"):
            bench = count
        elif label == "IR":
            continue
        else:
            pos = Position.normalize(label)
            if pos:
                slots[pos] = slots.get(pos, 0) + count
    if slots:
        settings.slots = slots
    settings.flex = flex
    settings.superflex = superflex
    settings.bench = bench
    return settings


def get_draft_state() -> DraftState:
    query = _query()
    settings = _roster_settings(query)

    # Draft results: which player_keys are already gone, and my roster.
    drafted_keys: set[str] = set()
    picks: list[DraftPick] = []
    try:
        results = query.get_league_draft_results() or []
    except Exception:
        results = []

    # Build a player_key -> Player lookup from the league player pool so drafted
    # players carry position/name, and undrafted players form the available pool.
    pool = _load_player_pool(query)
    key_to_player = {pkey: pl for pkey, (_, pl) in pool.items()}

    for dr in results:
        pkey = getattr(dr, "player_key", "")
        drafted_keys.add(pkey)
        team_key = getattr(dr, "team_key", "")
        player = key_to_player.get(pkey) or Player(id=pkey, name=pkey, position=None)
        picks.append(
            DraftPick(
                overall=int(getattr(dr, "pick", len(picks) + 1) or len(picks) + 1),
                round=int(getattr(dr, "round", 0) or 0),
                team_id=_team_num(team_key),
                player=player,
            )
        )

    available = [
        pl for pkey, pl in key_to_player.items() if pkey not in drafted_keys
    ]
    available.sort(key=lambda p: (p.overall_rank or p.adp or 1e9))

    num_teams = _num_teams(query)
    my_team_id = os.environ.get("YAHOO_TEAM_ID", "")
    slot = os.environ.get("YAHOO_DRAFT_SLOT")

    return DraftState(
        settings=settings,
        picks=picks,
        available=available,
        my_team_id=my_team_id,
        my_draft_slot=int(slot) if slot else None,
        num_teams=num_teams,
    )


def _num_teams(query: YahooFantasySportsQuery) -> int:
    try:
        info = query.get_league_info()
        return int(getattr(info, "num_teams", 0) or 0)
    except Exception:
        return 0


def _load_player_pool(query: YahooFantasySportsQuery) -> dict:
    """Fetch the league player pool, paginated. Returns {player_key: (raw, Player)}."""
    pool: dict = {}
    start = 0
    page = 25
    max_players = int(os.environ.get("YAHOO_MAX_PLAYERS", "300"))
    while start < max_players:
        try:
            players = query.get_league_players(
                player_count_limit=page, player_count_start=start
            )
        except Exception:
            break
        if not players:
            break
        for raw in players:
            pkey = getattr(raw, "player_key", None) or str(getattr(raw, "player_id", ""))
            pool[pkey] = (raw, _to_player(raw))
        if len(players) < page:
            break
        start += page
    return pool


def get_available_players(position: str | None = None, limit: int = 50) -> list[Player]:
    state = get_draft_state()
    players = state.available
    if position:
        want = Position.normalize(position)
        players = [p for p in players if p.position == want]
    return players[:limit]
