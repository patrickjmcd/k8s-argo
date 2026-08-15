"""ESPN Fantasy Football client.

ESPN has no official public API. This wraps the community ``espn-api`` library,
which hits the same private ``fantasy.espn.com/apis/v3`` endpoints the website
uses. Private leagues require two browser cookies, ``espn_s2`` and ``SWID``,
which you copy from a logged-in ESPN session.
"""

from __future__ import annotations

import os

from espn_api.football import League

from fantasy_core import DraftPick, DraftState, Player, Position, RosterSettings

# ESPN's numeric roster-slot ids -> our positions. Slot ids are stable across
# seasons; the multi-position slots (FLEX etc.) are handled separately.
_SLOT_TO_POSITION = {
    0: Position.QB,
    2: Position.RB,
    4: Position.WR,
    6: Position.TE,
    16: Position.DST,
    17: Position.K,
}
_FLEX_SLOT = 23  # RB/WR/TE
_SUPERFLEX_SLOT = 7  # OP (QB/RB/WR/TE) in some leagues


def _config() -> dict:
    return {
        "league_id": os.environ.get("ESPN_LEAGUE_ID"),
        "year": int(os.environ.get("ESPN_SEASON", "2026")),
        "espn_s2": os.environ.get("ESPN_S2"),
        "swid": os.environ.get("ESPN_SWID"),
        "team_id": os.environ.get("ESPN_TEAM_ID"),
    }


def _connect() -> League:
    cfg = _config()
    if not cfg["league_id"]:
        raise RuntimeError("ESPN_LEAGUE_ID is not set")
    return League(
        league_id=int(cfg["league_id"]),
        year=cfg["year"],
        espn_s2=cfg["espn_s2"],
        swid=cfg["swid"],
    )


def _to_player(p) -> Player:
    pos = Position.normalize(getattr(p, "position", "") or "")
    return Player(
        id=str(getattr(p, "playerId", getattr(p, "name", ""))),
        name=getattr(p, "name", "unknown"),
        position=pos,
        nfl_team=getattr(p, "proTeam", "") or "",
        projected_points=getattr(p, "projected_total_points", None),
        percent_rostered=getattr(p, "percent_owned", None),
        overall_rank=_espn_rank(p),
        injury_status=(getattr(p, "injuryStatus", "") or "").title().replace("_", " "),
    )


def _espn_rank(p) -> int | None:
    ranks = getattr(p, "posRank", None)
    if isinstance(ranks, int):
        return ranks
    return None


def _roster_settings(league: League) -> RosterSettings:
    """Read starting-lineup requirements from the league's slot counts."""
    settings = RosterSettings.standard()
    slotcounts = getattr(getattr(league, "settings", None), "position_slot_counts", None)
    if not slotcounts:
        return settings
    # espn-api exposes counts keyed by position abbreviation strings.
    mapping = {
        "QB": Position.QB,
        "RB": Position.RB,
        "WR": Position.WR,
        "TE": Position.TE,
        "K": Position.K,
        "D/ST": Position.DST,
    }
    new_slots = {}
    for label, count in slotcounts.items():
        pos = mapping.get(label)
        if pos and count:
            new_slots[pos] = int(count)
    if new_slots:
        settings.slots = new_slots
    settings.flex = int(slotcounts.get("RB/WR/TE", slotcounts.get("FLEX", 1)) or 0)
    settings.superflex = int(slotcounts.get("OP", 0) or 0)
    settings.bench = int(slotcounts.get("BE", slotcounts.get("BN", 6)) or 6)
    return settings


def get_draft_state() -> DraftState:
    league = _connect()
    cfg = _config()
    settings = _roster_settings(league)

    picks: list[DraftPick] = []
    drafted_ids: set[str] = set()
    for pk in getattr(league, "draft", []) or []:
        # espn-api Pick objects expose playerId/playerName/round_num/round_pick/team.
        # They don't carry position, so drafted players are recorded by name/id only.
        player = Player(
            id=str(getattr(pk, "playerId", getattr(pk, "playerName", ""))),
            name=getattr(pk, "playerName", "unknown"),
            position=None,
            nfl_team="",
        )
        team = getattr(pk, "team", None)
        team_id = str(getattr(team, "team_id", "")) if team else ""
        overall = getattr(pk, "round_num", 0) * league.settings.team_count - (
            league.settings.team_count - getattr(pk, "round_pick", 0)
        )
        picks.append(
            DraftPick(
                overall=overall or len(picks) + 1,
                round=getattr(pk, "round_num", 0),
                team_id=team_id,
                player=player,
            )
        )
        drafted_ids.add(player.id)

    # Available pool = free agents (large size to cover an early draft board).
    available: list[Player] = []
    try:
        for p in league.free_agents(size=300):
            pl = _to_player(p)
            if pl.id not in drafted_ids:
                available.append(pl)
    except Exception:
        # Before/around draft time free_agents can be empty; fall back to players.
        pass

    draft_slot = os.environ.get("ESPN_DRAFT_SLOT")
    return DraftState(
        settings=settings,
        picks=picks,
        available=available,
        my_team_id=cfg.get("team_id") or "",
        my_draft_slot=int(draft_slot) if draft_slot else None,
        num_teams=getattr(league.settings, "team_count", 0),
    )


def get_available_players(position: str | None = None, limit: int = 50) -> list[Player]:
    league = _connect()
    players = [_to_player(p) for p in league.free_agents(size=max(limit * 3, 100))]
    if position:
        want = Position.normalize(position)
        players = [p for p in players if p.position == want]
    players.sort(
        key=lambda p: (p.projected_points is None, -(p.projected_points or 0))
    )
    return players[:limit]
