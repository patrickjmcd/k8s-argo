"""Draft-intelligence: value-based drafting, tiers, roster needs, pick advice.

This is the shared "brain" both MCP servers call. It operates entirely on the
normalized models in ``models.py`` and never touches a platform API.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .models import (
    FLEX_ELIGIBLE,
    SUPERFLEX_ELIGIBLE,
    DraftFormat,
    DraftState,
    Player,
    Position,
    RosterSettings,
)


def _sort_key(p: Player) -> tuple:
    """Best-to-worst ordering when we have to rank a pool of players.

    Prefer projected points, then platform overall rank, then ADP. Missing
    values sort last so real data always wins.
    """
    proj = -(p.projected_points if p.projected_points is not None else -1e9)
    rank = p.overall_rank if p.overall_rank is not None else 1e9
    adp = p.adp if p.adp is not None else 1e9
    return (proj, rank, adp)


def replacement_levels(
    available: list[Player], settings: RosterSettings, num_teams: int
) -> dict[Position, float]:
    """Projected points of the "replacement" player at each position.

    Value-based drafting compares a player not to zero but to what you could get
    for free later at the same position. The replacement level is the projected
    points of the last startable player at that position across the league —
    i.e. (starting slots per team x number of teams), plus a share of flex.
    """
    by_pos: dict[Position, list[float]] = defaultdict(list)
    for p in available:
        if p.position and p.projected_points is not None:
            by_pos[p.position].append(p.projected_points)

    teams = max(num_teams, 1)
    flex_share = settings.flex + settings.superflex
    levels: dict[Position, float] = {}
    for pos, points in by_pos.items():
        points.sort(reverse=True)
        starters = settings.starters_at(pos) * teams
        # Distribute flex demand across flex-eligible positions.
        if pos in FLEX_ELIGIBLE:
            starters += int(flex_share * teams / max(len(FLEX_ELIGIBLE), 1))
        if pos in SUPERFLEX_ELIGIBLE and settings.superflex:
            starters += int(settings.superflex * teams / max(len(SUPERFLEX_ELIGIBLE), 1))
        idx = min(max(starters, 1), len(points)) - 1
        levels[pos] = points[idx]
    return levels


def value_over_replacement(
    player: Player, levels: dict[Position, float]
) -> float | None:
    if player.position is None or player.projected_points is None:
        return None
    baseline = levels.get(player.position)
    if baseline is None:
        return None
    return round(player.projected_points - baseline, 1)


def auction_values(
    pool: list[Player],
    settings: RosterSettings,
    num_teams: int,
    budget: int,
) -> dict[str, int]:
    """Par dollar values for an auction, keyed by player id.

    Classic value-based auction math: spread the league's total discretionary
    money (total budget minus a $1 minimum per roster spot) across players in
    proportion to their value over replacement. Players at or below replacement
    are worth the $1 minimum and are omitted from the returned dict.

    Should be computed once over the full pre-draft universe so values stay
    stable as players leave the board. Falls back to a rank/ADP signal when the
    pool has no projections (e.g. Yahoo).
    """
    if num_teams <= 0 or budget <= 0 or not pool:
        return {}

    has_projections = any(p.projected_points is not None for p in pool)
    levels = replacement_levels(pool, settings, num_teams) if has_projections else {}

    signals: dict[str, float] = {}
    for p in pool:
        if has_projections:
            v = value_over_replacement(p, levels)
        else:
            # Center rank value on the last starter so only startable players
            # carry positive (biddable) value.
            v = _rank_value(p) - 40.0
        if v is not None and v > 0:
            signals[p.id] = v

    total_signal = sum(signals.values())
    rosterable = num_teams * (settings.total_starters() + settings.bench)
    total_money = num_teams * budget
    discretionary = total_money - rosterable
    if total_signal <= 0 or discretionary <= 0:
        return {}

    per_unit = discretionary / total_signal
    return {pid: max(1, round(1 + sig * per_unit)) for pid, sig in signals.items()}


@dataclass
class RosterNeed:
    position: Position
    starters_required: int
    starters_filled: int

    @property
    def remaining(self) -> int:
        return max(self.starters_required - self.starters_filled, 0)


def roster_needs(roster: list[Player], settings: RosterSettings) -> list[RosterNeed]:
    """How many dedicated starting slots remain open at each position.

    This reports only the position-specific slots (2 RB, 2 WR, ...). Flex and
    superflex are a shared pool handled by ``flex_remaining`` so a single flex
    slot never gets double-counted as a need at every eligible position.
    """
    have: dict[Position, int] = defaultdict(int)
    for p in roster:
        if p.position:
            have[p.position] += 1

    needs: list[RosterNeed] = []
    for pos, required in settings.slots.items():
        needs.append(RosterNeed(pos, required, min(have[pos], required)))
    needs.sort(key=lambda n: (-n.remaining, n.position.value))
    return needs


def flex_remaining(roster: list[Player], settings: RosterSettings) -> int:
    """Unfilled flex + superflex slots after dedicated starters are seated.

    Players beyond a position's dedicated starters spill into flex; whatever
    flex capacity they don't cover is still open.
    """
    have: dict[Position, int] = defaultdict(int)
    for p in roster:
        if p.position:
            have[p.position] += 1

    spill = 0
    for pos in SUPERFLEX_ELIGIBLE:
        spill += max(have[pos] - settings.starters_at(pos), 0)
    capacity = settings.flex + settings.superflex
    return max(capacity - min(spill, capacity), 0)


@dataclass
class Recommendation:
    player: Player
    vor: float | None
    score: float
    reasons: list[str]
    # Auction only: the player's par dollar value and the most you should bid.
    auction_value: int | None = None
    suggested_max_bid: int | None = None


def recommend(
    state: DraftState,
    strategy: str = "bpa",
    limit: int = 8,
) -> list[Recommendation]:
    """Rank the best players to draft right now.

    strategy:
      - "bpa"  : best player available (value first, need as a light tiebreak)
      - "need" : weight positional need heavily
      - "balanced": split the difference (default-ish behaviour many drafters want)

    In an auction draft the returned recommendations are "who to target next,"
    each carrying a par dollar value and a suggested max bid.
    """
    settings = state.settings
    available = [p for p in state.available if p.position is not None]
    if not available:
        return []

    if state.is_auction:
        return _recommend_auction(state, strategy, limit)

    # Some platforms (notably Yahoo) hand us ADP/rank but no season point
    # projections. When the pool has no projections, value-over-replacement is
    # undefined, so we fall back to a rank/ADP-derived value so the advisor still
    # produces a sensible ordering instead of collapsing to need-only.
    has_projections = any(p.projected_points is not None for p in available)

    levels = replacement_levels(available, settings, state.num_teams or 10)
    needs = {n.position: n for n in roster_needs(state.my_roster, settings)}
    flex_open = flex_remaining(state.my_roster, settings)
    have: dict[Position, int] = defaultdict(int)
    for p in state.my_roster:
        if p.position:
            have[p.position] += 1

    # Positional scarcity: how big is the drop to the next tier at this position?
    scarcity = _scarcity_by_position(available) if has_projections else _scarcity_by_rank(available)

    need_weight = {"bpa": 0.15, "balanced": 0.5, "need": 1.0}.get(strategy, 0.5)

    recs: list[Recommendation] = []
    for p in available:
        vor = value_over_replacement(p, levels) if has_projections else None
        if vor is not None:
            base = vor
        else:
            base = _rank_value(p)
        reasons: list[str] = []

        need = needs.get(p.position)
        need_bonus = 0.0
        if need and need.remaining > 0:
            need_bonus = need.remaining * 15 * need_weight
            reasons.append(f"fills a starting {p.position.value} need")
        elif flex_open > 0 and p.position in FLEX_ELIGIBLE:
            need_bonus = 8 * need_weight  # can still start in a flex slot
            reasons.append("fills a flex slot")
        elif have.get(p.position, 0) >= settings.starters_at(p.position) + settings.flex:
            need_bonus = -20 * need_weight  # already deep here
            reasons.append(f"you're already deep at {p.position.value}")

        scarcity_bonus = scarcity.get(p.position, 0.0) * 0.5
        if scarcity_bonus > 8:
            reasons.append(f"{p.position.value} tier is about to fall off")

        if p.adp is not None and p.adp <= state.next_overall_pick - 3:
            reasons.append(f"value: ADP {p.adp:.0f}, still here at {state.next_overall_pick}")

        if p.injury_status:
            reasons.append(f"injury: {p.injury_status}")

        score = base + need_bonus + scarcity_bonus
        if vor is not None:
            reasons.insert(0, f"+{vor:.0f} pts over replacement")
        elif p.overall_rank is not None:
            reasons.insert(0, f"rank #{p.overall_rank}")
        elif p.adp is not None:
            reasons.insert(0, f"ADP {p.adp:.0f}")
        recs.append(Recommendation(p, vor, round(score, 1), reasons))

    recs.sort(key=lambda r: r.score, reverse=True)
    return recs[:limit]


def _recommend_auction(
    state: DraftState, strategy: str, limit: int
) -> list[Recommendation]:
    """Auction targets: par dollar value plus a suggested max bid.

    The suggested max bid starts from the player's par value, adds a small
    premium for positions you still need to start, and is always capped by what
    you can actually spend while keeping $1 for every other open roster spot.
    """
    settings = state.settings
    # Value the full known universe (available + already-rostered) so par values
    # don't inflate as players leave the board.
    universe = list(state.available) + [p.player for p in state.picks]
    values = auction_values(universe, settings, state.num_teams or 10, state.budget)

    needs = {n.position: n for n in roster_needs(state.my_roster, settings)}
    flex_open = flex_remaining(state.my_roster, settings)
    my_max = state.max_bid(state.my_team_id)
    my_remaining = state.budget_remaining(state.my_team_id)
    open_slots = state.open_roster_slots(state.my_team_id)

    need_weight = {"bpa": 0.1, "balanced": 0.35, "need": 0.7}.get(strategy, 0.35)

    recs: list[Recommendation] = []
    for p in state.available:
        if p.position is None:
            continue
        par = values.get(p.id, 1)
        reasons = [f"par ${par}"]

        premium = 0
        need = needs.get(p.position)
        if need and need.remaining > 0:
            premium = round(par * 0.25 * need_weight) + 1
            reasons.append(f"fills a starting {p.position.value} need")
        elif flex_open > 0 and p.position in FLEX_ELIGIBLE:
            premium = round(par * 0.1 * need_weight)
            reasons.append("fills a flex slot")
        elif p.position in needs and need and need.remaining == 0:
            reasons.append(f"depth at {p.position.value}")

        target_bid = par + premium
        capped = min(target_bid, my_max)
        if capped < target_bid:
            reasons.append(f"capped by your budget (max ${my_max})")
        reasons.append(f"bid up to ${capped}")

        # Score targets by par value, nudged by need, so the list is "who matters."
        score = par + premium * 2
        recs.append(
            Recommendation(
                player=p,
                vor=None,
                score=float(round(score, 1)),
                reasons=reasons,
                auction_value=par,
                suggested_max_bid=capped,
            )
        )

    recs.sort(key=lambda r: r.score, reverse=True)
    # Annotate the top rec with overall budget context.
    if recs:
        recs[0].reasons.append(
            f"you have ${my_remaining} for {open_slots} slots (max bid ${my_max})"
        )
    return recs[:limit]


def _scarcity_by_position(available: list[Player]) -> dict[Position, float]:
    """Estimate the projected-point drop between the next player and the one a
    round-ish later at each position. Bigger drop = more urgent to draft now."""
    by_pos: dict[Position, list[Player]] = defaultdict(list)
    for p in available:
        if p.position and p.projected_points is not None:
            by_pos[p.position].append(p)
    out: dict[Position, float] = {}
    for pos, players in by_pos.items():
        players.sort(key=_sort_key)
        if len(players) >= 2:
            top = players[0].projected_points or 0
            later = players[min(11, len(players) - 1)].projected_points or 0
            out[pos] = max(top - later, 0.0)
    return out


def _rank_value(p: Player) -> float:
    """A projection-free value proxy from overall rank or ADP.

    Higher is better. We invert rank/ADP onto a rough points-like scale so the
    numbers sit in the same ballpark as VOR bonuses elsewhere in the scorer.
    """
    signal = p.overall_rank if p.overall_rank is not None else p.adp
    if signal is None:
        return 0.0
    # Rank 1 -> ~100, rank 200 -> ~0, gently curved so the top of the board
    # separates more than the deep bench.
    return max(100.0 - (signal ** 0.85), 0.0)


def _scarcity_by_rank(available: list[Player]) -> dict[Position, float]:
    """Projection-free scarcity: how many quality players remain at a position.

    Fewer remaining near the top of the board => scarcer => higher urgency.
    """
    by_pos: dict[Position, list[Player]] = defaultdict(list)
    for p in available:
        if p.position:
            by_pos[p.position].append(p)
    out: dict[Position, float] = {}
    for pos, players in by_pos.items():
        players.sort(key=lambda x: (x.overall_rank or x.adp or 1e9))
        # Count how many sit in the "startable near-term" window (top ~24 overall).
        near = sum(1 for p in players[:40] if (p.overall_rank or p.adp or 999) <= 80)
        out[pos] = max(24 - near, 0) * 0.8
    return out


def tiers(players: list[Player], drop_threshold: float = 12.0) -> list[list[Player]]:
    """Group players into tiers by finding gaps in projected points.

    A new tier starts whenever projected points drop by more than
    ``drop_threshold`` from the previous player.
    """
    projected = [p for p in players if p.projected_points is not None]
    if projected:
        ranked = sorted(projected, key=_sort_key)
        result: list[list[Player]] = [[ranked[0]]]
        for prev, cur in zip(ranked, ranked[1:]):
            gap = (prev.projected_points or 0) - (cur.projected_points or 0)
            if gap > drop_threshold:
                result.append([])
            result[-1].append(cur)
        return result

    # Projection-free fallback: tier by gaps in overall rank / ADP.
    ranked = sorted(
        [p for p in players if (p.overall_rank or p.adp) is not None],
        key=lambda p: (p.overall_rank or p.adp),
    )
    if not ranked:
        return []
    rank_gap = 6  # a 6+ spot jump in rank starts a new tier
    result = [[ranked[0]]]
    for prev, cur in zip(ranked, ranked[1:]):
        pv = prev.overall_rank or prev.adp
        cv = cur.overall_rank or cur.adp
        if (cv - pv) > rank_gap:
            result.append([])
        result[-1].append(cur)
    return result
