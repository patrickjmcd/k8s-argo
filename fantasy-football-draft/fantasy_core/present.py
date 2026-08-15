"""Human-readable formatting of draft state and advice.

Shared by both MCP servers so Yahoo and ESPN output read identically. MCP tools
return text to Claude, so these produce compact, scannable strings.
"""

from __future__ import annotations

from .advisor import Recommendation, roster_needs, tiers
from .models import DraftState, Player


def format_player(p: Player) -> str:
    bits = [p.name]
    if p.position:
        bits.append(p.position.value)
    if p.nfl_team:
        bits.append(p.nfl_team)
    tail = []
    if p.projected_points is not None:
        tail.append(f"proj {p.projected_points:.0f}")
    if p.adp is not None:
        tail.append(f"ADP {p.adp:.0f}")
    if p.percent_rostered is not None:
        tail.append(f"{p.percent_rostered:.0f}% rost")
    if p.injury_status:
        tail.append(p.injury_status)
    line = " ".join(bits)
    if tail:
        line += "  (" + ", ".join(tail) + ")"
    return line


def format_recommendations(recs: list[Recommendation]) -> str:
    if not recs:
        return "No available players to recommend (draft pool empty or not yet loaded)."
    lines = []
    for i, r in enumerate(recs, 1):
        head = f"{i}. {format_player(r.player)}"
        lines.append(head)
        if r.reasons:
            lines.append("     " + "; ".join(r.reasons))
    return "\n".join(lines)


def format_draft_summary(state: DraftState) -> str:
    lines = []
    lines.append(f"Pick #{state.next_overall_pick} of the draft.")
    if state.num_teams:
        wait = state.picks_until_my_turn()
        if wait == 0:
            lines.append("You are ON THE CLOCK.")
        elif wait is not None:
            lines.append(f"{wait} pick(s) until your turn.")
    roster = state.my_roster
    if roster:
        lines.append("\nYour roster:")
        for p in roster:
            lines.append("  - " + format_player(p))
    needs = [n for n in roster_needs(roster, state.settings) if n.remaining > 0]
    if needs:
        lines.append(
            "\nStarting needs: "
            + ", ".join(f"{n.position.value} x{n.remaining}" for n in needs)
        )
    return "\n".join(lines)


def format_tiers(players: list[Player], limit_per_tier: int = 8) -> str:
    grouped = tiers(players)
    if not grouped:
        return "No projection data available to build tiers."
    lines = []
    for i, tier in enumerate(grouped[:10], 1):
        lines.append(f"Tier {i}:")
        for p in tier[:limit_per_tier]:
            lines.append("  - " + format_player(p))
    return "\n".join(lines)
