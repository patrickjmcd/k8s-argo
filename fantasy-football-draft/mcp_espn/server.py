"""ESPN fantasy-football draft MCP server (stdio).

Exposes live-draft tools backed by the ESPN Fantasy API. Run locally next to
Claude; requires ESPN_LEAGUE_ID, ESPN_SEASON, and (for private leagues)
ESPN_S2 + ESPN_SWID cookies in the environment.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from fantasy_core import recommend
from fantasy_core.present import (
    format_draft_summary,
    format_player,
    format_recommendations,
    format_tiers,
)

from . import client

mcp = FastMCP("espn-fantasy-draft")


@mcp.tool()
def draft_status() -> str:
    """Current state of the ESPN draft: pick number, your turn, your roster, and
    remaining starting-lineup needs."""
    state = client.get_draft_state()
    return format_draft_summary(state)


@mcp.tool()
def recommend_pick(strategy: str = "balanced", limit: int = 8) -> str:
    """Recommend who to draft right now from your ESPN league.

    strategy: "bpa" (best player available), "need" (weight roster holes), or
    "balanced" (default). Considers value-over-replacement, positional scarcity,
    and your current roster.
    """
    state = client.get_draft_state()
    recs = recommend(state, strategy=strategy, limit=limit)
    return format_draft_summary(state) + "\n\nRecommended picks:\n" + format_recommendations(recs)


@mcp.tool()
def best_available(position: str = "", limit: int = 15) -> str:
    """List the best available (undrafted) players, optionally filtered by
    position (QB/RB/WR/TE/K/DST)."""
    players = client.get_available_players(position=position or None, limit=limit)
    if not players:
        return "No available players found."
    return "\n".join(f"{i}. {format_player(p)}" for i, p in enumerate(players, 1))


@mcp.tool()
def position_tiers(position: str, limit: int = 60) -> str:
    """Show tier breaks (by projected points) for available players at a
    position, so you can see when a talent cliff is coming."""
    players = client.get_available_players(position=position, limit=limit)
    return format_tiers(players)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
