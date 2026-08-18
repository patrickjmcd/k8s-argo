"""Yahoo fantasy-football draft MCP server (stdio).

Exposes live-draft tools backed by the official Yahoo Fantasy Sports API. Run
locally next to Claude. Requires a one-time OAuth bootstrap:
``python -m mcp_yahoo.auth``.
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

mcp = FastMCP("yahoo-fantasy-draft")


@mcp.tool()
def draft_status() -> str:
    """Current state of the Yahoo draft: pick number, your turn, your roster, and
    remaining starting-lineup needs."""
    state = client.get_draft_state()
    return format_draft_summary(state)


@mcp.tool()
def recommend_pick(strategy: str = "balanced", limit: int = 8) -> str:
    """Recommend who to draft right now from your Yahoo league.

    strategy: "bpa" (best player available), "need" (weight roster holes), or
    "balanced" (default). Yahoo does not expose season projections, so
    recommendations are driven by preseason rank, ADP, positional scarcity, and
    your current roster.
    """
    state = client.get_draft_state()
    recs = recommend(state, strategy=strategy, limit=limit)
    return format_draft_summary(state) + "\n\nRecommended picks:\n" + format_recommendations(recs)


@mcp.tool()
def best_available(position: str = "", limit: int = 15) -> str:
    """List the best available (undrafted) players, optionally filtered by
    position (QB/RB/WR/TE/K/DST), ordered by preseason rank / ADP."""
    players = client.get_available_players(position=position or None, limit=limit)
    if not players:
        return "No available players found."
    return "\n".join(f"{i}. {format_player(p)}" for i, p in enumerate(players, 1))


@mcp.tool()
def position_tiers(position: str, limit: int = 60) -> str:
    """Show tier breaks for available players at a position (by rank/ADP gaps),
    so you can see when a talent cliff is coming."""
    players = client.get_available_players(position=position, limit=limit)
    return format_tiers(players)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
