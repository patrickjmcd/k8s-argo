"""Mock-draft simulator MCP server (stdio).

A local rehearsal tool: run a full snake or auction draft against bots and get
recommendations from the same brain the live Yahoo/ESPN servers use. No
credentials or network needed. The draft is held in memory for the life of the
server process, so a session's picks accumulate across tool calls.

Start here to test the whole thing before draft day:
    mock-draft-mcp   (or: python -m mcp_mock.server)
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from fantasy_core import recommend
from fantasy_core.mock import MockDraft
from fantasy_core.models import DraftFormat
from fantasy_core.present import (
    format_draft_summary,
    format_player,
    format_recommendations,
    format_tiers,
)

mcp = FastMCP("mock-fantasy-draft")

# The single in-progress mock draft for this server process.
_draft: MockDraft | None = None


def _require() -> MockDraft:
    if _draft is None:
        raise RuntimeError("No mock draft started. Call start_mock_draft first.")
    return _draft


@mcp.tool()
def start_mock_draft(
    teams: int = 10,
    my_slot: int = 3,
    draft_format: str = "snake",
    budget: int = 200,
) -> str:
    """Start (or restart) a mock draft.

    teams: number of teams (default 10).
    my_slot: your draft position, 1..teams (snake). Also used as your team id.
    draft_format: "snake" or "auction".
    budget: per-team auction budget (auction only, default $200).
    """
    global _draft
    fmt = DraftFormat.AUCTION if draft_format.lower().startswith("auc") else DraftFormat.SNAKE
    my_slot = max(1, min(my_slot, teams))
    _draft = MockDraft(
        num_teams=teams,
        my_slot=my_slot,
        draft_format=fmt,
        budget=budget,
    )
    header = (
        f"Started a {teams}-team {fmt.value} mock draft. You are team {my_slot}."
    )
    if fmt == DraftFormat.SNAKE:
        # Advance bots so you're on the clock at your first pick.
        log = _draft.advance_bots_snake()
        body = ("\n".join(log) + "\n\n") if log else ""
        return header + "\n" + body + format_draft_summary(_draft.state())
    return (
        header
        + f"\nAuction: each team has ${budget}. Use buy_player to win players and "
        "sim_opponents to let bots spend. recommend_pick shows targets and max bids.\n\n"
        + format_draft_summary(_draft.state())
    )


@mcp.tool()
def draft_status() -> str:
    """Current mock-draft state: your turn/budget, roster, and remaining needs."""
    return format_draft_summary(_require().state())


@mcp.tool()
def recommend_pick(strategy: str = "balanced", limit: int = 8) -> str:
    """Recommend who to draft/target right now (snake) or nominate with a max bid
    (auction). Strategies: bpa, need, balanced."""
    draft = _require()
    state = draft.state()
    recs = recommend(state, strategy=strategy, limit=limit)
    label = "Targets" if state.is_auction else "Recommended picks"
    return format_draft_summary(state) + f"\n\n{label}:\n" + format_recommendations(recs)


@mcp.tool()
def draft_player(name: str) -> str:
    """(Snake) Draft an available player by name for your team. Bots then pick up
    to your next turn."""
    draft = _require()
    if draft.draft_format != DraftFormat.SNAKE:
        return "This is an auction draft — use buy_player instead."
    result = draft.user_pick_snake(name)
    return result + "\n\n" + format_draft_summary(draft.state())


@mcp.tool()
def buy_player(name: str, price: int) -> str:
    """(Auction) Win an available player for your team at the given price, then let
    opponents buy a wave of players."""
    draft = _require()
    if draft.draft_format != DraftFormat.AUCTION:
        return "This is a snake draft — use draft_player instead."
    result = draft.user_buy_auction(name, price)
    return result + "\n\n" + format_draft_summary(draft.state())


@mcp.tool()
def sim_opponents(count: int = 5) -> str:
    """Let bot opponents make moves: advance to your next pick (snake) or run
    ``count`` opponent purchases (auction)."""
    draft = _require()
    if draft.draft_format == DraftFormat.SNAKE:
        log = draft.advance_bots_snake()
        body = "\n".join(log) if log else "It's already your turn."
    else:
        body = draft.sim_opponents_auction(count)
    return body + "\n\n" + format_draft_summary(draft.state())


@mcp.tool()
def best_available(position: str = "", limit: int = 15) -> str:
    """List the best available players, optionally filtered by position."""
    draft = _require()
    from fantasy_core.models import Position

    players = draft.available
    if position:
        want = Position.normalize(position)
        players = [p for p in players if p.position == want]
    players = sorted(players, key=lambda p: (p.overall_rank or p.adp or 1e9))[:limit]
    if not players:
        return "No available players found."
    return "\n".join(f"{i}. {format_player(p)}" for i, p in enumerate(players, 1))


@mcp.tool()
def position_tiers(position: str, limit: int = 60) -> str:
    """Show tier breaks for available players at a position."""
    draft = _require()
    from fantasy_core.models import Position

    want = Position.normalize(position)
    players = [p for p in draft.available if p.position == want][:limit]
    return format_tiers(players)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
