"""Platform-agnostic data models.

Both the Yahoo and ESPN MCP servers normalize their platform-specific payloads
into these types, so all the draft-intelligence logic in this package can be
written once and reused by both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Position(str, Enum):
    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    K = "K"
    DST = "DST"

    @classmethod
    def normalize(cls, raw: str) -> "Position | None":
        """Map the many strings platforms use onto our canonical positions."""
        if not raw:
            return None
        key = raw.strip().upper()
        aliases = {
            "D/ST": cls.DST,
            "DEF": cls.DST,
            "DST": cls.DST,
            "PK": cls.K,
            "K": cls.K,
        }
        if key in aliases:
            return aliases[key]
        try:
            return cls(key)
        except ValueError:
            return None


# Roster slots that can be filled by more than one position.
FLEX_ELIGIBLE = {Position.RB, Position.WR, Position.TE}
SUPERFLEX_ELIGIBLE = {Position.QB, Position.RB, Position.WR, Position.TE}


@dataclass
class Player:
    """A single NFL player, normalized across platforms."""

    id: str
    name: str
    position: Position | None
    nfl_team: str = ""
    # Average draft position across the platform's drafts. Lower = drafted earlier.
    adp: float | None = None
    # Platform's projected season fantasy points (league scoring where available).
    projected_points: float | None = None
    # Percent of leagues where the player is rostered (0-100).
    percent_rostered: float | None = None
    # Bye week, when the platform provides it.
    bye_week: int | None = None
    # Platform's own overall rank for the player (1 = best), if provided.
    overall_rank: int | None = None
    # Injury designation string straight from the platform (e.g. "Q", "IR").
    injury_status: str = ""

    @property
    def is_available(self) -> bool:  # overridden by DraftState bookkeeping
        return True


@dataclass
class RosterSettings:
    """Starting-lineup requirements for the league.

    ``slots`` maps a position to the number of dedicated starting slots. ``flex``
    and ``superflex`` are extra slots fillable by multiple positions. ``bench`` is
    informational (affects how deep it's worth reaching).
    """

    slots: dict[Position, int] = field(
        default_factory=lambda: {
            Position.QB: 1,
            Position.RB: 2,
            Position.WR: 2,
            Position.TE: 1,
            Position.K: 1,
            Position.DST: 1,
        }
    )
    flex: int = 1
    superflex: int = 0
    bench: int = 6

    @classmethod
    def standard(cls) -> "RosterSettings":
        return cls()

    def total_starters(self) -> int:
        return sum(self.slots.values()) + self.flex + self.superflex

    def starters_at(self, position: Position) -> int:
        return self.slots.get(position, 0)


@dataclass
class DraftPick:
    overall: int
    round: int
    team_id: str
    player: Player


@dataclass
class DraftState:
    """A snapshot of the draft at a point in time."""

    settings: RosterSettings
    picks: list[DraftPick] = field(default_factory=list)
    # Players not yet drafted, best-guess ordered by the platform.
    available: list[Player] = field(default_factory=list)
    # The requesting user's team id, so we can compute their roster.
    my_team_id: str = ""
    # The user's draft slot (1..num_teams). Needed for snake turn math because a
    # team id is not the same as its position in the draft order.
    my_draft_slot: int | None = None
    # Total number of teams, for snake-draft turn math.
    num_teams: int = 0

    @property
    def my_roster(self) -> list[Player]:
        return [p.player for p in self.picks if p.team_id == self.my_team_id]

    @property
    def next_overall_pick(self) -> int:
        return len(self.picks) + 1

    def picks_until_my_turn(self) -> int | None:
        """How many picks until the requesting user is on the clock (snake)."""
        if not self.num_teams or not self.my_draft_slot:
            return None
        my_slot = self.my_draft_slot
        # Walk forward from the next pick until it lands on my slot in the snake.
        upcoming = self.next_overall_pick
        for offset in range(self.num_teams * 2 + 1):
            overall = upcoming + offset
            rnd = (overall - 1) // self.num_teams  # 0-indexed round
            pos_in_round = (overall - 1) % self.num_teams
            slot = pos_in_round if rnd % 2 == 0 else self.num_teams - 1 - pos_in_round
            if slot + 1 == my_slot:
                return offset
        return None
