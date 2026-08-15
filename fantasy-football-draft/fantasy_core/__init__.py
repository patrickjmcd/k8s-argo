"""Platform-agnostic fantasy-football draft intelligence.

The Yahoo and ESPN MCP servers both normalize their data into the models here
and call the shared advisor so the draft logic lives in exactly one place.
"""

from .advisor import (
    Recommendation,
    RosterNeed,
    flex_remaining,
    recommend,
    replacement_levels,
    roster_needs,
    tiers,
    value_over_replacement,
)
from .models import (
    DraftPick,
    DraftState,
    Player,
    Position,
    RosterSettings,
)

__all__ = [
    "DraftPick",
    "DraftState",
    "Player",
    "Position",
    "RosterSettings",
    "Recommendation",
    "RosterNeed",
    "flex_remaining",
    "recommend",
    "replacement_levels",
    "roster_needs",
    "tiers",
    "value_over_replacement",
]
