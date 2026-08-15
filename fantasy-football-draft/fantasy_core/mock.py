"""Local mock-draft simulator.

ESPN and Yahoo mock-draft *rooms* are ephemeral practice lobbies with no
API-queryable league id, so you can't point the platform servers at one. This
module instead simulates a full draft locally against bot opponents, driving the
exact same ``fantasy_core`` brain the live servers use — so rehearsing here also
exercises the real recommendation logic. Supports snake and auction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .advisor import auction_values, recommend
from .models import (
    DraftFormat,
    DraftPick,
    DraftState,
    Player,
    Position,
    RosterSettings,
)

# Rough NFL team rotation just to give players a team label.
_NFL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LV", "LAC", "LAR", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SF", "SEA", "TB",
    "TEN", "WAS",
]

# (position, count, top_projection, bottom_projection) — a believable shape.
_POOL_SHAPE = [
    (Position.RB, 60, 320.0, 70.0),
    (Position.WR, 72, 300.0, 60.0),
    (Position.QB, 32, 390.0, 210.0),
    (Position.TE, 30, 210.0, 55.0),
    (Position.K, 20, 155.0, 105.0),
    (Position.DST, 20, 150.0, 85.0),
]


def generate_pool() -> list[Player]:
    """A deterministic, realistically-shaped synthetic player pool.

    Names are synthetic (``RB1``, ``WR2``, ...) on purpose — they carry no real
    2026 projections and are meant for rehearsing mechanics and decisions, not as
    a cheat sheet. Projections descend within each position; ADP is derived later
    from value-based ranking so elite RB/WR go early and QB/TE slide, like real
    drafts.
    """
    players: list[Player] = []
    t = 0
    for pos, count, top, bottom in _POOL_SHAPE:
        for i in range(1, count + 1):
            frac = (i - 1) / max(count - 1, 1)
            proj = round(top - (top - bottom) * frac, 1)
            players.append(
                Player(
                    id=f"{pos.value}{i}",
                    name=f"{pos.value}{i}",
                    position=pos,
                    nfl_team=_NFL_TEAMS[t % len(_NFL_TEAMS)],
                    projected_points=proj,
                    bye_week=4 + (t % 11),
                )
            )
            t += 1

    _assign_adp(players)
    return players


def _assign_adp(players: list[Player]) -> None:
    """Derive ADP/overall rank from value-based ranking across the whole pool."""
    settings = RosterSettings.standard()
    from .advisor import replacement_levels, value_over_replacement

    levels = replacement_levels(players, settings, num_teams=10)
    ranked = sorted(
        players,
        key=lambda p: -(value_over_replacement(p, levels) or -1e9),
    )
    for i, p in enumerate(ranked, 1):
        p.overall_rank = i
        p.adp = float(i)


@dataclass
class MockDraft:
    """A running mock draft. Held in memory by the mock MCP server across calls."""

    num_teams: int = 10
    my_slot: int = 3
    draft_format: DraftFormat = DraftFormat.SNAKE
    budget: int = 200
    settings: RosterSettings = field(default_factory=RosterSettings.standard)
    pool: list[Player] = field(default_factory=generate_pool)
    picks: list[DraftPick] = field(default_factory=list)

    @property
    def my_team_id(self) -> str:
        return str(self.my_slot)

    @property
    def team_ids(self) -> list[str]:
        return [str(i) for i in range(1, self.num_teams + 1)]

    def _drafted_ids(self) -> set[str]:
        return {pk.player.id for pk in self.picks}

    @property
    def available(self) -> list[Player]:
        drafted = self._drafted_ids()
        return [p for p in self.pool if p.id not in drafted]

    def state(self) -> DraftState:
        return DraftState(
            settings=self.settings,
            picks=list(self.picks),
            available=self.available,
            my_team_id=self.my_team_id,
            my_draft_slot=self.my_slot,
            num_teams=self.num_teams,
            draft_format=self.draft_format,
            budget=self.budget,
        )

    def _state_for(self, team_id: str) -> DraftState:
        s = self.state()
        s.my_team_id = team_id
        s.my_draft_slot = int(team_id)
        return s

    # -- lookup ---------------------------------------------------------------

    def find_available(self, name: str) -> tuple[Player | None, list[Player]]:
        """Return (exact/unique match, list of candidates) for a fuzzy name."""
        q = name.strip().lower()
        avail = self.available
        exact = [p for p in avail if p.name.lower() == q]
        if exact:
            return exact[0], exact
        partial = [p for p in avail if q in p.name.lower()]
        if len(partial) == 1:
            return partial[0], partial
        return None, partial

    def drafted_by_name(self, name: str) -> DraftPick | None:
        """An exact-name match among already-drafted players, if any."""
        q = name.strip().lower()
        for pk in self.picks:
            if pk.player.name.lower() == q:
                return pk
        return None

    def _resolve_or_message(self, name: str) -> tuple[Player | None, str]:
        """Shared name resolution used by both snake and auction paths."""
        player, candidates = self.find_available(name)
        if player is not None:
            return player, ""
        taken = self.drafted_by_name(name)
        if taken is not None:
            cost = f" for ${taken.cost}" if taken.cost is not None else ""
            return None, f"{taken.player.name} is already drafted (Team {taken.team_id}{cost})."
        if not candidates:
            return None, f"No available player matches '{name}'."
        return None, "Ambiguous — did you mean: " + ", ".join(p.name for p in candidates[:8]) + "?"

    # -- snake ----------------------------------------------------------------

    def _team_on_clock(self) -> str:
        overall = len(self.picks) + 1
        rnd = (overall - 1) // self.num_teams
        pos_in_round = (overall - 1) % self.num_teams
        slot = pos_in_round if rnd % 2 == 0 else self.num_teams - 1 - pos_in_round
        return str(slot + 1)

    def is_complete(self) -> bool:
        return len(self.picks) >= self.num_teams * self.settings_rounds()

    def settings_rounds(self) -> int:
        return self.settings.total_starters() + self.settings.bench

    def _record(self, team_id: str, player: Player, cost: int | None = None) -> None:
        overall = len(self.picks) + 1
        self.picks.append(
            DraftPick(
                overall=overall,
                round=(overall - 1) // self.num_teams + 1,
                team_id=team_id,
                player=player,
                cost=cost,
            )
        )

    def _bot_choice(self, team_id: str) -> Player | None:
        recs = recommend(self._state_for(team_id), strategy="balanced", limit=1)
        if recs:
            return recs[0].player
        avail = self.available
        return avail[0] if avail else None

    def advance_bots_snake(self) -> list[str]:
        """Run bot picks until it's the user's turn (or the draft ends)."""
        log: list[str] = []
        while not self.is_complete():
            team = self._team_on_clock()
            if team == self.my_team_id:
                break
            player = self._bot_choice(team)
            if player is None:
                break
            self._record(team, player)
            log.append(f"Team {team} drafted {player.name} ({_pos(player)})")
        return log

    def user_pick_snake(self, name: str) -> str:
        if self.is_complete():
            return "The draft is complete."
        if self._team_on_clock() != self.my_team_id:
            return f"It's not your pick yet — Team {self._team_on_clock()} is on the clock. Run advance first."
        player, message = self._resolve_or_message(name)
        if player is None:
            return message
        self._record(self.my_team_id, player)
        log = [f"You drafted {player.name} ({_pos(player)})."]
        log += self.advance_bots_snake()
        if self.is_complete():
            log.append("Draft complete.")
        return "\n".join(log)

    # -- auction --------------------------------------------------------------

    def _auction_values(self) -> dict[str, int]:
        return auction_values(self.pool, self.settings, self.num_teams, self.budget)

    def _bot_buy(self, values: dict[str, int]) -> str | None:
        """Have the neediest bot with budget buy a sensible player."""
        state = self.state()
        # Pick the bot with the most open slots and at least $1 to spend.
        bots = [
            t for t in self.team_ids
            if t != self.my_team_id and state.open_roster_slots(t) > 0
        ]
        if not bots:
            return None
        bots.sort(key=lambda t: (-state.open_roster_slots(t), -state.budget_remaining(t)))
        team = bots[0]
        recs = recommend(self._state_for(team), strategy="balanced", limit=1)
        if not recs:
            return None
        player = recs[0].player
        par = values.get(player.id, 1)
        max_bid = state.max_bid(team)
        # Bots pay roughly par, occasionally a touch under when short on cash.
        price = max(1, min(par, max_bid))
        self._record(team, player, cost=price)
        return f"Team {team} won {player.name} ({_pos(player)}) for ${price}"

    def user_buy_auction(self, name: str, price: int) -> str:
        state = self.state()
        if state.open_roster_slots(self.my_team_id) <= 0:
            return "Your roster is full."
        max_bid = state.max_bid(self.my_team_id)
        if price < 1:
            return "Bid must be at least $1."
        if price > max_bid:
            return (
                f"You can't bid ${price} — your max bid is ${max_bid} "
                f"(${state.budget_remaining(self.my_team_id)} left, "
                f"{state.open_roster_slots(self.my_team_id)} slots to fill)."
            )
        player, message = self._resolve_or_message(name)
        if player is None:
            return message
        self._record(self.my_team_id, player, cost=price)
        log = [f"You won {player.name} ({_pos(player)}) for ${price}."]
        # Let opponents buy a wave of players so the board and budgets move.
        values = self._auction_values()
        for _ in range(self.num_teams - 1):
            msg = self._bot_buy(values)
            if msg:
                log.append(msg)
        return "\n".join(log)

    def sim_opponents_auction(self, n: int) -> str:
        values = self._auction_values()
        log = []
        for _ in range(max(n, 0)):
            msg = self._bot_buy(values)
            if not msg:
                break
            log.append(msg)
        return "\n".join(log) if log else "No opponent purchases were possible."


def _pos(p: Player) -> str:
    return p.position.value if p.position else "?"
