"""Tests for the platform-agnostic draft brain.

These run without any Yahoo/ESPN credentials — they exercise the shared logic
that both MCP servers depend on.
"""

from fantasy_core import (
    DraftPick,
    DraftState,
    Player,
    Position,
    RosterSettings,
    recommend,
    roster_needs,
    tiers,
)


def mk(name, pos, proj=None, adp=None, rank=None):
    return Player(
        id=name,
        name=name,
        position=Position.normalize(pos),
        projected_points=proj,
        adp=adp,
        overall_rank=rank,
    )


def projected_pool():
    # A believable pool with projections at each position.
    players = []
    for i in range(1, 13):
        players.append(mk(f"RB{i}", "RB", proj=260 - i * 12, adp=i * 2))
    for i in range(1, 13):
        players.append(mk(f"WR{i}", "WR", proj=250 - i * 10, adp=i * 2 + 1))
    for i in range(1, 6):
        players.append(mk(f"QB{i}", "QB", proj=340 - i * 8, adp=i * 20))
    for i in range(1, 5):
        players.append(mk(f"TE{i}", "TE", proj=180 - i * 20, adp=i * 25))
    return players


def test_position_normalization():
    assert Position.normalize("D/ST") is Position.DST
    assert Position.normalize("def") is Position.DST
    assert Position.normalize("pk") is Position.K
    assert Position.normalize("wr") is Position.WR
    assert Position.normalize("garbage") is None


def test_roster_needs_start_empty():
    settings = RosterSettings.standard()
    needs = {n.position: n for n in roster_needs([], settings)}
    assert needs[Position.RB].remaining == 2
    assert needs[Position.QB].remaining == 1


def test_roster_needs_fill():
    settings = RosterSettings.standard()
    roster = [mk("A", "RB"), mk("B", "RB")]
    needs = {n.position: n for n in roster_needs(roster, settings)}
    assert needs[Position.RB].remaining == 0
    assert needs[Position.WR].remaining == 2


def test_recommend_with_projections_prefers_value():
    state = DraftState(
        settings=RosterSettings.standard(),
        available=projected_pool(),
        num_teams=10,
        my_team_id="1",
    )
    recs = recommend(state, strategy="balanced", limit=5)
    assert recs, "should produce recommendations"
    # The top rec should carry a value-over-replacement reason.
    assert any("replacement" in r.lower() for r in recs[0].reasons)
    # Scores should be sorted descending.
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_recommend_need_strategy_shifts_toward_holes():
    pool = projected_pool()
    # Already have two elite RBs; a "need" strategy should stop piling on RB.
    roster_picks = [
        DraftPick(1, 1, "1", mk("RB1", "RB", proj=248)),
        DraftPick(2, 1, "1", mk("RB2", "RB", proj=236)),
    ]
    state = DraftState(
        settings=RosterSettings.standard(),
        picks=roster_picks,
        available=[p for p in pool if p.name not in {"RB1", "RB2"}],
        num_teams=10,
        my_team_id="1",
    )
    need_recs = recommend(state, strategy="need", limit=3)
    top_positions = {r.player.position for r in need_recs}
    # With RB starters filled, the need strategy should surface non-RB starters.
    assert top_positions != {Position.RB}


def test_recommend_without_projections_uses_rank():
    # Yahoo-style pool: no projections, only ranks/ADP.
    pool = [mk(f"P{i}", "RB" if i % 2 else "WR", rank=i, adp=i) for i in range(1, 30)]
    state = DraftState(
        settings=RosterSettings.standard(),
        available=pool,
        num_teams=10,
        my_team_id="1",
    )
    recs = recommend(state, strategy="balanced", limit=5)
    assert recs
    # Best rank should come out on or near top and cite its rank.
    assert any("rank" in r.lower() for r in recs[0].reasons)


def test_tiers_with_projections_breaks_on_gaps():
    players = [
        mk("A", "RB", proj=300),
        mk("B", "RB", proj=298),
        mk("C", "RB", proj=260),  # big gap -> new tier
        mk("D", "RB", proj=258),
    ]
    grouped = tiers(players, drop_threshold=12)
    assert len(grouped) == 2
    assert [p.name for p in grouped[0]] == ["A", "B"]


def test_tiers_without_projections_falls_back_to_rank():
    players = [mk(f"P{i}", "WR", rank=i) for i in [1, 2, 3, 20, 21]]
    grouped = tiers(players)
    assert len(grouped) == 2
    assert {p.name for p in grouped[0]} == {"P1", "P2", "P3"}


def test_snake_picks_until_my_turn():
    # 10-team snake, I'm slot 3. After 2 picks made, I'm on the clock next.
    state = DraftState(
        settings=RosterSettings.standard(),
        picks=[
            DraftPick(1, 1, "1", mk("x", "RB")),
            DraftPick(2, 1, "2", mk("y", "RB")),
        ],
        num_teams=10,
        my_draft_slot=3,
    )
    assert state.next_overall_pick == 3
    assert state.picks_until_my_turn() == 0

    # Slot 1 in round 1: after pick 3 done, the snake turns at picks 20/21.
    state2 = DraftState(
        settings=RosterSettings.standard(),
        picks=[DraftPick(i, 1, str(i), mk(f"p{i}", "RB")) for i in range(1, 4)],
        num_teams=10,
        my_draft_slot=1,
    )
    # next pick is 4; slot 1 picks again at overall 20 (end of round 2 snake).
    assert state2.picks_until_my_turn() == 16
