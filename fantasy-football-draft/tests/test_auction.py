"""Tests for auction values, budget math, and auction recommendations."""

from fantasy_core import (
    DraftFormat,
    DraftPick,
    DraftState,
    Player,
    Position,
    RosterSettings,
    auction_values,
    recommend,
)


def mk(name, pos, proj):
    return Player(id=name, name=name, position=Position.normalize(pos), projected_points=proj)


def big_pool():
    players = []
    for i in range(1, 40):
        players.append(mk(f"RB{i}", "RB", 300 - i * 6))
    for i in range(1, 40):
        players.append(mk(f"WR{i}", "WR", 290 - i * 5))
    for i in range(1, 16):
        players.append(mk(f"QB{i}", "QB", 380 - i * 8))
    for i in range(1, 16):
        players.append(mk(f"TE{i}", "TE", 200 - i * 8))
    for i in range(1, 12):
        players.append(mk(f"K{i}", "K", 150 - i * 3))
    for i in range(1, 12):
        players.append(mk(f"DST{i}", "DST", 140 - i * 3))
    return players


def test_auction_values_basic_properties():
    settings = RosterSettings.standard()
    values = auction_values(big_pool(), settings, num_teams=10, budget=200)
    assert values, "should produce dollar values"
    # Every value is at least $1.
    assert all(v >= 1 for v in values.values())
    # The best RB should be worth more than a mid RB.
    assert values.get("RB1", 1) > values.get("RB20", 1)
    # Total assigned value should not exceed the league's total money.
    assert sum(values.values()) <= 10 * 200


def test_max_bid_reserves_a_dollar_per_slot():
    settings = RosterSettings.standard()  # 8 starters + 6 bench = 14 roster spots
    state = DraftState(
        settings=settings,
        draft_format=DraftFormat.AUCTION,
        budget=200,
        my_team_id="1",
        num_teams=10,
    )
    # Fresh team: 14 open slots, $200. Max bid keeps $1 for the other 13.
    assert state.roster_size == settings.total_starters() + settings.bench
    assert state.max_bid("1") == 200 - (state.roster_size - 1)

    # After spending $50 on one player, budget and open slots both drop.
    state.picks.append(
        DraftPick(1, 1, "1", Player(id="x", name="x", position=Position.RB), cost=50)
    )
    assert state.budget_remaining("1") == 150
    assert state.open_roster_slots("1") == state.roster_size - 1
    assert state.max_bid("1") == 150 - (state.open_roster_slots("1") - 1)


def test_auction_recommend_returns_bids_within_budget():
    settings = RosterSettings.standard()
    state = DraftState(
        settings=settings,
        available=big_pool(),
        draft_format=DraftFormat.AUCTION,
        budget=200,
        my_team_id="1",
        num_teams=10,
    )
    recs = recommend(state, strategy="balanced", limit=5)
    assert recs
    my_max = state.max_bid("1")
    for r in recs:
        assert r.auction_value is not None and r.auction_value >= 1
        assert r.suggested_max_bid is not None
        # Never suggest bidding more than you can actually afford.
        assert r.suggested_max_bid <= my_max
    # Top target should be an elite player carrying real value.
    assert recs[0].auction_value >= 2


def test_auction_recommend_respects_small_budget():
    settings = RosterSettings.standard()
    state = DraftState(
        settings=settings,
        available=big_pool(),
        draft_format=DraftFormat.AUCTION,
        budget=200,
        my_team_id="1",
        num_teams=10,
    )
    # Blow most of the budget so only a few dollars remain.
    for i in range(10):
        state.picks.append(
            DraftPick(i + 1, 1, "1", Player(id=f"spent{i}", name=f"s{i}", position=Position.RB), cost=19)
        )
    my_max = state.max_bid("1")
    recs = recommend(state, strategy="balanced", limit=3)
    assert all(r.suggested_max_bid <= my_max for r in recs)
