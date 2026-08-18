"""Tests for the local mock-draft simulator (snake + auction)."""

from fantasy_core.mock import MockDraft, generate_pool
from fantasy_core.models import DraftFormat


def test_pool_is_reasonable():
    pool = generate_pool()
    assert len(pool) > 150
    # Every player has a projection, a position, and a derived ADP/rank.
    assert all(p.projected_points is not None for p in pool)
    assert all(p.position is not None for p in pool)
    assert all(p.adp is not None and p.overall_rank is not None for p in pool)
    # ADP should be unique per player (a proper ordering).
    adps = [p.adp for p in pool]
    assert len(set(adps)) == len(adps)


def test_snake_draft_flow():
    d = MockDraft(num_teams=10, my_slot=3, draft_format=DraftFormat.SNAKE)
    # Advancing bots should bring the draft to the user's first pick.
    d.advance_bots_snake()
    assert d._team_on_clock() == d.my_team_id
    before = len(d.available)

    top = d.available[0].name
    msg = d.user_pick_snake(top)
    assert "You drafted" in msg
    # One player left our board for us plus however many bots took.
    assert len(d.available) < before
    # We own exactly one player and it's the one we picked.
    assert len(d.state().my_roster) == 1
    assert d.state().my_roster[0].name == top


def test_snake_rejects_out_of_turn_and_bad_name():
    d = MockDraft(num_teams=10, my_slot=1, draft_format=DraftFormat.SNAKE)
    # Slot 1 is on the clock immediately (no bots ahead).
    assert d._team_on_clock() == "1"
    assert "No available player" in d.user_pick_snake("Nonexistent Player 999")


def test_auction_flow_tracks_budget():
    d = MockDraft(num_teams=10, my_slot=2, draft_format=DraftFormat.AUCTION, budget=200)
    state = d.state()
    assert state.is_auction
    start_remaining = state.budget_remaining(d.my_team_id)
    assert start_remaining == 200

    target = d.available[0].name
    msg = d.user_buy_auction(target, 45)
    assert "You won" in msg
    after = d.state()
    assert after.budget_remaining(d.my_team_id) == 155
    assert after.my_roster and after.my_roster[0].name == target
    # Opponents should have spent too (their budgets dropped below 200).
    other_spend = sum(
        200 - after.budget_remaining(t) for t in d.team_ids if t != d.my_team_id
    )
    assert other_spend > 0


def test_auction_rejects_overbid():
    d = MockDraft(num_teams=10, my_slot=1, draft_format=DraftFormat.AUCTION, budget=200)
    over = d.state().max_bid("1") + 50
    msg = d.user_buy_auction(d.available[0].name, over)
    assert "max bid" in msg.lower()
    # Nothing was purchased.
    assert len(d.state().my_roster) == 0


def test_bots_fill_rosters_without_going_negative():
    d = MockDraft(num_teams=8, my_slot=4, draft_format=DraftFormat.AUCTION, budget=200)
    # Run a lot of opponent buys; budgets must never go negative.
    d.sim_opponents_auction(80)
    state = d.state()
    for t in d.team_ids:
        assert state.budget_remaining(t) >= 0
