from pathlib import Path
import pandas as pd
import pytest
from fpl.config import Config
from fpl.pipeline import run
from fpl.report.weekly import Recommendation

BOOTSTRAP = {
    "teams": [
        {"id": t, "name": f"Team{t}", "short_name": f"T{t}",
         "strength_overall_home": 3, "strength_overall_away": 3}
        for t in range(1, 9)
    ],
    "element_types": [
        {"id": 1, "singular_name_short": "GKP"}, {"id": 2, "singular_name_short": "DEF"},
        {"id": 3, "singular_name_short": "MID"}, {"id": 4, "singular_name_short": "FWD"},
    ],
    "elements": [],
}
_pid = 1
for _t in range(1, 9):
    for _et, _n in [(1, 3), (2, 6), (3, 6), (4, 4)]:
        for _k in range(_n):
            BOOTSTRAP["elements"].append({
                "id": _pid, "web_name": f"P{_pid}", "team": _t, "element_type": _et,
                "now_cost": 45 + (_pid % 5) * 10, "status": "a", "news": "",
                "chance_of_playing_next_round": None, "minutes": 2500, "starts": 28,
                "total_points": 100 + _pid % 40, "goals_scored": _pid % 8,
                "assists": _pid % 5, "clean_sheets": 10, "goals_conceded": 30,
                "saves": 60 if _et == 1 else 0, "bonus": 10, "bps": 400,
                "yellow_cards": 2, "red_cards": 0, "own_goals": 0,
                "expected_goals": str(_pid % 8), "expected_assists": str(_pid % 5),
                "expected_goals_conceded": "30.0", "selected_by_percent": "5.0",
                "defensive_contribution": 100 + _pid % 20,
            })
            _pid += 1

FIXTURES = []
_fid = 1
for _ev in range(1, 7):
    for _h, _a in [(1, 2), (3, 4), (5, 6), (7, 8)]:
        FIXTURES.append({
            "id": _fid, "event": _ev, "team_h": _h, "team_a": _a,
            "team_h_difficulty": 3, "team_a_difficulty": 3,
            "kickoff_time": f"2026-08-2{_ev}T19:00:00Z", "finished": False,
        })
        _fid += 1


class FakeClient:
    stale = False

    def bootstrap(self):
        return BOOTSTRAP

    def fixtures(self):
        return FIXTURES

    def element_summaries(self, player_ids, ttl_hours=None, progress=None,
                          not_before=None, **kwargs):
        """Echo each element's totals back as a completed season.

        The BOOTSTRAP fixture above is written as a full season of history --
        which is what bootstrap-static actually shows pre-season. The pipeline now
        re-sources that baseline from element-summary history_past, so the fake
        client has to serve the same numbers there for the fixture to keep its
        meaning.
        """
        by_id = {e["id"]: e for e in BOOTSTRAP["elements"]}
        return {
            int(pid): {"history_past": [dict(by_id[int(pid)], season_name="2025/26")]}
            for pid in player_ids if int(pid) in by_id
        }


def test_mode_one_produces_a_valid_recommendation(tmp_path):
    rec, xp = run(Config(rank_sims=0, budget=100.0), mode=1, from_event=1, root=tmp_path,
                  client=FakeClient())
    assert isinstance(rec, Recommendation)
    assert len(rec.squad_ids) == 15
    assert len(rec.lineup.xi) == 11
    assert len(rec.lineup.bench) == 4
    assert rec.lineup.captain in rec.lineup.xi


def test_mode_one_respects_budget(tmp_path):
    rec, _ = run(Config(rank_sims=0, budget=100.0), mode=1, from_event=1, root=tmp_path,
                 client=FakeClient())
    assert rec.squad_value <= 100.0 + 1e-6


def test_returns_contract_frame(tmp_path):
    from fpl.model.xp import CONTRACT_COLUMNS
    cfg = Config(rank_sims=0, horizon_gw=5)
    _, xp = run(cfg, mode=1, from_event=1, root=tmp_path, client=FakeClient())
    # The fixed contract, then the per-gameweek breakdown the optimizers use to
    # move the armband week by week.
    assert list(xp.columns) == CONTRACT_COLUMNS + [f"xp_gw{e}" for e in range(1, 6)]


def test_pipeline_imports_no_mcp_or_archive():
    """The headless path must not depend on MCP, skills, or the backtest archive."""
    project_root = Path(__file__).resolve().parent.parent
    src = ((project_root / "fpl" / "pipeline.py").read_text()
           + (project_root / "run_gameweek.py").read_text())
    for banned in ["mcp", "fantasy_pl", "fpl_mcp", "archive", "Skill"]:
        assert banned not in src, f"weekly path must not reference {banned!r}"


def test_stale_flag_propagates_to_report(tmp_path):
    class StaleClient(FakeClient):
        stale = True

    rec, _ = run(Config(rank_sims=0), mode=1, from_event=1, root=tmp_path, client=StaleClient())
    assert rec.stale is True


def test_mode_two_bank_reflects_pre_transfer_squad_value(tmp_path):
    """Regression: bank must be recomputed from the CURRENT squad's value,
    not passed through unchanged -- otherwise a squad whose value changes
    after transfers reports a bank that doesn't reconcile against budget."""
    from fpl.data.normalize import normalize_players
    players = normalize_players(BOOTSTRAP)
    need = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    current, club_count = [], {}
    cheap = players[players.player_id % 5 == 0].sort_values("player_id")
    for _, row in cheap.iterrows():
        pos, team_id, pid = row["position"], int(row["team_id"]), int(row["player_id"])
        if need.get(pos, 0) > 0 and club_count.get(team_id, 0) < 3:
            current.append(pid)
            need[pos] -= 1
            club_count[team_id] = club_count.get(team_id, 0) + 1
        if all(v == 0 for v in need.values()):
            break

    rec, xp = run(Config(rank_sims=0, max_paid_hits=2), mode=2, from_event=1, root=tmp_path,
                  client=FakeClient(), current_squad=current, bank=5.0, free_transfers=2)
    prices = dict(zip(xp["player_id"].astype(int), xp["price"].astype(float)))
    value_before = round(sum(prices[i] for i in current), 1)
    expected_bank = round(5.0 + value_before - rec.squad_value, 1)
    assert abs(rec.bank - expected_bank) < 1e-6
    # Fixture is chosen so the optimizer actually changes squad value --
    # otherwise the old (buggy) pass-through formula would coincidentally match.
    assert value_before != rec.squad_value


def test_mode_falls_back_honestly_when_no_current_squad_available(tmp_path):
    """Regression: requesting mode=2 with no current_squad (the CLI never
    fetches one today) must report BOTH mode and trust as reflecting the
    Mode 1 rebuild that actually ran, not the requested mode=2 label with
    the trust caveat silently dropped."""
    rec, _ = run(Config(rank_sims=0), mode=2, from_event=1, root=tmp_path, client=FakeClient())
    assert rec.mode == 1
    assert rec.trust != ""
    assert rec.transfers is None


def test_news_override_reaches_the_minutes_model(tmp_path):
    """A p_start override handed to run() must actually move that player's xP.

    minutes_model's own override handling is unit-tested, but nothing checked
    that the pipeline forwards `news` at all -- so a broken wire here would be
    invisible: the run still succeeds and just silently ignores the correction.
    """
    cfg = Config(rank_sims=0, budget=100.0)
    _, base = run(cfg, mode=1, from_event=1, root=tmp_path, client=FakeClient())
    _, cut = run(cfg, mode=1, from_event=1, root=tmp_path, client=FakeClient(),
                 news={5: {"p_start_override": 0.0, "note": "benched",
                           "source": "test"}})

    def xp_of(frame, pid):
        return float(frame.loc[frame["player_id"] == pid].iloc[0]["xp_next5"])

    def p_start_of(frame, pid):
        return float(frame.loc[frame["player_id"] == pid].iloc[0]["p_start"])

    assert p_start_of(cut, 5) < p_start_of(base, 5)
    assert xp_of(cut, 5) < xp_of(base, 5)
    # A TEAMMATE may move: under the team-total cap (model.xp.team_goal_scales)
    # a benched attacker's share of the side's goals passes to the others, and
    # it should. A player at another club has no such link and must not move.
    assert xp_of(cut, 6) >= xp_of(base, 6)          # same club as player 5
    assert xp_of(cut, 25) == xp_of(base, 25), "other clubs must be untouched"


def test_clean_sheet_value_tracks_the_baseline_league_goal_rate(tmp_path):
    """Doubling every team's goals conceded in the baseline season must make
    clean sheets rarer, and so make defenders worth less.

    All teams move together, so the att/dfn ratios are unchanged and the only
    thing that can move is the league goal rate the pipeline estimates.
    """
    import copy
    leaky = copy.deepcopy(BOOTSTRAP)
    for e in leaky["elements"]:
        e["goals_conceded"] = e["goals_conceded"] * 2

    class LeakyClient(FakeClient):
        def bootstrap(self):
            return leaky

        def element_summaries(self, player_ids, ttl_hours=None, progress=None,
                              not_before=None, **kwargs):
            by_id = {e["id"]: e for e in leaky["elements"]}
            return {
                int(pid): {"history_past": [dict(by_id[int(pid)], season_name="2025/26")]}
                for pid in player_ids if int(pid) in by_id
            }

    cfg = Config(rank_sims=0, budget=100.0, horizon_gw=3)
    _, base_xp = run(cfg, mode=1, from_event=1, root=tmp_path, client=FakeClient())
    _, leaky_xp = run(cfg, mode=1, from_event=1, root=tmp_path, client=LeakyClient())

    base_def = base_xp[base_xp.position == "DEF"].xp_next1.mean()
    leaky_def = leaky_xp[leaky_xp.position == "DEF"].xp_next1.mean()
    assert leaky_def < base_def


def test_every_run_records_its_forecast_for_later_scoring(tmp_path):
    """Nothing else in the pipeline persists the xP frame, so without this the
    week's forecast is gone before the gameweek it predicts is played."""
    from fpl.backtest.ledger import load_predictions
    cfg = Config(rank_sims=0, budget=100.0, horizon_gw=3)
    _, xp = run(cfg, mode=1, from_event=4, root=tmp_path, client=FakeClient())

    ledger = load_predictions(gw=4, root=tmp_path)
    assert (ledger["gw"] == 4).all()
    assert sorted(ledger.player_id) == sorted(xp.player_id)


def test_trust_text_prefers_a_real_measurement_over_the_hard_coded_note(tmp_path):
    """The static TRUST_SUMMARY quotes a backtest of a simplified proxy. Once a
    real gameweek has been scored, that measurement is what the user should see."""
    from fpl.backtest.ledger import save_scored_summary
    save_scored_summary("GW1 scored: rank quality +0.472 overall.", root=tmp_path)
    cfg = Config(rank_sims=0, budget=100.0, horizon_gw=3)
    rec, _ = run(cfg, mode=1, from_event=2, root=tmp_path, client=FakeClient())
    assert rec.trust == "GW1 scored: rank quality +0.472 overall."


def _legal_squad_from_bootstrap():
    from fpl.data.normalize import normalize_players
    players = normalize_players(BOOTSTRAP)
    need = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    current, club_count = [], {}
    cheap = players[players.player_id % 5 == 0].sort_values("player_id")
    for _, row in cheap.iterrows():
        pos, team_id, pid = row["position"], int(row["team_id"]), int(row["player_id"])
        if need.get(pos, 0) > 0 and club_count.get(team_id, 0) < 3:
            current.append(pid)
            need[pos] -= 1
            club_count[team_id] = club_count.get(team_id, 0) + 1
        if all(v == 0 for v in need.values()):
            break
    return current, players


def test_mode_two_bank_credits_selling_value_not_market_value(tmp_path):
    """Sold players return purchase price plus half their rise. Crediting the
    full market price invents money the squad never had."""
    from fpl.optimize.transfers import selling_price
    current, players = _legal_squad_from_bootstrap()
    price = dict(zip(players.player_id.astype(int), players.price.astype(float)))
    # every player bought 0.4 below today's price -> sells 0.2 below it
    purchase = {pid: round(price[pid] - 0.4, 1) for pid in current}

    rec, xp = run(Config(rank_sims=0, max_paid_hits=2), mode=2, from_event=1, root=tmp_path,
                  client=FakeClient(), current_squad=current, bank=5.0,
                  free_transfers=2, purchase_prices=purchase)

    assert rec.transfers is not None and rec.transfers.n_transfers > 0
    proceeds = sum(selling_price(purchase[i], price[i]) for i in rec.transfers.out_ids)
    spent = sum(price[i] for i in rec.transfers.in_ids)
    assert rec.bank == pytest.approx(round(5.0 + proceeds - spent, 1), abs=0.051)


# --- P5a: the optimizer maximises the discounted horizon (2026-08-27 audit) ---

SKEWED_FIXTURES = FIXTURES + [
    # teams 1 and 2 play twice in event 1; teams 7 and 8 play twice in event 5
    {"id": 900, "event": 1, "team_h": 1, "team_a": 2, "team_h_difficulty": 3,
     "team_a_difficulty": 3, "kickoff_time": "2026-08-21T19:00:00Z", "finished": False},
    {"id": 901, "event": 5, "team_h": 7, "team_a": 8, "team_h_difficulty": 3,
     "team_a_difficulty": 3, "kickoff_time": "2026-09-21T19:00:00Z", "finished": False},
]


class SkewedClient(FakeClient):
    def fixtures(self):
        return SKEWED_FIXTURES


def _points_available_this_week(rec, xp):
    return float(xp[xp.player_id.isin(rec.squad_ids)].xp_next1.sum())


def test_squad_prefers_points_available_sooner_when_the_horizon_is_discounted(tmp_path):
    """A double gameweek five weeks out is worth less than one this week — the
    squad can be changed before then. Without a discount the solver treats them
    as identical."""
    patient = Config(rank_sims=0, budget=100.0, horizon_gw=5, horizon_decay=1.0)
    impatient = Config(rank_sims=0, budget=100.0, horizon_gw=5, horizon_decay=0.3)

    rec_p, xp_p = run(patient, mode=1, from_event=1, root=tmp_path, client=SkewedClient())
    rec_i, xp_i = run(impatient, mode=1, from_event=1, root=tmp_path, client=SkewedClient())

    assert _points_available_this_week(rec_i, xp_i) > _points_available_this_week(rec_p, xp_p)


# --- P3: the model can finally see the current season (2026-08-27 audit) ---

class InFormClient(FakeClient):
    """Player 20 has started and scored in every gameweek so far this season."""

    def element_summaries(self, player_ids, ttl_hours=None, progress=None,
                          not_before=None, **kwargs):
        out = FakeClient.element_summaries(self, player_ids, ttl_hours, progress)
        for pid in out:
            out[pid] = dict(out[pid], history=[
                {"round": r, "minutes": 90, "starts": 1,
                 "goals_scored": 2 if pid == 20 else 0, "assists": 0, "bonus": 3,
                 "defensive_contribution": 0, "saves": 0, "yellow_cards": 0,
                 "red_cards": 0, "own_goals": 0, "clean_sheets": 0,
                 "goals_conceded": 1, "bps": 30, "total_points": 12,
                 "expected_goals": "1.5" if pid == 20 else "0.0",
                 "expected_assists": "0.0", "expected_goals_conceded": "1.0"}
                for r in range(1, 5)
            ])
        return out


class NotBeforeRecordingClient(FakeClient):
    """Captures the cache floor the pipeline demands of element-summaries."""

    def __init__(self, fixtures):
        self._fixtures = fixtures
        self.not_before = "never called"

    def fixtures(self):
        return self._fixtures

    def element_summaries(self, player_ids, ttl_hours=None, progress=None,
                          not_before=None, **kwargs):
        self.not_before = not_before
        return FakeClient.element_summaries(self, player_ids, ttl_hours, progress)


def test_pipeline_requires_summaries_newer_than_the_last_finished_match(tmp_path):
    """The current season is read out of element-summaries, which cache for 30
    days. Age alone let a GW1 snapshot serve GW3, so the pipeline must also
    demand the cache postdate the last finished match."""
    from datetime import datetime, timedelta, timezone

    played = [dict(f, finished=True) for f in FIXTURES if f["event"] <= 2]
    upcoming = [f for f in FIXTURES if f["event"] > 2]
    client = NotBeforeRecordingClient(played + upcoming)

    run(Config(rank_sims=0, budget=100.0, horizon_gw=3), mode=1, from_event=3, root=tmp_path,
        client=client)

    last_ko = max(datetime.fromisoformat(f["kickoff_time"].replace("Z", "+00:00"))
                  for f in played)
    assert client.not_before == last_ko + timedelta(hours=3)


def test_pipeline_leaves_ttl_in_charge_before_any_match_is_played(tmp_path):
    """Pre-season there is nothing to be stale against — demanding a floor here
    would force a needless refetch of every player."""
    client = NotBeforeRecordingClient(FIXTURES)
    run(Config(rank_sims=0, budget=100.0, horizon_gw=3), mode=1, from_event=1, root=tmp_path,
        client=client)
    assert client.not_before is None


# --- Chip state and data freshness reach the pipeline (2026-09-07 review) ---

def test_a_spent_chip_is_never_recommended_again(tmp_path):
    """pipeline.run passed a literal [] for chips_used, so the advisor could not
    know a chip was gone and cheerfully suggested it every week."""
    # NearDoubleClient gives teams 1 and 2 a double gameweek in event 1, which
    # is what makes the advisor reach for a Triple Captain. SkewedClient no
    # longer works here: its event-5 double is beyond a 3-gameweek horizon, so
    # the advisor now correctly HOLDS the chip rather than advising one.
    cfg = Config(rank_sims=0, budget=100.0, horizon_gw=3)
    rec, _ = run(cfg, mode=1, from_event=1, root=tmp_path, client=NearDoubleClient())
    assert rec.chip.chip is not None, "fixture must advise some chip to be a test"

    spent, _ = run(cfg, mode=1, from_event=1, root=tmp_path, client=NearDoubleClient(),
                   chip_events=[{"chip": rec.chip.chip, "event": 1}])
    assert spent.chip.chip != rec.chip.chip


def test_a_spent_chip_is_explained_when_nothing_is_left_to_advise(tmp_path):
    """The "already used" note is the fallback for advising NOTHING, so it only
    appears once every qualifying chip is gone. Asserting it right after a
    single chip was spent conflated two behaviours, and broke the moment a
    fixture change let a second chip qualify -- the advisor was then correctly
    recommending that one instead of explaining itself."""
    cfg = Config(rank_sims=0, budget=100.0, horizon_gw=3)
    # Dated to GW1 so they land in the same chip window as the run.
    all_chips = [{"chip": c, "event": 1} for c in
                 ("wildcard", "freehit", "benchboost", "triplecaptain")]
    spent, _ = run(cfg, mode=1, from_event=1, root=tmp_path, client=NearDoubleClient(),
                   chip_events=all_chips)
    assert spent.chip.chip is None
    assert "was played in GW1" in spent.chip.reason


def test_a_matchday_shortens_the_cache_ttl(tmp_path):
    """Prices, news and availability move hour to hour on a matchday. The
    shorter TTL was configured from the start and never wired to anything."""
    from datetime import datetime, timezone

    today = datetime.now(timezone.utc).strftime("%Y-%m-%dT18:00:00Z")

    class TodayClient(FakeClient):
        ttl_hours = 6.0

        def fixtures(self):
            return [dict(f, kickoff_time=today) for f in FIXTURES]

    client = TodayClient()
    run(Config(rank_sims=0, budget=100.0, horizon_gw=3, cache_ttl_hours=6,
               cache_ttl_matchday_hours=1), mode=1, from_event=1, root=tmp_path,
        client=client)
    assert client.ttl_hours == 1.0


def test_off_matchday_the_ordinary_ttl_stands(tmp_path):
    class QuietClient(FakeClient):
        ttl_hours = 6.0

    client = QuietClient()
    run(Config(rank_sims=0, budget=100.0, horizon_gw=3, cache_ttl_hours=6,
               cache_ttl_matchday_hours=1), mode=1, from_event=1, root=tmp_path,
        client=client)
    assert client.ttl_hours == 6.0


def test_current_season_form_reaches_the_projection(tmp_path):
    """blend_form existed and was unit-tested for a month while nothing called
    it — the model ran on last season alone."""
    cfg = Config(rank_sims=0, budget=100.0, horizon_gw=3)
    _, cold = run(cfg, mode=1, from_event=5, root=tmp_path, client=FakeClient())
    _, hot = run(cfg, mode=1, from_event=5, root=tmp_path, client=InFormClient())

    cold_xp = float(cold.set_index("player_id").loc[20, "xp_next1"])
    hot_xp = float(hot.set_index("player_id").loc[20, "xp_next1"])
    assert hot_xp > cold_xp


def test_no_refresh_survives_a_matchday(tmp_path):
    """--no-refresh means "use cached data only". The matchday TTL must not
    quietly shorten the cache it exists to keep serving and send the run to the
    network anyway."""
    from datetime import datetime, timezone

    today = datetime.now(timezone.utc).strftime("%Y-%m-%dT18:00:00Z")

    class TodayClient(FakeClient):
        ttl_hours = 24 * 365

        def fixtures(self):
            return [dict(f, kickoff_time=today) for f in FIXTURES]

    client = TodayClient()
    # what run_gameweek.py --no-refresh builds
    cfg = Config(rank_sims=0, budget=100.0, horizon_gw=3, cache_ttl_hours=24 * 365,
                 cache_ttl_matchday_hours=24 * 365)
    run(cfg, mode=1, from_event=1, root=tmp_path, client=client)
    assert client.ttl_hours == 24 * 365


def test_history_cached_before_fpls_data_check_is_flagged_in_the_report(tmp_path):
    """A snapshot taken between a gameweek's last whistle and FPL confirming
    bonus carries numbers that were still moving. Snapshots written from now on
    say which gameweeks were checked; older ones cannot, so the report says the
    forecast may be running on provisional returns."""
    from fpl.pipeline import freshness_flags

    played = [dict(f, finished=True) for f in FIXTURES if f["event"] <= 2]
    upcoming = [f for f in FIXTURES if f["event"] > 2]

    class Unverified:
        # last GW2 kickoff is 2026-08-22T19:00Z; the snapshot predates the
        # +6h settle window
        unverified = {"element-summary-1"}
        sources = {"element-summary-1": "2026-08-22T21:00:00Z"}

    class Verified(Unverified):
        unverified = set()

    assert freshness_flags(Unverified(), played + upcoming, 2)
    assert "provisional" in freshness_flags(Unverified(), played + upcoming, 2)[0].lower() \
        or "corrections" in freshness_flags(Unverified(), played + upcoming, 2)[0].lower()
    assert freshness_flags(Verified(), played + upcoming, 2) == []


def test_history_cached_after_the_check_window_is_not_flagged(tmp_path):
    from fpl.pipeline import freshness_flags

    played = [dict(f, finished=True) for f in FIXTURES if f["event"] <= 2]

    class Late:
        unverified = {"element-summary-1"}
        sources = {"element-summary-1": "2026-08-23T09:00:00Z"}  # well past +6h

    assert freshness_flags(Late(), played, 2) == []


def test_nothing_is_flagged_before_any_gameweek_is_checked(tmp_path):
    from fpl.pipeline import freshness_flags

    class Any:
        unverified = {"element-summary-1"}
        sources = {"element-summary-1": "2026-08-01T09:00:00Z"}

    assert freshness_flags(Any(), FIXTURES, 0) == []


def test_fallback_trust_note_does_not_repeat_the_refuted_goalkeeper_claim():
    """TRUST_SUMMARY asserted GK rank quality of 0.034 ("no measurable skill")
    from a points-per-90 PROXY backtest. Two scored gameweeks of the production
    model contradicted it (+0.524, +0.529). Telling a user to distrust their
    keeper on a number the model's own ledger refutes is worse than saying
    nothing."""
    from fpl.pipeline import TRUST_SUMMARY
    assert "0.034" not in TRUST_SUMMARY
    assert "no measurable skill" not in TRUST_SUMMARY.lower()
    # It must still say the fallback is not a measurement of this model.
    assert "proxy" in TRUST_SUMMARY.lower()


# --- the distributional layer in the pipeline (2026-09-09) -----------------

def test_the_report_says_how_often_the_squad_beats_the_field(tmp_path):
    """A squad's expected points mean nothing on their own -- the user needs to
    know where that lands against the people they are competing with."""
    cfg = Config(horizon_gw=2, rank_sims=600, rank_candidates=3)
    rec, _ = run(cfg, mode=1, from_event=1, root=tmp_path, client=FakeClient())
    assert rec.rank is not None
    assert 0.0 <= rec.rank["p_beat_target"] <= 1.0
    assert 0.0 <= rec.rank["rank_percentile"] <= 1.0
    assert rec.rank["sd_points"] > 0
    assert rec.rank["n_candidates"] == 3


def test_turning_the_simulation_off_falls_back_to_the_plain_optimum(tmp_path):
    cfg = Config(horizon_gw=2, rank_sims=0)
    rec, _ = run(cfg, mode=1, from_event=1, root=tmp_path, client=FakeClient())
    assert rec.rank is None
    assert len(rec.squad_ids) == 15


def test_the_rank_chosen_squad_is_still_a_legal_fifteen(tmp_path):
    cfg = Config(horizon_gw=2, rank_sims=600, rank_candidates=4)
    rec, xp = run(cfg, mode=1, from_event=1, root=tmp_path, client=FakeClient())
    pos = xp.set_index("player_id").loc[rec.squad_ids, "position"].value_counts()
    assert len(rec.squad_ids) == 15
    assert pos.get("GKP", 0) == 2 and pos.get("DEF", 0) == 5
    assert pos.get("MID", 0) == 5 and pos.get("FWD", 0) == 3
    assert len(rec.lineup.xi) == 11


def test_calibration_is_skipped_until_enough_gameweeks_are_scored(tmp_path):
    """Early season there is nothing to fit on, and the run must proceed
    uncalibrated rather than applying a correction built from noise."""
    cfg = Config(rank_sims=0, horizon_gw=2)
    rec, _ = run(cfg, mode=1, from_event=1, root=tmp_path, client=FakeClient())
    assert rec.calibration is None


def test_calibration_can_be_switched_off(tmp_path):
    cfg = Config(rank_sims=0, horizon_gw=2, calibrate=False)
    rec, _ = run(cfg, mode=1, from_event=1, root=tmp_path, client=FakeClient())
    assert rec.calibration is None


def test_the_report_does_not_present_the_field_comparison_as_a_forecast(tmp_path):
    """`p_beat_target` is computed by simulating BOTH sides from the model's
    own projections, so it inherits any optimism in them. On real GW1-3 data
    the simulation implied +15 pts/GW against a realised +1.3. It is valid for
    ranking candidate squads against each other and must not be read as a
    prediction of where you will finish."""
    from fpl.report.weekly import render, Recommendation
    cfg = Config(rank_sims=400, rank_candidates=2, horizon_gw=2)
    rec, xp = run(cfg, mode=1, from_event=1, root=tmp_path, client=FakeClient())
    text = render(rec, xp)
    low = text.lower()
    assert "own projection" in low or "not a forecast" in low
    assert "relative" in low


def test_the_weekly_transfer_run_also_reports_against_the_field(tmp_path):
    """The distributional layer was wired into the squad build only, so the
    mode anyone actually runs every week never used it."""
    cfg = Config(horizon_gw=2, rank_sims=600, rank_candidates=3)
    current = [1, 20, 4, 5, 23, 24, 42, 48, 49, 67, 68, 69, 92, 93, 94]   # 2/5/5/3 across five clubs, 3 each
    rec, _ = run(cfg, mode=2, from_event=1, root=tmp_path, client=FakeClient(),
                 current_squad=current, bank=2.0, free_transfers=1)
    assert rec.mode == 2
    assert rec.rank is not None
    assert 0.0 <= rec.rank["p_beat_target"] <= 1.0
    assert rec.rank["n_candidates"] >= 1


def test_the_weekly_run_is_unchanged_when_the_rank_layer_is_off(tmp_path):
    """Regression guard: rank_sims=0 must reproduce the plain MILP plan."""
    from fpl.optimize.transfers import optimize_transfers
    cfg = Config(horizon_gw=2, rank_sims=0)
    current = [1, 20, 4, 5, 23, 24, 42, 48, 49, 67, 68, 69, 92, 93, 94]   # 2/5/5/3 across five clubs, 3 each
    rec, xp = run(cfg, mode=2, from_event=1, root=tmp_path, client=FakeClient(),
                  current_squad=current, bank=2.0, free_transfers=1)
    assert rec.rank is None
    best, _ = optimize_transfers(xp, current, 2.0, 1, cfg, xp_col="xp_horizon")
    assert set(rec.squad_ids) == set(best.squad_ids)


# --- Chip timing sees past the xP horizon (2026-09-13) ---

# Teams 1 and 2 double in event 1 and nothing else is unusual. SKEWED_FIXTURES
# is unsuitable as a baseline here: it also doubles teams 7 and 8 in event 5,
# which is itself beyond a 3-gameweek horizon and would hold the chip.
NEAR_DOUBLE_FIXTURES = FIXTURES + [
    {"id": 900, "event": 1, "team_h": 1, "team_a": 2, "team_h_difficulty": 3,
     "team_a_difficulty": 3, "kickoff_time": "2026-08-21T19:00:00Z", "finished": False},
]

# The same, plus a full-squad double in event 10 -- seven gameweeks past what a
# 3-gameweek projection can see.
FAR_DOUBLE_FIXTURES = NEAR_DOUBLE_FIXTURES + [
    {"id": 950 + i, "event": 10, "team_h": h, "team_a": a, "team_h_difficulty": 3,
     "team_a_difficulty": 3, "kickoff_time": "2026-11-21T19:00:00Z", "finished": False}
    for i, (h, a) in enumerate([(1, 2), (3, 4), (5, 6), (7, 8),
                                (1, 3), (2, 4), (5, 7), (6, 8)])
]


class NearDoubleClient(FakeClient):
    def fixtures(self):
        return NEAR_DOUBLE_FIXTURES


class FarDoubleClient(FakeClient):
    def fixtures(self):
        return FAR_DOUBLE_FIXTURES


def test_a_double_beyond_the_projection_horizon_holds_the_chip(tmp_path):
    """An event-1 double is enough to trigger a Triple Captain. The same squad
    facing a FULL-SQUAD double in event 10 must hold the chip instead of
    burning it now -- which the advisor could not see before, because it was
    handed fixture counts for the projection horizon only."""
    cfg = Config(rank_sims=0, budget=100.0, horizon_gw=3)

    now, _ = run(cfg, mode=1, from_event=1, root=tmp_path, client=NearDoubleClient())
    later, _ = run(cfg, mode=1, from_event=1, root=tmp_path, client=FarDoubleClient())

    assert now.chip.chip == "triplecaptain", "baseline must play the chip now"
    assert now.chip.hold_until is None
    assert later.chip.chip != "triplecaptain"
    assert later.chip.hold_until == 10


def test_a_structural_hold_never_claims_to_have_a_projection(tmp_path):
    """The two hold reasons must read differently. A gameweek 7 weeks past the
    horizon has a fixture COUNT and nothing else, and saying it 'projects
    better' would invent a number the model never computed."""
    cfg = Config(rank_sims=0, budget=100.0, horizon_gw=3)

    later, _ = run(cfg, mode=1, from_event=1, root=tmp_path, client=FarDoubleClient())

    assert "no projection" in later.chip.reason
    assert "projects better" not in later.chip.reason


# --- P0/B4: a named chip returns its own squad (2026-09-17 audit) ---

def _legal_current_squad():
    """A cheap but legal 15 to run Mode 2 transfers from."""
    from fpl.data.normalize import normalize_players
    players = normalize_players(BOOTSTRAP)
    need = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    current, club_count = [], {}
    cheap = players[players.player_id % 5 == 0].sort_values("player_id")
    for _, row in cheap.iterrows():
        pos, team_id, pid = row["position"], int(row["team_id"]), int(row["player_id"])
        if need.get(pos, 0) > 0 and club_count.get(team_id, 0) < 3:
            current.append(pid)
            need[pos] -= 1
            club_count[team_id] = club_count.get(team_id, 0) + 1
        if all(v == 0 for v in need.values()):
            break
    return current


def test_a_recommended_wildcard_returns_the_rebuilt_squad(tmp_path, monkeypatch):
    """Confirming a Wildcard used to apply the limited-transfer squad, which is
    not the team the chip chose."""
    import fpl.pipeline as pipeline
    from fpl.optimize.chips import ChipAdvice

    monkeypatch.setattr(pipeline, "advise_chips",
                        lambda *a, **k: ChipAdvice("wildcard", "test"))
    rec, _ = run(Config(rank_sims=0, max_paid_hits=2), mode=2, from_event=1,
                 root=tmp_path, client=FakeClient(),
                 current_squad=_legal_current_squad(), bank=5.0, free_transfers=1)
    assert rec.chip.chip == "wildcard"
    assert rec.chip_squad is True
    assert rec.chip_temporary is False
    # Unlimited free transfers: a rebuild can never be charged a hit.
    assert rec.transfers is not None and rec.transfers.hit_cost == 0
    assert set(rec.lineup.xi) <= set(rec.squad_ids)


def test_a_recommended_free_hit_is_flagged_temporary(tmp_path, monkeypatch):
    import fpl.pipeline as pipeline
    from fpl.optimize.chips import ChipAdvice

    monkeypatch.setattr(pipeline, "advise_chips",
                        lambda *a, **k: ChipAdvice("freehit", "test"))
    rec, _ = run(Config(rank_sims=0, max_paid_hits=2), mode=2, from_event=1,
                 root=tmp_path, client=FakeClient(),
                 current_squad=_legal_current_squad(), bank=5.0, free_transfers=1)
    assert rec.chip_squad is True
    assert rec.chip_temporary is True
    assert len(rec.squad_ids) == 15


def test_an_ordinary_week_is_not_flagged_as_a_chip_squad(tmp_path):
    rec, _ = run(Config(rank_sims=0, max_paid_hits=2), mode=2, from_event=1,
                 root=tmp_path, client=FakeClient(),
                 current_squad=_legal_current_squad(), bank=5.0, free_transfers=1)
    assert rec.chip_squad is False
    assert rec.chip_temporary is False


# --- B7: the report shows the captain the rank layer scored ---

def test_the_reported_captain_is_the_rank_layers_when_rank_decided():
    from fpl.optimize.lineup import Lineup
    from fpl.pipeline import _honour_rank_captain
    xp = pd.DataFrame({"player_id": [1, 2, 3], "xp_next1": [5.0, 6.0, 4.0]})
    lineup = Lineup(xi=[1, 2, 3], bench=[], formation="x", captain=2, vice=1, xp=15.0)
    out, stats = _honour_rank_captain(lineup, {"captain": 1, "decided_by": "rank"}, xp)
    assert out.captain == 1 and out.vice == 2
    assert stats["captain_reported"] is True


def test_the_lineups_captain_stands_when_rank_only_reported():
    from fpl.optimize.lineup import Lineup
    from fpl.pipeline import _honour_rank_captain
    xp = pd.DataFrame({"player_id": [1, 2, 3], "xp_next1": [5.0, 6.0, 4.0]})
    lineup = Lineup(xi=[1, 2, 3], bench=[], formation="x", captain=2, vice=1, xp=15.0)
    out, stats = _honour_rank_captain(
        lineup, {"captain": 1, "decided_by": "expected points over the horizon"}, xp)
    assert out.captain == 2
    assert stats["captain_reported"] is False


def test_a_rank_captain_outside_the_exact_xi_is_not_forced_in():
    from fpl.optimize.lineup import Lineup
    from fpl.pipeline import _honour_rank_captain
    xp = pd.DataFrame({"player_id": [1, 2, 3, 9], "xp_next1": [5.0, 6.0, 4.0, 1.0]})
    lineup = Lineup(xi=[1, 2, 3], bench=[9], formation="x", captain=2, vice=1, xp=15.0)
    out, stats = _honour_rank_captain(lineup, {"captain": 9, "decided_by": "rank"}, xp)
    assert out.captain == 2 and stats["captain_reported"] is False


# --- R1: Mode 1 decides on expected points, rank reports ---

def test_mode_one_returns_the_xp_optimum_and_reports_rank(tmp_path):
    rec, _ = run(Config(rank_sims=300, rank_candidates=3, budget=100.0),
                 mode=1, from_event=1, root=tmp_path, client=FakeClient())
    assert rec.rank is not None
    assert rec.rank["decided_by"] == "expected points over the horizon"
    assert "rank_would_choose" in rec.rank


def test_mode_one_lets_rank_decide_only_when_asked(tmp_path):
    rec, _ = run(Config(rank_sims=300, rank_candidates=3, budget=100.0, rank_squad=True),
                 mode=1, from_event=1, root=tmp_path, client=FakeClient())
    assert rec.rank["decided_by"] == "rank"
