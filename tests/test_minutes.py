import pandas as pd
from fpl.config import Config
from fpl.model.minutes import minutes_model, M_START, M_SUB

CFG = Config(shrinkage_minutes=900, news_weight=0.5)

PLAYERS = pd.DataFrame({
    "player_id": [1, 2, 3, 4, 5],
    "web_name": ["Nailed", "Rotation", "Injured", "Doubt", "NewSigning"],
    "position": ["MID", "MID", "DEF", "FWD", "MID"],
    "team_id": [1, 1, 2, 2, 3],
    "price": [9.0, 5.0, 4.5, 7.0, 8.5],
    "status": ["a", "a", "i", "d", "a"],
    "chance_of_playing": [None, None, 0, 25, None],
    "news": ["", "", "Knee injury", "Knock - 25% chance", ""],
    "available": [True, True, False, False, True],
    "minutes": [3200, 900, 2000, 2500, 0],
    "starts": [36, 8, 24, 29, 0],
})


def test_nailed_starter_has_high_p_start():
    df = minutes_model(PLAYERS, CFG).set_index("player_id")
    assert df.loc[1, "p_start"] > 0.75


def test_rotation_risk_has_lower_p_start_than_nailed():
    df = minutes_model(PLAYERS, CFG).set_index("player_id")
    assert df.loc[2, "p_start"] < df.loc[1, "p_start"]


def test_injured_player_is_zeroed():
    df = minutes_model(PLAYERS, CFG).set_index("player_id")
    assert df.loc[3, "p_start"] == 0.0
    assert df.loc[3, "e_minutes"] == 0.0
    assert any("unavailable" in f.lower() or "injur" in f.lower() for f in df.loc[3, "flags"])


def test_doubtful_player_scaled_by_chance_of_playing():
    df = minutes_model(PLAYERS, CFG).set_index("player_id")
    assert 0 < df.loc[4, "p_start"] < 0.4
    assert any("25%" in f for f in df.loc[4, "flags"])


def test_new_signing_gets_price_prior_and_low_confidence():
    df = minutes_model(PLAYERS, CFG).set_index("player_id")
    assert df.loc[5, "confidence"] == "low"
    assert df.loc[5, "p_start"] > 0
    assert any("limited data" in f.lower() for f in df.loc[5, "flags"])


def test_expected_minutes_bounded_by_start_and_sub_values():
    df = minutes_model(PLAYERS, CFG).set_index("player_id")
    assert 0 <= df.loc[1, "e_minutes"] <= M_START + M_SUB
    assert df.loc[1, "e_minutes"] > df.loc[2, "e_minutes"]


def test_p_60_never_exceeds_p_play():
    df = minutes_model(PLAYERS, CFG)
    assert (df["p_60"] <= df["p_play"] + 1e-9).all()


def test_news_override_blended_by_weight_and_flagged():
    news = {2: {"p_start_override": 1.0, "note": "confirmed to start", "source": "example.com"}}
    base = minutes_model(PLAYERS, CFG).set_index("player_id").loc[2, "p_start"]
    out = minutes_model(PLAYERS, CFG, news=news).set_index("player_id")
    assert out.loc[2, "p_start"] > base
    assert any("example.com" in f for f in out.loc[2, "flags"])


def test_news_weight_zero_ignores_news():
    news = {2: {"p_start_override": 1.0, "note": "confirmed", "source": "example.com"}}
    cfg = Config(shrinkage_minutes=900, news_weight=0.0)
    base = minutes_model(PLAYERS, cfg).set_index("player_id").loc[2, "p_start"]
    out = minutes_model(PLAYERS, cfg, news=news).set_index("player_id").loc[2, "p_start"]
    assert abs(out - base) < 1e-9
