"""Load and validate user preferences from config.yaml."""
from dataclasses import dataclass, field
from pathlib import Path
import yaml

VALID_PROFILES = {"balanced", "template", "differential"}
# Empty since 2026-09-09: ownership_weight now reaches the squad and transfer
# objectives through optimize.objective.effective_xp, so "template" and
# "differential" genuinely tilt picks instead of silently behaving like
# "balanced". Anything added back here is rejected at load rather than
# quietly no-opping.
NOT_YET_IMPLEMENTED_PROFILES: set[str] = set()
FT_CAP = 5


@dataclass
class Config:
    budget: float = 100.0
    horizon_gw: int = 5
    risk_profile: str = "balanced"
    ownership_weight: float = 0.0
    news_weight: float = 0.5
    news_max_age_hours: int = 48
    form_half_life_gw: float = 3.0
    form_max_weight: float = 0.6
    shrinkage_minutes: float = 900.0
    start_prior_games: float = 4.0
    horizon_decay: float = 0.85
    max_paid_hits: int = 2
    hit_cost: int = 4
    bench_weight: list[float] = field(default_factory=lambda: [0.15, 0.10, 0.05, 0.02])
    # Lowest projection allowed to SIT on the bench in a squad build. This is a
    # Bench Boost READINESS constraint, not an ordinary-week one: forcing every
    # bench slot over a floor spends XI budget on players who, in a normal week,
    # score nothing. Measured on the real GW5 pool it cost 0.07 xP of XI to gain
    # 9.94 xP of bench -- a good trade the week you play the chip and a pure
    # loss every week you do not, including after it has been spent.
    #
    # So it is OFF by default and belongs in config.yaml only while preparing a
    # Wildcard or a Bench Boost. `pipeline._squad_quality` and the Wildcard
    # action still honour whatever is configured.
    bench_floor_xp: float = 0.0
    odds_provider: str | None = None
    cache_ttl_hours: int = 6
    cache_ttl_matchday_hours: int = 1
    entry_id: int | None = None
    free_transfers: int = 1
    # The distributional layer (model.simulate + optimize.rank). Squads are
    # compared by how often they beat a simulated field rather than by
    # expected points alone, which is the only way correlated bets -- three
    # defenders sharing one clean sheet -- get priced as the single bet they
    # are. 0 sims turns it off and falls back to the plain MILP optimum.
    rank_sims: int = 4000
    rank_candidates: int = 8
    # How many of the fifteen must change between candidate squads. At 1 the
    # candidates are one swap apart and the choice between them is noise; 4
    # makes them genuinely different teams, which is the only setting at which
    # rank selection changed the answer on real data.
    rank_diversity: int = 4
    # Which slice of the field to try to beat. 0.5 is the median manager and
    # tracks expected points closely; raise it toward 0.9 to chase a green
    # arrow, which is the only regime where taking variance is correct.
    rank_target: float = 0.5
    # Recalibrate xP per position against scored gameweeks (model.calibration).
    # Self-limiting: it refuses to fit below MIN_GAMEWEEKS, so early in a season
    # this is a no-op rather than a correction built from noise.
    calibrate: bool = True


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    risk = raw.get("risk") or {}
    news = raw.get("news") or {}
    model = raw.get("model") or {}
    opt = raw.get("optimizer") or {}
    odds = raw.get("odds") or {}
    data = raw.get("data") or {}

    d = Config()
    cfg = Config(
        budget=float(raw.get("budget", d.budget)),
        horizon_gw=int(raw.get("horizon_gw", d.horizon_gw)),
        risk_profile=risk.get("profile", d.risk_profile),
        ownership_weight=float(risk.get("ownership_weight", d.ownership_weight)),
        news_weight=float(news.get("weight", d.news_weight)),
        news_max_age_hours=int(news.get("max_age_hours", d.news_max_age_hours)),
        form_half_life_gw=float(model.get("form_half_life_gw", d.form_half_life_gw)),
        form_max_weight=float(model.get("form_max_weight", d.form_max_weight)),
        shrinkage_minutes=float(model.get("shrinkage_minutes", d.shrinkage_minutes)),
        start_prior_games=float(model.get("start_prior_games", d.start_prior_games)),
        horizon_decay=float(model.get("horizon_decay", d.horizon_decay)),
        max_paid_hits=int(opt.get("max_paid_hits", d.max_paid_hits)),
        hit_cost=int(opt.get("hit_cost", d.hit_cost)),
        bench_weight=list(opt.get("bench_weight", d.bench_weight)),
        bench_floor_xp=float(opt.get("bench_floor_xp", d.bench_floor_xp)),
        odds_provider=odds.get("provider", d.odds_provider),
        cache_ttl_hours=int(data.get("cache_ttl_hours", d.cache_ttl_hours)),
        cache_ttl_matchday_hours=int(data.get("cache_ttl_matchday_hours", d.cache_ttl_matchday_hours)),
        entry_id=raw.get("entry_id", d.entry_id),
        rank_sims=int(opt.get("rank_sims", d.rank_sims)),
        rank_candidates=int(opt.get("rank_candidates", d.rank_candidates)),
        rank_diversity=int(opt.get("rank_diversity", d.rank_diversity)),
        calibrate=bool(model.get("calibrate", d.calibrate)),
        rank_target=float(opt.get("rank_target", d.rank_target)),
        free_transfers=int(raw.get("free_transfers", d.free_transfers)),
    )

    if cfg.risk_profile not in VALID_PROFILES:
        raise ValueError(f"risk.profile must be one of {sorted(VALID_PROFILES)}, got {cfg.risk_profile!r}")
    if cfg.risk_profile in NOT_YET_IMPLEMENTED_PROFILES:
        raise ValueError(
            f"risk.profile={cfg.risk_profile!r} is accepted by the schema but not yet "
            f"wired into the optimizer -- ownership_weight has no effect yet. "
            f"Use 'balanced' for now."
        )
    if cfg.odds_provider is not None:
        raise ValueError(
            f"odds.provider={cfg.odds_provider!r} is accepted by the schema but no "
            f"provider is implemented -- the pipeline never passes one to "
            f"model.strength.team_ratings, so setting this would silently do "
            f"nothing. Leave it null (team_ratings still takes an injected "
            f"provider object for tests and future use)."
        )
    if not 0.0 < cfg.rank_target < 1.0:
        raise ValueError(
            f"optimizer.rank_target is the quantile of the field to try to "
            f"beat and must be in (0, 1) -- 0.5 is the median manager, 0.9 a "
            f"top-tenth week, got {cfg.rank_target}"
        )
    if cfg.rank_sims > 0:
        # Fail here rather than twenty minutes into a run: locating an extreme
        # quantile of the field needs rivals in proportion to 1/(1 - target),
        # and past a point that draw costs more than the answer is worth.
        from .optimize.rank import required_rivals
        required_rivals(cfg.rank_target)
    if cfg.rank_sims < 0:
        raise ValueError(f"optimizer.rank_sims must be >= 0, got {cfg.rank_sims}")
    if not 1 <= cfg.rank_diversity <= 15:
        raise ValueError(
            f"optimizer.rank_diversity is how many of the fifteen must differ "
            f"between candidate squads and must be in 1..15, got {cfg.rank_diversity}")
    if cfg.rank_candidates < 1:
        raise ValueError(
            f"optimizer.rank_candidates must be at least 1, got {cfg.rank_candidates}")
    if not 0.0 <= cfg.ownership_weight <= 1.0:
        raise ValueError(
            f"risk.ownership_weight scales the ownership tilt from 0 (pure "
            f"expected points) to 1 (the strongest tilt on offer) and must be "
            f"in 0..1, got {cfg.ownership_weight}"
        )
    if not 0 <= cfg.free_transfers <= FT_CAP:
        raise ValueError(f"free_transfers must be 0..{FT_CAP}, got {cfg.free_transfers}")
    if cfg.start_prior_games <= 0:
        raise ValueError(
            f"model.start_prior_games is the beta-binomial prior strength in team "
            f"games and must be positive, got {cfg.start_prior_games}"
        )
    if not 0 < cfg.horizon_decay <= 1:
        raise ValueError(
            f"model.horizon_decay discounts each further gameweek and must be in "
            f"(0, 1] -- 1.0 means no discount, got {cfg.horizon_decay}"
        )
    if cfg.budget <= 0:
        raise ValueError(f"budget must be positive, got {cfg.budget}")
    if len(cfg.bench_weight) != 4:
        raise ValueError(f"optimizer.bench_weight needs exactly 4 values, got {len(cfg.bench_weight)}")
    if cfg.bench_floor_xp < 0:
        raise ValueError(
            f"optimizer.bench_floor_xp is the lowest projection allowed to sit on "
            f"the bench and cannot be negative, got {cfg.bench_floor_xp}")
    return cfg
