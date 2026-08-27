"""Load and validate user preferences from config.yaml."""
from dataclasses import dataclass, field
from pathlib import Path
import yaml

VALID_PROFILES = {"balanced", "template", "differential"}
# Accepted by the schema (forward-compatible), but the optimizer never reads
# ownership_weight yet -- so these two would silently behave identically to
# "balanced" rather than actually tilting picks. Reject them until they're
# wired into the MILP objective, so the config never quietly no-ops.
NOT_YET_IMPLEMENTED_PROFILES = {"template", "differential"}
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
    odds_provider: str | None = None
    cache_ttl_hours: int = 6
    cache_ttl_matchday_hours: int = 1
    entry_id: int | None = None
    free_transfers: int = 1


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
        odds_provider=odds.get("provider", d.odds_provider),
        cache_ttl_hours=int(data.get("cache_ttl_hours", d.cache_ttl_hours)),
        cache_ttl_matchday_hours=int(data.get("cache_ttl_matchday_hours", d.cache_ttl_matchday_hours)),
        entry_id=raw.get("entry_id", d.entry_id),
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
    return cfg
