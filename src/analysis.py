"""End-to-end analysis pipeline shared by the dashboard, the notebook and tests.

``run_analysis`` turns a raw price table and an ``AnalysisConfig`` into every
number the dashboard displays. The Streamlit layer only formats and plots
the result, so all analytics stay testable without a browser.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .data_loader import (
    DEFAULT_MIN_OBSERVATIONS,
    DataQualityReport,
    Frequency,
    compute_returns,
    infer_frequency,
    prepare_prices,
)
from .exceptions import InsufficientDataError
from .performance import (
    DrawdownStats,
    drawdown_series,
    drawdown_stats,
    growth_of_investment,
    performance_summary,
    period_returns,
    relative_performance,
)
from .portfolio import PortfolioResult, WeightValidation, construct_portfolio, validate_weights
from .risk_metrics import (
    VaRBacktest,
    backtest_historical_var,
    calendar_year_risk,
    concentration_summary,
    correlation_matrix,
    risk_contributions,
    risk_summary,
    rolling_sharpe,
    rolling_volatility,
    tail_count,
    var_es_table,
    volatility_regime_analysis,
)


@dataclass(frozen=True)
class AnalysisConfig:
    assets: tuple[str, ...]
    weights: Mapping[str, float]
    benchmark: str | None = None
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    rebalance: str = "monthly"
    risk_free_rate: float = 0.02          # annual, effective
    confidence: float = 0.95
    portfolio_value: float = 1_000_000.0
    vol_window: int = 63                  # periods
    sharpe_window: int = 252              # periods
    var_window: int = 252                 # periods used by the VaR backtest
    missing_policy: str = "drop"
    ffill_limit: int = 3
    periods_per_year: int | None = None   # None -> inferred from the dates
    regime_window: int = 21
    regime_thresholds: tuple[float, float] = (0.15, 0.25)
    min_observations: int = DEFAULT_MIN_OBSERVATIONS


@dataclass
class AnalysisResult:
    config: AnalysisConfig
    weights: WeightValidation
    quality: DataQualityReport
    frequency: Frequency
    periods_per_year: int
    prices: pd.DataFrame
    asset_returns: pd.DataFrame
    portfolio: PortfolioResult
    benchmark_returns: pd.Series | None
    summary: pd.DataFrame                    # metrics x [Portfolio, Benchmark]
    relative: dict[str, float] | None
    drawdown: DrawdownStats
    benchmark_drawdown: DrawdownStats | None
    var_table: pd.DataFrame
    growth: pd.DataFrame
    drawdowns: pd.DataFrame
    rolling_volatility: pd.DataFrame | None
    rolling_sharpe: pd.DataFrame | None
    backtest: VaRBacktest | None
    correlation: pd.DataFrame | None
    concentration: dict[str, float]
    risk_contributions: pd.DataFrame | None
    calendar_years: pd.DataFrame | None
    benchmark_years: pd.Series | None
    regimes: pd.DataFrame | None
    monthly_returns: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    @property
    def portfolio_returns(self) -> pd.Series:
        return self.portfolio.returns


def _summary_column(returns: pd.Series, cfg: AnalysisConfig, ppy: int) -> dict[str, float]:
    perf = performance_summary(returns, cfg.risk_free_rate, ppy)
    risk = risk_summary(returns, cfg.confidence, cfg.portfolio_value, ppy)
    perf.update(
        historical_var=risk["historical_var"],
        historical_es=risk["historical_es"],
        historical_var_amount=risk["historical_var_amount"],
        historical_es_amount=risk["historical_es_amount"],
        skewness=risk["skewness"],
        excess_kurtosis=risk["excess_kurtosis"],
    )
    return perf


def _monthly_table(returns: pd.Series) -> pd.DataFrame:
    monthly = period_returns(returns, "M")
    table = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.to_numpy()})
    table = table.pivot(index="year", columns="month", values="ret").reindex(columns=range(1, 13))
    table.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    table["Year"] = period_returns(returns, "Y").to_numpy()
    return table


def run_analysis(prices: pd.DataFrame, cfg: AnalysisConfig) -> AnalysisResult:
    notes: list[str] = []

    weights = validate_weights({a: cfg.weights.get(a, 0.0) for a in cfg.assets}, prices.columns)
    held = list(weights.weights.index)
    tickers = held + ([cfg.benchmark] if cfg.benchmark and cfg.benchmark not in held else [])

    aligned, quality = prepare_prices(
        prices, tickers, cfg.start, cfg.end,
        missing_policy=cfg.missing_policy, ffill_limit=cfg.ffill_limit, min_observations=cfg.min_observations,
    )
    frequency = infer_frequency(aligned.index)
    ppy = cfg.periods_per_year or frequency.periods_per_year
    returns = compute_returns(aligned)
    asset_returns = returns[held]
    bench = returns[cfg.benchmark].rename("Benchmark") if cfg.benchmark else None

    portfolio = construct_portfolio(asset_returns, weights.weights, cfg.rebalance)
    port = portfolio.returns
    n = len(port)
    start_date = aligned.index[0]

    # --- performance ------------------------------------------------------- #
    columns = {"Portfolio": _summary_column(port, cfg, ppy)}
    if bench is not None:
        columns["Benchmark"] = _summary_column(bench, cfg, ppy)
    summary = pd.DataFrame(columns)
    relative = relative_performance(port, bench, cfg.risk_free_rate, ppy) if bench is not None else None

    growth = pd.DataFrame({"Portfolio": growth_of_investment(port, cfg.portfolio_value, start_date)})
    drawdowns = pd.DataFrame({"Portfolio": drawdown_series(port)})
    if bench is not None:
        growth["Benchmark"] = growth_of_investment(bench, cfg.portfolio_value, start_date)
        drawdowns["Benchmark"] = drawdown_series(bench)

    if n < ppy:
        notes.append(
            f"The sample covers less than one year ({n} periods). Annualised figures extrapolate a short "
            "history and should be read with caution."
        )

    # --- risk -------------------------------------------------------------- #
    var_table = var_es_table(port, sorted({0.90, 0.95, 0.99, cfg.confidence}), cfg.portfolio_value)
    k = tail_count(n, cfg.confidence)
    if k < 10:
        notes.append(
            f"Only {k} observation(s) lie in the {cfg.confidence:.1%} tail, so VaR and Expected Shortfall "
            "rest on very few data points and are statistically fragile."
        )

    def _pair(func, window, **kwargs):
        frame = pd.DataFrame({"Portfolio": func(port, window, **kwargs)})
        if bench is not None:
            frame["Benchmark"] = func(bench, window, **kwargs)
        return frame

    roll_vol = roll_sharpe = None
    if cfg.vol_window < n:
        roll_vol = _pair(rolling_volatility, cfg.vol_window, periods_per_year=ppy)
    else:
        notes.append(f"Rolling volatility skipped: the {cfg.vol_window}-period window exceeds the sample.")
    if n >= 2 * cfg.sharpe_window:
        roll_sharpe = _pair(rolling_sharpe, cfg.sharpe_window, risk_free_rate=cfg.risk_free_rate, periods_per_year=ppy)
    else:
        notes.append(
            f"Rolling Sharpe skipped: it needs at least twice the {cfg.sharpe_window}-period window "
            "to be informative (short-window Sharpe ratios are extremely noisy)."
        )

    backtest = None
    if n >= cfg.var_window + 60:
        backtest = backtest_historical_var(port, cfg.var_window, cfg.confidence)
    else:
        notes.append(f"VaR backtest skipped: it needs at least {cfg.var_window + 60} periods.")

    # --- diversification --------------------------------------------------- #
    corr_input = asset_returns
    if bench is not None and cfg.benchmark not in held:
        corr_input = asset_returns.join(bench.rename(cfg.benchmark))
    correlation = correlation_matrix(corr_input) if corr_input.shape[1] >= 2 else None
    concentration = concentration_summary(weights.weights, asset_returns)
    try:
        contributions = risk_contributions(weights.weights, asset_returns, ppy)
    except InsufficientDataError as exc:
        contributions = None
        notes.append(str(exc))

    # --- market periods ---------------------------------------------------- #
    try:
        years = calendar_year_risk(port, cfg.risk_free_rate, ppy, cfg.confidence, min_periods=max(3, round(ppy / 12)))
    except InsufficientDataError:
        years = None
    bench_years = period_returns(bench, "Y") if bench is not None else None

    regimes = None
    if ppy >= 252:
        reference = bench if bench is not None else port
        if bench is None:
            notes.append("No benchmark selected: market regimes are classified from the portfolio's own volatility.")
        try:
            regimes = volatility_regime_analysis(
                port, reference, cfg.regime_window, cfg.regime_thresholds, ppy, cfg.confidence
            )
        except InsufficientDataError:
            notes.append("Not enough history to classify volatility regimes.")

    return AnalysisResult(
        config=cfg,
        weights=weights,
        quality=quality,
        frequency=frequency,
        periods_per_year=ppy,
        prices=aligned,
        asset_returns=asset_returns,
        portfolio=portfolio,
        benchmark_returns=bench,
        summary=summary,
        relative=relative,
        drawdown=drawdown_stats(port),
        benchmark_drawdown=drawdown_stats(bench) if bench is not None else None,
        var_table=var_table,
        growth=growth,
        drawdowns=drawdowns,
        rolling_volatility=roll_vol,
        rolling_sharpe=roll_sharpe,
        backtest=backtest,
        correlation=correlation,
        concentration=concentration,
        risk_contributions=contributions,
        calendar_years=years,
        benchmark_years=bench_years,
        regimes=regimes,
        monthly_returns=_monthly_table(port),
        notes=list(weights.messages) + notes,
    )


# --------------------------------------------------------------------------- #
# Export helpers
# --------------------------------------------------------------------------- #
def returns_export(result: AnalysisResult) -> pd.DataFrame:
    """Per-period returns, wealth and drawdown for download."""
    out = pd.DataFrame({"portfolio_return": result.portfolio_returns})
    if result.benchmark_returns is not None:
        out["benchmark_return"] = result.benchmark_returns
    out["portfolio_value"] = result.growth["Portfolio"].iloc[1:]
    out["portfolio_drawdown"] = result.drawdowns["Portfolio"]
    out = out.join(result.asset_returns.add_suffix("_return"))
    out.index.name = "date"
    return out


def summary_export(result: AnalysisResult) -> pd.DataFrame:
    """Long-format table of every headline number plus the assumptions used."""
    rows: list[tuple[str, str, object]] = []
    for metric, values in result.summary.iterrows():
        for column, value in values.items():
            rows.append((column.lower(), metric, value))
    for metric, value in (result.relative or {}).items():
        rows.append(("relative", metric, value))
    for metric, value in result.concentration.items():
        rows.append(("concentration", metric, value))
    dd = result.drawdown
    rows += [
        ("portfolio", "max_drawdown_peak_date", dd.peak_date),
        ("portfolio", "max_drawdown_trough_date", dd.trough_date),
        ("portfolio", "max_drawdown_recovery_date", dd.recovery_date),
    ]
    if result.backtest is not None:
        bt = result.backtest
        rows += [
            ("var_backtest", "forecasts", bt.n_forecasts),
            ("var_backtest", "exceptions", bt.n_exceptions),
            ("var_backtest", "expected_exceptions", bt.expected_exceptions),
            ("var_backtest", "kupiec_p_value", bt.kupiec_p_value),
        ]
    for asset, weight in result.weights.weights.items():
        rows.append(("weights", f"target_{asset}", weight))
    cfg = asdict(result.config)
    cfg.pop("weights")
    cfg["assets"] = ", ".join(cfg["assets"])
    cfg["effective_start"] = result.quality.effective_start
    cfg["effective_end"] = result.quality.effective_end
    cfg["periods_per_year_used"] = result.periods_per_year
    cfg["detected_frequency"] = result.frequency.label
    cfg["var_definition"] = "1-period historical VaR/ES as positive loss fraction of portfolio value"
    for key, value in cfg.items():
        rows.append(("assumption", key, value))
    frame = pd.DataFrame(rows, columns=["section", "metric", "value"])
    frame["value"] = frame["value"].map(lambda v: v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else v)
    return frame.replace({np.nan: None})
