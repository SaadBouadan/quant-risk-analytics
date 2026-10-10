"""Return-based performance statistics.

Conventions (see docs/METHODOLOGY.md):

* Inputs are periodic **simple** returns (0.01 = +1%) indexed by date.
* ``periods_per_year`` is 252 for daily trading-day data.
* Annualised return is **geometric** (CAGR): (1 + R_cum)^(P/N) - 1.
* Annualised volatility is the sample standard deviation (ddof=1) x sqrt(P),
  which assumes returns are serially uncorrelated.
* Sharpe ratio = mean(r - rf_p) / std(r - rf_p) x sqrt(P), where the annual
  risk-free rate is converted to a per-period rate by compounding:
  rf_p = (1 + rf)^(1/P) - 1.
* Undefined statistics (e.g. a Sharpe ratio with zero volatility) are returned
  as NaN rather than infinity or zero.
* Inputs containing NaN are rejected - nothing is dropped silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .exceptions import DataValidationError, InsufficientDataError

TRADING_DAYS_PER_YEAR = 252
ZERO_TOLERANCE = 1e-12


def _as_series(returns, min_obs: int = 1, name: str = "returns") -> pd.Series:
    series = returns if isinstance(returns, pd.Series) else pd.Series(returns, dtype=float)
    series = series.astype(float)
    if series.isna().any():
        raise DataValidationError(f"{name} contain missing values; clean the data first.")
    if len(series) < min_obs:
        raise InsufficientDataError(f"At least {min_obs} observation(s) required, got {len(series)}.")
    return series


def cumulative_return(returns) -> float:
    """Total compounded return: prod(1 + r_t) - 1."""
    r = _as_series(returns)
    return float(np.prod(1.0 + r.to_numpy()) - 1.0)


def wealth_index(returns, initial_value: float = 1.0) -> pd.Series:
    """Value of ``initial_value`` invested at the start, at the end of each period."""
    r = _as_series(returns)
    return initial_value * (1.0 + r).cumprod()


def growth_of_investment(returns, initial_value: float = 10_000.0, start_date=None) -> pd.Series:
    """Wealth index with the starting value prepended at ``start_date`` (for charts)."""
    wealth = wealth_index(returns, initial_value)
    if start_date is None:
        return wealth
    start = pd.Series([initial_value], index=pd.DatetimeIndex([pd.Timestamp(start_date)]))
    return pd.concat([start, wealth]).rename(wealth.name)


def annualized_return(returns, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Geometric average annual return (CAGR): (1 + R_cum)^(P / N) - 1."""
    r = _as_series(returns)
    growth = 1.0 + cumulative_return(r)
    if growth <= 0:
        return -1.0
    return float(growth ** (periods_per_year / len(r)) - 1.0)


def annualized_volatility(returns, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Sample standard deviation of periodic returns x sqrt(periods_per_year)."""
    r = _as_series(returns, min_obs=2)
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def periodic_risk_free_rate(annual_rate: float, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Convert an annual (effective) risk-free rate to a per-period rate by compounding."""
    if annual_rate <= -1:
        raise ValueError("The annual risk-free rate must be greater than -100%.")
    return float((1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0)


def sharpe_ratio(
    returns,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sharpe ratio of periodic excess returns.

    ``risk_free_rate`` is an annual effective rate (0.03 = 3%). Returns NaN
    when the excess returns have (numerically) zero dispersion.
    """
    r = _as_series(returns, min_obs=2)
    excess = r - periodic_risk_free_rate(risk_free_rate, periods_per_year)
    sd = float(excess.std(ddof=1))
    if sd <= ZERO_TOLERANCE:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def drawdown_series(returns) -> pd.Series:
    """Percentage decline of the wealth index from its running peak (<= 0).

    The initial investment (value 1 before the first return) counts as a peak,
    so a loss in the very first period is a drawdown.
    """
    wealth = wealth_index(returns)
    peak = wealth.cummax().clip(lower=1.0)
    return (wealth / peak - 1.0).rename("drawdown")


def max_drawdown(returns) -> float:
    """Largest peak-to-trough decline, as a negative fraction (e.g. -0.25)."""
    return float(min(drawdown_series(returns).min(), 0.0))


@dataclass(frozen=True)
class DrawdownStats:
    max_drawdown: float                    # negative fraction, 0 if never below a peak
    peak_date: pd.Timestamp | None         # None => peak was the initial investment
    trough_date: pd.Timestamp | None
    recovery_date: pd.Timestamp | None     # None => not recovered by the end of the sample
    peak_to_trough_periods: int
    peak_to_recovery_periods: int | None
    longest_underwater_periods: int        # longest run of consecutive periods below a prior peak
    currently_underwater: bool
    current_drawdown: float


def drawdown_stats(returns) -> DrawdownStats:
    """Depth, dates and duration (in periods) of drawdowns."""
    r = _as_series(returns)
    values = np.concatenate([[1.0], (1.0 + r.to_numpy()).cumprod()])  # position 0 = initial value
    peaks = np.maximum.accumulate(values)
    dd = values / peaks - 1.0
    dates = [None] + list(r.index)

    trough = int(np.argmin(dd))
    mdd = float(dd[trough])
    underwater = dd < -ZERO_TOLERANCE

    longest = run = 0
    for flag in underwater:
        run = run + 1 if flag else 0
        longest = max(longest, run)

    if mdd >= -ZERO_TOLERANCE:
        return DrawdownStats(0.0, None, None, None, 0, None, 0, False, 0.0)

    peak = int(np.argmax(values[: trough + 1]))  # first time the pre-trough high was reached
    recovered = np.flatnonzero(values[trough + 1 :] >= values[peak] * (1 - ZERO_TOLERANCE))
    recovery = trough + 1 + int(recovered[0]) if recovered.size else None

    return DrawdownStats(
        max_drawdown=mdd,
        peak_date=dates[peak],
        trough_date=dates[trough],
        recovery_date=dates[recovery] if recovery is not None else None,
        peak_to_trough_periods=trough - peak,
        peak_to_recovery_periods=(recovery - peak) if recovery is not None else None,
        longest_underwater_periods=longest,
        currently_underwater=bool(underwater[-1]),
        current_drawdown=float(dd[-1]),
    )


def period_returns(returns, freq: str = "M") -> pd.Series:
    """Compound periodic returns into calendar periods ('M' month, 'Q' quarter, 'Y' year).

    Partial periods at the edges of the sample are included as-is.
    """
    r = _as_series(returns)
    if not isinstance(r.index, pd.DatetimeIndex):
        raise DataValidationError("Returns need a DatetimeIndex to group by calendar period.")
    grouped = (1.0 + r).groupby(r.index.to_period(freq)).prod() - 1.0
    return grouped.rename(r.name)


def best_worst_periods(returns) -> dict[str, tuple[pd.Timestamp | pd.Period, float]]:
    """Best/worst single period and best/worst calendar month."""
    r = _as_series(returns)
    monthly = period_returns(r, "M")
    return {
        "best_period": (r.idxmax(), float(r.max())),
        "worst_period": (r.idxmin(), float(r.min())),
        "best_month": (monthly.idxmax(), float(monthly.max())),
        "worst_month": (monthly.idxmin(), float(monthly.min())),
    }


def relative_performance(
    portfolio_returns,
    benchmark_returns,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Benchmark-relative statistics. Both series must share identical dates.

    * excess cumulative return  = R_cum(p) - R_cum(b)
    * excess annualised return  = CAGR(p) - CAGR(b)
    * tracking error            = std(r_p - r_b) x sqrt(P)
    * information ratio         = mean(r_p - r_b) x P / tracking error
    * beta                      = cov(r_p, r_b) / var(r_b)
    * Jensen's alpha (annual)   = [mean(r_p - rf) - beta x mean(r_b - rf)] x P
    """
    p = _as_series(portfolio_returns, min_obs=2, name="portfolio returns")
    b = _as_series(benchmark_returns, min_obs=2, name="benchmark returns")
    if not p.index.equals(b.index):
        raise DataValidationError("Portfolio and benchmark returns must cover identical dates.")

    active = p - b
    te = float(active.std(ddof=1) * np.sqrt(periods_per_year))
    var_b = float(b.var(ddof=1))
    beta = float(np.cov(p, b, ddof=1)[0, 1] / var_b) if var_b > ZERO_TOLERANCE else float("nan")
    sd_p = float(p.std(ddof=1))
    corr = float(np.corrcoef(p, b)[0, 1]) if sd_p > ZERO_TOLERANCE and var_b > ZERO_TOLERANCE else float("nan")
    rf_p = periodic_risk_free_rate(risk_free_rate, periods_per_year)
    alpha = float("nan")
    if np.isfinite(beta):
        alpha = float(((p - rf_p).mean() - beta * (b - rf_p).mean()) * periods_per_year)

    return {
        "excess_cumulative_return": cumulative_return(p) - cumulative_return(b),
        "excess_annualized_return": annualized_return(p, periods_per_year) - annualized_return(b, periods_per_year),
        "tracking_error": te,
        "information_ratio": float(active.mean() * periods_per_year / te) if te > ZERO_TOLERANCE else float("nan"),
        "beta": beta,
        "correlation": corr,
        "jensens_alpha": alpha,
    }


def performance_summary(
    returns,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Headline statistics used by the dashboard and the downloadable report."""
    r = _as_series(returns, min_obs=2)
    dd = drawdown_stats(r)
    bw = best_worst_periods(r)
    return {
        "cumulative_return": cumulative_return(r),
        "annualized_return": annualized_return(r, periods_per_year),
        "annualized_volatility": annualized_volatility(r, periods_per_year),
        "sharpe_ratio": sharpe_ratio(r, risk_free_rate, periods_per_year),
        "max_drawdown": dd.max_drawdown,
        "longest_underwater_periods": dd.longest_underwater_periods,
        "best_period_return": bw["best_period"][1],
        "worst_period_return": bw["worst_period"][1],
        "best_month_return": bw["best_month"][1],
        "worst_month_return": bw["worst_month"][1],
        "positive_period_share": float((r > 0).mean()),
        "n_periods": len(r),
        "years": len(r) / periods_per_year,
    }
