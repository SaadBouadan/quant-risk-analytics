"""Risk statistics: VaR, Expected Shortfall, rolling risk, diversification.

Sign convention - read this before using the numbers:

* VaR and Expected Shortfall are reported as **positive loss fractions of
  portfolio value over one return period** (one trading day for daily data).
  ``historical_var(r, 0.95) == 0.021`` means "on 95% of days in the sample the
  loss was no larger than 2.1% of portfolio value".
* Multiply by the portfolio value to get a monetary amount.
* The underlying return quantile is the *negative* of VaR (here -2.1%).

Historical VaR definition (empirical / "inverted CDF" quantile)::

    k        = ceil(n * (1 - c))          # number of tail observations
    VaR_c    = - (k-th smallest return)
    ES_c     = - mean(k smallest returns)  # always >= VaR_c

All rolling statistics use trailing windows only: the value on date t uses
returns up to and including t, never later data. Backtests additionally lag
the VaR estimate by one period so it is known before the loss it is tested on.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from .exceptions import InsufficientDataError
from .performance import (
    TRADING_DAYS_PER_YEAR,
    ZERO_TOLERANCE,
    _as_series,
    annualized_volatility,
    cumulative_return,
    max_drawdown,
    periodic_risk_free_rate,
    sharpe_ratio,
)


# --------------------------------------------------------------------------- #
# Value at Risk and Expected Shortfall
# --------------------------------------------------------------------------- #
def _validate_confidence(confidence: float) -> float:
    if not 0.5 <= confidence < 1.0:
        raise ValueError(f"Confidence level must be in [0.5, 1); got {confidence}.")
    return float(confidence)


def tail_count(n_obs: int, confidence: float) -> int:
    """Number of observations in the loss tail: ceil(n * (1 - c)), at least 1.

    The product is rounded first so floating-point noise (100 * 0.05 =
    5.000000000000004) does not push the count up by one.
    """
    _validate_confidence(confidence)
    if n_obs < 1:
        raise InsufficientDataError("VaR needs at least one return observation.")
    return max(1, math.ceil(round(n_obs * (1.0 - confidence), 9)))


def historical_var(returns, confidence: float = 0.95) -> float:
    """One-period historical VaR as a positive loss fraction."""
    r = np.sort(_as_series(returns).to_numpy())
    k = tail_count(len(r), confidence)
    return float(-r[k - 1])


def historical_expected_shortfall(returns, confidence: float = 0.95) -> float:
    """One-period historical Expected Shortfall (CVaR): average loss in the worst k periods."""
    r = np.sort(_as_series(returns).to_numpy())
    k = tail_count(len(r), confidence)
    return float(-r[:k].mean())


def parametric_var(returns, confidence: float = 0.95) -> float:
    """Gaussian (variance-covariance) VaR: -(mu + z_{1-c} * sigma). For comparison only."""
    r = _as_series(returns, min_obs=2)
    z = NormalDist().inv_cdf(1.0 - _validate_confidence(confidence))
    return float(-(r.mean() + z * r.std(ddof=1)))


def parametric_expected_shortfall(returns, confidence: float = 0.95) -> float:
    """Gaussian ES: -(mu - sigma * phi(z) / (1 - c)). For comparison only."""
    r = _as_series(returns, min_obs=2)
    c = _validate_confidence(confidence)
    z = NormalDist().inv_cdf(1.0 - c)
    return float(-(r.mean() - r.std(ddof=1) * NormalDist().pdf(z) / (1.0 - c)))


def var_es_table(
    returns,
    confidences: Sequence[float] = (0.90, 0.95, 0.99),
    portfolio_value: float = 1_000_000.0,
) -> pd.DataFrame:
    """VaR/ES at several confidence levels, as fractions and monetary amounts."""
    r = _as_series(returns, min_obs=2)
    rows = []
    for c in confidences:
        h_var, h_es = historical_var(r, c), historical_expected_shortfall(r, c)
        rows.append(
            {
                "confidence": c,
                "tail_observations": tail_count(len(r), c),
                "historical_var": h_var,
                "historical_es": h_es,
                "historical_var_amount": h_var * portfolio_value,
                "historical_es_amount": h_es * portfolio_value,
                "gaussian_var": parametric_var(r, c),
                "gaussian_es": parametric_expected_shortfall(r, c),
            }
        )
    return pd.DataFrame(rows).set_index("confidence")


# --------------------------------------------------------------------------- #
# Rolling (trailing-window) statistics
# --------------------------------------------------------------------------- #
def _check_window(window: int, n: int) -> None:
    if window < 2:
        raise ValueError("Rolling window must be at least 2 periods.")
    if window > n:
        raise InsufficientDataError(f"Rolling window ({window}) is longer than the sample ({n} periods).")


def rolling_volatility(returns, window: int = 63, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    """Trailing annualised volatility; NaN until ``window`` observations exist."""
    r = _as_series(returns)
    _check_window(window, len(r))
    return (r.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)).rename("rolling_volatility")


def rolling_sharpe(
    returns,
    window: int = 126,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Trailing annualised Sharpe ratio (NaN where volatility is zero)."""
    r = _as_series(returns)
    _check_window(window, len(r))
    excess = r - periodic_risk_free_rate(risk_free_rate, periods_per_year)
    mean = excess.rolling(window).mean()
    sd = excess.rolling(window).std(ddof=1)
    sharpe = mean / sd.where(sd > ZERO_TOLERANCE) * np.sqrt(periods_per_year)
    return sharpe.rename("rolling_sharpe")


def rolling_historical_var(returns, window: int = 252, confidence: float = 0.95) -> pd.Series:
    """Trailing historical VaR, using exactly the same quantile rule as ``historical_var``."""
    r = _as_series(returns)
    _check_window(window, len(r))
    k = tail_count(window, confidence)
    return r.rolling(window).apply(lambda x: -np.partition(x, k - 1)[k - 1], raw=True).rename("rolling_var")


# --------------------------------------------------------------------------- #
# VaR backtest
# --------------------------------------------------------------------------- #
def kupiec_pof_test(n_obs: int, n_exceptions: int, confidence: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures likelihood-ratio test.

    H0: the true exception probability equals p = 1 - c.
    LR = -2 ln[(1-p)^(n-x) p^x] + 2 ln[(1-x/n)^(n-x) (x/n)^x]  ~  chi^2(1).
    Returns (LR statistic, p-value). A small p-value (e.g. < 0.05) means the
    observed exception frequency is inconsistent with the VaR model.
    """
    if n_obs < 1 or not 0 <= n_exceptions <= n_obs:
        raise ValueError("Need n_obs >= 1 and 0 <= n_exceptions <= n_obs.")
    p = 1.0 - _validate_confidence(confidence)
    x, n = n_exceptions, n_obs
    phat = x / n

    def _xlogy(a: float, b: float) -> float:
        return 0.0 if a == 0 else a * math.log(b)

    log_null = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    log_alt = _xlogy(n - x, 1 - phat) + _xlogy(x, phat)
    lr = max(0.0, -2.0 * (log_null - log_alt))
    p_value = math.erfc(math.sqrt(lr / 2.0))  # survival function of chi^2 with 1 d.o.f.
    return lr, p_value


@dataclass(frozen=True)
class VaRBacktest:
    confidence: float
    window: int
    n_forecasts: int
    n_exceptions: int
    expected_exceptions: float
    exception_rate: float
    kupiec_lr: float
    kupiec_p_value: float
    var_forecast: pd.Series   # VaR known at the start of each date (estimated through t-1)
    exceptions: pd.Series     # True where the realised loss exceeded the forecast


def backtest_historical_var(returns, window: int = 252, confidence: float = 0.95) -> VaRBacktest:
    """Out-of-sample test of rolling historical VaR.

    The VaR used for date t is estimated from the ``window`` returns ending at
    t-1, so the forecast never sees the return it is evaluated against.
    """
    r = _as_series(returns)
    if len(r) <= window:
        raise InsufficientDataError(
            f"A VaR backtest with a {window}-period window needs more than {window} observations."
        )
    forecast = rolling_historical_var(r, window, confidence).shift(1).dropna()
    realised_loss = -r.loc[forecast.index]
    exceptions = (realised_loss > forecast).rename("exception")
    n, x = len(forecast), int(exceptions.sum())
    lr, p_value = kupiec_pof_test(n, x, confidence)
    return VaRBacktest(
        confidence=confidence,
        window=window,
        n_forecasts=n,
        n_exceptions=x,
        expected_exceptions=n * (1.0 - confidence),
        exception_rate=x / n,
        kupiec_lr=lr,
        kupiec_p_value=p_value,
        var_forecast=forecast.rename("var_forecast"),
        exceptions=exceptions,
    )


# --------------------------------------------------------------------------- #
# Correlation, concentration and diversification
# --------------------------------------------------------------------------- #
def correlation_matrix(asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of aligned asset returns."""
    if asset_returns.isna().any().any():
        raise ValueError("Asset returns contain missing values; align the data first.")
    if len(asset_returns) < 3:
        raise InsufficientDataError("At least three observations are needed for a correlation matrix.")
    return asset_returns.corr(method="pearson")


def _weights_array(weights: Mapping[str, float] | pd.Series) -> pd.Series:
    w = pd.Series(weights, dtype=float)
    if (w < 0).any() or not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError("Weights must be non-negative and sum to 1.")
    return w


def herfindahl_index(weights: Mapping[str, float] | pd.Series) -> float:
    """HHI = sum(w_i^2). 1/N for equal weights, 1.0 for a single asset."""
    w = _weights_array(weights)
    return float((w**2).sum())


def effective_number_of_assets(weights: Mapping[str, float] | pd.Series) -> float:
    """1 / HHI: the number of equally weighted assets with the same concentration."""
    return 1.0 / herfindahl_index(weights)


def _portfolio_cov(weights, asset_returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    w = _weights_array(weights)
    missing = [a for a in w.index if a not in asset_returns.columns]
    if missing:
        raise ValueError(f"No returns for weighted assets {missing}.")
    sub = asset_returns.loc[:, w.index]
    if len(sub) < 2:
        raise InsufficientDataError("At least two observations are needed for a covariance matrix.")
    return w, sub.cov(ddof=1)


def diversification_ratio(weights, asset_returns: pd.DataFrame) -> float:
    """sum(w_i * sigma_i) / sigma_p. Equals 1 with no diversification benefit; higher is better."""
    w, cov = _portfolio_cov(weights, asset_returns)
    port_var = float(w.to_numpy() @ cov.to_numpy() @ w.to_numpy())
    if port_var <= ZERO_TOLERANCE**2:
        return float("nan")
    weighted_vol = float((w * np.sqrt(np.diag(cov.to_numpy()))).sum())
    return weighted_vol / math.sqrt(port_var)


def risk_contributions(
    weights, asset_returns: pd.DataFrame, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> pd.DataFrame:
    """Euler decomposition of portfolio volatility.

    RC_i = w_i * (Sigma w)_i / sigma_p, so sum_i RC_i = sigma_p. The
    ``pct_of_risk`` column (RC_i / sigma_p) sums to 100%. Uses the sample
    covariance and the target weights (exact for constant-mix portfolios,
    an approximation when weights drift between rebalances).
    """
    w, cov = _portfolio_cov(weights, asset_returns)
    w_vec, sigma = w.to_numpy(), cov.to_numpy()
    port_vol = math.sqrt(float(w_vec @ sigma @ w_vec))
    if port_vol <= ZERO_TOLERANCE:
        raise InsufficientDataError("Portfolio volatility is zero; risk contributions are undefined.")
    contrib = w_vec * (sigma @ w_vec) / port_vol
    scale = math.sqrt(periods_per_year)
    return pd.DataFrame(
        {
            "weight": w_vec,
            "asset_volatility": np.sqrt(np.diag(sigma)) * scale,
            "risk_contribution": contrib * scale,
            "pct_of_risk": contrib / port_vol,
        },
        index=w.index,
    )


def concentration_summary(weights, asset_returns: pd.DataFrame) -> dict[str, float]:
    w = _weights_array(weights)
    return {
        "n_assets": int((w > 0).sum()),
        "max_weight": float(w.max()),
        "herfindahl_index": herfindahl_index(w),
        "effective_number_of_assets": effective_number_of_assets(w),
        "diversification_ratio": diversification_ratio(w, asset_returns),
    }


# --------------------------------------------------------------------------- #
# Risk across market periods (historical, descriptive)
# --------------------------------------------------------------------------- #
def calendar_year_risk(
    returns,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    confidence: float = 0.95,
    min_periods: int = 20,
) -> pd.DataFrame:
    """Return and risk statistics for each calendar year in the sample.

    Drawdowns are measured within the year (starting from the value on the
    first day of that year). Years with fewer than ``min_periods`` observations
    are skipped; partial years are flagged.
    """
    r = _as_series(returns, min_obs=2)
    rows = []
    for year, chunk in r.groupby(r.index.year):
        if len(chunk) < min_periods:
            continue
        rows.append(
            {
                "year": int(year),
                "observations": len(chunk),
                "partial_year": len(chunk) < 0.9 * periods_per_year,
                "return": cumulative_return(chunk),
                "volatility": annualized_volatility(chunk, periods_per_year),
                "sharpe_ratio": sharpe_ratio(chunk, risk_free_rate, periods_per_year),
                "max_drawdown": max_drawdown(chunk),
                "historical_var": historical_var(chunk, confidence),
                "worst_period": float(chunk.min()),
            }
        )
    if not rows:
        raise InsufficientDataError("No calendar year has enough observations.")
    return pd.DataFrame(rows).set_index("year")


REGIME_ORDER = ("Calm", "Normal", "Stressed")


def classify_volatility_regimes(
    reference_returns,
    window: int = 21,
    thresholds: tuple[float, float] = (0.15, 0.25),
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Label each date Calm / Normal / Stressed from *prior* realised volatility.

    The label for date t uses the annualised volatility of the ``window``
    returns ending at t-1 (shifted by one period), so it is known before t's
    return occurs - no look-ahead. Thresholds are fixed numbers, not sample
    quantiles, for the same reason. Dates without enough history are NaN.
    """
    low, high = thresholds
    if not 0 < low < high:
        raise ValueError("Thresholds must satisfy 0 < low < high.")
    vol = rolling_volatility(reference_returns, window, periods_per_year).shift(1)
    labels = pd.Series(np.nan, index=vol.index, dtype="object")
    labels[vol < low] = "Calm"
    labels[(vol >= low) & (vol < high)] = "Normal"
    labels[vol >= high] = "Stressed"
    return labels.rename("regime")


def volatility_regime_analysis(
    portfolio_returns,
    reference_returns,
    window: int = 21,
    thresholds: tuple[float, float] = (0.15, 0.25),
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Portfolio behaviour conditional on the market's prior volatility regime.

    Regime days are not contiguous, so returns are annualised arithmetically
    (mean x P) rather than compounded.
    """
    p = _as_series(portfolio_returns, min_obs=2)
    ref = _as_series(reference_returns, min_obs=2)
    if not p.index.equals(ref.index):
        raise ValueError("Portfolio and reference returns must cover identical dates.")
    regimes = classify_volatility_regimes(ref, window, thresholds, periods_per_year)

    rows = []
    for regime in REGIME_ORDER:
        chunk = p[regimes == regime]
        if len(chunk) < 2:
            continue
        rows.append(
            {
                "regime": regime,
                "days": len(chunk),
                "share_of_days": len(chunk) / int(regimes.notna().sum()),
                "annualized_mean_return": float(chunk.mean() * periods_per_year),
                "annualized_volatility": annualized_volatility(chunk, periods_per_year),
                "historical_var": historical_var(chunk, confidence),
                "historical_es": historical_expected_shortfall(chunk, confidence),
                "worst_period": float(chunk.min()),
            }
        )
    if not rows:
        raise InsufficientDataError("Not enough history to classify volatility regimes.")
    return pd.DataFrame(rows).set_index("regime")


def risk_summary(
    returns,
    confidence: float = 0.95,
    portfolio_value: float = 1_000_000.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Headline risk numbers for the dashboard and downloadable report."""
    r = _as_series(returns, min_obs=2)
    h_var = historical_var(r, confidence)
    h_es = historical_expected_shortfall(r, confidence)
    return {
        "confidence": confidence,
        "historical_var": h_var,
        "historical_es": h_es,
        "historical_var_amount": h_var * portfolio_value,
        "historical_es_amount": h_es * portfolio_value,
        "gaussian_var": parametric_var(r, confidence),
        "tail_observations": tail_count(len(r), confidence),
        "annualized_volatility": annualized_volatility(r, periods_per_year),
        "skewness": float(r.skew()),
        "excess_kurtosis": float(r.kurt()),
    }
