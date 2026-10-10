"""Portfolio weight validation and return construction.

Rebalancing assumption (explicit, user-selectable):

* ``daily``     - weights are reset to target at every close (constant-mix).
                  Portfolio return = sum_i w_i * r_i,t. This is the textbook
                  formula, but it implies trading every day.
* ``monthly`` / ``quarterly`` / ``annually`` - weights drift with relative
                  performance and are reset to target at the close of the last
                  trading day of each calendar period.
* ``none``      - buy-and-hold: weights drift for the whole sample.

Transaction costs, taxes and cash drag are ignored. Turnover at each
rebalance is reported so the trading implied by each choice is visible.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .exceptions import PortfolioError

REBALANCE_OPTIONS: dict[str, str] = {
    "daily": "Daily (constant weights)",
    "monthly": "Monthly",
    "quarterly": "Quarterly",
    "annually": "Annually",
    "none": "Never (buy-and-hold)",
}
_PERIOD_CODES = {"monthly": "M", "quarterly": "Q", "annually": "Y"}


@dataclass(frozen=True)
class WeightValidation:
    weights: pd.Series
    original_sum: float
    was_normalized: bool
    messages: tuple[str, ...]


def validate_weights(
    weights: Mapping[str, float] | pd.Series,
    available_assets: Iterable[str] | None = None,
    *,
    normalize: bool = True,
    tolerance: float = 1e-6,
) -> WeightValidation:
    """Validate long-only portfolio weights.

    Rejected (``PortfolioError``): no weights, unknown assets, non-finite
    values, negative weights (short positions are out of scope), and weights
    that sum to zero.

    If the weights do not sum to 1 within ``tolerance``:
      * ``normalize=True``  - rescale proportionally (w_i / sum w) and say so;
      * ``normalize=False`` - reject.

    Zero-weight assets are removed (and reported) so they do not affect the
    sample window or the diversification statistics.
    """
    series = pd.Series(weights, dtype=float) if not isinstance(weights, pd.Series) else weights.astype(float)
    series.index = series.index.astype(str)
    if series.empty:
        raise PortfolioError("No portfolio weights were provided.")
    if series.index.duplicated().any():
        raise PortfolioError(f"Duplicate assets in weights: {series.index[series.index.duplicated()].tolist()}.")
    if available_assets is not None:
        unknown = sorted(set(series.index) - set(available_assets))
        if unknown:
            raise PortfolioError(f"Weights reference assets without price data: {unknown}.")
    if not np.isfinite(series.to_numpy()).all():
        raise PortfolioError("Weights must be finite numbers.")
    negative = series[series < 0]
    if not negative.empty:
        raise PortfolioError(
            f"Negative weights are not supported (long-only portfolios): {negative.round(4).to_dict()}."
        )

    total = float(series.sum())
    if total <= tolerance:
        raise PortfolioError("Weights sum to zero; assign a positive weight to at least one asset.")

    messages: list[str] = []
    zero = series.index[series == 0].tolist()
    if zero:
        messages.append(f"Assets with zero weight were excluded: {', '.join(zero)}.")
        series = series[series > 0]

    was_normalized = abs(total - 1.0) > tolerance
    if was_normalized:
        if not normalize:
            raise PortfolioError(f"Weights must sum to 100%; they sum to {total:.4%}.")
        series = series / total
        messages.append(f"Weights summed to {total:.2%} and were rescaled proportionally to 100%.")

    return WeightValidation(series, total, was_normalized, tuple(messages))


@dataclass
class PortfolioResult:
    returns: pd.Series            # portfolio simple return for each period
    weights_history: pd.DataFrame  # weights held at the start of each period
    end_weights: pd.Series        # drifted weights after the final period
    turnover: pd.Series           # one-way turnover at each rebalance date
    target_weights: pd.Series
    rebalance: str


def construct_portfolio(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series,
    rebalance: str = "monthly",
) -> PortfolioResult:
    """Simulate a long-only portfolio under the chosen rebalancing rule.

    For each period t, with start-of-period weights w_t and asset returns r_t::

        R_p,t      = w_t . r_t
        w_drift    = w_t * (1 + r_t) / (1 + R_p,t)       # weights after the move
        w_{t+1}    = target   if t closes a rebalancing period
                     w_drift  otherwise
        turnover_t = 0.5 * sum |target - w_drift|          # one-way, at rebalance

    Only information available at the close of t is used to set w_{t+1}.
    """
    if rebalance not in REBALANCE_OPTIONS:
        raise PortfolioError(f"Unknown rebalancing rule {rebalance!r}; choose from {list(REBALANCE_OPTIONS)}.")
    target = validate_weights(weights, asset_returns.columns, normalize=False).weights
    if asset_returns.empty:
        raise PortfolioError("Asset returns are empty.")
    sub = asset_returns.loc[:, target.index]
    if sub.isna().any().any():
        raise PortfolioError("Asset returns contain missing values; align prices before building the portfolio.")

    r = sub.to_numpy(dtype=float)
    w_target = target.to_numpy(dtype=float)
    n_periods = len(sub)
    dates = sub.index

    if rebalance in _PERIOD_CODES:
        keys = dates.to_period(_PERIOD_CODES[rebalance])
        closes_period = np.append(keys[1:] != keys[:-1], False)
    elif rebalance == "daily":
        closes_period = np.ones(n_periods, dtype=bool)
        closes_period[-1] = False
    else:
        closes_period = np.zeros(n_periods, dtype=bool)

    port = np.empty(n_periods)
    w_hist = np.empty_like(r)
    turnover_dates, turnover_values = [], []
    w = w_target.copy()
    w_drift = w

    for t in range(n_periods):
        w_hist[t] = w
        gross = 1.0 + r[t]
        port_gross = float(w @ gross)
        if port_gross <= 0:
            raise PortfolioError(f"Portfolio value reached zero on {dates[t]:%Y-%m-%d}.")
        port[t] = port_gross - 1.0
        w_drift = w * gross / port_gross
        if closes_period[t]:
            turnover_dates.append(dates[t])
            turnover_values.append(0.5 * float(np.abs(w_target - w_drift).sum()))
            w = w_target.copy()
        else:
            w = w_drift

    return PortfolioResult(
        returns=pd.Series(port, index=dates, name="Portfolio"),
        weights_history=pd.DataFrame(w_hist, index=dates, columns=target.index),
        end_weights=pd.Series(w_drift, index=target.index, name="end_weight"),
        turnover=pd.Series(turnover_values, index=pd.DatetimeIndex(turnover_dates), name="turnover", dtype=float),
        target_weights=target,
        rebalance=rebalance,
    )
