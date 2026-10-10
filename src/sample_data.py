"""Deterministic **synthetic** price dataset for offline use and demos.

The data is generated, not observed. Every ticker carries a ``SYN_`` prefix
so it can never be mistaken for a real security, and the stress episodes are
stylised regimes, not a reproduction of actual market history.

Model (daily log returns, weekday calendar without holidays)::

    r_i,t = mu_i / 252 + beta_i * m_t + gamma_i * b_t + s_i * e_i,t

* ``m_t`` - equity market factor, Student-t(5) shocks scaled to unit variance,
  with regime-dependent volatility and drift.
* ``b_t`` - interest-rate (bond-price) factor; it sells off in the 2022-style
  regime so that bonds and equities fall together.
* ``e_i,t`` - idiosyncratic Student-t(5) shocks.

Deliberate imperfections, used to demonstrate the missing-data handling:

* ``SYN_EM_EQUITY`` starts trading on 2017-03-01 (late listing).
* ``SYN_INTL_EQUITY`` has no price on 26 December each year (local holiday).

Regenerate with ``python -m src.sample_data``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Seed selected (from a scan of 0-299) so the synthetic path has plausible long-run
# statistics: ~9% market CAGR, a ~34% market drawdown, low single-digit bond returns.
SEED = 87
START_DATE = "2015-01-02"
END_DATE = "2024-12-31"
TRADING_DAYS = 252
STUDENT_T_DF = 5

# name: (annual drift, market beta, bond-factor loading, idiosyncratic annual vol)
ASSETS: dict[str, tuple[float, float, float, float]] = {
    "SYN_MARKET": (0.000, 1.00, 0.00, 0.010),
    "SYN_US_EQUITY": (0.010, 1.00, 0.00, 0.040),
    "SYN_TECH_GROWTH": (0.030, 1.30, 0.20, 0.150),
    "SYN_INTL_EQUITY": (-0.015, 0.85, 0.00, 0.080),
    "SYN_EM_EQUITY": (0.000, 1.10, 0.00, 0.140),
    "SYN_TREASURY": (0.020, -0.10, 1.00, 0.010),
    "SYN_CORP_BOND": (0.030, 0.15, 0.70, 0.020),
    "SYN_GOLD": (0.045, 0.05, 0.30, 0.140),
    "SYN_REIT": (0.010, 0.90, 0.40, 0.120),
}

LATE_LISTINGS = {"SYN_EM_EQUITY": "2017-03-01"}
HOLIDAY_GAPS = {"SYN_INTL_EQUITY": (12, 26)}  # (month, day) with no local trading

# Market factor: base annual vol 15%, base annual drift 9%.
MARKET_VOL, MARKET_DRIFT = 0.15, 0.10
# Bond factor: base annual vol 5%, base annual drift 0%.
BOND_VOL, BOND_DRIFT = 0.05, 0.00

# (start, end, market vol multiplier, market annual drift, bond vol multiplier, bond annual drift)
REGIMES = [
    ("2016-01-04", "2016-02-26", 1.5, -0.50, 1.0, 0.10),
    ("2018-10-01", "2018-12-31", 1.7, -0.50, 1.0, 0.08),
    ("2020-02-24", "2020-04-03", 3.5, -2.50, 2.0, 0.30),
    ("2020-04-06", "2020-08-31", 1.6, 0.60, 1.2, 0.05),
    ("2022-01-03", "2022-10-14", 1.5, -0.30, 1.8, -0.15),
]


def _unit_t(rng: np.random.Generator, size) -> np.ndarray:
    """Student-t draws rescaled to unit variance (fat tails, same scale as N(0,1))."""
    return rng.standard_t(STUDENT_T_DF, size=size) / np.sqrt(STUDENT_T_DF / (STUDENT_T_DF - 2))


def generate_sample_prices(seed: int = SEED) -> pd.DataFrame:
    """Return the synthetic adjusted-price table (deterministic for a given seed)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(START_DATE, END_DATE, name="Date")
    n = len(dates)

    m_vol = np.full(n, MARKET_VOL)
    m_drift = np.full(n, MARKET_DRIFT)
    b_vol = np.full(n, BOND_VOL)
    b_drift = np.full(n, BOND_DRIFT)
    for start, end, mv, md, bv, bd in REGIMES:
        mask = (dates >= start) & (dates <= end)
        m_vol[mask] = MARKET_VOL * mv
        m_drift[mask] = md
        b_vol[mask] = BOND_VOL * bv
        b_drift[mask] = bd

    dt = 1.0 / TRADING_DAYS
    market = m_drift * dt + m_vol * np.sqrt(dt) * _unit_t(rng, n)
    bond = b_drift * dt + b_vol * np.sqrt(dt) * _unit_t(rng, n)
    idio = _unit_t(rng, (n, len(ASSETS)))

    columns = {}
    for j, (name, (mu, beta, gamma, s)) in enumerate(ASSETS.items()):
        log_ret = mu * dt + beta * market + gamma * bond + s * np.sqrt(dt) * idio[:, j]
        log_ret[0] = 0.0  # first date is the 100.0 base
        columns[name] = 100.0 * np.exp(np.cumsum(log_ret))

    prices = pd.DataFrame(columns, index=dates).round(4)

    for name, listing_date in LATE_LISTINGS.items():
        listed = prices.index >= listing_date
        prices.loc[listed, name] = (prices.loc[listed, name] / prices.loc[listed, name].iloc[0] * 100.0).round(4)
        prices.loc[~listed, name] = np.nan
    for name, (month, day) in HOLIDAY_GAPS.items():
        holiday = (prices.index.month == month) & (prices.index.day == day)
        prices.loc[holiday, name] = np.nan
    return prices


def write_sample_prices(path: Path | str, seed: int = SEED) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_sample_prices(seed).to_csv(path, date_format="%Y-%m-%d", float_format="%.4f")
    return path


if __name__ == "__main__":
    from .data_loader import SAMPLE_DATA_PATH

    out = write_sample_prices(SAMPLE_DATA_PATH)
    print(f"Wrote synthetic sample prices to {out}")
