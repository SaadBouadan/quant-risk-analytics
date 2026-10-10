"""Performance statistics, verified against hand-calculated values."""

import math

import numpy as np
import pandas as pd
import pytest

from src.exceptions import DataValidationError, InsufficientDataError
from src.performance import (
    annualized_return,
    annualized_volatility,
    best_worst_periods,
    cumulative_return,
    drawdown_series,
    drawdown_stats,
    growth_of_investment,
    max_drawdown,
    performance_summary,
    period_returns,
    periodic_risk_free_rate,
    relative_performance,
    sharpe_ratio,
    wealth_index,
)


def series(values, start="2024-01-01", freq="B"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=freq), dtype=float)


# --------------------------------------------------------------------------- #
# Cumulative return
# --------------------------------------------------------------------------- #
class TestCumulativeReturn:
    def test_compounds_returns(self):
        # 1.10 * 0.95 * 1.02 = 1.0659
        assert cumulative_return(series([0.10, -0.05, 0.02])) == pytest.approx(0.0659, abs=1e-12)

    def test_gain_then_equal_loss_is_not_zero(self):
        # +10% then -10%: 1.1 * 0.9 = 0.99 -> -1%, not 0%
        assert cumulative_return(series([0.10, -0.10])) == pytest.approx(-0.01, abs=1e-12)

    def test_single_observation(self):
        assert cumulative_return(series([0.037])) == pytest.approx(0.037)

    def test_wealth_index_tracks_value(self):
        wealth = wealth_index(series([0.10, -0.05, 0.02]), initial_value=100.0)
        assert wealth.tolist() == pytest.approx([110.0, 104.5, 106.59])

    def test_growth_of_investment_prepends_start_value(self):
        r = series([0.10, -0.05])
        growth = growth_of_investment(r, 10_000, start_date="2023-12-29")
        assert growth.index[0] == pd.Timestamp("2023-12-29")
        assert growth.tolist() == pytest.approx([10_000.0, 11_000.0, 10_450.0])

    def test_missing_values_are_rejected_not_dropped(self):
        with pytest.raises(DataValidationError):
            cumulative_return(series([0.01, np.nan, 0.02]))

    def test_empty_input_rejected(self):
        with pytest.raises(InsufficientDataError):
            cumulative_return(pd.Series([], dtype=float))


# --------------------------------------------------------------------------- #
# Annualised return and volatility
# --------------------------------------------------------------------------- #
class TestAnnualization:
    def test_two_years_of_daily_data(self):
        # 21% over 504 trading days (2 years) -> 10% per year: 1.21 ** (1/2) = 1.10
        daily = 1.21 ** (1 / 504) - 1
        assert annualized_return(series([daily] * 504)) == pytest.approx(0.10, abs=1e-10)

    def test_half_year_is_scaled_up_geometrically(self):
        # 5% over 126 days -> 1.05 ** 2 - 1 = 10.25% (not 2 x 5% = 10%)
        daily = 1.05 ** (1 / 126) - 1
        assert annualized_return(series([daily] * 126)) == pytest.approx(0.1025, abs=1e-10)

    def test_monthly_periods_per_year(self):
        # 44% over 24 months -> 1.44 ** (12/24) - 1 = 20%
        monthly = 1.44 ** (1 / 24) - 1
        r = series([monthly] * 24, freq="ME")
        assert annualized_return(r, periods_per_year=12) == pytest.approx(0.20, abs=1e-10)

    def test_total_loss(self):
        assert annualized_return(series([0.5, -1.0])) == -1.0

    def test_volatility_hand_calculation(self):
        # mean 0, squared deviations 4 x 1e-4, sample variance 4e-4 / 3
        r = series([0.01, -0.01, 0.01, -0.01])
        expected_daily = 0.01 * math.sqrt(4 / 3)
        assert annualized_volatility(r) == pytest.approx(expected_daily * math.sqrt(252), rel=1e-12)
        assert annualized_volatility(r, periods_per_year=12) == pytest.approx(expected_daily * math.sqrt(12))

    def test_volatility_requires_two_observations(self):
        with pytest.raises(InsufficientDataError):
            annualized_volatility(series([0.01]))

    def test_constant_returns_have_zero_volatility(self):
        r = series([0.001] * 10)
        assert annualized_volatility(r) == pytest.approx(0.0, abs=1e-12)
        assert cumulative_return(r) == pytest.approx(1.001**10 - 1)
        assert annualized_return(r) == pytest.approx(1.001**252 - 1)


# --------------------------------------------------------------------------- #
# Sharpe ratio
# --------------------------------------------------------------------------- #
class TestSharpe:
    RETURNS = [0.01, 0.02, 0.03, 0.00, -0.01]  # mean 0.01, sample std sqrt(2.5e-4)

    def test_zero_risk_free_rate(self):
        expected = 0.01 / math.sqrt(2.5e-4) * math.sqrt(252)  # = 10.0399...
        assert sharpe_ratio(series(self.RETURNS)) == pytest.approx(expected, rel=1e-12)
        assert expected == pytest.approx(10.03992, abs=1e-5)

    def test_risk_free_rate_is_compounded_to_daily(self):
        # choose an annual rate whose daily equivalent is exactly 0.5%
        annual = 1.005**252 - 1
        assert periodic_risk_free_rate(annual) == pytest.approx(0.005, rel=1e-12)
        expected = 0.005 / math.sqrt(2.5e-4) * math.sqrt(252)
        assert sharpe_ratio(series(self.RETURNS), risk_free_rate=annual) == pytest.approx(expected, rel=1e-10)

    def test_periodic_rate_round_trips(self):
        assert (1 + periodic_risk_free_rate(0.03, 252)) ** 252 == pytest.approx(1.03, rel=1e-12)

    def test_zero_volatility_gives_nan_not_infinity(self):
        assert math.isnan(sharpe_ratio(series([0.001] * 20), risk_free_rate=0.02))

    def test_negative_excess_return_gives_negative_sharpe(self):
        # daily risk-free rate of 1.5% exceeds the 1% mean return -> excess mean -0.5%
        annual = 1.015**252 - 1
        expected = -0.005 / math.sqrt(2.5e-4) * math.sqrt(252)
        assert sharpe_ratio(series(self.RETURNS), risk_free_rate=annual) == pytest.approx(expected, rel=1e-9)

    def test_short_history_rejected(self):
        with pytest.raises(InsufficientDataError):
            sharpe_ratio(series([0.01]))


# --------------------------------------------------------------------------- #
# Drawdowns
# --------------------------------------------------------------------------- #
class TestDrawdown:
    # wealth: 1 -> 1.10 -> 0.88 -> 0.968 -> 1.1616 -> 1.10352
    RETURNS = [0.10, -0.20, 0.10, 0.20, -0.05]

    def test_max_drawdown(self):
        assert max_drawdown(series(self.RETURNS)) == pytest.approx(-0.20, abs=1e-12)

    def test_drawdown_series(self):
        dd = drawdown_series(series(self.RETURNS))
        assert dd.tolist() == pytest.approx([0.0, -0.20, -0.12, 0.0, -0.05], abs=1e-12)

    def test_drawdown_dates_and_durations(self):
        r = series(self.RETURNS)
        stats = drawdown_stats(r)
        assert stats.max_drawdown == pytest.approx(-0.20)
        assert stats.peak_date == r.index[0]
        assert stats.trough_date == r.index[1]
        assert stats.recovery_date == r.index[3]
        assert stats.peak_to_trough_periods == 1
        assert stats.peak_to_recovery_periods == 3
        assert stats.longest_underwater_periods == 2
        assert stats.currently_underwater is True
        assert stats.current_drawdown == pytest.approx(-0.05)

    def test_loss_in_first_period_counts(self):
        r = series([-0.10, 0.05])
        # wealth 0.90 then 0.945, measured against the initial 1.00
        assert max_drawdown(r) == pytest.approx(-0.10)
        assert drawdown_series(r).tolist() == pytest.approx([-0.10, -0.055])
        stats = drawdown_stats(r)
        assert stats.max_drawdown == pytest.approx(-0.10)
        assert stats.peak_date is None  # the initial investment was the peak
        assert stats.trough_date == r.index[0]
        assert stats.recovery_date is None and stats.peak_to_recovery_periods is None

    def test_monotonic_rise_has_no_drawdown(self):
        stats = drawdown_stats(series([0.01, 0.02, 0.0, 0.03]))
        assert stats.max_drawdown == 0.0
        assert stats.longest_underwater_periods == 0
        assert not stats.currently_underwater

    def test_drawdown_is_independent_of_scale(self):
        # a 50% fall from any peak is a 50% drawdown
        assert max_drawdown(series([1.0, -0.5, 0.2])) == pytest.approx(-0.5)


# --------------------------------------------------------------------------- #
# Best / worst periods
# --------------------------------------------------------------------------- #
def test_best_worst_periods_and_calendar_months():
    r = pd.Series(
        [0.02, -0.01, 0.03, -0.04],
        index=pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"]),
    )
    bw = best_worst_periods(r)
    assert bw["best_period"] == (pd.Timestamp("2024-02-01"), pytest.approx(0.03))
    assert bw["worst_period"] == (pd.Timestamp("2024-02-02"), pytest.approx(-0.04))
    # January: 1.02 * 0.99 - 1 = 0.0098; February: 1.03 * 0.96 - 1 = -0.0112
    assert bw["best_month"][0] == pd.Period("2024-01", "M")
    assert bw["best_month"][1] == pytest.approx(0.0098)
    assert bw["worst_month"][1] == pytest.approx(-0.0112)


def test_period_returns_compound_by_year():
    r = pd.Series([0.10, 0.10, -0.50], index=pd.to_datetime(["2023-12-28", "2023-12-29", "2024-01-02"]))
    yearly = period_returns(r, "Y")
    assert yearly.tolist() == pytest.approx([0.21, -0.50])


# --------------------------------------------------------------------------- #
# Benchmark-relative statistics
# --------------------------------------------------------------------------- #
class TestRelativePerformance:
    BENCH = [0.01, -0.02, 0.015, 0.005, -0.01]

    def test_levered_benchmark_has_beta_two(self):
        b = series(self.BENCH)
        p = 2 * b
        rel = relative_performance(p, b)
        assert rel["beta"] == pytest.approx(2.0)
        assert rel["correlation"] == pytest.approx(1.0)
        # active return = b, so tracking error = vol(b)
        assert rel["tracking_error"] == pytest.approx(b.std(ddof=1) * math.sqrt(252))
        assert rel["information_ratio"] == pytest.approx(b.mean() * 252 / rel["tracking_error"])
        assert rel["jensens_alpha"] == pytest.approx(0.0, abs=1e-12)
        assert rel["excess_cumulative_return"] == pytest.approx(cumulative_return(p) - cumulative_return(b))

    def test_jensens_alpha_with_risk_free_rate(self):
        # CAPM-required return for beta 2 is 2*E[b] - rf, actual is 2*E[b] -> alpha = rf
        b = series(self.BENCH)
        annual = 1.0001**252 - 1
        rel = relative_performance(2 * b, b, risk_free_rate=annual)
        assert rel["jensens_alpha"] == pytest.approx(0.0001 * 252, rel=1e-9)

    def test_identical_series(self):
        b = series(self.BENCH)
        rel = relative_performance(b, b)
        assert rel["tracking_error"] == pytest.approx(0.0, abs=1e-15)
        assert math.isnan(rel["information_ratio"])
        assert rel["beta"] == pytest.approx(1.0)
        assert rel["excess_annualized_return"] == pytest.approx(0.0, abs=1e-15)

    def test_misaligned_dates_rejected(self):
        b = series(self.BENCH)
        p = series(self.BENCH, start="2024-02-01")
        with pytest.raises(DataValidationError):
            relative_performance(p, b)


def test_performance_summary_contains_consistent_values():
    r = series(TestSharpe.RETURNS * 10)
    summary = performance_summary(r, risk_free_rate=0.0)
    assert summary["cumulative_return"] == pytest.approx(cumulative_return(r))
    assert summary["sharpe_ratio"] == pytest.approx(sharpe_ratio(r))
    assert summary["n_periods"] == 50
    assert summary["positive_period_share"] == pytest.approx(0.6)  # 3 of every 5 returns are > 0
