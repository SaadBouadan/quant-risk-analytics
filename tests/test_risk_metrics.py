"""Risk statistics, verified against hand-calculated values and independent references."""

import math

import numpy as np
import pandas as pd
import pytest

from src.exceptions import InsufficientDataError
from src.performance import annualized_volatility, sharpe_ratio
from src.risk_metrics import (
    backtest_historical_var,
    calendar_year_risk,
    classify_volatility_regimes,
    concentration_summary,
    correlation_matrix,
    diversification_ratio,
    effective_number_of_assets,
    herfindahl_index,
    historical_expected_shortfall,
    historical_var,
    kupiec_pof_test,
    parametric_expected_shortfall,
    parametric_var,
    risk_contributions,
    rolling_historical_var,
    rolling_sharpe,
    rolling_volatility,
    tail_count,
    var_es_table,
    volatility_regime_analysis,
)


def series(values, start="2024-01-01", freq="B"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=freq), dtype=float)


# The worked example used in the README and in the app's Methodology tab.
WORKED_EXAMPLE = [0.005, -0.012, 0.003, 0.021, -0.004, -0.030, 0.008, 0.011, -0.007, 0.002,
                  -0.022, 0.009, 0.015, -0.001, 0.006, -0.018, 0.004, 0.010, -0.009, 0.007]


# --------------------------------------------------------------------------- #
# Historical VaR
# --------------------------------------------------------------------------- #
class TestHistoricalVaR:
    def test_worked_example(self):
        r = series(WORKED_EXAMPLE)
        # 90%: k = ceil(20 * 0.10) = 2 -> 2nd-worst return is -2.2% -> VaR = 2.2% loss
        assert historical_var(r, 0.90) == pytest.approx(0.022)
        # 95%: k = ceil(20 * 0.05) = 1 -> worst return -3.0%
        assert historical_var(r, 0.95) == pytest.approx(0.030)

    def test_var_is_a_positive_loss_not_a_return_quantile(self):
        r = series(WORKED_EXAMPLE)
        assert historical_var(r, 0.90) > 0
        assert np.sort(r)[1] == pytest.approx(-historical_var(r, 0.90))

    def test_hundred_evenly_spaced_returns(self):
        r = series(np.arange(1, 101) / 1000 - 0.05)  # -4.9%, -4.8%, ..., +5.0%
        assert historical_var(r, 0.95) == pytest.approx(0.045)  # 5th worst
        assert historical_var(r, 0.99) == pytest.approx(0.049)  # worst
        assert historical_var(r, 0.90) == pytest.approx(0.040)  # 10th worst

    def test_order_of_observations_is_irrelevant(self):
        values = np.arange(1, 101) / 1000 - 0.05
        shuffled = np.random.default_rng(0).permutation(values)
        assert historical_var(series(shuffled), 0.95) == pytest.approx(historical_var(series(values), 0.95))

    def test_tail_count_is_robust_to_floating_point(self):
        # 100 * (1 - 0.95) evaluates to 5.000000000000004; the tail must still be 5, not 6
        assert tail_count(100, 0.95) == 5
        assert tail_count(1000, 0.99) == 10
        assert tail_count(20, 0.90) == 2
        assert tail_count(10, 0.95) == 1  # short history: never fewer than one observation

    @pytest.mark.parametrize("n", [250, 1003])
    @pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
    def test_matches_numpy_inverted_cdf_quantile(self, n, confidence):
        r = np.random.default_rng(n).standard_t(4, size=n) * 0.01
        expected = -np.quantile(r, 1 - confidence, method="inverted_cdf")
        assert historical_var(series(r), confidence) == pytest.approx(expected)

    def test_all_positive_returns_give_negative_var(self):
        # Even the worst day is a gain, so the "loss" at the 95% level is -1%.
        assert historical_var(series([0.01, 0.02, 0.03] * 10), 0.95) == pytest.approx(-0.01)

    def test_constant_returns(self):
        r = series([-0.002] * 50)
        assert historical_var(r, 0.99) == pytest.approx(0.002)
        assert historical_expected_shortfall(r, 0.99) == pytest.approx(0.002)

    def test_short_history_uses_worst_observation(self):
        r = series([0.01, -0.03, 0.02, -0.01, 0.0, 0.005, -0.002, 0.004, 0.001, 0.003])
        assert historical_var(r, 0.95) == pytest.approx(0.03)

    def test_empty_returns_rejected(self):
        with pytest.raises(InsufficientDataError):
            historical_var(pd.Series([], dtype=float))

    @pytest.mark.parametrize("bad", [1.0, 0.3, 95])
    def test_invalid_confidence_rejected(self, bad):
        with pytest.raises(ValueError):
            historical_var(series(WORKED_EXAMPLE), bad)


# --------------------------------------------------------------------------- #
# Expected Shortfall
# --------------------------------------------------------------------------- #
class TestExpectedShortfall:
    def test_worked_example(self):
        r = series(WORKED_EXAMPLE)
        # 90%: mean of the two worst losses (3.0% and 2.2%) = 2.6%
        assert historical_expected_shortfall(r, 0.90) == pytest.approx(0.026)
        assert historical_expected_shortfall(r, 0.95) == pytest.approx(0.030)

    def test_hundred_evenly_spaced_returns(self):
        r = series(np.arange(1, 101) / 1000 - 0.05)
        # mean of the 5 worst: (4.9 + 4.8 + 4.7 + 4.6 + 4.5) / 5 = 4.7%
        assert historical_expected_shortfall(r, 0.95) == pytest.approx(0.047)
        # mean of the 10 worst: 4.9% ... 4.0% -> 4.45%
        assert historical_expected_shortfall(r, 0.90) == pytest.approx(0.0445)

    @pytest.mark.parametrize("confidence", [0.90, 0.95, 0.975, 0.99])
    def test_es_is_never_smaller_than_var(self, confidence):
        r = series(np.random.default_rng(7).standard_t(3, size=500) * 0.01)
        assert historical_expected_shortfall(r, confidence) >= historical_var(r, confidence)

    def test_monetary_table(self):
        table = var_es_table(series(WORKED_EXAMPLE), confidences=(0.90, 0.95), portfolio_value=1_000_000)
        assert table.loc[0.90, "historical_var_amount"] == pytest.approx(22_000)
        assert table.loc[0.90, "historical_es_amount"] == pytest.approx(26_000)
        assert table.loc[0.95, "tail_observations"] == 1


class TestParametric:
    def test_gaussian_var_and_es(self):
        r = series([0.01, -0.01] * 50)  # mean 0
        sd = 0.01 * math.sqrt(100 / 99)
        assert parametric_var(r, 0.95) == pytest.approx(1.6448536 * sd, rel=1e-6)
        # ES = sigma * phi(z) / (1 - c) = sigma * 2.0627128 at 95%
        assert parametric_expected_shortfall(r, 0.95) == pytest.approx(2.0627128 * sd, rel=1e-6)


# --------------------------------------------------------------------------- #
# Rolling statistics and look-ahead bias
# --------------------------------------------------------------------------- #
@pytest.fixture
def noisy_returns():
    return series(np.random.default_rng(1).normal(0.0004, 0.01, 300))


class TestRolling:
    def test_rolling_volatility_matches_full_window_calculation(self, noisy_returns):
        rv = rolling_volatility(noisy_returns, window=63)
        assert rv.iloc[:62].isna().all()
        assert rv.iloc[62] == pytest.approx(annualized_volatility(noisy_returns.iloc[:63]))
        assert rv.iloc[-1] == pytest.approx(annualized_volatility(noisy_returns.iloc[-63:]))

    def test_rolling_values_do_not_use_future_data(self, noisy_returns):
        t = 150
        changed = noisy_returns.copy()
        changed.iloc[t + 1 :] = 0.25  # wildly different future
        for func in (rolling_volatility, rolling_sharpe, rolling_historical_var):
            original = func(noisy_returns, window=63)
            modified = func(changed, window=63)
            pd.testing.assert_series_equal(original.iloc[: t + 1], modified.iloc[: t + 1])

    def test_rolling_sharpe_matches_point_calculation(self, noisy_returns):
        rs = rolling_sharpe(noisy_returns, window=126, risk_free_rate=0.02)
        assert rs.iloc[200] == pytest.approx(sharpe_ratio(noisy_returns.iloc[75:201], 0.02))

    def test_rolling_sharpe_is_nan_when_volatility_is_zero(self):
        rs = rolling_sharpe(series([0.001] * 30 + [0.01, -0.01] * 10), window=10)
        assert rs.iloc[9:30].isna().all()
        assert rs.iloc[-1] == pytest.approx(sharpe_ratio(series([0.01, -0.01] * 5)))

    def test_rolling_var_uses_the_same_quantile_rule(self, noisy_returns):
        rv = rolling_historical_var(noisy_returns, window=100, confidence=0.95)
        for end in (99, 180, 299):
            window = noisy_returns.iloc[end - 99 : end + 1]
            assert rv.iloc[end] == pytest.approx(historical_var(window, 0.95))

    def test_window_longer_than_sample_rejected(self):
        with pytest.raises(InsufficientDataError):
            rolling_volatility(series([0.01, 0.02, 0.03]), window=10)


# --------------------------------------------------------------------------- #
# VaR backtest
# --------------------------------------------------------------------------- #
class TestBacktest:
    def test_hand_constructed_exceptions(self):
        # 20 alternating +/-1% days, then -2%, 0%, -1.5%; window 20, 95% -> VaR = worst prior loss
        r = series([0.01, -0.01] * 10 + [-0.02, 0.0, -0.015])
        bt = backtest_historical_var(r, window=20, confidence=0.95)
        assert bt.n_forecasts == 3
        assert bt.var_forecast.tolist() == pytest.approx([0.01, 0.02, 0.02])
        assert bt.exceptions.tolist() == [True, False, False]  # 2% > 1%; 0% and 1.5% < 2%
        assert bt.n_exceptions == 1
        assert bt.expected_exceptions == pytest.approx(0.15)

    def test_forecast_does_not_see_the_return_it_is_tested_on(self, noisy_returns):
        bt = backtest_historical_var(noisy_returns, window=100)
        date = noisy_returns.index[200]
        shocked = noisy_returns.copy()
        shocked.loc[date] = -0.50
        assert backtest_historical_var(shocked, window=100).var_forecast.loc[date] == bt.var_forecast.loc[date]
        assert bt.var_forecast.loc[date] == pytest.approx(historical_var(noisy_returns.iloc[100:200], 0.95))

    def test_too_short_for_backtest(self):
        with pytest.raises(InsufficientDataError):
            backtest_historical_var(series([0.01] * 50), window=50)


class TestKupiec:
    def test_exactly_expected_exceptions(self):
        lr, p = kupiec_pof_test(100, 5, 0.95)
        assert lr == pytest.approx(0.0, abs=1e-12)
        assert p == pytest.approx(1.0)

    def test_zero_exceptions(self):
        # LR = -2 * 100 * ln(0.95) = 10.2587
        lr, p = kupiec_pof_test(100, 0, 0.95)
        assert lr == pytest.approx(-200 * math.log(0.95))
        assert lr == pytest.approx(10.2587, abs=1e-4)
        assert p == pytest.approx(0.00136, abs=2e-5)

    def test_too_many_exceptions(self):
        # x = 10, n = 100, p = 5%: LR = 4.1308 (hand calculation), just significant at 5%
        lr, p = kupiec_pof_test(100, 10, 0.95)
        assert lr == pytest.approx(4.1308, abs=1e-4)
        assert 0.04 < p < 0.045


# --------------------------------------------------------------------------- #
# Correlation, concentration, diversification
# --------------------------------------------------------------------------- #
# Two zero-mean, uncorrelated return streams; B is twice as volatile as A.
A = [0.01, -0.01, 0.01, -0.01]
B = [0.02, 0.02, -0.02, -0.02]


class TestDiversification:
    def test_correlation_matrix(self):
        frame = pd.DataFrame({"A": A, "B": B, "C": [-x for x in A]})
        corr = correlation_matrix(frame)
        assert corr.loc["A", "B"] == pytest.approx(0.0, abs=1e-12)
        assert corr.loc["A", "C"] == pytest.approx(-1.0)
        assert np.allclose(np.diag(corr), 1.0)

    def test_correlation_rejects_missing_values(self):
        with pytest.raises(ValueError):
            correlation_matrix(pd.DataFrame({"A": [0.01, np.nan, 0.02, 0.0], "B": [0.0, 0.01, 0.02, 0.03]}))

    def test_herfindahl_and_effective_n(self):
        assert herfindahl_index({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}) == pytest.approx(0.25)
        assert effective_number_of_assets({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}) == pytest.approx(4.0)
        assert herfindahl_index({"A": 1.0}) == pytest.approx(1.0)
        # 0.5^2 + 0.3^2 + 0.2^2 = 0.38 -> effective N = 2.63
        assert herfindahl_index({"A": 0.5, "B": 0.3, "C": 0.2}) == pytest.approx(0.38)
        assert effective_number_of_assets({"A": 0.5, "B": 0.3, "C": 0.2}) == pytest.approx(1 / 0.38)

    def test_diversification_ratio_uncorrelated_equal_vol(self):
        frame = pd.DataFrame({"A": A, "B": [0.01, 0.01, -0.01, -0.01]})
        assert diversification_ratio({"A": 0.5, "B": 0.5}, frame) == pytest.approx(math.sqrt(2))

    def test_diversification_ratio_perfect_correlation_is_one(self):
        frame = pd.DataFrame({"A": A, "B": [2 * x for x in A]})
        assert diversification_ratio({"A": 0.5, "B": 0.5}, frame) == pytest.approx(1.0)

    def test_risk_contributions(self):
        frame = pd.DataFrame({"A": A, "B": B})
        rc = risk_contributions({"A": 0.5, "B": 0.5}, frame)
        # uncorrelated: share_i = w_i^2 s_i^2 / sum -> 1 : 4
        assert rc["pct_of_risk"].tolist() == pytest.approx([0.2, 0.8])
        assert rc["pct_of_risk"].sum() == pytest.approx(1.0)
        port_vol = annualized_volatility(0.5 * frame["A"] + 0.5 * frame["B"])
        assert rc["risk_contribution"].sum() == pytest.approx(port_vol)

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError):
            herfindahl_index({"A": 0.6, "B": 0.6})

    def test_concentration_summary(self):
        frame = pd.DataFrame({"A": A, "B": B})
        summary = concentration_summary({"A": 0.7, "B": 0.3}, frame)
        assert summary["max_weight"] == pytest.approx(0.7)
        assert summary["n_assets"] == 2


# --------------------------------------------------------------------------- #
# Market-period analysis
# --------------------------------------------------------------------------- #
class TestMarketPeriods:
    def test_calendar_year_table(self):
        idx = pd.bdate_range("2022-12-01", "2023-12-29")
        r = pd.Series(0.001, index=idx)
        r.loc["2023"] = [0.002, -0.001] * (len(r.loc["2023"]) // 2) + [0.002] * (len(r.loc["2023"]) % 2)
        table = calendar_year_risk(r)
        assert list(table.index) == [2022, 2023]
        assert table.loc[2022, "partial_year"]
        assert not table.loc[2023, "partial_year"]
        assert table.loc[2022, "return"] == pytest.approx(1.001 ** len(r.loc["2022"]) - 1)
        assert table.loc[2022, "max_drawdown"] == 0.0
        assert table.loc[2023, "historical_var"] == pytest.approx(0.001)

    def test_regimes_use_only_prior_volatility(self):
        rng = np.random.default_rng(3)
        calm = rng.normal(0, 0.002, 60)      # ~3% annualised
        stressed = rng.normal(0, 0.03, 60)   # ~48% annualised
        ref = series(np.concatenate([calm, stressed]))
        regimes = classify_volatility_regimes(ref, window=21)
        assert regimes.iloc[:21].isna().all()  # warm-up: no prior window yet
        assert (regimes.iloc[21:60] == "Calm").all()
        # the first stressed day is still labelled from the calm window before it
        assert regimes.iloc[60] == "Calm"
        assert regimes.iloc[-1] == "Stressed"

        shocked = ref.copy()
        shocked.iloc[40] = 0.5  # changing today's return cannot change today's label
        assert classify_volatility_regimes(shocked, window=21).iloc[40] == regimes.iloc[40]

    def test_regime_table(self):
        rng = np.random.default_rng(4)
        ref = series(np.concatenate([rng.normal(0, 0.002, 100), rng.normal(0, 0.03, 100)]))
        table = volatility_regime_analysis(ref, ref, window=21)
        assert {"Calm", "Stressed"} <= set(table.index) <= {"Calm", "Normal", "Stressed"}
        assert table["days"].sum() <= len(ref) - 21  # warm-up days are never classified
        assert table["share_of_days"].sum() <= 1.0 + 1e-12
        assert table.loc["Stressed", "annualized_volatility"] > table.loc["Calm", "annualized_volatility"]
