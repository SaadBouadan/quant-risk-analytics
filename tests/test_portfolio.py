"""Weight validation and portfolio construction under each rebalancing rule."""

import numpy as np
import pandas as pd
import pytest

from src.exceptions import PortfolioError
from src.performance import cumulative_return
from src.portfolio import construct_portfolio, validate_weights


# --------------------------------------------------------------------------- #
# Weight validation
# --------------------------------------------------------------------------- #
class TestValidateWeights:
    def test_valid_weights_unchanged(self):
        result = validate_weights({"A": 0.6, "B": 0.4}, ["A", "B"])
        assert result.weights.to_dict() == pytest.approx({"A": 0.6, "B": 0.4})
        assert not result.was_normalized
        assert result.messages == ()

    def test_percentages_are_normalized_with_message(self):
        result = validate_weights({"A": 60, "B": 40})
        assert result.weights.to_dict() == pytest.approx({"A": 0.6, "B": 0.4})
        assert result.was_normalized
        assert result.original_sum == pytest.approx(100)
        assert "rescaled" in result.messages[0]

    def test_unnormalized_weights_rejected_when_normalization_disabled(self):
        with pytest.raises(PortfolioError, match="sum to 100%"):
            validate_weights({"A": 0.7, "B": 0.4}, normalize=False)

    def test_tiny_rounding_error_is_tolerated(self):
        result = validate_weights({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
        assert not result.was_normalized

    @pytest.mark.parametrize(
        "weights, match",
        [
            ({"A": 0.5, "B": -0.5}, "Negative"),
            ({"A": 0.0, "B": 0.0}, "sum to zero"),
            ({"A": float("nan"), "B": 1.0}, "finite"),
            ({"A": float("inf"), "B": 1.0}, "finite"),
            ({}, "No portfolio weights"),
        ],
    )
    def test_invalid_weights_rejected(self, weights, match):
        with pytest.raises(PortfolioError, match=match):
            validate_weights(weights)

    def test_unknown_asset_rejected(self):
        with pytest.raises(PortfolioError, match="without price data"):
            validate_weights({"A": 0.5, "ZZZ": 0.5}, ["A", "B"])

    def test_zero_weight_assets_are_dropped_and_reported(self):
        result = validate_weights({"A": 0.5, "B": 0.0, "C": 0.5})
        assert list(result.weights.index) == ["A", "C"]
        assert any("zero weight" in m for m in result.messages)


# --------------------------------------------------------------------------- #
# Portfolio construction
# --------------------------------------------------------------------------- #
def frame(data, dates=None):
    n = len(next(iter(data.values())))
    index = pd.DatetimeIndex(dates) if dates is not None else pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(data, index=index, dtype=float)


class TestRebalancing:
    # Asset A doubles then halves; B is flat. 50/50 target.
    RETURNS = {"A": [1.0, -0.5], "B": [0.0, 0.0]}
    WEIGHTS = {"A": 0.5, "B": 0.5}

    def test_daily_rebalancing_is_weighted_sum(self):
        result = construct_portfolio(frame(self.RETURNS), self.WEIGHTS, "daily")
        # day 1: 0.5*100% = 50%; day 2: 0.5*(-50%) = -25%; total 1.5*0.75 - 1 = 12.5%
        assert result.returns.tolist() == pytest.approx([0.5, -0.25])
        assert cumulative_return(result.returns) == pytest.approx(0.125)

    def test_buy_and_hold_lets_weights_drift(self):
        result = construct_portfolio(frame(self.RETURNS), self.WEIGHTS, "none")
        # after day 1: A = 1.0, B = 0.5 -> weights 2/3, 1/3; day 2: 2/3 * -50% = -1/3
        assert result.returns.tolist() == pytest.approx([0.5, -1 / 3])
        assert result.weights_history.iloc[1].tolist() == pytest.approx([2 / 3, 1 / 3])
        assert cumulative_return(result.returns) == pytest.approx(0.0)  # A round-trips to its start
        assert result.turnover.empty

    def test_buy_and_hold_equals_weighted_asset_growth(self):
        rng = np.random.default_rng(11)
        returns = frame({"A": rng.normal(0.001, 0.02, 300), "B": rng.normal(0.0, 0.01, 300)})
        weights = {"A": 0.3, "B": 0.7}
        result = construct_portfolio(returns, weights, "none")
        asset_growth = (1 + returns).prod()
        expected = 0.3 * asset_growth["A"] + 0.7 * asset_growth["B"] - 1
        assert cumulative_return(result.returns) == pytest.approx(expected, rel=1e-10)

    def test_daily_matches_matrix_product(self):
        rng = np.random.default_rng(12)
        returns = frame({"A": rng.normal(0, 0.01, 50), "B": rng.normal(0, 0.02, 50), "C": rng.normal(0, 0.03, 50)})
        weights = pd.Series({"A": 0.2, "B": 0.3, "C": 0.5})
        result = construct_portfolio(returns, weights, "daily")
        np.testing.assert_allclose(result.returns.to_numpy(), returns.to_numpy() @ weights.to_numpy())

    def test_monthly_rebalance_resets_weights_at_month_end(self):
        dates = ["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"]
        returns = frame({"A": [0.10, 0.10, 0.0, 0.10], "B": [0.0, 0.0, 0.0, 0.0]}, dates)
        result = construct_portfolio(returns, {"A": 0.5, "B": 0.5}, "monthly")

        # 31 Jan close: A = 0.5 * 1.21 = 0.605, B = 0.5 -> drifted weight of A = 0.605 / 1.105
        drifted_a = 0.605 / 1.105
        assert result.weights_history.loc["2024-01-31", "A"] == pytest.approx(0.55 / 1.05)
        assert result.weights_history.loc["2024-02-01"].tolist() == pytest.approx([0.5, 0.5])
        assert list(result.turnover.index) == [pd.Timestamp("2024-01-31")]
        assert result.turnover.iloc[0] == pytest.approx(drifted_a - 0.5)
        # after the reset A is back to 50%, so a 10% move adds exactly 5%
        assert result.returns.loc["2024-02-02"] == pytest.approx(0.05)

        hold = construct_portfolio(returns, {"A": 0.5, "B": 0.5}, "none")
        assert hold.returns.loc["2024-02-02"] == pytest.approx(drifted_a * 0.10)

    def test_quarterly_does_not_rebalance_within_a_quarter(self):
        dates = ["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"]
        returns = frame({"A": [0.10, 0.10, 0.0, 0.10], "B": [0.0, 0.0, 0.0, 0.0]}, dates)
        quarterly = construct_portfolio(returns, {"A": 0.5, "B": 0.5}, "quarterly")
        hold = construct_portfolio(returns, {"A": 0.5, "B": 0.5}, "none")
        pd.testing.assert_series_equal(quarterly.returns, hold.returns)

    def test_end_weights_reflect_final_drift(self):
        result = construct_portfolio(frame(self.RETURNS), self.WEIGHTS, "daily")
        # day 2 starts at 50/50: A -> 0.25, B -> 0.5 -> 1/3, 2/3
        assert result.end_weights.tolist() == pytest.approx([1 / 3, 2 / 3])

    def test_missing_returns_rejected(self):
        returns = frame({"A": [0.01, np.nan], "B": [0.0, 0.01]})
        with pytest.raises(PortfolioError, match="missing"):
            construct_portfolio(returns, {"A": 0.5, "B": 0.5})

    def test_weights_must_already_sum_to_one(self):
        with pytest.raises(PortfolioError):
            construct_portfolio(frame(self.RETURNS), {"A": 0.6, "B": 0.6})

    def test_unknown_rebalance_rule_rejected(self):
        with pytest.raises(PortfolioError, match="rebalancing"):
            construct_portfolio(frame(self.RETURNS), self.WEIGHTS, "weekly")
