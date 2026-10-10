"""Integration tests of the end-to-end pipeline on the bundled sample data."""

import numpy as np
import pandas as pd
import pytest

from src.analysis import AnalysisConfig, returns_export, run_analysis, summary_export
from src.data_loader import load_sample_prices
from src.exceptions import InsufficientDataError, PortfolioError
from src.performance import cumulative_return

WEIGHTS = {"SYN_US_EQUITY": 0.4, "SYN_TREASURY": 0.4, "SYN_GOLD": 0.2}


@pytest.fixture(scope="module")
def prices():
    return load_sample_prices()


@pytest.fixture(scope="module")
def result(prices):
    cfg = AnalysisConfig(assets=tuple(WEIGHTS), weights=WEIGHTS, benchmark="SYN_MARKET")
    return run_analysis(prices, cfg)


def test_pipeline_outputs_are_consistent(result):
    port = result.portfolio_returns
    assert not port.isna().any()
    assert result.periods_per_year == 252
    # growth chart ends at initial value x (1 + cumulative return)
    assert result.growth["Portfolio"].iloc[-1] == pytest.approx(
        result.config.portfolio_value * (1 + cumulative_return(port))
    )
    assert result.growth.index[0] == result.prices.index[0]
    # the headline VaR row equals the table row for the chosen confidence
    assert result.summary.loc["historical_var", "Portfolio"] == pytest.approx(
        result.var_table.loc[0.95, "historical_var"]
    )
    assert result.risk_contributions["pct_of_risk"].sum() == pytest.approx(1.0)
    assert result.backtest is not None and result.backtest.n_forecasts == len(port) - 252
    assert np.allclose(np.diag(result.correlation), 1.0)


def test_benchmark_and_portfolio_share_dates(result):
    pd.testing.assert_index_equal(result.portfolio_returns.index, result.benchmark_returns.index)


def test_monthly_table_years_compound_to_total(result):
    yearly = result.monthly_returns["Year"]
    assert (1 + yearly).prod() - 1 == pytest.approx(cumulative_return(result.portfolio_returns))


def test_late_listing_trims_window_with_warning(prices):
    weights = {"SYN_US_EQUITY": 0.5, "SYN_EM_EQUITY": 0.5}
    cfg = AnalysisConfig(assets=tuple(weights), weights=weights)
    res = run_analysis(prices, cfg)
    assert res.quality.effective_start == pd.Timestamp("2017-03-01")
    assert any("SYN_EM_EQUITY" in w for w in res.quality.warnings)


def test_holiday_gaps_are_dropped_and_reported(prices):
    cfg = AnalysisConfig(assets=("SYN_INTL_EQUITY",), weights={"SYN_INTL_EQUITY": 1.0})
    res = run_analysis(prices, cfg)
    assert res.quality.rows_dropped == int(prices["SYN_INTL_EQUITY"].isna().sum())
    assert res.benchmark_returns is None and res.relative is None


def test_weights_are_normalized_and_reported(prices):
    cfg = AnalysisConfig(assets=("SYN_US_EQUITY", "SYN_GOLD"), weights={"SYN_US_EQUITY": 0.9, "SYN_GOLD": 0.3})
    res = run_analysis(prices, cfg)
    assert res.weights.weights.to_dict() == pytest.approx({"SYN_US_EQUITY": 0.75, "SYN_GOLD": 0.25})
    assert any("rescaled" in n for n in res.notes)


def test_invalid_inputs_raise_user_facing_errors(prices):
    with pytest.raises(PortfolioError):
        run_analysis(prices, AnalysisConfig(assets=("SYN_GOLD",), weights={"SYN_GOLD": -1.0}))
    with pytest.raises(InsufficientDataError):
        run_analysis(prices, AnalysisConfig(assets=("SYN_GOLD",), weights={"SYN_GOLD": 1.0},
                                            start=pd.Timestamp("2024-12-01"), end=pd.Timestamp("2024-12-31")))


def test_short_sample_skips_long_window_analytics_with_notes(prices):
    cfg = AnalysisConfig(assets=("SYN_GOLD",), weights={"SYN_GOLD": 1.0},
                         start=pd.Timestamp("2024-06-01"), end=pd.Timestamp("2024-12-31"))
    res = run_analysis(prices, cfg)
    assert res.rolling_sharpe is None and res.backtest is None
    assert any("less than one year" in n for n in res.notes)


def test_exports(result):
    returns = returns_export(result)
    assert returns.index.name == "date"
    assert {"portfolio_return", "benchmark_return", "portfolio_value", "portfolio_drawdown"} <= set(returns.columns)
    summary = summary_export(result)
    assert {"section", "metric", "value"} == set(summary.columns)
    assumptions = summary[summary.section == "assumption"].set_index("metric")["value"]
    assert assumptions["rebalance"] == "monthly"
    assert assumptions["var_definition"].startswith("1-period historical")
