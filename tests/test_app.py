"""Headless smoke tests of the Streamlit app (no browser, no network).

Uses Streamlit's AppTest harness to execute app.py, interact with widgets and
check that results and validation messages reach the page.
"""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.analysis import AnalysisConfig, run_analysis  # noqa: E402
from src.data_loader import SAMPLE_BENCHMARK, load_sample_prices  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")
DEFAULT_WEIGHTS = {"SYN_US_EQUITY": 0.35, "SYN_INTL_EQUITY": 0.15, "SYN_TREASURY": 0.25,
                   "SYN_CORP_BOND": 0.15, "SYN_GOLD": 0.10}


@pytest.fixture
def app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    return at


def metric(at: AppTest, label_start: str):
    return next(m for m in at.metric if m.label.startswith(label_start))


def test_app_starts_with_sample_data(app):
    assert not app.exception
    assert not app.error
    assert len(app.tabs) == 6
    assert any("Synthetic" in w.value for w in app.warning)  # sample data is clearly labelled


def test_headline_numbers_match_the_analytics_layer(app):
    expected = run_analysis(
        load_sample_prices(),
        AnalysisConfig(assets=tuple(DEFAULT_WEIGHTS), weights=DEFAULT_WEIGHTS, benchmark=SAMPLE_BENCHMARK),
    ).summary["Portfolio"]
    assert metric(app, "Cumulative return").value == f"{expected.cumulative_return:.2%}"
    assert metric(app, "Annualised volatility").value == f"{expected.annualized_volatility:.2%}"
    assert metric(app, "1-period VaR 95.0%").value == f"{expected.historical_var:.2%}"


def test_all_zero_weights_are_rejected_with_a_message(app):
    for box in app.number_input:
        if box.key and box.key.startswith("w::"):
            box.set_value(0.0)
    app.run()
    assert not app.exception
    assert any("sum to zero" in e.value for e in app.error)


def test_assumption_changes_rerun_cleanly(app):
    next(s for s in app.selectbox if s.label == "Rebalancing").set_value("none")
    app.select_slider[0].set_value(0.99)
    next(n for n in app.number_input if n.label.startswith("Risk-free")).set_value(5.0)
    app.run()
    assert not app.exception
    assert metric(app, "1-period VaR 99.0%")
    assert metric(app, "Sharpe ratio (rf 5.00%)")


def test_short_date_range_reports_insufficient_data(app):
    end = load_sample_prices().index[-1]
    next(d for d in app.date_input if d.label == "Start").set_value((end - pd.Timedelta(10, unit="D")).date())
    app.run()
    assert not app.exception
    assert any("return observations" in e.value for e in app.error)


@pytest.mark.parametrize("tab", ["risk", "methodology", "not-a-tab"])
def test_tab_deep_links_never_break_the_page(tab):
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.query_params["tab"] = tab
    at.run()
    assert not at.exception
    assert len(at.tabs) == 6


def test_upload_source_without_file_shows_instructions(app):
    app.radio(key="source").set_value("upload").run()
    assert not app.exception
    assert any("Upload a CSV" in i.value for i in app.info)
