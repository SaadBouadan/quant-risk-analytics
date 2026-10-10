"""CSV parsing, validation, missing-data handling, frequency detection and data sources."""

import io

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    SAMPLE_DATA_PATH,
    clean_tickers,
    compute_returns,
    fetch_yfinance_prices,
    infer_frequency,
    load_sample_prices,
    make_csv_template,
    prepare_prices,
    read_price_csv,
)
from src.exceptions import DataSourceError, DataValidationError, InsufficientDataError
from src.sample_data import generate_sample_prices


def csv(text: str) -> io.StringIO:
    return io.StringIO(text.strip() + "\n")


# --------------------------------------------------------------------------- #
# CSV parsing and validation
# --------------------------------------------------------------------------- #
class TestReadPriceCsv:
    def test_wide_format_is_parsed_and_sorted(self):
        prices = read_price_csv(csv("Date,aaa,BBB\n2024-01-03,101,51\n2024-01-02,100,50"))
        assert list(prices.columns) == ["AAA", "BBB"]  # tickers are upper-cased
        assert list(prices.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
        assert prices.loc["2024-01-03", "AAA"] == 101.0

    def test_missing_tokens_become_nan_not_zero(self):
        prices = read_price_csv(csv("Date,A,B\n2024-01-02,100,\n2024-01-03,N/A,51\n2024-01-04,102,null"))
        assert prices["A"].isna().sum() == 1
        assert prices["B"].isna().sum() == 2
        assert (prices.fillna(1).values > 0).all()

    def test_long_format_is_pivoted(self):
        text = "Date,Ticker,Adj Close\n2024-01-02,AAA,100\n2024-01-02,BBB,50\n2024-01-03,AAA,101"
        prices = read_price_csv(csv(text))
        assert prices.shape == (2, 2)
        assert np.isnan(prices.loc["2024-01-03", "BBB"])  # absent row -> missing, not invented

    def test_long_format_duplicates_rejected(self):
        text = "Date,Ticker,Price\n2024-01-02,AAA,100\n2024-01-02,AAA,101"
        with pytest.raises(DataValidationError, match="Duplicate"):
            read_price_csv(csv(text))

    @pytest.mark.parametrize(
        "text, match",
        [
            ("Date,A\n2024-01-02,100\n2024-01-03,abc", "non-numeric"),
            ("Date,A\n2024-01-02,100\n2024-01-03,-5", "strictly positive"),
            ("Date,A\n2024-01-02,100\n2024-01-03,0", "strictly positive"),
            ("Date,A\n2024-01-02,100\n2024-01-02,101", "Duplicate dates"),
            ("Day,A\n2024-01-02,100", "Date"),
            ("Date,A\n03/04/2024,100", "ISO 8601"),
            ("Date,A,B\n2024-01-02,100,\n2024-01-03,101,", "no valid prices"),
            ("Date\n2024-01-02", "at least one price column"),
        ],
    )
    def test_invalid_files_rejected_with_clear_message(self, text, match):
        with pytest.raises(DataValidationError, match=match):
            read_price_csv(csv(text))

    def test_empty_file_rejected(self):
        with pytest.raises(DataValidationError):
            read_price_csv(io.StringIO(""))

    def test_template_round_trips(self):
        prices = read_price_csv(io.StringIO(make_csv_template()))
        assert list(prices.columns) == ["TICKER_A", "TICKER_B", "BENCHMARK"]
        assert len(prices) == 3


# --------------------------------------------------------------------------- #
# Alignment and missing-data policy
# --------------------------------------------------------------------------- #
def price_frame(n=40, gaps=None, start_nan=None):
    idx = pd.bdate_range("2024-01-01", periods=n)
    prices = pd.DataFrame(
        {"A": 100 * 1.01 ** np.arange(n), "B": 50 * 0.995 ** np.arange(n)}, index=idx
    )
    for col, positions in (gaps or {}).items():
        prices.iloc[positions, prices.columns.get_loc(col)] = np.nan
    for col, k in (start_nan or {}).items():
        prices.iloc[:k, prices.columns.get_loc(col)] = np.nan
    return prices


class TestPreparePrices:
    def test_clean_data_passes_through(self):
        prices = price_frame()
        aligned, report = prepare_prices(prices, ["A", "B"], min_observations=10)
        pd.testing.assert_frame_equal(aligned, prices)
        assert report.rows_dropped == 0 and report.cells_forward_filled == 0
        assert report.n_return_periods == 39

    def test_drop_policy_removes_incomplete_dates(self):
        prices = price_frame(gaps={"B": [10, 11]})
        aligned, report = prepare_prices(prices, ["A", "B"], min_observations=10)
        assert report.rows_dropped == 2
        assert report.dropped_dates == list(prices.index[[10, 11]])
        assert not aligned.isna().any().any()
        # no fabricated values: every remaining price is an original observation
        pd.testing.assert_frame_equal(aligned, prices.drop(prices.index[[10, 11]]))

    def test_return_across_a_dropped_gap_is_exact(self):
        prices = price_frame(gaps={"B": [10]})
        aligned, _ = prepare_prices(prices, ["A", "B"], min_observations=10)
        returns = compute_returns(aligned)
        # A's return on day 11 spans days 9 -> 11: 1.01^2 - 1
        assert returns.loc[prices.index[11], "A"] == pytest.approx(1.01**2 - 1)
        # cumulative return is unchanged by dropping the date
        assert (1 + returns["A"]).prod() - 1 == pytest.approx(prices["A"].iloc[-1] / prices["A"].iloc[0] - 1)

    def test_ffill_policy_is_bounded_and_reported(self):
        prices = price_frame(gaps={"B": [5, 20, 21, 22, 23, 24]})
        aligned, report = prepare_prices(prices, ["A", "B"], missing_policy="ffill", ffill_limit=3, min_observations=10)
        # single-day gap filled; the 5-day gap has 3 filled and 2 left missing -> dropped
        assert report.cells_forward_filled == 4
        assert report.rows_dropped == 2
        assert aligned.loc[prices.index[5], "B"] == prices["B"].iloc[4]
        assert report.per_asset.loc["B", "forward_filled"] == 4
        assert any("forward-filled" in w for w in report.warnings)

    def test_late_start_trims_window_and_warns(self):
        prices = price_frame(start_nan={"B": 5})
        aligned, report = prepare_prices(prices, ["A", "B"], min_observations=10)
        assert aligned.index[0] == prices.index[5]
        assert report.effective_start == prices.index[5]
        assert any("B" in w and "no earlier" in w for w in report.warnings)

    def test_date_range_filter(self):
        prices = price_frame()
        aligned, _ = prepare_prices(prices, ["A"], start=prices.index[5], end=prices.index[30], min_observations=10)
        assert aligned.index[0] == prices.index[5] and aligned.index[-1] == prices.index[30]

    def test_insufficient_history_rejected(self):
        with pytest.raises(InsufficientDataError, match="at least 30"):
            prepare_prices(price_frame(n=20), ["A", "B"])

    def test_no_overlap_rejected(self):
        prices = price_frame()
        prices.iloc[20:, 0] = np.nan  # A stops
        prices.iloc[:20, 1] = np.nan  # B starts after
        with pytest.raises(InsufficientDataError, match="overlapping"):
            prepare_prices(prices, ["A", "B"], min_observations=1)

    def test_unknown_ticker_rejected(self):
        with pytest.raises(DataValidationError, match="not in the price data"):
            prepare_prices(price_frame(), ["A", "ZZZ"])

    def test_invalid_policy_rejected(self):
        with pytest.raises(ValueError):
            prepare_prices(price_frame(), ["A"], missing_policy="interpolate")


class TestComputeReturns:
    def test_simple_returns(self):
        prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=pd.bdate_range("2024-01-01", periods=3))
        returns = compute_returns(prices)
        assert returns["A"].tolist() == pytest.approx([0.10, -0.10])
        assert returns.index[0] == prices.index[1]

    def test_missing_prices_rejected(self):
        prices = pd.DataFrame({"A": [100.0, np.nan, 99.0]}, index=pd.bdate_range("2024-01-01", periods=3))
        with pytest.raises(DataValidationError):
            compute_returns(prices)


# --------------------------------------------------------------------------- #
# Frequency detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "index, expected",
    [
        (pd.bdate_range("2024-01-01", periods=100), 252),
        (pd.date_range("2024-01-01", periods=100, freq="D"), 365),
        (pd.date_range("2024-01-05", periods=60, freq="W-FRI"), 52),
        (pd.date_range("2020-01-31", periods=36, freq="ME"), 12),
        (pd.date_range("2015-03-31", periods=20, freq="QE"), 4),
    ],
)
def test_infer_frequency(index, expected):
    assert infer_frequency(index).periods_per_year == expected


def test_irregular_frequency_rejected():
    with pytest.raises(DataValidationError):
        infer_frequency(pd.DatetimeIndex(["2024-01-01", "2024-01-20", "2024-03-01", "2024-04-15"]))


# --------------------------------------------------------------------------- #
# Sample dataset
# --------------------------------------------------------------------------- #
class TestSampleData:
    def test_sample_file_matches_generator(self):
        on_disk = load_sample_prices()
        regenerated = generate_sample_prices()
        assert list(on_disk.columns) == list(regenerated.columns)
        pd.testing.assert_index_equal(on_disk.index, regenerated.index, check_names=False)
        np.testing.assert_allclose(on_disk.to_numpy(), regenerated.to_numpy(), atol=2e-4, equal_nan=True)

    def test_sample_is_clearly_synthetic(self):
        prices = load_sample_prices()
        assert all(c.startswith("SYN_") for c in prices.columns)
        assert SAMPLE_DATA_PATH.exists()

    def test_documented_gaps_are_present(self):
        prices = load_sample_prices()
        assert prices["SYN_EM_EQUITY"].first_valid_index() == pd.Timestamp("2017-03-01")
        dec_26 = prices.index[(prices.index.month == 12) & (prices.index.day == 26)]
        assert prices.loc[dec_26, "SYN_INTL_EQUITY"].isna().all()
        assert prices.drop(columns=["SYN_EM_EQUITY", "SYN_INTL_EQUITY"]).notna().all().all()

    def test_generator_is_deterministic(self):
        pd.testing.assert_frame_equal(generate_sample_prices(), generate_sample_prices())


# --------------------------------------------------------------------------- #
# Yahoo Finance adapter (network-free: the downloader is stubbed)
# --------------------------------------------------------------------------- #
def fake_download(frame):
    def _download(*args, **kwargs):
        assert kwargs["auto_adjust"] is True
        return frame

    return _download


class TestYahooAdapter:
    def test_multiindex_close_extracted_and_failed_tickers_reported(self):
        idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], tz="America/New_York")
        cols = pd.MultiIndex.from_product([["Close", "Volume"], ["AAA", "BAD"]])
        raw = pd.DataFrame([[10.0, np.nan, 1, 1], [11.0, np.nan, 1, 1]], index=idx, columns=cols)
        prices, failed = fetch_yfinance_prices(
            ["aaa", "bad"], "2024-01-01", "2024-01-05", download_fn=fake_download(raw)
        )
        assert list(prices.columns) == ["AAA"]
        assert failed == ["BAD"]
        assert prices.index.tz is None
        assert prices.index[0] == pd.Timestamp("2024-01-02")

    def test_single_level_columns(self):
        raw = pd.DataFrame({"Close": [10.0, 10.5]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
        prices, failed = fetch_yfinance_prices("AAA", "2024-01-01", "2024-01-05", download_fn=fake_download(raw))
        assert prices["AAA"].tolist() == [10.0, 10.5] and failed == []

    def test_empty_download_raises(self):
        with pytest.raises(DataSourceError, match="no price data"):
            fetch_yfinance_prices("AAA", "2024-01-01", "2024-01-05", download_fn=fake_download(pd.DataFrame()))

    def test_network_error_wrapped(self):
        def broken(*args, **kwargs):
            raise ConnectionError("offline")

        with pytest.raises(DataSourceError, match="offline"):
            fetch_yfinance_prices("AAA", "2024-01-01", "2024-01-05", download_fn=broken)

    def test_end_date_is_made_inclusive(self):
        captured = {}

        def capture(*args, **kwargs):
            captured.update(kwargs)
            return pd.DataFrame({"Close": [1.0, 1.1]}, index=pd.to_datetime(["2024-01-02", "2024-01-05"]))

        fetch_yfinance_prices("AAA", "2024-01-01", "2024-01-05", download_fn=capture)
        assert captured["end"] == "2024-01-06"

    def test_ticker_cleaning(self):
        assert clean_tickers(" spy, qqq ;SPY  brk-b ^gspc") == ["SPY", "QQQ", "BRK-B", "^GSPC"]
        with pytest.raises(DataValidationError):
            clean_tickers("SPY, BAD$TICKER")
        with pytest.raises(DataValidationError):
            clean_tickers("   ")
