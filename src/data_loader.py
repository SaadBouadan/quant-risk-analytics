"""Loading, validating and aligning price data.

Methodology (also documented in docs/METHODOLOGY.md):

* Prices are expected to be **adjusted closing prices** (split- and
  dividend-adjusted), so simple returns approximate total returns.
* Dates must be ISO 8601 (``YYYY-MM-DD``). Ambiguous formats such as
  ``03/04/2024`` are rejected rather than guessed.
* Missing observations are **never silently replaced**. The default policy
  drops any date on which a selected asset has no price. Forward-filling is
  available as an explicit, bounded, reported choice.
* Non-positive or non-numeric prices are rejected with a message naming the
  offending column and date.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .exceptions import DataSourceError, DataValidationError, InsufficientDataError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample_prices.csv"
SAMPLE_BENCHMARK = "SYN_MARKET"

MISSING_POLICIES = ("drop", "ffill")
DEFAULT_MIN_OBSERVATIONS = 30

# Cell contents that conventionally mean "no value" in spreadsheet / vendor exports.
_NA_TOKENS = {"", "na", "n/a", "nan", "null", "none", "#n/a", "-", "--", "."}
_LONG_PRICE_COLUMNS = ("adj close", "adj_close", "adjclose", "adjusted close", "price", "close")
_TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-^=]{1,20}$")


# --------------------------------------------------------------------------- #
# CSV input
# --------------------------------------------------------------------------- #
def read_price_csv(source) -> pd.DataFrame:
    """Read a price file in *wide* or *long* format and return a validated table.

    Wide format (preferred)::

        Date,AAA,BBB
        2024-01-02,100.0,50.0

    Long format::

        Date,Ticker,Adj Close
        2024-01-02,AAA,100.0

    Returns a float DataFrame indexed by a sorted, unique ``DatetimeIndex``
    named ``Date`` with one column per ticker. Missing cells stay ``NaN``.
    """
    try:
        raw = pd.read_csv(source, dtype=str, keep_default_na=False, skipinitialspace=True)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError("The CSV file is empty.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"The CSV file could not be parsed: {exc}") from exc

    raw.columns = [str(c).strip() for c in raw.columns]
    if raw.empty:
        raise DataValidationError("The CSV file has a header but no data rows.")

    if _is_long_format(raw.columns):
        raw = _long_to_wide(raw)
    return _parse_wide(raw)


def _find_column(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    lookup = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def _is_long_format(columns: Iterable[str]) -> bool:
    columns = list(columns)
    return (
        _find_column(columns, ["date"]) is not None
        and _find_column(columns, ["ticker", "symbol"]) is not None
        and _find_column(columns, _LONG_PRICE_COLUMNS) is not None
    )


def _long_to_wide(raw: pd.DataFrame) -> pd.DataFrame:
    date_col = _find_column(raw.columns, ["date"])
    ticker_col = _find_column(raw.columns, ["ticker", "symbol"])
    price_col = _find_column(raw.columns, _LONG_PRICE_COLUMNS)

    frame = raw[[date_col, ticker_col, price_col]].copy()
    frame[ticker_col] = frame[ticker_col].str.strip().str.upper()
    duplicated = frame.duplicated(subset=[date_col, ticker_col], keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, [date_col, ticker_col]].head(3).values.tolist()
        raise DataValidationError(
            f"Duplicate (date, ticker) rows found, e.g. {examples}. Each ticker may have one price per date."
        )
    wide = frame.pivot(index=date_col, columns=ticker_col, values=price_col)
    wide.columns.name = None
    return wide.reset_index().rename(columns={date_col: "Date"})


def _parse_dates(values: pd.Series) -> pd.DatetimeIndex:
    text = values.astype(str).str.strip()
    try:
        parsed = pd.to_datetime(text, format="ISO8601", errors="coerce")
    except (ValueError, TypeError) as exc:  # e.g. mixed timezone offsets
        raise DataValidationError(f"Dates could not be parsed as ISO 8601 (YYYY-MM-DD): {exc}") from exc

    bad = parsed.isna()
    if bad.any():
        examples = text[bad].head(3).tolist()
        raise DataValidationError(
            f"{int(bad.sum())} date value(s) are not valid ISO 8601 dates (YYYY-MM-DD), e.g. {examples}. "
            "Ambiguous formats such as 03/04/2024 are not accepted."
        )
    index = pd.DatetimeIndex(parsed)
    if index.tz is not None:
        index = index.tz_localize(None)  # keep the local trading date
    return index.normalize()


def _parse_wide(raw: pd.DataFrame) -> pd.DataFrame:
    date_col = _find_column(raw.columns, ["date"])
    if date_col is None:
        raise DataValidationError("The CSV must contain a 'Date' column.")
    value_cols = [c for c in raw.columns if c != date_col and c != ""]
    if not value_cols:
        raise DataValidationError("The CSV must contain at least one price column besides 'Date'.")

    index = _parse_dates(raw[date_col])
    if index.duplicated().any():
        dupes = index[index.duplicated()].strftime("%Y-%m-%d").unique()[:5].tolist()
        raise DataValidationError(f"Duplicate dates found: {dupes}. Each date may appear only once.")

    prices = {}
    for col in value_cols:
        column = raw[col]
        text = column.where(column.notna(), "").astype(str).str.strip()
        is_blank = text.str.lower().isin(_NA_TOKENS)
        numeric = pd.to_numeric(text.mask(is_blank), errors="coerce")
        invalid = numeric.isna() & ~is_blank
        if invalid.any():
            pos = np.flatnonzero(invalid.to_numpy())[:3]
            examples = [f"{index[p]:%Y-%m-%d}: '{text.iloc[p]}'" for p in pos]
            raise DataValidationError(f"Column '{col}' contains non-numeric values ({', '.join(examples)}).")
        prices[col.strip().upper()] = numeric.to_numpy(dtype=float)

    frame = pd.DataFrame(prices, index=index)
    frame.index.name = "Date"
    frame = frame.sort_index()
    validate_price_values(frame)
    return frame


def validate_price_values(prices: pd.DataFrame) -> None:
    """Reject empty columns, infinite values and non-positive prices."""
    empty = [c for c in prices.columns if prices[c].notna().sum() == 0]
    if empty:
        raise DataValidationError(f"Column(s) {empty} contain no valid prices.")
    values = prices.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise DataValidationError("Prices contain infinite values.")
    non_positive = prices.le(0) & prices.notna()
    if non_positive.any().any():
        col = non_positive.any().idxmax()
        date = non_positive[col].idxmax()
        raise DataValidationError(
            f"Prices must be strictly positive; '{col}' has {prices.at[date, col]} on {date:%Y-%m-%d}."
        )


def make_csv_template() -> str:
    """Return a small wide-format CSV template users can fill in."""
    return (
        "Date,TICKER_A,TICKER_B,BENCHMARK\n"
        "2024-01-02,100.00,50.00,400.00\n"
        "2024-01-03,101.25,49.80,402.10\n"
        "2024-01-04,100.90,50.35,401.55\n"
    )


def load_sample_prices(path: Path | str = SAMPLE_DATA_PATH) -> pd.DataFrame:
    """Load the bundled synthetic dataset, regenerating it if the file is absent."""
    path = Path(path)
    if not path.exists():
        from .sample_data import write_sample_prices

        write_sample_prices(path)
    return read_price_csv(path)


# --------------------------------------------------------------------------- #
# Yahoo Finance (optional)
# --------------------------------------------------------------------------- #
def clean_tickers(tickers: Iterable[str] | str) -> list[str]:
    """Upper-case, de-duplicate and validate ticker symbols."""
    if isinstance(tickers, str):
        tickers = re.split(r"[,\s;]+", tickers)
    cleaned = list(dict.fromkeys(t.strip().upper() for t in tickers if t and t.strip()))
    invalid = [t for t in cleaned if not _TICKER_PATTERN.match(t)]
    if invalid:
        raise DataValidationError(f"Invalid ticker symbol(s): {invalid}.")
    if not cleaned:
        raise DataValidationError("Enter at least one ticker symbol.")
    return cleaned


def _extract_close(raw: pd.DataFrame | None, tickers: Sequence[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
        elif "Close" in raw.columns.get_level_values(-1):
            close = raw.xs("Close", axis=1, level=-1)
        else:
            return pd.DataFrame()
    elif "Close" in raw.columns:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        return pd.DataFrame()

    close = close.copy()
    close.columns = [str(c).upper() for c in close.columns]
    index = pd.DatetimeIndex(close.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    close.index = index.normalize()
    close.index.name = "Date"
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close.apply(pd.to_numeric, errors="coerce").astype(float)


def fetch_yfinance_prices(
    tickers: Iterable[str] | str,
    start,
    end,
    *,
    download_fn: Callable | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Download split/dividend-adjusted closing prices from Yahoo Finance.

    ``auto_adjust=True`` makes yfinance return adjusted prices in the ``Close``
    field. ``end`` is treated as inclusive (yfinance's own ``end`` is exclusive).
    Returns ``(prices, failed_tickers)``. Raises :class:`DataSourceError` if no
    ticker returns data. ``download_fn`` allows tests to inject a stub.
    """
    symbols = clean_tickers(tickers)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if start_ts >= end_ts:
        raise DataValidationError("The start date must be before the end date.")

    if download_fn is None:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise DataSourceError("yfinance is not installed. Run `pip install yfinance`.") from exc
        download_fn = yf.download

    try:
        raw = download_fn(
            symbols,
            start=start_ts.strftime("%Y-%m-%d"),
            end=(end_ts + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            actions=False,
            progress=False,
            group_by="column",
        )
    except Exception as exc:  # network errors, rate limits, API changes
        raise DataSourceError(f"Yahoo Finance download failed: {exc}") from exc

    prices = _extract_close(raw, symbols)
    available = [t for t in symbols if t in prices.columns and prices[t].notna().any()]
    failed = [t for t in symbols if t not in available]
    if not available:
        raise DataSourceError(
            f"Yahoo Finance returned no price data for {symbols}. Check the symbols, the date range "
            "and your internet connection, or use the bundled sample data."
        )
    prices = prices[available]
    validate_price_values(prices)
    return prices, failed


# --------------------------------------------------------------------------- #
# Alignment and missing-data handling
# --------------------------------------------------------------------------- #
@dataclass
class DataQualityReport:
    """Everything that happened between raw prices and the analysed sample."""

    requested_start: pd.Timestamp | None
    requested_end: pd.Timestamp | None
    effective_start: pd.Timestamp
    effective_end: pd.Timestamp
    n_price_rows: int
    n_return_periods: int
    missing_policy: str
    ffill_limit: int | None
    rows_dropped: int
    dropped_dates: list[pd.Timestamp]
    cells_forward_filled: int
    per_asset: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def prepare_prices(
    prices: pd.DataFrame,
    tickers: Sequence[str],
    start=None,
    end=None,
    *,
    missing_policy: str = "drop",
    ffill_limit: int = 3,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Select assets and dates, align histories and apply the missing-data policy.

    Steps:
      1. Restrict to the requested tickers and ``[start, end]``.
      2. Trim to the **common window** in which every asset has started and not
         yet ended trading (an asset that lists later shortens the sample; this is
         reported, never back-filled).
      3. Inside the window, handle gaps according to ``missing_policy``:
         ``"drop"``  - remove dates where any asset is missing. The next return
                       then spans the gap, so cumulative returns stay exact.
         ``"ffill"`` - carry the last price forward for at most ``ffill_limit``
                       consecutive dates (reported); longer gaps are dropped.
      4. Require at least ``min_observations`` return periods.
    """
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        raise DataValidationError("Select at least one asset.")
    unknown = [t for t in tickers if t not in prices.columns]
    if unknown:
        raise DataValidationError(f"Ticker(s) {unknown} are not in the price data.")
    if missing_policy not in MISSING_POLICIES:
        raise ValueError(f"missing_policy must be one of {MISSING_POLICIES}, got {missing_policy!r}.")
    if missing_policy == "ffill" and ffill_limit < 1:
        raise ValueError("ffill_limit must be at least 1.")

    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None
    if start_ts is not None and end_ts is not None and start_ts >= end_ts:
        raise DataValidationError("The start date must be before the end date.")

    frame = prices.loc[:, tickers].sort_index()
    if start_ts is not None:
        frame = frame.loc[frame.index >= start_ts]
    if end_ts is not None:
        frame = frame.loc[frame.index <= end_ts]
    if frame.empty:
        raise InsufficientDataError("There are no price observations in the selected date range.")

    first_valid = {t: frame[t].first_valid_index() for t in tickers}
    last_valid = {t: frame[t].last_valid_index() for t in tickers}
    no_data = [t for t, d in first_valid.items() if d is None]
    if no_data:
        raise InsufficientDataError(f"No prices for {no_data} in the selected date range.")

    window_start = max(first_valid.values())
    window_end = min(last_valid.values())
    if window_start >= window_end:
        raise InsufficientDataError("The selected assets have no overlapping price history.")

    warnings: list[str] = []
    if window_start > frame.index[0]:
        late = [t for t, d in first_valid.items() if d == window_start]
        warnings.append(
            f"Analysis starts on {window_start:%Y-%m-%d} because {', '.join(late)} has no earlier "
            "prices in the selected range. Earlier data is not back-filled."
        )
    if window_end < frame.index[-1]:
        early = [t for t, d in last_valid.items() if d == window_end]
        warnings.append(f"Analysis ends on {window_end:%Y-%m-%d} because {', '.join(early)} has no later prices.")

    window = frame.loc[window_start:window_end]
    missing_in_window = window.isna().sum()
    filled_per_asset = pd.Series(0, index=tickers, dtype=int)

    if missing_policy == "ffill":
        filled = window.ffill(limit=ffill_limit)
        filled_per_asset = (window.isna() & filled.notna()).sum().astype(int)
        window = filled

    incomplete = window.isna().any(axis=1)
    dropped_dates = list(window.index[incomplete])
    aligned = window.loc[~incomplete]

    n_returns = len(aligned) - 1
    if n_returns < min_observations:
        raise InsufficientDataError(
            f"Only {max(n_returns, 0)} return observations remain after cleaning; at least "
            f"{min_observations} are required. Widen the date range or choose assets with longer histories."
        )

    cells_filled = int(filled_per_asset.sum())
    if cells_filled:
        warnings.append(
            f"{cells_filled} missing price(s) were forward-filled (at most {ffill_limit} consecutive dates). "
            "A forward-filled date shows a 0% return for that asset, which slightly understates volatility."
        )
    if dropped_dates:
        warnings.append(
            f"{len(dropped_dates)} date(s) were removed because at least one selected asset had no price. "
            "The following return spans the gap, so cumulative performance is unaffected."
        )

    per_asset = pd.DataFrame(
        {
            "first_price": [first_valid[t] for t in tickers],
            "last_price": [last_valid[t] for t in tickers],
            "missing_in_window": missing_in_window.reindex(tickers).astype(int).to_numpy(),
            "forward_filled": filled_per_asset.reindex(tickers).to_numpy(),
        },
        index=pd.Index(tickers, name="asset"),
    )

    report = DataQualityReport(
        requested_start=start_ts,
        requested_end=end_ts,
        effective_start=aligned.index[0],
        effective_end=aligned.index[-1],
        n_price_rows=len(aligned),
        n_return_periods=n_returns,
        missing_policy=missing_policy,
        ffill_limit=ffill_limit if missing_policy == "ffill" else None,
        rows_dropped=len(dropped_dates),
        dropped_dates=dropped_dates,
        cells_forward_filled=cells_filled,
        per_asset=per_asset,
        warnings=warnings,
    )
    return aligned, report


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple (arithmetic) returns ``P_t / P_{t-1} - 1`` from aligned prices.

    Simple returns are used because they aggregate across assets
    (portfolio return = weighted sum), unlike log returns.
    """
    if prices.isna().any().any():
        raise DataValidationError("Prices contain missing values; align them with prepare_prices() first.")
    if len(prices) < 2:
        raise InsufficientDataError("At least two price observations are needed to compute a return.")
    returns = prices / prices.shift(1) - 1.0
    return returns.iloc[1:]


# --------------------------------------------------------------------------- #
# Frequency detection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Frequency:
    label: str
    periods_per_year: int


def infer_frequency(index: pd.DatetimeIndex) -> Frequency:
    """Infer the sampling frequency and the matching annualisation factor.

    Daily data on weekdays only -> 252 trading days per year. Daily data that
    includes weekends (e.g. crypto) -> 365. Weekly -> 52, monthly -> 12,
    quarterly -> 4, annual -> 1.
    """
    index = pd.DatetimeIndex(index)
    if len(index) < 3:
        raise InsufficientDataError("At least three dates are needed to infer the data frequency.")
    gaps = np.diff(index.values).astype("timedelta64[s]").astype(float) / 86_400.0
    median_gap = float(np.median(gaps))

    if median_gap <= 1.5:
        weekend_share = float((index.dayofweek >= 5).mean())
        if weekend_share > 0.10:
            return Frequency("daily (calendar days, incl. weekends)", 365)
        return Frequency("daily (trading days)", 252)
    if 5 <= median_gap <= 9:
        return Frequency("weekly", 52)
    if 25 <= median_gap <= 35:
        return Frequency("monthly", 12)
    if 80 <= median_gap <= 100:
        return Frequency("quarterly", 4)
    if 350 <= median_gap <= 380:
        return Frequency("annual", 1)
    raise DataValidationError(
        f"Could not infer a regular data frequency (median gap {median_gap:.1f} days). "
        "Use daily, weekly, monthly, quarterly or annual prices."
    )
