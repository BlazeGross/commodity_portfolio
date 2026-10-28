from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from risk_engine.analytics import daily_returns
from risk_engine.market import (
    INSTRUMENTS,
    MarketDataError,
    download_prices,
    read_prices_csv,
    validate_prices,
)


def prices():
    return pd.DataFrame(
        {"A": np.linspace(1, 2, 70), "B": np.linspace(2, 4, 70)},
        index=pd.bdate_range("2024-01-01", periods=70),
    ).rename_axis("Date")


def test_csv_roundtrip():
    frame = prices()
    clean, dropped = read_prices_csv(frame.to_csv().encode())
    pd.testing.assert_frame_equal(clean, frame, check_freq=False)
    assert dropped == 0


def test_no_forward_fill_and_sorted():
    frame = prices().iloc[::-1]
    frame.iloc[1, 0] = np.nan
    clean = validate_prices(frame)
    assert len(clean) == 69
    assert clean.index.is_monotonic_increasing
    assert len(daily_returns(frame)) == 68
    assert not clean.isna().any().any()


@pytest.mark.parametrize("invalid", [0, -1, np.inf, -np.inf])
def test_invalid_price(invalid):
    frame = prices()
    frame.iloc[0, 0] = invalid
    with pytest.raises(MarketDataError):
        validate_prices(frame)


def test_invalid_dates_and_short_history():
    frame = prices()
    frame.index = ["not-a-date"] * len(frame)
    with pytest.raises(MarketDataError):
        validate_prices(frame)
    with pytest.raises(MarketDataError):
        validate_prices(prices().iloc[:20])
    with pytest.raises(MarketDataError):
        validate_prices(pd.concat([prices(), prices().iloc[[0]]]))


@pytest.mark.parametrize(
    "content",
    [b"", b"x,y\n1,2", b"Date,A\n2024-01-01,bad", b"Date,A,A\n2024-01-01,1,2"],
)
def test_bad_csv(content):
    with pytest.raises(MarketDataError):
        read_prices_csv(content)


def test_yahoo_mapping_and_no_missing_symbol_fallback():
    frame = pd.DataFrame(
        {item.ticker: np.linspace(1, 2, 70) for item in INSTRUMENTS},
        index=prices().index,
    )
    raw = pd.concat({"Close": frame}, axis=1)
    with patch("risk_engine.market.yf.download", return_value=raw) as download:
        clean, dropped = download_prices("2024-01-01", "2025-01-01")
    assert list(clean.columns) == [item.ticker for item in INSTRUMENTS]
    assert dropped == 0
    assert download.call_args.kwargs["auto_adjust"] is True
    raw[("Close", "NG=F")] = np.nan
    with (
        patch("risk_engine.market.yf.download", return_value=raw),
        pytest.raises(MarketDataError, match="NG=F"),
    ):
        download_prices("2024-01-01", "2025-01-01")
    with (
        patch("risk_engine.market.yf.download", return_value=pd.DataFrame()),
        pytest.raises(MarketDataError),
    ):
        download_prices("2024-01-01", "2025-01-01")
