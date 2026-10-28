from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from risk_engine.market import INSTRUMENTS, MarketDataError

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_all_app_sections_and_horizon():
    st.cache_data.clear()
    rng = np.random.default_rng(44)
    prices = pd.DataFrame(
        np.exp(np.cumsum(rng.normal(0, 0.01, (420, 4)), axis=0)) * np.array([1.1, 1.3, 3.0, 80.0]),
        index=pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=420),
        columns=[item.ticker for item in INSTRUMENTS],
    ).rename_axis("Date")
    with patch("risk_engine.market.download_prices", return_value=(prices, 0)):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        assert not app.exception
        assert len(app.metric) == 4
        for section in ["Dekompozycja", "Stress i limity", "Backtest", "Dane i eksport"]:
            app.segmented_control[0].set_value(section).run()
            assert not app.exception, section
        app.sidebar.select_slider[1].set_value(10).run()
        assert not app.exception
        assert len(app.get("download_button")) == 4
        app.sidebar.selectbox[0].select("Import CSV").run()
        assert not app.exception
        assert any("Wybierz plik" in item.value for item in app.info)


def test_yahoo_failure_is_visible():
    st.cache_data.clear()
    with patch(
        "risk_engine.market.download_prices", side_effect=MarketDataError("Yahoo niedostepne")
    ):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        assert not app.exception
        assert any("Yahoo niedostepne" in error.value for error in app.error)


def test_saved_yahoo_sample_is_usable_without_network():
    st.cache_data.clear()
    with patch(
        "risk_engine.market.download_prices", side_effect=MarketDataError("Offline")
    ) as download:
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        download.reset_mock()
        app.sidebar.selectbox[0].select("Yahoo Finance (zapisany przyklad)").run()
        assert not app.exception
        assert not app.error
        assert len(app.metric) == 4
        download.assert_not_called()
