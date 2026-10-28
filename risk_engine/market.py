"""Yahoo Finance ingestion and strict, reusable price validation."""

import csv
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import pandas as pd
import yfinance as yf


class MarketDataError(ValueError):
    """Market data cannot be used safely."""


@dataclass(frozen=True)
class Instrument:
    ticker: str
    name: str
    asset_class: str
    unit: str
    multiplier: float
    quantity: float


INSTRUMENTS = (
    Instrument("EURUSD=X", "EUR / USD", "FX", "EUR", 1.0, 250_000.0),
    Instrument("GBPUSD=X", "GBP / USD", "FX", "GBP", 1.0, -100_000.0),
    Instrument("NG=F", "Henry Hub natural gas", "Gas", "contract", 10_000.0, 8.0),
    Instrument("XLU", "Utilities Select Sector SPDR (proxy)", "Power proxy", "share", 1.0, 2_000.0),
)


def default_portfolio() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": item.ticker,
                "name": item.name,
                "asset_class": item.asset_class,
                "unit": item.unit,
                "quantity": item.quantity,
                "multiplier": item.multiplier,
            }
            for item in INSTRUMENTS
        ]
    )


def validate_prices(raw: pd.DataFrame, minimum_rows: int = 61) -> pd.DataFrame:
    if raw.empty or raw.columns.empty:
        raise MarketDataError("Brak cen. Sprawdz symbole i zakres dat.")
    if raw.columns.duplicated().any():
        raise MarketDataError("Nazwy instrumentow musza byc unikalne.")
    frame = raw.copy()
    try:
        dates = pd.to_datetime(frame.index, errors="raise", utc=True)
        frame.index = dates.tz_convert(None).normalize()
        frame = frame.apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise MarketDataError("Niepoprawne daty lub nienumeryczne ceny.") from exc
    if frame.index.isna().any() or frame.index.duplicated().any():
        raise MarketDataError("Daty musza byc poprawne i unikalne (jeden wiersz na dzien).")
    frame = frame.sort_index()
    values = frame.to_numpy(dtype=float)
    if np.isinf(values).any() or (values <= 0).any():
        raise MarketDataError("Ceny musza byc dodatnie i skonczone; brak ceny oznacz pustym polem.")
    frame = frame.dropna(how="any")
    if len(frame) < minimum_rows:
        raise MarketDataError(
            f"Potrzeba co najmniej {minimum_rows} wspolnych dni cen; dostepne: {len(frame)}."
        )
    frame.index.name = "Date"
    return frame.astype(float)


def read_prices_csv(content: bytes) -> tuple[pd.DataFrame, int]:
    try:
        header = next(csv.reader(content.decode("utf-8-sig").splitlines()), [])
        if len(header) != len(set(header)) or any(not name.strip() for name in header):
            raise MarketDataError("Nazwy kolumn CSV musza byc niepuste i unikalne.")
        frame = pd.read_csv(BytesIO(content))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise MarketDataError(
            "Nie mozna odczytac CSV. Uzyj UTF-8 i przecinka jako separatora."
        ) from exc
    if "Date" not in frame or len(frame.columns) < 2:
        raise MarketDataError("CSV musi zawierac kolumne Date i co najmniej jeden instrument.")
    if any(str(column).startswith("Unnamed:") for column in frame.columns):
        raise MarketDataError("CSV zawiera kolumne bez nazwy.")
    frame = frame.set_index("Date")
    clean = validate_prices(frame)
    return clean, len(frame) - len(clean)


def download_prices(start: str, end: str) -> tuple[pd.DataFrame, int]:
    """Download adjusted USD closes. Yahoo's end date is exclusive."""
    symbols = [item.ticker for item in INSTRUMENTS]
    try:
        raw = yf.download(
            symbols,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=20,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise MarketDataError(f"Yahoo Finance: {exc}") from exc
    if raw.empty or "Close" not in raw:
        raise MarketDataError("Yahoo Finance nie zwrocilo cen. Sprobuj ponownie lub uzyj CSV.")
    closes = raw["Close"]
    missing = [symbol for symbol in symbols if symbol not in closes or closes[symbol].isna().all()]
    if missing:
        raise MarketDataError(f"Yahoo Finance: brak notowan dla {', '.join(missing)}.")
    frame = closes[symbols]
    clean = validate_prices(frame)
    return clean, len(frame) - len(clean)
