"""Persist an explicitly requested, real Yahoo Finance sample for offline use."""

import argparse
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from risk_engine.market import default_portfolio, download_prices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=(date.today() - timedelta(days=3 * 365)).isoformat())
    parser.add_argument("--end", default=date.today().isoformat(), help="Inclusive ISO date")
    args = parser.parse_args()
    try:
        start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
        if start >= end:
            parser.error("--start musi poprzedzac --end.")
        prices, dropped = download_prices(start.isoformat(), (end + timedelta(days=1)).isoformat())
    except ValueError as exc:
        parser.exit(1, f"Blad pobierania: {exc}\n")
    target = Path(__file__).resolve().parents[1] / "data"
    target.mkdir(exist_ok=True)
    prices.to_csv(target / "yahoo_prices.csv", float_format="%.12g")
    default_portfolio().to_csv(target / "sample_portfolio.csv", index=False)
    metadata = {
        "source": "Yahoo Finance via yfinance",
        "retrieved_utc": datetime.now(UTC).isoformat(),
        "requested_start": start.isoformat(),
        "requested_end_inclusive": end.isoformat(),
        "first_observation": str(prices.index.min().date()),
        "last_observation": str(prices.index.max().date()),
        "observations": len(prices),
        "dropped_rows": dropped,
        "auto_adjust": True,
        "currency": "USD",
        "note": "XLU is a utilities equity proxy, not an electricity spot/forward price.",
    }
    (target / "yahoo_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Zapisano {len(prices)} wspolnych obserwacji do {target}")
    print(prices.tail(1).to_string())


if __name__ == "__main__":
    main()
