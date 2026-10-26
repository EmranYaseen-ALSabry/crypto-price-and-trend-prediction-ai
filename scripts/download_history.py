"""Download multi-year OHLCV history from Binance (live data only, no synthetic fallback).

Usage:
    python scripts/download_history.py --years 3
    python scripts/download_history.py --symbols BTCUSDT ETHUSDT --years 2 --interval 1h

Run it on a machine with internet access. Data is cached to data/raw/<SYMBOL>_<interval>.csv,
which the pipeline then reuses. If Binance is unreachable (e.g. HTTP 451 in restricted regions),
the script fails loudly instead of silently substituting sample or synthetic data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crypto_prediction.data.collector import DataCollector  # noqa: E402
from crypto_prediction.data.validator import DataValidator  # noqa: E402

BARS_PER_DAY = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "4h": 6, "1d": 1}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Binance OHLCV history.")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--interval", default="1h", choices=list(BARS_PER_DAY))
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--min-bars", type=int, default=8000,
                        help="Fail if fewer closed bars than this are returned (guards against partial downloads).")
    args = parser.parse_args()

    total_bars = int(args.years * 365 * BARS_PER_DAY[args.interval])
    collector = DataCollector(base_dir=ROOT / "data")
    validator = DataValidator(expected_interval=args.interval)

    failures: list[str] = []
    for symbol in args.symbols:
        print(f"\n=== {symbol}: requesting {total_bars} x {args.interval} bars ===")
        try:
            df = collector.fetch_historical_bars(
                symbol=symbol, interval=args.interval, total_bars=total_bars, fallback_to_sample=False,
            )
        except Exception as exc:  # network error, HTTP 451/429, etc.
            print(f"[FAIL] {symbol}: {exc}")
            failures.append(symbol)
            continue

        source = df.attrs.get("data_source", "unknown")
        if source != "binance_live" or df.attrs.get("is_synthetic", False):
            print(f"[FAIL] {symbol}: data source is '{source}', refusing to treat it as real history.")
            failures.append(symbol)
            continue

        report = validator.validate(df)
        print(report.summary())
        print(f"Range: {df['open_time'].min()} -> {df['open_time'].max()} ({len(df)} bars)")
        if len(df) < args.min_bars:
            print(f"[FAIL] {symbol}: only {len(df)} bars (< {args.min_bars}).")
            failures.append(symbol)
        elif not report.is_valid:
            print(f"[WARN] {symbol}: validation reported errors; the cleaner will repair small gaps only.")

    if failures:
        print(f"\nFailed: {', '.join(failures)}")
        return 1
    print("\nAll downloads complete. Files are in data/raw/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
