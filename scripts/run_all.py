"""Regenerate every result from scratch with honest, multi-fold evaluation.

For each symbol x horizon this runs:
  1. (optional) the classic single-split experiment (PipelineTrainer) -> refreshed artifacts/
  2. walk-forward REGRESSION     -> skill vs. the train-mean baseline, per fold, mean +/- std
  3. walk-forward CLASSIFICATION -> direction AUC / accuracy edge on moves larger than trading costs
  4. out-of-fold TRADING backtest of the classifier signals with fees + slippage and a min-trades flag

Usage (after scripts/download_history.py):
    python scripts/run_all.py --clean
    python scripts/run_all.py --symbols BTCUSDT --horizons 1 4 --folds 6 --test-window 500
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crypto_prediction.data.cleaner import DataCleaner  # noqa: E402
from crypto_prediction.data.collector import DataCollector  # noqa: E402
from crypto_prediction.data.validator import DataValidator  # noqa: E402
from crypto_prediction.features.engineer import FeatureEngineer  # noqa: E402
from crypto_prediction.models.classifiers import DEFAULT_THRESHOLD, default_classifier_factories  # noqa: E402
from crypto_prediction.pipeline.trainer import PipelineTrainer  # noqa: E402
from crypto_prediction.pipeline.walk_forward import (  # noqa: E402
    run_walk_forward_classification,
    run_walk_forward_regression,
    run_walk_forward_trading,
)


def archive_old_artifacts(artifacts: Path) -> None:
    """Moves the previous artifacts aside (never deletes) so stale numbers cannot be mixed with new ones."""
    if artifacts.exists() and any(artifacts.iterdir()):
        dest = artifacts.with_name(f"{artifacts.name}_old_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.move(str(artifacts), str(dest))
        print(f"Archived previous artifacts -> {dest.name}")
    artifacts.mkdir(parents=True, exist_ok=True)


def load_clean_data(collector: DataCollector, symbol: str, interval: str, bars: int) -> pd.DataFrame:
    raw = collector.fetch_historical_bars(symbol=symbol, interval=interval, total_bars=bars, fallback_to_sample=False)
    source = raw.attrs.get("data_source", "unknown")
    if raw.attrs.get("is_synthetic", False) or source in {"synthetic", "local_sample_csv"}:
        raise RuntimeError(f"{symbol}: data source '{source}' is not real history; run scripts/download_history.py.")
    report = DataValidator(expected_interval=interval).validate(raw)
    if not report.is_valid:
        raise RuntimeError(f"{symbol}: data validation failed: {report.errors}")
    print(f"{symbol}: {len(raw)} bars from '{source}' ({raw['open_time'].min()} -> {raw['open_time'].max()})")
    return DataCleaner().clean(raw)


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate all results with walk-forward evaluation.")
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--bars", type=int, default=26000, help="Bars to load (26000 ~ 3 years of 1h).")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--test-window", type=int, default=500)
    ap.add_argument("--min-train", type=int, default=2000)
    ap.add_argument("--p-long", type=float, default=0.55, help="Go long when P(up) >= this.")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Neutral band for direction labels.")
    ap.add_argument("--clean", action="store_true", help="Archive old artifacts/ before running.")
    ap.add_argument("--skip-single", action="store_true", help="Skip the single-split PipelineTrainer experiment.")
    ap.add_argument("--with-lstm", action="store_true", help="Include LSTM in walk-forward regression (slow).")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--artifacts-dir", default=str(ROOT / "artifacts"))
    args = ap.parse_args()

    artifacts = Path(args.artifacts_dir)
    if args.clean:
        archive_old_artifacts(artifacts)
    wf_dir = artifacts / "walk_forward"
    wf_dir.mkdir(parents=True, exist_ok=True)

    collector = DataCollector(base_dir=args.data_dir)
    trainer = PipelineTrainer(artifacts_dir=artifacts)
    clf_factories = default_classifier_factories(threshold=args.threshold)
    all_trading: list[pd.DataFrame] = []

    for symbol in args.symbols:
        try:
            clean_df = load_clean_data(collector, symbol, args.interval, args.bars)
        except Exception as exc:
            print(f"[SKIP] {symbol}: {exc}")
            continue

        for h in args.horizons:
            print(f"\n######## {symbol} | horizon {h}h ########")
            if not args.skip_single:
                try:
                    trainer.run_experiment(symbol=symbol, interval=args.interval, horizon=h, total_bars=args.bars)
                except Exception as exc:
                    print(f"[WARN] single-split experiment failed for {symbol} h{h}: {exc}")

            X, y, _ = FeatureEngineer(primary_horizon=h, forecast_horizons=[1, 4, 24]).prepare_dataset(clean_df, target_horizon=h)
            needed = args.folds * args.test_window + args.min_train + h
            if len(X) < needed:
                print(f"[SKIP] {symbol} h{h}: {len(X)} rows < {needed} needed for {args.folds} walk-forward folds.")
                continue

            names = [n for n in trainer.instantiate_models() if args.with_lstm or n != "LSTM"]
            reg_factories = {n: (lambda n=n: trainer.instantiate_models()[n]) for n in names}
            kw = dict(n_folds=args.folds, test_window=args.test_window, min_train=args.min_train)

            reg_folds, reg_sum = run_walk_forward_regression(X, y, reg_factories, purge_bars=h, **kw)
            clf_folds, clf_sum = run_walk_forward_classification(X, y, clf_factories, threshold=args.threshold, purge_bars=h, **kw)
            trading = run_walk_forward_trading(X, y, clf_factories, horizon=h, p_long=args.p_long, **kw)
            trading.insert(0, "horizon", h)
            trading.insert(0, "symbol", symbol)
            all_trading.append(trading)

            tag = f"{symbol}_h{h}"
            reg_folds.to_csv(wf_dir / f"{tag}_regression_folds.csv", index=False)
            reg_sum.to_csv(wf_dir / f"{tag}_regression_summary.csv", index=False)
            clf_folds.to_csv(wf_dir / f"{tag}_classification_folds.csv", index=False)
            clf_sum.to_csv(wf_dir / f"{tag}_classification_summary.csv", index=False)
            trading.to_csv(wf_dir / f"{tag}_trading.csv", index=False)

            pd.set_option("display.width", 200)
            print("\nRegression (rmse_skill > 0 beats the train-mean baseline):")
            print(reg_sum[["model_name", "rmse_skill_mean", "rmse_skill_std", "directional_accuracy_mean", "folds_beating_baseline", "n_folds"]].round(4).to_string(index=False))
            print("\nDirection classification (AUC 0.5 = random; edge > 0 beats majority class):")
            print(clf_sum.round(3).to_string(index=False))
            print("\nOut-of-fold trading (net of fees + slippage):")
            print(trading[["model_name", "total_trades", "cumulative_return", "benchmark_return", "sharpe_ratio", "max_drawdown", "statistically_reliable"]].round(4).to_string(index=False))

    if all_trading:
        pd.concat(all_trading).to_csv(wf_dir / "ALL_trading_summary.csv", index=False)
    print(f"\nDone. Results in {wf_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
