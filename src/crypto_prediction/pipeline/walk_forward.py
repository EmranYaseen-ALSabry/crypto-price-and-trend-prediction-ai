"""Multi-fold walk-forward evaluation (regression and direction classification).

A single chronological split gives one noisy number. Here the last ``n_folds * test_window`` bars are
cut into consecutive, NON-overlapping test windows; each fold trains on everything before its window
(expanding, minus a purge gap equal to the forecast horizon) and is scored on its own window. The
summary reports mean and std across folds so a lucky split cannot masquerade as skill.
"""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_squared_error, r2_score, roc_auc_score

from crypto_prediction.models.classifiers import DEFAULT_THRESHOLD, make_direction_labels


def make_folds(
    n: int,
    n_folds: int = 6,
    test_window: int = 500,
    purge_bars: int = 1,
    min_train: int = 2000,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yields (train_idx, test_idx) with disjoint, consecutive test windows and an expanding train set."""
    first_test_start = n - n_folds * test_window
    if first_test_start - purge_bars < min_train:
        raise ValueError(
            f"Not enough data: n={n} but {n_folds} folds x {test_window} bars + min_train={min_train} "
            f"+ purge={purge_bars} need at least {n_folds * test_window + min_train + purge_bars} rows."
        )
    for k in range(n_folds):
        test_start = first_test_start + k * test_window
        test_end = test_start + test_window
        yield np.arange(0, test_start - purge_bars), np.arange(test_start, test_end)


def _summarise(fold_df: pd.DataFrame, group_col: str, metric_cols: list[str]) -> pd.DataFrame:
    agg = fold_df.groupby(group_col)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_folds"] = fold_df.groupby(group_col).size()
    return agg.reset_index()


def run_walk_forward_regression(
    X: pd.DataFrame,
    y: pd.Series,
    model_factories: dict[str, Callable[[], object]],
    n_folds: int = 6,
    test_window: int = 500,
    purge_bars: int = 1,
    min_train: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-fold metrics, summary). ``rmse_skill`` = 1 - rmse / rmse_of_train_mean_baseline;
    values <= 0 mean the model does not beat predicting the training-window mean return."""
    rows = []
    for fold, (tr, te) in enumerate(make_folds(len(X), n_folds, test_window, purge_bars, min_train)):
        X_tr, y_tr, X_te, y_te = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
        base_rmse = float(np.sqrt(np.mean((y_te.to_numpy() - y_tr.mean()) ** 2)))
        for name, factory in model_factories.items():
            model = factory()
            model.fit(X_tr, y_tr)
            pred = np.asarray(model.predict(X_te), dtype=np.float64)
            rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
            active = pred != 0
            dir_acc = float(np.mean(np.sign(pred[active]) == np.sign(y_te.to_numpy()[active]))) if active.any() else np.nan
            rows.append({
                "fold": fold, "model_name": name, "test_start": X.index[te[0]], "test_end": X.index[te[-1]],
                "rmse": rmse, "baseline_rmse": base_rmse, "rmse_skill": 1.0 - rmse / base_rmse,
                "r2": float(r2_score(y_te, pred)), "directional_accuracy": dir_acc,
            })
    fold_df = pd.DataFrame(rows)
    summary = _summarise(fold_df, "model_name", ["rmse_skill", "r2", "directional_accuracy"])
    summary["folds_beating_baseline"] = fold_df.groupby("model_name")["rmse_skill"].apply(lambda s: int((s > 0).sum())).values
    return fold_df, summary.sort_values("rmse_skill_mean", ascending=False).reset_index(drop=True)


def run_walk_forward_classification(
    X: pd.DataFrame,
    y: pd.Series,
    classifier_factories: dict[str, Callable[[], object]],
    threshold: float = DEFAULT_THRESHOLD,
    n_folds: int = 6,
    test_window: int = 500,
    purge_bars: int = 1,
    min_train: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Direction skill on test bars whose |forward return| exceeds ``threshold`` (neutral band excluded).

    ``accuracy_edge`` = accuracy - share of the majority class in that fold; edge <= 0 means no
    better than always guessing the more frequent direction. AUC 0.5 = random.
    """
    rows = []
    for fold, (tr, te) in enumerate(make_folds(len(X), n_folds, test_window, purge_bars, min_train)):
        X_tr, y_tr, X_te, y_te = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
        lab_te = make_direction_labels(y_te, threshold)
        keep = lab_te >= 0
        if keep.sum() < 20 or len(np.unique(lab_te[keep])) < 2:
            continue
        majority = float(max(lab_te[keep].mean(), 1 - lab_te[keep].mean()))
        for name, factory in classifier_factories.items():
            clf = factory()
            clf.fit(X_tr, y_tr)
            p_up = clf.predict_proba_up(X_te.iloc[np.flatnonzero(keep)])
            pred_lab = (p_up >= 0.5).astype(int)
            acc = float(np.mean(pred_lab == lab_te[keep]))
            rows.append({
                "fold": fold, "model_name": name, "n_eval_bars": int(keep.sum()),
                "accuracy": acc, "majority_share": majority, "accuracy_edge": acc - majority,
                "balanced_accuracy": float(balanced_accuracy_score(lab_te[keep], pred_lab)),
                "auc": float(roc_auc_score(lab_te[keep], p_up)),
            })
    fold_df = pd.DataFrame(rows)
    if fold_df.empty:
        return fold_df, fold_df
    summary = _summarise(fold_df, "model_name", ["accuracy_edge", "balanced_accuracy", "auc"])
    return fold_df, summary.sort_values("auc_mean", ascending=False).reset_index(drop=True)


def run_walk_forward_trading(
    X: pd.DataFrame,
    y: pd.Series,
    classifier_factories: dict[str, Callable[[], object]],
    horizon: int = 1,
    p_long: float = 0.55,
    n_folds: int = 6,
    test_window: int = 500,
    min_train: int = 2000,
    fee_rate: float = 0.0010,
    slippage_rate: float = 0.0005,
    min_trades: int = 30,
) -> pd.DataFrame:
    """Backtests out-of-fold direction signals (long when P(up) >= p_long, else cash) with fees and slippage.

    Folds are stitched in time order, so the result is one genuinely out-of-sample equity path.
    ``statistically_reliable`` is False when fewer than ``min_trades`` trades were taken.
    """
    from crypto_prediction.pipeline.backtester import StrategyBacktester

    folds = list(make_folds(len(X), n_folds, test_window, horizon, min_train))
    te_all = np.concatenate([te for _, te in folds])
    y_te_all = y.iloc[te_all].to_numpy()
    backtester = StrategyBacktester(fee_rate=fee_rate, slippage_rate=slippage_rate, min_trades=min_trades)

    rows = []
    for name, factory in classifier_factories.items():
        p_all = []
        for tr, te in folds:
            clf = factory()
            clf.fit(X.iloc[tr], y.iloc[tr])
            p_all.append(clf.predict_proba_up(X.iloc[te]))
        signal = (np.concatenate(p_all) >= p_long).astype(np.float64)  # 1.0 -> long, 0.0 -> cash
        res = backtester.run_backtest(y_te_all, signal, model_name=name, horizon=horizon)
        rows.append(res.to_dict())
    return pd.DataFrame(rows)
