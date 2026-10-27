# Phase 2 Baseline Report: Data Coverage, Integrity, and Architecture Audit

**Project:** Crypto Price and Trend Prediction AI  
**Author:** Senior AI Engineering & Quantitative Research Team  
**Date:** 2026-10-27  
**Status:** Baseline Established  
**Reference Phase:** Phase 1 Remediation (`reports/phase1_remediation.md`)

---

## 1. Executive Summary

Phase 2 investigates the core empirical hypothesis of the project:
> **Research Hypothesis:** Does training machine learning models on a pooled multi-asset cryptocurrency panel improve out-of-sample prediction quality and generalization over single-asset models, and can such models generalize out-of-sample to unseen assets (Leave-One-Asset-Out)?

Before conducting any multi-asset experiments, Stage 0 audits the baseline environment, directly verifies the Phase 1 remediations in source code and runtime behavior, evaluates data availability and timestamp overlap across candidate assets (BTC, ETH, SOL), and documents data constraints.

---

## 2. Environment & Test Suite Verification

### 2.1 Execution Environment
- **Operating System:** Windows (win32)
- **Python Version:** 3.14.5
- **Virtual Environment:** `C:\Users\OMRRAN\Desktop\Cource AI\#4- ML\envairoment\.venv`
- **Core Dependencies:**
  - `numpy`: 2.2.3
  - `pandas`: 2.2.3
  - `scikit-learn`: 1.6.1
  - `xgboost`: 3.4.1
  - `lightgbm`: 4.7.0
  - `catboost`: 1.2.10
  - `optuna`: 5.0.0
  - `pytest`: 9.1.1
  - `torch`: **Not installed** (deliberately tested under explicit fallback governance)

### 2.2 Direct Test Suite Execution
The complete existing test suite was executed from the repository root:
```powershell
& "..\..\..\envairoment\.venv\Scripts\python.exe" -m pytest --tb=short
```

**Results:**
- Collected: 60 items
- Passed: **60 passed in 45.99s**
- Failed: **0**
- Regressions: **None**

---

## 3. Independent Verification of Phase 1 Remediations

Rather than relying on documentation claims, critical Phase 1 implementations were inspected and verified directly in the codebase:

1. **LSTM Identity Governance (`src/crypto_prediction/models/lstm.py`):**
   - Verified that `allow_mlp_fallback` defaults to `False`.
   - Verified that when PyTorch is missing, instantiation raises `ImportError` unless explicitly opted-in.
   - Verified `@property is_genuine_lstm` returns `False` when using MLP fallback.
   - Verified `LSTMModel.load()` detects legacy MLP artifacts and assigns `"MLPRegressor (Legacy LSTM Fallback)"`.

2. **Booster Split & Collapse Diagnostics (`src/crypto_prediction/models/boosters.py`):**
   - Verified `get_split_diagnostics()` inspects internal trees for XGBoost, LightGBM, and CatBoost.
   - Confirmed `is_degenerate_constant` and `is_collapsed_to_mean` flags alert if variance collapses to zero.
   - Verified target standardization (`standardize_target=True`) properly inverse-transforms predictions during `predict()`.

3. **Artifact Naming & Schema Integrity (`predict.py`, `trainer.py`):**
   - Documented naming standard established: `{symbol}_{interval}_{name}_h{horizon}.joblib`.
   - Confirmed `predict.py` defaults to `auto_train_if_missing=False`, requiring `--train-if-missing` for on-the-fly fitting.
   - Confirmed schema validation rejects feature dimension mismatches (e.g. 35 vs 46 features).

4. **Backtest Cost Friction (`src/crypto_prediction/pipeline/backtester.py`):**
   - Verified unit friction is applied per order leg: $0.10\%$ fee $+ 0.05\%$ slippage $= 0.0015$.
   - Verified round-trip entry threshold from cash $= 2 \times 0.0015 = 0.0030$.
   - Verified separate reporting of gross return, transaction costs, and net return.

---

## 4. Data Coverage Audit by Asset and Interval

The repository maintains data in two tiers: `data/raw/` (unprocessed real market downloads) and `data/sample/` (synchronized benchmark datasets). All files were audited for OHLCV invariants, completeness, timestamps, and feature readiness:

| Asset & Source File | Interval | Row Count | First Timestamp (UTC) | Last Timestamp (UTC) | Duplicates | Gaps / Invariants | Feature Readiness (Warmup 168) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`data/raw/BTCUSDT_1h.csv`** | 1h | 399 | 2026-10-10 21:00:00 | 2026-10-27 11:00:00 | 0 | 0 gaps, valid OHLCV | 230 usable bars (46 features) |
| **`data/raw/ETHUSDT_1h.csv`** | 1h | 49,999 | 2021-02-09 10:00:00 | 2026-10-25 06:00:00 | 0 | 7 historical gaps | 49,830 usable bars (46 features) |
| **`data/raw/SOLUSDT_1h.csv`** | 1h | 1,000 | 2026-09-13 01:00:00 | 2026-10-24 16:00:00 | 0 | 0 gaps, valid OHLCV | 831 usable bars (46 features) |
| **`data/sample/BTCUSDT_1h.csv`** | 1h | 1,500 | 2024-01-01 00:00:00 | 2024-03-03 11:00:00 | 0 | 0 gaps, valid OHLCV | 1,331 usable bars (46 features) |
| **`data/sample/ETHUSDT_1h.csv`** | 1h | 1,500 | 2024-01-01 00:00:00 | 2024-03-03 11:00:00 | 0 | 0 gaps, valid OHLCV | 1,331 usable bars (46 features) |
| **`data/sample/SOLUSDT_1h.csv`** | 1h | 1,500 | 2024-01-01 00:00:00 | 2024-03-03 11:00:00 | 0 | 0 gaps, valid OHLCV | 1,331 usable bars (46 features) |

### 4.1 Statistical Target Distribution (1-Hour Return Horizon)
When prepared with `FeatureEngineer(primary_horizon=1, include_patterns=True)`:
- **Sample BTCUSDT:** $\mu = -0.000095$, $\sigma = 0.008339$, range $[-0.0293, +0.0268]$
- **Sample ETHUSDT:** $\mu = +0.000176$, $\sigma = 0.010340$, range $[-0.0346, +0.0357]$
- **Sample SOLUSDT:** $\mu = +0.000247$, $\sigma = 0.013048$, range $[-0.0447, +0.0490]$

As expected in financial theory, SOL exhibits higher volatility ($\sigma \approx 1.30\%$) than ETH ($\sigma \approx 1.03\%$) and BTC ($\sigma \approx 0.83\%$).

---

## 5. Overlapping Historical Periods & Comparative Feasibility

### 5.1 The `data/raw/` Coverage Bottleneck
In `data/raw/`, the three assets have extreme historical asymmetry:
- Overlap start: $\max(\text{BTC: 2026-10-10 21:00}, \text{ETH: 2021-02-09}, \text{SOL: 2026-09-13}) = \mathbf{2026-10-10\ 21:00:00\ UTC}$
- Overlap end: $\min(\text{BTC: 2026-10-27 11:00}, \text{ETH: 2026-10-25 06:00}, \text{SOL: 2026-10-24 16:00}) = \mathbf{2026-10-24\ 16:00:00\ UTC}$
- Total mutual overlap: **332 hours (13.8 days)**.
- After standard 168-hour rolling warmup: **only 164 observations**.

> [!WARNING]
> Attempting to train single-asset models, a pooled multi-asset model, and perform leave-one-out cross-validation across 164 time steps would leave only $\sim 115$ training bars, $\sim 25$ validation bars, and $\sim 24$ test bars per asset. This is statistically insufficient for robust model evaluation and would suffer from severe small-sample noise.

### 5.2 The `data/sample/` Standardized Benchmark Panel
In contrast, `data/sample/` provides 1,500 hourly candles across BTC, ETH, and SOL over an identical time horizon (`2024-01-01 00:00:00` to `2024-03-03 11:00:00 UTC`):
- Mutual overlap: **1,500 hours (62.5 days)**.
- Post-warmup observations per asset: **1,331 bars**.
- Total pooled panel: **3,993 observations**.
- Chronological split (70% train, 15% val, 15% test with 1-bar purging):
  - Train: 931 bars per asset (2,793 panel rows)
  - Validation: 199 bars per asset (597 panel rows)
  - Test: 200 bars per asset (600 panel rows)

Furthermore, `DataCollector` can synchronize fresh real data from Binance REST API on-demand if deeper real market coverage is desired, while remaining fully deterministic and offline-executable via `data/sample/`.

---

## 6. Baseline Assessment Conclusion

The baseline audit confirms:
1. All Phase 1 remediations are functionally operational and verified by 60 passing tests.
2. The codebase enforces zero data leakage, chronological ordering, and purging.
3. The experimental design for Phase 2 must use the synchronized 1,500-bar benchmark panel (`data/sample/`) to ensure identical time intervals, regime alignment, and fair cross-asset comparability without small-sample distortion.
4. Stage 1 (Leakage-Safe Experiment Design) can now proceed.
