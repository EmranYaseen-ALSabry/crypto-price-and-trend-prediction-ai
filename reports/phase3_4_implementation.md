# Phase 3.4 — Controlled Multi-Asset Model Benchmarking: Technical Implementation Report

**Role:** Senior Quantitative ML Engineer, Time-Series Researcher, and ML Validation Architect  
**Project:** `crypto-price-and-trend-prediction-ai`  
**Date:** 2026-10-27  
**Artifact Status:** APPROVED & REPRODUCED  

---

## 1. Overview & Architectural Components

Phase 3.4 delivers the benchmarking engine and evaluation framework for comparing single-asset and multi-asset cryptocurrency forecasting models under a rigorous 2x2 factorial experimental design.

### Key Components Implemented:
1. **`ControlledBenchmarkEngine` (`src/crypto_prediction/pipeline/benchmarking.py`):**
   - Coordinates multi-regime walk-forward splits, feature partitioning, data pooling, volatility weighting, model fitting, and metric calculations.
2. **Sample Weight Support across Forecasting Models:**
   - Updated `BaseModel`, `RidgeModel`, `LightGBMModel`, `CatBoostModel`, `XGBoostModel`, and `MeanReturnBaseline` to natively accept and process `sample_weight`.
3. **Execution Script (`scripts/run_phase3_4_benchmark.py`):**
   - Provides a CLI interface to execute benchmarks, output summary pivot tables, and serialize evaluation artifacts.
4. **Comprehensive Unit & Integration Test Suite (`tests/test_benchmarking_3_4.py`):**
   - Validates walk-forward boundaries, purge gaps, feature isolation, metric mathematics, sample weight formulas, and end-to-end benchmark execution.

---

## 2. Technical Details & Code Implementation

### 2.1 Walk-Forward Fold Definitions & Anti-Leakage Buffers

Walk-forward folds are defined by the `WalkForwardFoldSpec` dataclass:
```python
@dataclass
class WalkForwardFoldSpec:
    fold_id: int
    name: str
    train_start_idx: int
    train_end_idx: int
    purge_bars: int
    test_start_idx: int
    test_end_idx: int
    train_start_utc: str = ""
    train_end_utc: str = ""
    test_start_utc: str = ""
    test_end_utc: str = ""
```

Three folds were constructed across the 33,324 synchronized hourly bars:
- **Fold 1:** Train: 0 to 8,000; Purge: 24 bars; Test: 8,024 to 10,024 (2,000 bars).
- **Fold 2:** Train: 0 to 16,000; Purge: 24 bars; Test: 16,024 to 18,024 (2,000 bars).
- **Fold 3:** Train: 0 to 24,000; Purge: 24 bars; Test: 24,024 to 26,024 (2,000 bars).
- **Final Holdout:** 26,024 to 33,324 (7,300 bars = 304 days spanning 2026) is strictly quarantined.

### 2.2 Feature Space Partitioning & Unified Panel Schema

The engine partitions feature columns into:
- **Asset-Specific Features (49 columns):** Technical indicators, momentum, volatility, and lags derived solely from the asset's own candlestick history.
- **Cross-Asset Features (79 columns):** The 49 asset-specific features plus 30 cross-asset peer features (`btc_return_lag_*`, `eth_return_lag_*`, `sol_return_lag_*`, `cross_vol_ratio_*`, `cross_corr_*`, `cross_mom_spread_*`, and `market_*` aggregates).

To ensure complete mathematical consistency across pooled data rows:
- Pairwise features for self-relationships are mapped to identity constants:
  - $\text{corr}(A, A) = 1.0$
  - $\text{vol\_ratio}(A, A) = 1.0$
  - $\text{mom\_spread}(A, A) = 0.0$

### 2.3 Volatility-Inverse Sample Weighting in Pooled Models

In Approaches C and D, sample weights are computed from historical 24-hour volatility:
$$w_{i,t} = \frac{1}{\max(\sigma_{i,24h,t}^2, 10^{-8})}$$
To prevent an asset with lower volatility (e.g. BTC) from completely dominating an asset with higher volatility (e.g. SOL) in aggregate sample count, weights are normalized per asset:
$$w'_{i,t} = w_{i,t} \cdot \frac{N_i}{\sum_{k=1}^{N_i} w_{i,k}}$$
This ensures $\sum_{t} w'_{i,t} = N_i$ for all assets, guaranteeing equal asset-level representation in the pooled objective function.

### 2.4 Model Support for `sample_weight`

- **`RidgeModel` (`src/crypto_prediction/models/linear.py`):**
  Uses `Ridge.fit(X, y, sample_weight=sample_weight)`. Standardizes features dynamically within the fold.
- **`LightGBMModel` (`src/crypto_prediction/models/boosters.py`):**
  Passes `sample_weight` directly to `LGBMRegressor.fit(X, y_target, sample_weight=sample_weight)`.
- **`CatBoostModel` (`src/crypto_prediction/models/boosters.py`):**
  Passes `sample_weight` directly to `CatBoostRegressor.fit(X, y, sample_weight=sample_weight)`.
- **`XGBoostModel` (`src/crypto_prediction/models/boosters.py`):**
  Passes `sample_weight` directly to `XGBRegressor.fit(X, y_target, sample_weight=sample_weight)`.
- **`MeanReturnBaseline` (`src/crypto_prediction/models/baselines.py`):**
  Computes weighted mean return: $\bar{y} = \frac{\sum w_i y_i}{\sum w_i}$.

---

## 3. Test Suite Verification & Quality Assurance

A dedicated unit test module (`tests/test_benchmarking_3_4.py`) was created and verified:
1. `test_walk_forward_fold_spec_and_boundaries`:
   - Validates 3 folds, `purge_bars >= 24`, strict zero-lookahead, strictly expanding training sets, and untouched 2026 holdout ($\ge 7,000$ bars).
2. `test_feature_partitioning`:
   - Validates that target and metadata columns are excluded and that asset-specific columns contain zero cross-asset prefixes.
3. `test_metric_calculations`:
   - Validates RMSE, MAE, financial OOS $R^2$, and directional accuracy against perfect, zero, and inverted predictions.
4. `test_sample_weight_inverse_volatility`:
   - Validates inverse quadratic scaling with respect to volatility.
5. `test_mini_benchmark_execution`:
   - Validates end-to-end benchmark execution on synthetic data, schema integrity, and output CSV artifact creation.

### Full Regression Test Run
Command:
```powershell
..\..\..\envairoment\.venv\Scripts\python.exe -m pytest --tb=short
```
Output:
```
======================= 94 passed in 160.38s (0:02:40) ========================
```
100% of all 94 project tests pass without any regressions or warnings.

---

## 4. Execution & Artifact Verification

The benchmark was executed via:
```powershell
..\..\..\envairoment\.venv\Scripts\python.exe scripts/run_phase3_4_benchmark.py
```
- Total execution time: 158.5 seconds.
- Total model evaluations: 216 records.
- Output artifact: `artifacts/phase3_4/phase3_4_benchmark_results.csv`
- Metadata sidecar: `artifacts/phase3_4/phase3_4_benchmark_results.csv.meta.json`

Metadata sidecar contents:
```json
{
  "total_evaluations": 216,
  "approaches": [
    "A_single_asset_baseline",
    "B_single_asset_cross_asset",
    "C_multi_asset_baseline",
    "D_multi_asset_cross_asset"
  ],
  "models": [
    "ZeroReturn",
    "MeanReturn",
    "Ridge",
    "LightGBM",
    "CatBoost",
    "XGBoost"
  ],
  "symbols": [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT"
  ],
  "n_folds": 3,
  "sha256": "5c68f1c07172fe4ba13289069d2d854497fbe13645bbab6dd040f92b7ef5d3b6",
  "generated_at_utc": "2026-10-27 19:16:17.382604+00:00"
}
```

---

## 5. Acceptance Criteria Sign-Off

| Criterion | Requirement | Status | Evidence |
| :--- | :--- | :---: | :--- |
| **Prerequisites** | Phase 3.3 datasets, schemas, and leakage tests verified | **MET** | Verified 33,324 bars, 79 features, zero leakage ($0.00\times 10^{-10}$). |
| **2x2 Factorial Design** | Approaches A, B, C, D implemented and benchmarked | **MET** | All 4 approaches evaluated across 6 models and 3 assets. |
| **Walk-Forward Validation** | Chronological folds with purging, no lookahead | **MET** | 3 folds evaluated with 24-bar purge gap; 2026 holdout quarantined. |
| **Sample Weighting** | Volatility-inverse weights for pooled models | **MET** | Weighted loss integration across Ridge, LightGBM, CatBoost, XGBoost. |
| **Model Diversity** | Linear, GBDT, Baselines | **MET** | ZeroReturn, MeanReturn, Ridge, LightGBM, CatBoost, XGBoost tested. |
| **Identical Test Slices** | Evaluated on identical test timestamps | **MET** | All 4 approaches evaluated on the exact same 2,000 bars per fold. |
| **Reports Generated** | Experiment design, model comparison, implementation | **MET** | `reports/phase3_4_experiment_design.md`, `reports/phase3_4_model_comparison.md`, `reports/phase3_4_implementation.md`. |
| **Unit Tests Passing** | Dedicated suite + zero regression in existing tests | **MET** | 5 new tests in `test_benchmarking_3_4.py`, 94/94 project tests green. |
