# Phase 1 Remediation Report: Evidence-Based Engineering Fixes

**Project:** Crypto Price and Trend Prediction AI  
**Author:** Senior AI Engineering & Quantitative Research Team  
**Date:** 2026-10-27  
**Status:** Complete (Verified: 60/60 tests passing)  
**Reference Document:** `reports/engineering_audit.md`

---

## 1. Executive Summary

Following the comprehensive audit documented in `reports/engineering_audit.md`, Phase 1 focused on evidence-based remediations across four critical engineering dimensions:
1. **Model Identity Integrity:** Eliminating silent misrepresentation of MLP regressors as genuine recurrent LSTMs.
2. **Booster Prediction Dynamics:** Diagnosing split collapse in tree-based models and embedding automated tree-split and variance diagnostics.
3. **Artifact Persistence & Schema Governance:** Establishing a unified, documented artifact naming convention (`{symbol}_{interval}_{name}_h{horizon}.joblib`) with sidecar metadata and strict schema validation to prevent silent retraining during inference.
4. **Execution Cost Accounting in Backtesting:** Tracing the complete lifecycle of order execution, eliminating frictional double-counting, calibrating realistic entry/exit thresholds, and reporting gross returns, transaction costs, and net returns separately.
5. **Portability & Dependency Hardening:** Resolving unstated dependencies (`statsmodels`), ensuring graceful notebook fallbacks, and parameterizing runtime scripts (`run_predict.bat`, `run_predict.ps1`) for dynamic environment resolution.

All changes were implemented in-place, without altering the research data partitions or breaking existing APIs. The test suite expanded from 55 to **60 comprehensive unit and integration tests**, achieving a **100% pass rate** in 46.64 seconds.

---

## 2. Task 1: LSTM Model Identity & Fallback Governance

### 2.1 The Issue
The audit identified that when PyTorch was not installed, `src/crypto_prediction/models/lstm.py` silently instantiated an ordinary `sklearn.neural_network.MLPRegressor`, yet labeled the model and saved artifacts as `"LSTM"`. This violated experimental integrity and misleadingly suggested deep sequence modeling was operational.

### 2.2 Implemented Remediation
- **Explicit Dependency Check & Opt-In Fallback:** In `src/crypto_prediction/models/lstm.py`, `LSTMModel` now defaults to `allow_mlp_fallback=False`. If PyTorch is unavailable and the user has not explicitly set `allow_mlp_fallback=True`, initialization raises an informative `ImportError`:
  ```python
  ImportError: PyTorch (torch) is not installed. To train a genuine recurrent LSTM model, install PyTorch (torch>=2.2.0). To deliberately use a scikit-learn MLP fallback in environments without PyTorch, pass allow_mlp_fallback=True.
  ```
- **Strict Labeling & Identity Properties:** Added properties:
  - `is_genuine_lstm`: Returns `True` if and only if PyTorch is available and recurrent `nn.LSTM` weights exist.
  - `model_family`: Returns `"recurrent_lstm"` or `"mlp_fallback"`.
  - Name reflects truth: `"PyTorch-LSTM"` or `"MLPRegressor (LSTM Fallback)"`.
- **Legacy Artifact Detection:** Overrode `LSTMModel.load()` to inspect serialized weights. If an artifact lacks recurrent tensors, it logs a warning and renames the instance to `"MLPRegressor (Legacy LSTM Fallback)"`.
- **Pipeline Integration:** In `src/crypto_prediction/pipeline/trainer.py`, `instantiate_models()` checks `self.config.get("allow_mlp_fallback", False)`. When PyTorch is absent and fallback is disabled, LSTM is cleanly omitted without breaking other models. When fallback is enabled, it registers under `"MLP_Fallback"`.

### 2.3 Verification
- Tested dependency absence: `LSTMModel()` raises `ImportError`.
- Tested explicit fallback: `LSTMModel(allow_mlp_fallback=True)` produces `is_genuine_lstm=False` and `model_family="mlp_fallback"`.
- Tested legacy artifact deserialization: flags MLP artifact accurately.

---

## 3. Task 2: Booster Collapse Diagnostics & Target Scaling

### 3.1 The Issue
On raw hourly returns ($O(10^{-3})$), standard split criteria in gradient boosted decision trees (XGBoost and LightGBM) can fail to find splits under default minimum hessian weights and regularization, causing trees to collapse into 0 splits (`0:leaf=0.0`) and predicting the constant training mean.

### 3.2 Implemented Remediation
- **Integrated Split & Prediction Diagnostics:** Added `get_split_diagnostics(X=None)` to `XGBoostModel`, `LightGBMModel`, and `CatBoostModel` in `src/crypto_prediction/models/boosters.py`:
  - `total_trees`: Total tree count.
  - `active_split_trees`: Trees possessing at least one decision split.
  - `leaf_only_trees`: Degenerate trees with zero splits.
  - `total_splits`: Cumulative decision splits across all trees.
  - `is_collapsed_to_mean`: Boolean flag alerting when `total_splits == 0`.
  - `prediction_variance` & `prediction_range`: When evaluation features $X$ are supplied.
  - `is_constant_prediction` & `is_degenerate_constant`: Boolean flag triggered if `range < 1e-9` or `variance < 1e-12`.
- **Target Standardization Control:** Added explicit `standardize_target: bool = True` parameter to `XGBoostModel` and `LightGBMModel`. When enabled, targets are standardized during `fit()` using training mean and standard deviation, and predictions are automatically inverse-transformed back to the return domain:
  $$\hat{r} = \hat{y}_{\text{std}} \cdot \sigma_y + \mu_y$$
- **Split Sensitivity Defaults:** Configured conservative minimum child sample thresholds (`min_child_weight=1.0` in XGBoost, `min_child_samples=15`, `min_split_gain=0.0` in LightGBM).

### 3.3 Verification
- Unit test `test_booster_split_diagnostics` verified `total_trees > 0`, `total_splits > 0`, and `is_degenerate_constant == False`.
- Unit test `test_booster_target_scaling_consistency` verified mathematical consistency of inverse transformation.

---

## 4. Task 3: Unified Model Artifact Naming & Schema Invariance

### 4.1 The Issue
The codebase exhibited naming asymmetry: `trainer.py` saved models as `{symbol}_{name}_h{horizon}.joblib` (omitting interval), whereas `predict.py` sought `{symbol}_{interval}_{name}_h{horizon}.joblib`. Furthermore, `predict.py` defaulted to `auto_train_if_missing=True`, which silently retrained models on the entire dataset during live inference, corrupting train/test separation. Additionally, loading older 35-feature artifacts into the current 46-feature pipeline caused unhandled runtime crashes.

### 4.2 Implemented Remediation
- **Documented Artifact Naming Standard:**
  $$\text{Artifact Path: } \texttt{artifacts/models/\{symbol\}\_\{interval\}\_\{model\_name\}\_h\{horizon\}.joblib}$$
  $$\text{Sidecar Metadata: } \texttt{artifacts/models/\{symbol\}\_\{interval\}\_\{model\_name\}\_h\{horizon\}\_meta.json}$$
- **Sidecar Metadata Specification:** Saved sidecar JSON files record:
  - `model_name` and `model_class`
  - `symbol`, `interval`, and `horizon`
  - `feature_names` (ordered list) and `feature_count`
  - `is_genuine_lstm` flag
  - UTC ISO timestamp
- **Safe Inference Governance in `predict.py`:**
  - `auto_train_if_missing` now defaults strictly to `False`.
  - Added CLI flag `--train-if-missing` to require explicit user authorization for dynamic fitting.
  - Implemented backwards-compatible fallback search (checks standard name first, then legacy name).
  - Implemented strict feature schema validation: compares `candidate.feature_names` or internal scaler dimensions against input features. If mismatched, loading is safely aborted with a clear diagnostic message rather than proceeding with corrupted inputs.

### 4.3 Verification
- Unit test `test_artifact_naming_convention_and_schema_validation` verified that artifacts following the new standard serialize/deserialize correctly, and mismatched column structures are rejected.

---

## 5. Task 4: Backtesting Execution Friction, Turnover & Accounting

### 5.1 The Issue
In `src/crypto_prediction/pipeline/backtester.py`:
- Friction accounting required clarity regarding order-level vs. round-trip costs.
- Entry thresholds in cash required formulation based on one-way or round-trip friction without prematurely liquidating marginally profitable positions.
- Gross returns, transaction costs, and net returns were not reported separately, obscuring the cost drag of high-turnover models.

### 5.2 Implemented Remediation
- **Order-Level Friction Model:** Documented that transaction fee ($0.10\%$) and slippage ($0.05\%$) apply per order leg:
  $$\text{unit\_cost} = \text{fee\_rate} + \text{slippage\_rate} = 0.0015 \quad (0.15\% \text{ per side})$$
  - Position opening ($0 \to \pm 1$): $\Delta \text{pos} = 1 \implies \text{friction} = 0.0015$.
  - Position holding ($\pm 1 \to \pm 1$): $\Delta \text{pos} = 0 \implies \text{friction} = 0$.
  - Position reversal ($+1 \to -1$): $\Delta \text{pos} = 2 \implies \text{friction} = 2 \times 0.0015 = 0.0030$.
  - Final liquidation ($\pm 1 \to 0$): $\Delta \text{pos} = 1 \implies \text{friction} = 0.0015$.
- **Calibrated Entry/Exit Logic:**
  - Entry from cash requires expected return to exceed expected round-trip friction ($2 \times \text{unit\_cost}$).
  - Active positions are maintained as long as the expected return remains positive ($\hat{r} > 0$), preventing wasteful whipsaw liquidation on small positive signals.
- **Separate Return Decomposition in `BacktestResult`:**
  - `gross_cumulative_return`: Compounded return without friction.
  - `total_transaction_costs`: Cumulative percentage points lost to execution drag.
  - `cumulative_return`: True net compounded return.
  - Summary table and dictionary export report both gross and net alpha distinctly.

### 5.3 Verification
- Unit test `test_backtest_friction_lifecycle_and_gross_net_separation` analytically validated opening, holding (zero cost), reversal ($2\times$ cost), and final close ($1\times$ cost), confirming separate reporting fields.

---

## 6. Task 5: Dependencies & Script Portability

### 6.1 The Issue
- `notebooks/Crypto_Price_Prediction.ipynb` Cell 8 imported `statsmodels.tsa.stattools.adfuller`, but `statsmodels` was absent from `requirements.txt` and `pyproject.toml`.
- CLI helper scripts `run_predict.bat` and `run_predict.ps1` contained hardcoded relative paths to a specific virtual environment.

### 6.2 Implemented Remediation
- **Notebook Resiliency:** Wrapped Cell 8's ADF test in a `try...except ImportError` block that prints an informative notice and falls back to variance diagnostics if `statsmodels` is not installed.
- **Dependency Specification:** Added `statsmodels>=0.14.0` to `requirements.txt` and `pyproject.toml`.
- **Dynamic Script Resolution:** Updated `run_predict.bat` and `run_predict.ps1` to dynamically check:
  1. Local virtual environment: `.\.venv\Scripts\python.exe`
  2. Outer development workspace: `..\..\..\envairoment\.venv\Scripts\python.exe`
  3. System PATH fallback: `python`

---

## 7. Comprehensive Verification & Test Results

The full test suite was executed in the clean project virtual environment:

```powershell
& "..\..\..\envairoment\.venv\Scripts\python.exe" -m pytest --tb=short
```

### Complete Test Run Summary:
```
============================= test session starts =============================
platform win32 -- Python 3.14.5, pytest-9.1.1, pluggy-1.7.0
rootdir: C:\Users\OMRRAN\Desktop\Cource AI\#4- ML\Competitions\trend ai\Crypto Price Prediction AI
configfile: pyproject.toml
testpaths: tests
plugins: platformdirs-4.13.0
collected 60 items

tests/test_data_validation.py .........                                  [ 15%]
tests/test_dynamic_horizon.py ...                                        [ 20%]
tests/test_feature_engineering.py ....                                   [ 26%]
tests/test_models.py .........                                           [ 41%]
tests/test_patterns.py ......                                            [ 51%]
tests/test_pipeline.py ................                                  [ 78%]
tests/test_temporal_leakage.py ......                                    [ 88%]
tests/test_walk_forward.py .......                                       [100%]

============================= 60 passed in 46.64s =============================
```

### Test Suite Distribution:
| Test Module | Tests | Status | Scope |
| :--- | :---: | :---: | :--- |
| `test_data_validation.py` | 9 | Passed | OHLCV invariants, cleaning, gap handling, duplicate removal |
| `test_dynamic_horizon.py` | 3 | Passed | Arbitrary horizons, dynamic purging, projection visualizer |
| `test_feature_engineering.py` | 4 | Passed | Indicator calculations, bounds, lag/rolling causality |
| `test_models.py` | 9 | Passed | Baselines, Boosters, LSTM fallback governance, diagnostics |
| `test_patterns.py` | 6 | Passed | Candlestick patterns, support/resistance, market regimes |
| `test_pipeline.py` | 16 | Passed | End-to-end trainer, artifact naming, backtest friction lifecycle |
| `test_temporal_leakage.py` | 6 | Passed | Future perturbation, target isolation, walk-forward CV |
| `test_walk_forward.py` | 7 | Passed | Purged chronological folds, noise baselines, signal detection |
| **Total** | **60** | **100% Passed** | **Zero failures, Zero regressions** |

---

## 8. Conclusion & Readiness for Multi-Asset Research

Phase 1 has established an uncompromised, reproducible engineering baseline:
- No model claims exist without genuine underlying architectures.
- Booster behavior is fully observable via internal split and variance telemetry.
- Artifacts are unambiguously identifiable by symbol, interval, model, and horizon.
- Realistic trading friction and turnover are properly accounted for.
- The pipeline runs reliably with zero data leakage.

The platform is now fully prepared for **Phase 2: Multi-Asset Generalization & Transferability Research**.
