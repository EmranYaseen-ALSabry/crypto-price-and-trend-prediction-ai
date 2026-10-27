# Phase 3.4 — Controlled Multi-Asset Model Benchmarking: Experiment Design

**Role:** Senior Quantitative ML Engineer, Time-Series Researcher, and ML Validation Architect  
**Project:** `crypto-price-and-trend-prediction-ai`  
**Date:** 2026-10-27  
**Artifact Status:** APPROVED & REPRODUCED  

---

## 1. Executive Summary & Research Hypothesis

The primary objective of Phase 3.4 is to execute a rigorous, leak-free, 2x2 factorial controlled experiment comparing single-asset and multi-asset forecasting models across identical targets, horizons, and evaluation timestamps.

### Research Hypothesis
Can a forecasting model trained on a pooled panel of multiple cryptocurrencies learn transferable market structures and lead-lag dynamics that outperform independently trained single-asset models out of sample?

To prevent cherry-picking and isolate confounding factors, the experiment isolates two orthogonal dimensions:
1. **Information Dimension:** Asset-specific features only vs. Cross-asset features (peer leads/lags, correlations, volatilities, and market aggregates).
2. **Architecture / Training Dimension:** Separate single-asset estimators vs. Shared pooled multi-asset estimators.

---

## 2. 2x2 Factorial Matrix Design

The 2x2 factorial design systematically decomposes the empirical performance into four distinct approaches:

```
                          Information Space
                     Asset-Specific     Cross-Asset
                   +------------------+------------------+
    Single-Asset   |   Approach A     |   Approach B     |
    Independent    |  (Single-Asset   |  (Single + Cross |
    Estimators     |    Baseline)     |    Features)     |
Model              +------------------+------------------+
Training           |   Approach C     |   Approach D     |
    Multi-Asset    |  (Pooled Multi-  |  (Pooled Multi + |
    Shared Model   | Asset Baseline)  | Cross Features)  |
                   +------------------+------------------+
```

### Approach Definitions

1. **Approach A: Single-Asset Baseline**
   - Independent model instance per cryptocurrency ($f_{\text{BTC}}, f_{\text{ETH}}, f_{\text{SOL}}$).
   - Input feature space: 49 asset-specific features (returns, momentum, volatility, RSI, MACD, ATR, volume spreads).
   - Training dataset: Only historical data of the target asset.
   - Serves as the primary benchmark against which all enhancements are judged.

2. **Approach B: Single-Asset + Cross-Asset**
   - Independent model instance per cryptocurrency ($f_{\text{BTC}}, f_{\text{ETH}}, f_{\text{SOL}}$).
   - Input feature space: 79 cross-asset features (49 asset-specific features + 30 cross-asset peer features, including pairwise lead-lag returns, correlation windows, volatility ratios, momentum spreads, and market aggregates).
   - Training dataset: Only historical data of the target asset.
   - Isolates the pure effect of cross-asset information without parameter sharing.

3. **Approach C: Multi-Asset Baseline**
   - Single shared model instance ($f_{\text{shared}}$) trained on the pooled panel of all three assets.
   - Input feature space: 49 asset-specific features + one-hot asset indicator variables (`asset_BTCUSDT`, `asset_ETHUSDT`, `asset_SOLUSDT`).
   - Training dataset: Concatenated historical rows across BTCUSDT, ETHUSDT, and SOLUSDT.
   - Volatility-inverse sample weighting ($w_i = 1 / \sigma_i^2$) applied to balance gradient contributions across assets with different volatility scales.
   - Isolates the pure effect of shared parameter learning without cross-asset features.

4. **Approach D: Multi-Asset + Cross-Asset**
   - Single shared model instance ($f_{\text{shared}}$) trained on the pooled panel of all three assets.
   - Input feature space: 79 cross-asset features + one-hot asset indicator variables.
   - Training dataset: Concatenated historical rows across all three assets.
   - Combines both shared representation learning and lead-lag feature interactions.

---

## 3. Strict Validation & Anti-Leakage Protocol

### 3.1 Target Definition & Horizon
- Target: $h=1$ forward log return defined strictly as:
  $$y_{t} = r_{t+1} = \ln\left(\frac{P_{t+1}}{P_t}\right)$$
- Feature availability timestamp: All features at bar $t$ are computed using information available strictly at or before $t$ (candle close timestamp).
- Target alignment: Shifted backward by 1 bar such that row $t$ contains $X_t$ and $y_t = r_{t+1}$.

### 3.2 Chronological Walk-Forward Regimes & 24-Bar Purging
To evaluate generalizability across distinct market conditions, the dataset is evaluated over 3 walk-forward folds:

| Fold | Regime Name | Train Bars | Train Date Range (UTC) | Purge Gap | Test Bars | Test Date Range (UTC) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 2023 Recovery Regime | 8,000 | 2023-01-08 to 2023-12-07 | 24 bars | 2,000 | 2023-12-08 to 2024-02-29 |
| **Fold 2** | 2024 ETF Momentum Regime | 16,000 | 2023-01-08 to 2024-11-04 | 24 bars | 2,000 | 2024-11-05 to 2025-01-28 |
| **Fold 3** | 2025 High-Volatility Regime | 24,000 | 2023-01-08 to 2025-10-04 | 24 bars | 2,000 | 2025-10-05 to 2025-12-27 |

- **Purging Gap:** A mandatory 24-bar buffer separates the training window from the test window in every fold to completely eliminate lookahead leakage and auto-correlation overlap.
- **Untouched Holdout Isolation:** Bars 26,024 to 33,324 (~7,300 bars spanning 2026) are strictly isolated and never accessed during model training or cross-validation.
- **Identical Evaluation Timestamps:** For any given fold and symbol, Approaches A, B, C, and D are evaluated on the exact same 2,000 test observations.

### 3.3 Volatility-Inverse Sample Weighting
In pooled training (Approaches C and D), assets exhibit markedly different volatility regimes:
- BTC 24h Volatility: $\sim 1.5\% - 3\%$
- ETH 24h Volatility: $\sim 2.5\% - 5\%$
- SOL 24h Volatility: $\sim 5\% - 12\%$

Unweighted squared-error loss ($\sum (y_i - \hat{y}_i)^2$) would be dominated by SOL's larger variance, biasing the tree splits and regression coefficients toward high-volatility moves. To ensure balanced multi-task learning, observations are weighted by inverse variance:
$$w_{i,t} = \frac{1}{\sigma_{i,24h,t}^2}$$
Weights are then normalized per asset so that each asset contributes equally to the total loss:
$$w'_{i,t} = w_{i,t} \cdot \frac{N_i}{\sum_{k=1}^{N_i} w_{i,k}}$$

---

## 4. Evaluated Model Families

Six distinct model families are benchmarked across all approaches and folds:

1. **ZeroReturnBaseline:** Always predicts 0.0. The universal financial floor benchmark.
2. **MeanReturnBaseline:** Predicts the historical training mean return ($\bar{y}_{\text{train}}$).
3. **Ridge Regression:** L2-regularized linear model ($\alpha = 10.0$) with closed-form sample weight scaling.
4. **LightGBM:** Fast gradient-boosted decision trees (`n_estimators=150`, `learning_rate=0.03`, `max_depth=4`, `num_leaves=15`, `colsample_bytree=0.8`).
5. **CatBoost:** Symmetric decision trees with native sample weighting (`iterations=150`, `learning_rate=0.03`, `depth=4`).
6. **XGBoost:** Depth-wise gradient-boosted trees (`n_estimators=150`, `learning_rate=0.03`, `max_depth=4`, `subsample=0.8`, `colsample_bytree=0.8`).

### Evaluation Scale
$$\text{Total Evaluations} = 3 \text{ Folds} \times 4 \text{ Approaches} \times 6 \text{ Models} \times 3 \text{ Symbols} = 216 \text{ Evaluations}$$

---

## 5. Evaluation Metrics

For every evaluation, the following metrics are recorded:
1. **RMSE (Root Mean Squared Error):** Primary loss metric.
2. **MAE (Mean Absolute Error):** Robust error metric.
3. **Financial OOS $R^2$ (vs. Train Mean):**
   $$R^2_{\text{OOS}} = 1 - \frac{\sum_{t} (y_t - \hat{y}_t)^2}{\sum_{t} (y_t - \bar{y}_{\text{train}})^2}$$
4. **OOS $R^2$ (vs. Zero Return):**
   $$R^2_{\text{zero}} = 1 - \frac{\sum_{t} (y_t - \hat{y}_t)^2}{\sum_{t} y_t^2}$$
5. **Directional Accuracy:**
   $$\text{DA} = \frac{1}{|T_{\text{active}}|} \sum_{t \in T_{\text{active}}} \mathbb{I}(\text{sign}(\hat{y}_t) == \text{sign}(y_t))$$
6. **Relative Improvement vs. Approach A ($\Delta_A\%$):**
   $$\Delta_A\% = \frac{\text{RMSE}_A - \text{RMSE}}{\text{RMSE}_A} \times 100\%$$
7. **Relative Improvement vs. ZeroReturn Baseline ($\Delta_{\text{zero}}\%$):**
   $$\Delta_{\text{zero}}\% = \frac{\text{RMSE}_{\text{zero}} - \text{RMSE}}{\text{RMSE}_{\text{zero}}} \times 100\%$$
