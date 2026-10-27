# Phase 3.5 — Cross-Asset Transfer Learning & Volatility Adaptation: Experiment Design

**Role:** Senior Quantitative ML Engineer, Transfer Learning Researcher, and Time-Series Validation Architect  
**Project:** `crypto-price-and-trend-prediction-ai`  
**Date:** 2026-10-27  
**Artifact Status:** APPROVED & REPRODUCED  

---

## 1. Executive Summary & Research Hypothesis

Phase 3.5 investigates whether machine learning models trained exclusively on a subset of cryptocurrencies (source assets) can learn generalizable market dynamics and transfer effectively to an unseen cryptocurrency (target asset) under strict Leave-One-Asset-Out (LOAO) conditions.

### Core Research Questions:
1. **Zero-Shot Transferability:** Can a model trained on source assets (e.g. BTC + ETH) predict forward returns on an unseen asset (e.g. SOL) without ever observing a single label from that asset?
2. **Transfer Asymmetry:** Does transferability depend on the relative liquidity, volatility, and market capitalization between source and target assets (e.g., transferring down from high-volatility to low-volatility vs. transferring up)?
3. **Volatility Adaptation:** Does training on volatility-standardized return targets ($z_{i,t} = r_{i,t+1} / \sigma_{i,24h,t}$) and restoring predictions using causal target-asset volatility at test time eliminate scale mismatches and improve out-of-sample RMSE?
4. **Few-Shot Adaptation:** Does fine-tuning a source pre-trained model on a small, causal historical training slice of the target asset (residual calibration) outperform pure zero-shot transfer and close the gap to asset-specific models?

---

## 2. Leave-One-Asset-Out (LOAO) Protocol & Strict Isolation

### 2.1 LOAO Target Permutations
Three distinct LOAO transfer scenarios were evaluated across all folds:

| Permutation | Source Assets (Training Only) | Target Asset (Held-Out Evaluation) | Market Dynamics Tested |
| :---: | :---: | :---: | :--- |
| **LOAO 1** | ETHUSDT + SOLUSDT | **BTCUSDT** | Transferring from higher-volatility altcoins to the market benchmark. |
| **LOAO 2** | BTCUSDT + SOLUSDT | **ETHUSDT** | Transferring from market leader and high-beta altcoin to smart contract platform. |
| **LOAO 3** | BTCUSDT + ETHUSDT | **SOLUSDT** | Transferring from mega-cap low-volatility assets to high-beta, high-volatility asset. |

### 2.2 Strict Target Isolation Contract
- **Zero Label Contamination:** The held-out target asset's labels ($y$) and forward returns are strictly excluded from source model training, hyperparameter configuration, and feature preprocessing.
- **Asset-Agnostic Feature Set:** Only 52 universal, scale-invariant technical indicators, momentum metrics, volatility estimators, and market-wide aggregates are provided to transfer models. Asset-specific ticker prefixes (such as `btc_return_lag_*` or `cross_corr_eth_*`) are pruned to ensure the model cannot overfit to specific ticker identities.
- **Causal Volatility Scaling:** All volatility normalizations and inverse transformations use only information available at or before candle close $t$ ($\sigma_{24h,t}$), guaranteeing zero lookahead bias.

---

## 3. Four Transfer Learning Modes

To isolate the contributions of pre-training, volatility adaptation, and target fine-tuning, four distinct modes were evaluated:

### Mode 1: Zero-Shot Raw (`zero_shot_raw`)
- **Training:** Fit base model $f_{\text{source}}$ purely on source assets using un-normalized raw log returns ($y_{\text{source}} = r_{\text{source},t+1}$).
- **Target Exposure:** **Zero.** The model never encounters any sample from the target asset.
- **Inference:** Directly evaluate on target test data: $\hat{y}_{\text{test}} = f_{\text{source}}(X_{\text{target, test}})$.
- **Purpose:** Measures baseline transferability without any adaptation or domain adjustment.

### Mode 2: Zero-Shot Volatility-Adapted (`zero_shot_vol_adapted`)
- **Training:** Standardize target returns by trailing 24h causal volatility:
  $$z_{\text{source},t} = \frac{r_{\text{source},t+1}}{\sigma_{\text{source},24h,t}}$$
  Fit model $f_{\text{source}}$ on source assets to predict standardized return score $z$.
- **Target Exposure:** **Zero.** No target training samples used.
- **Inference:** Model predicts standardized score $\hat{z}_{\text{test}} = f_{\text{source}}(X_{\text{target, test}})$.
- **Inverse Transformation:** Restore forecast to target asset's return units using target's causal trailing volatility:
  $$\hat{y}_{\text{test}} = \hat{z}_{\text{test}} \cdot \sigma_{\text{target},24h,t}$$
- **Purpose:** Tests whether dynamic volatility scaling overcomes the cross-asset scale mismatch.

### Mode 3: Few-Shot Volatility-Adapted (`few_shot_vol_adapted`)
- **Training:** Base model $f_{\text{source}}$ is pre-trained on source assets using standardized returns $z$.
- **Few-Shot Adaptation Slice:** An explicitly separated historical window of $N_{\text{few}} = 1,000$ bars from the end of the target asset's training partition is used:
  $$\text{Window} = [\text{train\_end} - 1000, \text{train\_end})$$
  *(Note: This slice lies strictly before the mandatory 24-bar purge gap and never touches the test set).*
- **Residual Fitting:** Compute source predictions on target training slice: $\hat{z}_{\text{few}} = f_{\text{source}}(X_{\text{target, few}})$.
  Compute residuals: $e_{\text{few}} = z_{\text{target, few}} - \hat{z}_{\text{few}}$.
  Fit a regularized adapter model $g_{\text{adapter}}$ on $(X_{\text{target, few}}, e_{\text{few}})$.
- **Inference:** 
  $$\hat{z}_{\text{adapted}} = f_{\text{source}}(X_{\text{target, test}}) + g_{\text{adapter}}(X_{\text{target, test}})$$
  $$\hat{y}_{\text{test}} = \hat{z}_{\text{adapted}} \cdot \sigma_{\text{target},24h,t}$$
- **Purpose:** Evaluates whether low-capacity target residual adaptation recovers asset-specific idiosyncrasies.

### Mode 4: Asset-Specific Baseline (`asset_specific_baseline`)
- **Training:** Model trained solely on the target asset's historical training partition ($[0, \text{train\_end})$).
- **Target Exposure:** 100% target training history.
- **Inference:** Evaluated on the exact same test window as the transfer models.
- **Purpose:** Serves as the primary counterfactual benchmark (equivalent to Phase 3.4 Approach A).

---

## 4. Chronological Walk-Forward Folds & Purging

To ensure direct comparability with Phase 3.4, the exact same 3 chronological walk-forward folds and 24-bar purge gaps were utilized:

| Fold | Regime Name | Train Bars | Train Range (UTC) | Purge Gap | Test Bars | Test Range (UTC) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 2023 Recovery Regime | 8,000 | 2023-01-08 to 2023-12-07 | 24 bars | 2,000 | 2023-12-08 to 2024-02-29 |
| **Fold 2** | 2024 ETF Momentum Regime | 16,000 | 2023-01-08 to 2024-11-04 | 24 bars | 2,000 | 2024-11-05 to 2025-01-28 |
| **Fold 3** | 2025 High-Volatility Regime | 24,000 | 2023-01-08 to 2025-10-04 | 24 bars | 2,000 | 2025-10-05 to 2025-12-27 |

- **Quarantined Holdout:** Bars 26,024 to 33,324 (~7,300 bars spanning 2026) are strictly isolated.
- **Test Set Uniformity:** For any given fold and target asset, all 4 modes and 6 models are evaluated on the exact same 2,000 out-of-sample bars.

---

## 5. Evaluated Model Families & Evaluation Scale

Six model families were benchmarked across all folds, targets, and modes:
1. `ZeroReturnBaseline`: Predicts 0.0.
2. `MeanReturnBaseline`: Predicts training mean.
3. `Ridge`: L2-regularized linear model ($\alpha=10.0$ for base, $\alpha=50.0$ for adapter).
4. `LightGBM`: GBDT with gradient boosting (`n_est=150` base, `n_est=50` adapter).
5. `CatBoost`: Symmetric decision trees (`iter=150` base, `iter=50` adapter).
6. `XGBoost`: Depth-wise decision trees (`n_est=150` base, `n_est=50` adapter).

$$\text{Total Evaluations} = 3 \text{ Folds} \times 3 \text{ Target Assets} \times 4 \text{ Modes} \times 6 \text{ Models} = 216 \text{ Evaluations}$$
