# Phase 3.3 — Causal Cross-Asset Feature Engineering & Anti-Leakage Validation

**Document ID:** `reports/phase3_3_feature_validation.md`  
**Author:** Quantitative Research Engineer & ML Validation Specialist  
**Date:** 2026-10-27  
**Status:** Audited & Causally Certified  
**Target Audience:** Quantitative Researchers, Machine Learning Engineers, and Risk Auditors  

---

## 1. Executive Summary

Phase 3.3 implemented and verified the causal multi-asset feature pipeline: [`CrossAssetFeatureEngineer`](file:///c:/Users/OMRRAN/Desktop/Cource%20AI/#4-%20ML/Competitions/trend%20ai/Crypto%20Price%20Prediction%20AI/src/crypto_prediction/features/cross_asset.py).

The engineered feature space expands the platform from purely isolated single-asset technicals to a rich, stationary, and scale-invariant **67-feature cross-asset matrix** per asset.

**Core Audit Results:**
1. **Mandatory Future Perturbation Causality Test:** **PASSED** (Max difference across all 67 features and all assets: `0.00e+00`, strictly $< 10^{-10}$).
2. **Target Isolation Audit:** **PASSED** (Zero target, forward return, or future price columns present in input features).
3. **Scale Invariance:** **PASSED** (Raw unscaled prices $O, H, L, C, V$ are completely excluded from features; all inputs are stationary returns, bounded ratios, or normalized spreads).
4. **Mathematical Bound Consistency:** **PASSED** (Correlations strictly in $[-1.0, 1.0]$, volatility ratios $> 0$, RSI in $[0, 100]$).
5. **Timeline and Usable Sample Count Reconciliation:**
   - Input Synchronized Panel: 33,493 bars (`2023-01-01 00:00:00` to `2026-10-27 13:00:00 UTC`).
   - Maximum Feature Warmup Lookback: 168 hours (7 days, required for `volatility_168h` and `cross_corr_168h`).
   - Primary Forecast Horizon: 1 hour (requires $t+1$ target).
   - Dropped Boundary Rows: Exactly $168 + 1 = 169$ rows.
   - Usable Output Matrix: **33,324 continuous rows per asset** (`2023-01-08 00:00:00` to `2026-10-27 12:00:00 UTC`) with **zero NaN values**.

---

## 2. Feature Space Taxonomy (67 Features per Asset)

The feature matrix is partitioned into three complementary functional tiers:

### Group A: Asset-Specific Causal Features (41 Features)
- **Returns and Lags (8):**
  - `return_1h`: Immediate 1-period return $(P_t - P_{t-1}) / P_{t-1}$.
  - `return_lag_{1, 2, 3, 6, 12, 24, 48}`: Multi-bar returns over trailing $k$ hours.
- **Realized Volatility (6):**
  - `volatility_{6, 12, 24, 48, 168}h`: Trailing sample standard deviation of hourly returns (`center=False`).
  - `volatility_parkinson_24h`: Parkinson (1980) extreme-value High-Low volatility estimator:
    $$\sigma_P = \sqrt{\frac{1}{4 \ln 2 \cdot 24} \sum_{\tau=0}^{23} \left(\ln\frac{H_{t-\tau}}{L_{t-\tau}}\right)^2}$$
- **Momentum & Moving Average Ratios (7):**
  - `momentum_{6, 12, 24, 168}h`: Cumulative trailing multi-period returns.
  - `return_mean_{12, 24, 48}h`: Trailing arithmetic mean of returns.
- **Volume Dynamics (3):**
  - `volume_change_1h`: Percentage change in volume clipped to $[-1.0, 10.0]$.
  - `volume_sma_ratio_{12, 24}`: Volume relative to trailing simple moving average $(V_t / \text{SMA}(V, w)) - 1.0$.
- **Candle Morphology (4):**
  - `candle_hl_spread`: $(H_t - L_t) / O_t$ (intra-bar range).
  - `candle_co_spread`: $(C_t - O_t) / O_t$ (normalized candle body).
  - `candle_upper_shadow`: $(H_t - \max(O_t, C_t)) / O_t$ (upper wick).
  - `candle_lower_shadow`: $(\min(O_t, C_t) - L_t) / O_t$ (lower wick).
- **Technical Indicators (9):**
  - `rsi_{7, 14, 28}`: Wilder's Relative Strength Index over 7, 14, 28 periods.
  - `macd_line`, `macd_signal`, `macd_hist`: MACD components normalized by close price ($/ C_t$).
  - `bb_percent_b`: Position within 20-period 2-std Bollinger Bands.
  - `bb_bandwidth`: Bollinger Bandwidth $(U - L) / \text{SMA}_{20}$.
  - `atr_normalized_14`: 14-period Average True Range divided by close price.
- **Cyclical Calendar Features (4):**
  - `hour_sin`, `hour_cos`, `day_sin`, `day_cos`: Sine/Cosine encodings of UTC hour and day-of-week.

---

### Group B: Cross-Asset Lead-Lag & Relative Features (23 Features)
For each target asset (e.g. SOLUSDT):
- **Peer Returns and Lags (9):**
  - `btc_return_lag_{1, 2, 3}`: Trailing returns of Bitcoin at $t, t-1, t-2$.
  - `eth_return_lag_{1, 2, 3}`: Trailing returns of Ethereum at $t, t-1, t-2$.
  - `sol_return_lag_{1, 2, 3}`: Trailing returns of Solana at $t, t-1, t-2$.
  *(Crucial for resolving negative transfer in altcoins by providing explicit Bitcoin lead information).*
- **Cross-Asset Volatility Ratios (4):**
  - `cross_vol_ratio_btc_24h` = $\sigma_{\text{asset}, 24\text{h}} / (\sigma_{\text{BTC}, 24\text{h}} + 10^{-9})$
  - `cross_vol_ratio_btc_168h` = $\sigma_{\text{asset}, 168\text{h}} / (\sigma_{\text{BTC}, 168\text{h}} + 10^{-9})$
  - `cross_vol_ratio_eth_24h`, `cross_vol_ratio_eth_168h`: Relative volatility versus Ethereum.
- **Rolling Pearson Correlations (4):**
  - `cross_corr_btc_24h`: Trailing 24-hour correlation of 1h returns with Bitcoin.
  - `cross_corr_btc_168h`: Trailing 168-hour (7-day) correlation with Bitcoin.
  - `cross_corr_eth_24h`, `cross_corr_eth_168h`: Trailing correlations with Ethereum.
- **Relative Momentum Spreads (3):**
  - `cross_mom_spread_btc_1h`: 1-hour return spread $r_{\text{asset}, 1\text{h}} - r_{\text{BTC}, 1\text{h}}$.
  - `cross_mom_spread_btc_24h`: 24-hour return spread $r_{\text{asset}, 24\text{h}} - r_{\text{BTC}, 24\text{h}}$.
  - `cross_mom_spread_eth_1h`, `cross_mom_spread_eth_24h`: Return spreads against Ethereum.
- **Market-Wide Aggregate Indicators (3):**
  - `market_mean_return_1h`: Equal-weighted cross-sectional market return across active assets at $t$.
  - `market_return_dispersion_1h`: Cross-sectional standard deviation of returns across assets at $t$.
  - `market_mean_volatility_24h`: Equal-weighted 24h market volatility at $t$.

---

### Group C: Volatility-Aware & Regime Normalization (3 Features)
- `return_vol_standardized_24h`: Realized return standardized by trailing volatility ($r_{1\text{h}} / \sigma_{24\text{h}}$).
- `vol_ratio_24h_168h`: Term-structure volatility ratio ($\sigma_{24\text{h}} / \sigma_{168\text{h}}$) capturing short-term volatility expansion/contraction relative to the weekly baseline.

---

## 3. Mandatory Future Perturbation Causality Test

To guarantee zero lookahead bias, the entire multi-asset pipeline was subjected to the **Future Price Shock Audit** implemented in [`tests/test_cross_asset_features.py`](file:///c:/Users/OMRRAN/Desktop/Cource%20AI/#4-%20ML/Competitions/trend%20ai/Crypto%20Price%20Prediction%20AI/tests/test_cross_asset_features.py):

### Methodology:
1. Select observation boundary $T_{\text{cutoff}} = 250$ in a 400-bar multi-asset synchronized dataset.
2. Compute baseline feature matrices for BTC, ETH, and SOL on unperturbed data.
3. Heavily perturb future raw market data strictly at $t > T_{\text{cutoff}}$:
   - All future prices ($O, H, L, C$) for ALL assets are multiplied by **$3.5\times$**.
   - All future volumes ($V$) are multiplied by **$10.0\times$**.
   - Candlestick boundary conditions ($H \ge \max(O,C), L \le \min(O,C)$) are maintained.
4. Recompute cross-asset feature matrices on the perturbed multi-asset panel.
5. Calculate maximum absolute difference:
   $$\Delta_{\max} = \max_{i \in \{\text{BTC}, \text{ETH}, \text{SOL}\}} \max_{f \in \text{Features}} \max_{t \le T_{\text{cutoff}}} |X_{\text{orig}, i, t, f} - X_{\text{pert}, i, t, f}|$$

### Results:
- **Maximum Absolute Discrepancy ($\Delta_{\max}$):** **`0.00e+00`**
- **Threshold Allowed:** $1.00\times 10^{-10}$
- **Outcome:** **PASSED** (100% causal invariance; zero lookahead detected).

---

## 4. Target Isolation Verification

Every column in feature matrix $X$ was inspected against the forbidden substring list:
`["target", "future", "ahead", "shift_-", "lead"]`.

- Leaked target columns detected: **0**
- Input features strictly represent information available at or before observation time $t$.
- Primary forward target $y = (C_{t+1} - C_t) / C_t$ is cleanly isolated in separate series $y$.
