# Phase 3.1 Architecture Review: Baseline Freeze, System Audit & Phase 3 Contract

**Project:** Crypto Price and Trend Prediction AI  
**Author:** Senior Software Architect, Quantitative ML Engineer, and Time-Series Research Lead  
**Date:** 2026-10-27  
**Status:** Completed & Approved  
**Base Commit / Tree State:** `ea1cd0871f64c741f291af82f192811c7237a9f5`  
**Deliverables Generated:**  
- `docs/PHASE3_SPEC.md`  
- `docs/PHASE3_HANDOFF.md`  
- `reports/phase3_1_architecture_review.md`  

---

## 1. Executive Summary

Phase 3.1 establishes the architectural foundation and experimental contract for Phase 3 without introducing new models, downloading large external datasets, or modifying core production logic.

### Objectives Achieved:
1. **Architecture Audit:** Systematically audited all existing pipeline components (`data/`, `features/`, `models/`, `pipeline/`, `validation/`), confirming modularity, testability, and zero data leakage.
2. **Phase 2 Baseline Freeze:** Formally froze the Phase 2 multi-asset experimental benchmark, including data ranges, model weights, evaluation metrics, and sample counts ($1,500$ raw $\to 168$ warmup $\to 1$ target shift $\to 1,331$ usable $\to 931$ train, $1$ purge, $199$ val, $1$ purge, $199$ test).
3. **Experimental Contract Defined:** Established invariant specifications for target definitions, information availability timestamps, causal execution order, purging requirements, holdout isolation, and artifact naming standards.
4. **Subphase Roadmaps (3.2–3.7):** Defined detailed inputs, outputs, dependencies, automated tests, and measurable acceptance criteria for each planned Phase 3 subphase.
5. **Zero-Regression Verification:** Confirmed that the repository passes **70/70 automated unit and integration tests** with 100% green status.

---

## 2. Review of Existing Architecture & Component Readiness

The current architecture, established in Phase 1 and extended in Phase 2, demonstrates strong structural maturity:

```
src/crypto_prediction/
├── data/
│   ├── collector.py        [High Maturity] REST fetch, CSV caching, synthetic isolation
│   ├── validator.py        [High Maturity] Strict OHLCV invariant & gap checking
│   └── cleaner.py          [High Maturity] Deduplication, sorting, forward-filling
├── features/
│   ├── engineer.py         [High Maturity] 46 scale-invariant technical & rolling features
│   └── patterns.py         [High Maturity] Algorithmic candlestick & regime identification
├── models/
│   ├── base.py             [High Maturity] Unified lifecycle contract (fit, predict, save, load)
│   ├── baselines.py        [High Maturity] Zero-Return, Mean-Return, Last-Value baselines
│   ├── boosters.py         [High Maturity] XGBoost, LightGBM, CatBoost + split diagnostics
│   ├── linear.py           [High Maturity] Ridge, LinearRegression
│   ├── trees.py            [High Maturity] RandomForest, HistGradientBoosting
│   └── lstm.py             [High Maturity] Explicit PyTorch check + opt-in MLP fallback
├── pipeline/
│   ├── multi_asset.py      [High Maturity] Synchronized Single, Pooled, and LOAO engine
│   ├── evaluator.py        [High Maturity] OOS R2, MAE, RMSE, Directional Accuracy
│   └── backtester.py       [High Maturity] Order-level friction (0.15%), turnover, reliability
└── validation/
    ├── splitter.py         [High Maturity] Chronological splitting with purging gaps
    ├── leakage_guard.py    [High Maturity] Future price perturbation test
    └── walk_forward.py     [High Maturity] Rolling fold generation and signal verification
```

### Key Architectural Strengths:
- **Strict Invariance:** All feature transformations are mathematically scale-invariant; no unnormalized raw dollar prices are passed into boosters.
- **Causality Enforcement:** `LeakageGuard` tests prove that future price shocks have zero effect on historical feature values.
- **Reproducibility:** Random seeds and configurations are preserved, and all 49 model artifacts reproduce reported metrics with zero numerical discrepancy ($\Delta < 10^{-6}$).

---

## 3. The Frozen Baseline & Empirical Benchmark

The Phase 2 baseline is locked. All Phase 3 innovations must demonstrate measurable improvement over these frozen numbers:

### 3.1 Dataset Benchmark (Synchronized 1-Hour Panel)
- **Timeframe:** `2024-01-01 00:00:00` to `2024-03-03 11:00:00 UTC` (1,500 hourly bars).
- **Usable Post-Warmup Timeline:** `2024-01-08 00:00:00` to `2024-03-03 10:00:00 UTC` (1,331 bars).
- **Test Set Horizon:** `2024-02-24 04:00:00` to `2024-03-03 10:00:00 UTC` (199 bars per asset).

### 3.2 Frozen Baseline Performance (Test Partition)
$$\begin{array}{lccccc}
\hline
\textbf{Asset} & \textbf{Strategy / Model} & \textbf{RMSE} & \textbf{OOS } R^2 & \textbf{Directional Acc.} & \textbf{Rank IC} \\
\hline
\text{BTCUSDT} & \text{ZeroReturnBaseline} & 0.008930 & +0.000446 & \text{NaN} & 0.0000 \\
\text{BTCUSDT} & \text{Single-Asset CatBoost} & 0.008992 & -0.013567 & 50.3\% & +0.0214 \\
\text{BTCUSDT} & \text{Pooled XGBoost} & 0.008888 & +0.008988 & 53.8\% & +0.1032 \\
\text{BTCUSDT} & \text{LOAO RandomForest} & \mathbf{0.008868} & \mathbf{+0.013641} & 51.8\% & +0.1021 \\
\text{BTCUSDT} & \text{LOAO CatBoost} & 0.008869 & +0.013409 & \mathbf{54.8\%} & +0.0894 \\
\hline
\text{ETHUSDT} & \text{ZeroReturnBaseline} & \mathbf{0.010216} & +0.000169 & \text{NaN} & 0.0000 \\
\text{ETHUSDT} & \text{Single-Asset CatBoost} & 0.010338 & -0.023753 & 50.8\% & +0.0118 \\
\text{ETHUSDT} & \text{Pooled CatBoost} & 0.010284 & -0.013674 & \mathbf{53.8\%} & +0.0451 \\
\text{ETHUSDT} & \text{LOAO CatBoost} & 0.010277 & -0.012305 & 52.3\% & +0.0382 \\
\hline
\text{SOLUSDT} & \text{ZeroReturnBaseline} & \mathbf{0.012822} & +0.002241 & \text{NaN} & 0.0000 \\
\text{SOLUSDT} & \text{Single-Asset CatBoost} & 0.013194 & -0.056442 & 47.7\% & -0.0152 \\
\text{SOLUSDT} & \text{Pooled CatBoost} & 0.013037 & -0.033621 & 45.7\% & +0.0210 \\
\text{SOLUSDT} & \text{LOAO CatBoost} & 0.013309 & -0.077427 & 49.2\% & +0.0184 \\
\hline
\end{array}$$

---

## 4. Gap Analysis: Limitations to Resolve in Phase 3

| Identified Limitation | Empirical Evidence | Proposed Phase 3 Solution | Target Subphase |
| :--- | :--- | :--- | :---: |
| **Short Test Horizon** | 199 bars ($\sim 8.3$ days) yields $< 30$ trades; classified as `statistically_reliable = False`. | Ingest 5,000+ hourly bars (2024–2026) across multiple macroeconomic market regimes. | Subphase 3.2 |
| **Altcoin Negative Transfer** | SOLUSDT failed to beat ZeroReturn baseline; LOAO RMSE ($0.013309$) was worse than pooled ($0.013037$). | Implement cross-sectional volatility conditioning ($\sigma_i / \sigma_{\text{BTC}}$) and inverse-variance sample weights ($1 / \sigma_i^2$). | Subphase 3.3, 3.4 |
| **Absence of Lead-Lag Signals** | Pooled models treat asset rows independently without concurrent Bitcoin market-leader indicators. | Introduce cross-asset lagged return features ($r_{\text{BTC}, t}$, $r_{\text{BTC}, t-1}$) as explicit altcoin inputs. | Subphase 3.3 |
| **Static Single-Split Evaluation** | Single split does not measure stability across bull, bear, and consolidation regimes. | Implement a 5-fold Purged Walk-Forward Cross-Validation engine. | Subphase 3.5 |
| **Isolated Single-Asset Trading** | Backtesting evaluates assets separately rather than simulating a unified multi-asset portfolio. | Build `PortfolioBacktester` with dynamic risk-parity capital allocation and turnover accounting. | Subphase 3.7 |

---

## 5. Architectural Decision Records (ADRs)

### ADR-01: Multi-Year Panel Ingestion Strategy
- **Decision:** Ingest 5,000 continuous hourly candles for BTCUSDT, ETHUSDT, and SOLUSDT into `data/processed/panel_5000h/`.
- **Rationale:** 5,000 hourly candles ($\approx 208$ days) provide sufficient depth to span multiple macro volatility cycles while preserving exact common timestamps across all assets.
- **Status:** Approved for Subphase 3.2.

### ADR-02: Cross-Asset Lead-Lag Feature Representation
- **Decision:** Construct explicit Bitcoin return lags ($r_{\text{BTC}, t}, r_{\text{BTC}, t-1}, r_{\text{BTC}, t-2}$) and cross-asset volatility spreads as features for altcoins.
- **Rationale:** Financial econometrics demonstrates that Bitcoin acts as the primary price-discovery engine in cryptocurrency markets; altcoins frequently react with a 1-to-3-hour lag.
- **Status:** Approved for Subphase 3.3.

### ADR-03: Volatility-Conditioned Heteroskedastic Loss Weighting
- **Decision:** Train pooled gradient boosters using inverse-variance sample weights:
  $$w_{i, t} = \frac{1}{\sigma_i^2 \sum_j (1 / \sigma_j^2)}$$
- **Rationale:** Without sample weighting, Solana's large variance ($\sigma = 1.30\%$) disproportionately dominates squared-error loss gradients, causing the tree splits to optimize for SOL outliers at the expense of BTC/ETH accuracy.
- **Status:** Approved for Subphase 3.4.

### ADR-04: Purged Rolling Walk-Forward Architecture
- **Decision:** Deploy a 5-fold rolling walk-forward validator where each fold uses a 1,000-bar training window, 250-bar validation window, and 250-bar out-of-sample test window, with a mandatory 24-bar purge gap.
- **Rationale:** Overcomes single-period regime bias and guarantees zero data leakage between consecutive validation cycles.
- **Status:** Approved for Subphase 3.5.

### ADR-05: Risk-Parity Portfolio Execution Engine
- **Decision:** Build `PortfolioBacktester` where available trading capital ($100\%$ equity, spot only) is dynamically allocated across predicted top-ranked assets weighted inversely by recent volatility.
- **Rationale:** Evaluating individual asset trading in silos ignores cross-asset correlation, portfolio diversification, and the cost of capital rebalancing.
- **Status:** Approved for Subphase 3.7.

---

## 6. Risk Assessment & Unresolved Vulnerabilities

1. **API Rate Limiting & Historical Paging:**
   - *Risk:* Fetching 5,000 continuous bars via Binance REST API requires multiple paginated requests (limit 1,000 per call). Network disruptions or rate limits could introduce unexpected gaps.
   - *Mitigation:* `DataCollector`'s automatic failover across 5 REST endpoints and `DataValidator`'s mandatory gap audit guarantee that corrupted panels will fail loudly before reaching the feature engine.
2. **Timestamp Alignment in Lead-Lag Features:**
   - *Risk:* If Bitcoin features are joined onto Ethereum or Solana rows with misaligned timestamps, future Bitcoin prices could leak into altcoin features.
   - *Mitigation:* `LeakageGuard` must be extended in Subphase 3.3 to perturb future Bitcoin prices and verify that historical altcoin features remain 100% invariant.
3. **Overfitting to Cross-Asset Parameters:**
   - *Risk:* Adding lead-lag features increases dimensionality (e.g. from 46 to 55 features), raising the risk of overfitting on short datasets.
   - *Mitigation:* The 5,000-bar panel provides $3.75\times$ more training data, and tree depth will remain constrained ($\text{depth} \le 4$).

---

## 7. Phase 3.1 Verification & Readiness Checklist

- [x] Repository state audited and recorded (Commit `ea1cd0871f64c741f291af82f192811c7237a9f5`).
- [x] Existing datasets, artifacts, and test outputs preserved intact.
- [x] Phase 2 baseline formally frozen across all models and metrics.
- [x] Experimental contract specified (targets, horizons, timestamps, purging, holdout isolation).
- [x] Detailed subphase specifications (3.2–3.7) created in `docs/PHASE3_SPEC.md`.
- [x] Engineering transition guide created in `docs/PHASE3_HANDOFF.md`.
- [x] All 70 unit and integration tests executed and verified passing.
- [x] Execution stopped; Subphase 3.2 is NOT started automatically.

**Phase 3.1 is officially complete and ready for handoff.**
