# Phase 3.6 — Realistic Trading Backtesting & Cost Sensitivity: Backtest Design

**Role:** Senior Quantitative Trading Engineer, Financial Risk Analyst, and Software Validation Architect  
**Project:** `crypto-price-and-trend-prediction-ai`  
**Date:** 2026-10-27  
**Artifact Status:** APPROVED & REPRODUCED  

---

## 1. Executive Summary & Design Principles

Phase 3.6 transitions predictive signals from statistical accuracy into practical financial execution. The objective is to design, implement, and validate an institutional-grade, leakage-free backtesting simulation framework that assesses whether the models from Phases 3.4 and 3.5 deliver genuine economic alpha after realistic execution friction, bid-ask spreads, and exchange fees.

### Core Architectural Principles:
1. **Strict Timestamp & Execution Causality:**
   - Feature close at $t$ $\rightarrow$ Signal generated at $t$ $\rightarrow$ Order submitted at $t$ $\rightarrow$ Execution occurs over $[t, t+1]$ $\rightarrow$ Trade exits at $t+1$ close.
   - Zero same-bar execution bias: The model never trades on prices used to compute its predictive features.
2. **Exhaustive Cost Modeling:**
   - Maker/taker exchange fees, bid-ask spread crossing costs, and execution slippage are accounted for per unit of turnover.
   - Position reversals (Long $+1 \rightarrow$ Short $-1$) incur double turnover costs ($2 \times \text{unit\_cost}$).
   - Any open position at the simulation boundary is liquidated to cash, incurring final exit friction.
3. **Friction-Aware Decision Policies:**
   - Incorporating expected round-trip costs directly into the order entry threshold to prevent churn in low-conviction signals.
4. **Counterfactual Benchmarks:**
   - Strict comparison against Buy-and-Hold and No-Trade Cash Baselines over identical chronological time slices.

---

## 2. Order Execution & Position Accounting Model

### 2.1 State-Machine Dynamics
At each hourly bar $t \in \{1, \dots, T\}$:
1. **Observation & Forecast:**
   The model consumes feature vector $X_t$ (available at candle close $t$) and generates forward log return forecast $\hat{y}_t = \widehat{r}_{t+1}$.
2. **Target Position Determination:**
   Given current position $pos_{t} \in \{0, 1\}$ (for Long/Cash spot) and policy $\pi$:
   - **Cash ($pos_t = 0$):**
     If $\hat{y}_t > \theta_{\text{entry}}$, set $pos_{t+1} = 1.0$ (Enter Long).
   - **Long ($pos_t = 1$):**
     If $\hat{y}_t \le \theta_{\text{exit}}$, set $pos_{t+1} = 0.0$ (Exit to Cash).
     Otherwise, maintain $pos_{t+1} = 1.0$ (Hold).
3. **Turnover & Order Sizing:**
   $$\Delta pos_{t+1} = |pos_{t+1} - pos_t|$$
   - Entering position ($0 \rightarrow 1$): $\Delta pos = 1.0$ (1 order).
   - Exiting position ($1 \rightarrow 0$): $\Delta pos = 1.0$ (1 order).
   - Reversing position ($1 \rightarrow -1$ if short enabled): $\Delta pos = 2.0$ (2 orders).
   - Holding position ($1 \rightarrow 1$): $\Delta pos = 0.0$ (0 orders).
4. **Transaction Cost Drag:**
   $$\text{cost}_t = \Delta pos_{t+1} \times (\text{fee\_rate} + \text{slippage\_rate})$$
5. **Net Period Return:**
   $$R_{\text{gross}, t+1} = pos_{t+1} \cdot r_{t+1}$$
   $$R_{\text{net}, t+1} = R_{\text{gross}, t+1} - \text{cost}_t$$
6. **Final Liquidation:**
   At bar $T$, if $pos_T \neq 0$:
   $$R_{\text{net}, T} \leftarrow R_{\text{net}, T} - |pos_T| \times (\text{fee\_rate} + \text{slippage\_rate})$$
7. **Wealth Compounding:**
   $$E_0 = 1.0, \quad E_t = E_{t-1} \times \max(0, 1 + R_{\text{net}, t})$$

---

## 3. Four Configurable Cost Scenarios

To quantify strategy resilience under varying fee structures, four standardized industry scenarios were benchmarked:

| Cost Scenario | Fee Rate (One-Way) | Slippage/Spread (One-Way) | Total Unit Cost (One-Way) | Round-Trip Cost (Entry + Exit) | Market Representation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Zero Cost (Gross)** | 0.00% (0 bps) | 0.00% (0 bps) | **0 bps** | **0 bps** | Theoretical signal efficiency before market friction. |
| **2. Institutional / VIP** | 0.05% (5 bps) | 0.02% (2 bps) | **7 bps** | **14 bps** | High-volume VIP tier with maker rebates and tight spreads. |
| **3. Standard Retail** | 0.10% (10 bps) | 0.05% (5 bps) | **15 bps** | **30 bps** | Standard Binance VIP0 spot taker fee with typical slippage. |
| **4. High Friction / Stress** | 0.15% (15 bps) | 0.10% (10 bps) | **25 bps** | **50 bps** | High-volatility illiquid regimes or adverse market impact. |

---

## 4. Evaluated Decision Policies

Three distinct decision policies were compared against counterfactual benchmarks:

1. **Policy 1: Friction-Aware Threshold Filter (`threshold_filter`) [Primary Recommended Strategy]**
   - **Entry Condition:** Enter Long only when expected return exceeds round-trip friction:
     $$\hat{y}_t > 2 \times (\text{fee\_rate} + \text{slippage\_rate})$$
   - **Exit Condition:** Exit to Cash when expected return drops to non-positive ($\hat{y}_t \le 0.0$).
   - **Rationale:** Prevents churning capital on marginal signals where transaction costs exceed expected profit.
2. **Policy 2: Pure Directional Sign (`pure_sign`)**
   - **Entry Condition:** Enter Long whenever $\hat{y}_t > 0.0$.
   - **Exit Condition:** Exit to Cash whenever $\hat{y}_t \le 0.0$.
   - **Rationale:** Establishes the cost drag incurred by naive sign-following without friction buffers.
3. **Policy 3: High-Conviction Threshold (`high_conviction`)**
   - **Entry Condition:** Enter Long only when $\hat{y}_t > \max(0.0025, 2 \times \text{unit\_cost})$ (25 bps threshold).
   - **Exit Condition:** Exit to Cash when $\hat{y}_t \le 0.0$.
   - **Rationale:** Maximizes signal quality and minimizes turnover.
4. **Benchmark 1: Buy-and-Hold (`buy_and_hold`)**
   - 100% Long from bar 0 to bar $T$, incurring entry and exit fees ($2 \times \text{unit\_cost}$).
5. **Benchmark 2: Cash Baseline (`cash_baseline`)**
   - 100% Cash throughout ($R_{\text{net}} = 0.0\%$, Drawdown = 0.0%, Trades = 0).

---

## 5. Risk & Performance Metrics Specifications

For every backtest simulation, the following metrics are tracked:
- **Gross Cumulative Return (%):** Return generated before friction.
- **Total Cost Drag (%):** Cumulative drag from exchange fees, spreads, and slippage.
- **Net Cumulative Return (%):** Realized equity growth after all friction ($E_T - 1.0$).
- **Net Alpha (%):** Excess net return over Buy-and-Hold benchmark ($\text{Net Return} - \text{Benchmark Return}$).
- **Annualized Sharpe Ratio:**
  $$\text{Sharpe} = \frac{\overline{R_{\text{net}}}}{\text{std}(R_{\text{net}})} \times \sqrt{8760}$$
  *(8,760 hourly periods per year; $\sqrt{8760} \approx 93.595$).*
- **Annualized Sortino Ratio:** Downside deviation risk-adjusted return.
- **Maximum Drawdown (MDD %):** Peak-to-trough decline of the equity curve.
- **Win Rate (%):** Percentage of active trading bars with positive net return.
- **Profit Factor:** Gross profits divided by gross losses.
- **Total Trades & Turnover:** Total number of order transitions and cumulative units of capital turned over.
- **Exposure (%):** Percentage of time with capital committed to active market risk.
- **Breakeven Cost (bps Round-Trip):** The round-trip friction level at which strategy gross alpha equals transaction costs:
  $$\text{Breakeven (bps)} = \frac{\text{Gross Cumulative Return}}{\text{Turnover}} \times 20,000$$
- **Statistical Reliability Flag:** Strategies with fewer than 30 trades are flagged as statistically unverified.
