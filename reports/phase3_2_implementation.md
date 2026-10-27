# Phase 3.2 — Data Pipeline Implementation & Architecture Guide

**Document ID:** `reports/phase3_2_implementation.md`  
**Author:** Senior Software Architect & Lead Financial Data Engineer  
**Date:** 2026-10-27  
**Status:** Complete & Production-Ready  
**Target Audience:** Machine Learning Engineers, Quant Researchers, and Pipeline Maintainers  

---

## 1. Architectural Overview & Objectives

In Phase 3.2, the data acquisition subsystem of `crypto-price-and-trend-prediction-ai` was elevated from an offline prototype with basic backwards pagination into a resilient, scalable, and reproducible financial time-series ingestion pipeline.

Key architectural improvements implemented:
1. **Forward Date-Range Pagination:** Systematic chronological ingestion from arbitrary start timestamps (`startTime`) up to target boundaries (`endTime`) in batches of 1,000 candles.
2. **Resumability & Incremental Updates:** Automated detection of existing local stores, retrieving only newly completed bars since `max(open_time)` without redundant bandwidth consumption.
3. **Atomic File Persistence:** Elimination of file corruption risks due to process interruption or abrupt termination via tempfile buffering and atomic `os.replace`.
4. **Cache Protection Against Clobbering:** Intelligent merging ensuring that short queries (e.g., test runs with `total_bars=400`) never truncate or overwrite multi-year historical stores.
5. **Cryptographic Provenance:** Automatic generation of JSON metadata sidecars with SHA-256 digests, byte counts, UTC retrieval timestamps, and parameters.
6. **Exchange Rate Limiting & Backoff:** Compliance with Binance public limits, implementing configurable delay (0.25s), exponential backoff on HTTP 429 / 418, and server error retries.
7. **Spot Market Isolation:** Strict enforcement of `market_type="spot"` to prevent contamination between spot and perpetual futures contract specifications.

---

## 2. Core Component Enhancements

### 2.1 `DataCollector` (`src/crypto_prediction/data/collector.py`)

#### A. Forward Pagination Engine (`fetch_historical_range`)
The forward pagination engine accepts arbitrary date formats (ISO string, `pd.Timestamp`, `datetime`, or integer epoch ms) and paginates through exchange records:

```python
def fetch_historical_range(
    self,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start_time: str | pd.Timestamp | datetime | int | None = None,
    end_time: str | pd.Timestamp | datetime | int | None = None,
    market_type: str = "spot",
    resumable: bool = True,
    save_to_raw: bool = True,
    exclude_unclosed_bar: bool = True,
) -> pd.DataFrame:
```

**Algorithm:**
1. Check if `raw_file` exists and `resumable=True`. If cached data covers the range, return cache directly. If cache covers up to $T_{\max} < T_{\text{end}}$, set $T_{\text{start}} = T_{\max} + \Delta t$ to fetch only the missing delta.
2. If $T_{\text{start}}$ precedes the cache minimum, initiate full chronological backfill starting at $T_{\text{start}}$.
3. Query `https://api.binance.com/api/v3/klines` with `startTime = current_start_ms` and `limit = 1000`.
4. Upon receiving batch, advance `current_start_ms = last_open_time_ms + interval_ms`.
5. Repeat until `current_start_ms >= end_time_ms` or batch size $< 1000$ (end of exchange history).
6. Merge new records with existing cache, deduplicate on `open_time`, and sort chronologically.
7. If `exclude_unclosed_bar=True`, check if `open_time[-1] + delta > now_utc`; if so, drop incomplete bar.
8. Atomically write raw CSV and generate `.meta.json` sidecar.

#### B. Atomic File Writing (`_atomic_save_csv`)
To guarantee that interruptions never leave corrupted or zero-byte files on disk:
```python
def _atomic_save_csv(self, df: pd.DataFrame, file_path: str | Path) -> Path:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out_df = df.reset_index() if isinstance(df.index, pd.DatetimeIndex) else df

    temp_file = path.parent / f".{path.name}.tmp_{int(time.time() * 1000)}_{os.getpid()}"
    try:
        out_df.to_csv(temp_file, index=False)
        os.replace(temp_file, path)
    finally:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass
    return path
```

#### C. Cache Preservation Logic
When legacy functions or test suites call `fetch_historical_bars(symbol="BTCUSDT", total_bars=400)`, the collector inspects the existing file on disk:
- If `len(existing_cache) > len(new_df)`, it merges new observations into the existing store without truncating older history.
- The caller receives their requested 400-bar slice, while the underlying 33,493-bar dataset remains intact.

#### D. Cryptographic Checksumming
Independent utility function `compute_file_checksum(file_path, algorithm="sha256")` computes streaming SHA-256 digests in 64 KB chunks to support arbitrarily large files with minimal RAM overhead.

---

### 2.2 `DataValidator` (`src/crypto_prediction/data/validator.py`)

#### A. Structured Gap Diagnostics
`ValidationReport` was extended with `gap_details: list[dict[str, Any]]`:
- Automatically detects any inter-candle gap exceeding $1.5 \times$ the median sampling frequency.
- Captures `gap_start_utc`, `gap_end_utc`, and `duration_hours`.

#### B. Real-World Anomaly Auditing
- Scans for single-bar price moves exceeding $\pm 20\%$ or high-low spreads $> 40\%$.
- Populates `anomaly_details` with index, timestamp, and return ratio for manual quantitative inspection.

---

## 3. Reproduction & Execution Instructions

### 3.1 Executing Automated Tests (79 Passing Tests)
To verify all collector features, invariant checks, atomic writes, and rate limit backoffs:
```powershell
$env:PYTHONPATH="src"
& "..\..\..\envairoment\.venv\Scripts\python.exe" -m pytest tests/test_collector_advanced.py tests/test_historical_pipeline.py -v
```

### 3.2 Executing Pilot Ingestion (1 Month / 744 Bars)
```powershell
$env:PYTHONPATH="src"
& "..\..\..\envairoment\.venv\Scripts\python.exe" scripts/ingest_historical_data.py --symbols BTCUSDT ETHUSDT SOLUSDT
```

### 3.3 Updating / Synchronizing Historical Data
To synchronize the multi-year panel from `2023-01-01` to the latest completed hourly candle:
```powershell
$env:PYTHONPATH="src"
& "..\..\..\envairoment\.venv\Scripts\python.exe" scripts/ingest_historical_data.py --start-date "2023-01-01 00:00:00" --skip-pilot
```

This updates `data/raw/`, validates the new candles, updates `data/processed/historical/`, and outputs the time-synchronized panel to `data/processed/panel_5000h/`.
