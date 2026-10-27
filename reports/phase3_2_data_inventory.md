# Phase 3.2 — Multi-Year Historical Data Inventory

**Document ID:** `reports/phase3_2_data_inventory.md`  
**Author:** Senior Data Engineer & Quantitative Software Architect  
**Date:** 2026-10-27  
**Status:** Complete & Verified  
**Target Audience:** Machine Learning Engineers, Quant Researchers, and Data Quality Auditors  

---

## 1. Executive Summary

As part of **Phase 3.2 (Multi-Year Historical Data Pipeline)**, we have successfully collected, verified, and synchronized multi-year continuous hourly OHLCV datasets across **BTCUSDT**, **ETHUSDT**, and **SOLUSDT**. 

Key inventory achievements:
1. **Pilot Dataset (30 Days / 744 Bars per symbol):** Clean, verified 1-month benchmark panel spanning `2024-01-01 00:00:00` to `2024-01-31 23:00:00 UTC`.
2. **Multi-Year Historical Stores (Up to 50,054 Bars per symbol):** Multi-year spot records starting as early as `2021-02-09` for ETH and `2023-01-01` for BTC and SOL, extending through `2026-10-27 13:00:00 UTC`.
3. **Synchronized Multi-Asset Overlapping Panel (`panel_5000h`):** **33,493 continuous hourly bars** with **100% identical timestamp alignment** across BTC, ETH, and SOL spanning `2023-01-01 00:00:00` to `2026-10-27 13:00:00 UTC`. This exceeds the Phase 3.2 minimum requirement of 5,000 bars by **6.7x**.
4. **Cryptographic Provenance:** Every CSV file is accompanied by an atomic JSON metadata sidecar (`.meta.json`) recording the exact SHA-256 checksum, byte count, query parameters, timestamp range, and retrieval status.

---

## 2. Storage Directory Layout

The data directory is strictly partitioned into distinct functional tiers to prevent data corruption, preserve raw exchange records without premature manipulation, and ensure reproducible downstream modeling:

```text
data/
├── raw/                              # Immutable raw market downloads (direct exchange responses)
│   ├── BTCUSDT_1h.csv                # 33,493 bars (2023-01-01 to 2026-10-27 UTC)
│   ├── BTCUSDT_1h.csv.meta.json      # Provenance metadata sidecar with SHA-256
│   ├── ETHUSDT_1h.csv                # 50,054 bars (2021-02-09 to 2026-10-27 UTC)
│   ├── ETHUSDT_1h.csv.meta.json
│   ├── SOLUSDT_1h.csv                # 33,493 bars (2023-01-01 to 2026-10-27 UTC)
│   └── SOLUSDT_1h.csv.meta.json
├── sample/                           # Frozen Phase 2 baseline panel (1,500 bars, Jan–Mar 2024)
│   ├── BTCUSDT_1h.csv
│   ├── ETHUSDT_1h.csv
│   └── SOLUSDT_1h.csv
├── synthetic/                        # Deterministic synthetic data (isolated fallback only)
│   └── SOLUSDT_1h_synthetic.csv
└── processed/                        # Formatted, validated, and synchronized datasets
    ├── pilot/                        # Phase 3.2 pre-flight pilot verification panel (744 bars)
    │   ├── BTCUSDT_1h_pilot.csv
    │   ├── BTCUSDT_1h_pilot.csv.meta.json
    │   ├── ETHUSDT_1h_pilot.csv
    │   ├── ETHUSDT_1h_pilot.csv.meta.json
    │   ├── SOLUSDT_1h_pilot.csv
    │   └── SOLUSDT_1h_pilot.csv.meta.json
    ├── historical/                   # Validated individual multi-year series (pass DataValidator)
    │   ├── BTCUSDT_1h.csv            # 33,493 bars
    │   ├── BTCUSDT_1h.csv.meta.json
    │   ├── ETHUSDT_1h.csv            # 50,054 bars
    │   ├── ETHUSDT_1h.csv.meta.json
    │   ├── SOLUSDT_1h.csv            # 33,493 bars
    │   └── SOLUSDT_1h.csv.meta.json
    └── panel_5000h/                  # Synchronized multi-asset panel for Phase 3.3+ ML pipelines
        ├── BTCUSDT_1h.csv            # 33,493 bars (identical DatetimeIndex)
        ├── BTCUSDT_1h.csv.meta.json
        ├── ETHUSDT_1h.csv            # 33,493 bars (identical DatetimeIndex)
        ├── ETHUSDT_1h.csv.meta.json
        ├── SOLUSDT_1h.csv            # 33,493 bars (identical DatetimeIndex)
        └── SOLUSDT_1h.csv.meta.json
```

---

## 3. Dataset Inventory Matrix

### 3.1 Synchronized Multi-Asset Panel (`data/processed/panel_5000h/`)

*Primary input dataset for Phase 3.3 feature engineering, Phase 3.4 sample weighting, and Phase 3.5 multi-regime walk-forward evaluation.*

| Symbol | Interval | Market Type | Total Bars | Date Range (UTC) | File Size (Bytes) | SHA-256 Checksum |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **BTCUSDT** | `1h` | Spot | 33,493 | 2023-01-01 00:00:00 &rarr; 2026-10-27 13:00:00 | 2,444,016 | `c8d2a6a6a3a9fec4cdb6e2ca56ee8aa80788e8e4dd784658bba6269b5fd2d9c0` |
| **ETHUSDT** | `1h` | Spot | 33,493 | 2023-01-01 00:00:00 &rarr; 2026-10-27 13:00:00 | 2,302,030 | `296b80251e448e90e3fc97709a5d1c98a20b76bb4d93b33e5a7121970227cd25` |
| **SOLUSDT** | `1h` | Spot | 33,493 | 2023-01-01 00:00:00 &rarr; 2026-10-27 13:00:00 | 2,108,470 | `d3f65267930983e438926da01e65cae9a28a3684add423c856d97f0507466209` |

*Alignment Verification:* `(BTC['open_time'] == ETH['open_time']).all() == True` and `(BTC['open_time'] == SOL['open_time']).all() == True`. Zero timestamp misalignments.

---

### 3.2 Individual Multi-Year Historical Series (`data/processed/historical/`)

| Symbol | Interval | Total Bars | Date Range (UTC) | File Size (Bytes) | SHA-256 Checksum | Validation Status |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **BTCUSDT** | `1h` | 33,493 | 2023-01-01 00:00:00 &rarr; 2026-10-27 13:00:00 | 2,444,016 | `c8d2a6a6a3a9fec4cdb6e2ca56ee8aa80788e8e4dd784658bba6269b5fd2d9c0` | **PASSED** |
| **ETHUSDT** | `1h` | 50,054 | 2021-02-09 10:00:00 &rarr; 2026-10-27 13:00:00 | 3,450,953 | `c00f3d4533bb28119b05b87df1bbd4918eaeff9d084222350246a3274c50c2f5` | **PASSED** |
| **SOLUSDT** | `1h` | 33,493 | 2023-01-01 00:00:00 &rarr; 2026-10-27 13:00:00 | 2,108,470 | `d3f65267930983e438926da01e65cae9a28a3684add423c856d97f0507466209` | **PASSED** |

---

### 3.3 Pilot Verification Datasets (`data/processed/pilot/`)

| Symbol | Interval | Total Bars | Date Range (UTC) | File Size (Bytes) | SHA-256 Checksum | Validation Status |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **BTCUSDT** | `1h` | 744 | 2024-01-01 00:00:00 &rarr; 2024-01-31 23:00:00 | 53,945 | `ec59dc6a786a5fd8f1f1f9afb5fd8c692cc91db2214912809d747a5794078e8f` | **PASSED** |
| **ETHUSDT** | `1h` | 744 | 2024-01-01 00:00:00 &rarr; 2024-01-31 23:00:00 | 51,186 | `178c0e09d8c8ef73ce5540d264d59d6839c9b162dcbff572c39fafb1a0dc576e` | **PASSED** |
| **SOLUSDT** | `1h` | 744 | 2024-01-01 00:00:00 &rarr; 2024-01-31 23:00:00 | 45,649 | `7780629b9f7901a0acdf4705a0012f1d9385e050fdf02481e0bf55e1d3ddbd78` | **PASSED** |

---

### 3.4 Raw Exchange Stores (`data/raw/`)

| File Name | Total Bars | Date Range (UTC) | File Size (Bytes) | SHA-256 Checksum |
| :--- | :---: | :---: | :---: | :--- |
| `BTCUSDT_1h.csv` | 33,493 | 2023-01-01 00:00:00 &rarr; 2026-10-27 13:00:00 | 2,444,016 | `c8d2a6a6a3a9fec4cdb6e2ca56ee8aa80788e8e4dd784658bba6269b5fd2d9c0` |
| `ETHUSDT_1h.csv` | 50,054 | 2021-02-09 10:00:00 &rarr; 2026-10-27 13:00:00 | 3,450,953 | `c00f3d4533bb28119b05b87df1bbd4918eaeff9d084222350246a3274c50c2f5` |
| `SOLUSDT_1h.csv` | 33,493 | 2023-01-01 00:00:00 &rarr; 2026-10-27 13:00:00 | 2,108,470 | `d3f65267930983e438926da01e65cae9a28a3684add423c856d97f0507466209` |
| `BTCUSDT_5m.csv` *(existing test store)* | 9,999 | 2026-09-21 19:55:00 &rarr; 2026-10-26 13:05:00 | 709,355 | `e4cbe758317e8f48885323d84e8412b7784b5efec56e74e57acb24036c273d78` |
| `ETHUSDT_15m.csv` *(existing test store)* | 999 | 2026-10-16 03:00:00 &rarr; 2026-10-26 12:30:00 | 68,005 | `b8cd1d97d21f4434d233b021e5109b906f05e2f3c415135fe7e497381130f95c` |

---

## 4. Metadata Sidecar Schema

All newly generated dataset files are linked to an atomic `.meta.json` sidecar matching the following JSON schema:

```json
{
  "file_name": "BTCUSDT_1h.csv",
  "symbol": "BTCUSDT",
  "interval": "1h",
  "market_type": "spot",
  "total_bars": 33493,
  "first_bar_utc": "2023-01-01 00:00:00+00:00",
  "last_bar_utc": "2026-10-27 13:00:00+00:00",
  "retrieved_at_utc": "2026-10-27 14:08:09.965000+00:00",
  "data_source": "binance_spot_live",
  "status": "raw",
  "sha256": "c8d2a6a6a3a9fec4cdb6e2ca56ee8aa80788e8e4dd784658bba6269b5fd2d9c0",
  "file_size_bytes": 2444016
}
```

This guarantees complete traceability and cryptographic verification across the entire modeling pipeline.
