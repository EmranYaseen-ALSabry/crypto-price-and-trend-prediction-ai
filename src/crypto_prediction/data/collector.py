"""Market data collector supporting Binance REST API, offline CSVs, and synthetic fallback."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import requests

from crypto_prediction.utils.logger import get_logger

logger = get_logger("data_collector")


def compute_file_checksum(file_path: str | Path, algorithm: str = "sha256") -> str:
    """Computes cryptographic checksum (SHA-256 by default) of a file on disk."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for checksum calculation: {path}")
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class DataCollector:
    """Collects OHLCV market data via REST API or local CSVs, with deterministic synthetic fallback.

    Supports:
    - Multi-symbol acquisition (BTCUSDT, ETHUSDT, SOLUSDT, etc.)
    - Configurable date-range forward pagination with exchange rate limiting and backoff
    - Incremental updates and resumable historical downloads
    - Atomic file saving to eliminate corruption on interruption
    - Provenance metadata tracking with SHA-256 checksums
    - Strict spot market isolation
    """

    BINANCE_KLINES_URLS = [
        "https://api.binance.com/api/v3/klines",
        "https://data-api.binance.vision/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
        "https://api2.binance.com/api/v3/klines",
        "https://api3.binance.com/api/v3/klines",
    ]

    INTERVAL_DELTAS = {
        "1m": pd.Timedelta(minutes=1),
        "3m": pd.Timedelta(minutes=3),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "30m": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1),
        "2h": pd.Timedelta(hours=2),
        "4h": pd.Timedelta(hours=4),
        "6h": pd.Timedelta(hours=6),
        "8h": pd.Timedelta(hours=8),
        "12h": pd.Timedelta(hours=12),
        "1d": pd.Timedelta(days=1),
        "3d": pd.Timedelta(days=3),
        "1w": pd.Timedelta(weeks=1),
    }

    def __init__(
        self,
        base_dir: str | Path = "data",
        request_timeout: int = 15,
        rate_limit_delay: float = 0.25,
        max_retries: int = 5,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.sample_dir = self.base_dir / "sample"
        self.synthetic_dir = self.base_dir / "synthetic"
        self.processed_dir = self.base_dir / "processed"
        self.request_timeout = request_timeout
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.synthetic_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_interval_delta(cls, interval: str) -> pd.Timedelta:
        """Returns standard pd.Timedelta for supported candlestick intervals."""
        if interval not in cls.INTERVAL_DELTAS:
            raise ValueError(
                f"Unsupported interval '{interval}'. Supported: {list(cls.INTERVAL_DELTAS.keys())}"
            )
        return cls.INTERVAL_DELTAS[interval]

    @staticmethod
    def parse_to_epoch_ms(value: str | pd.Timestamp | datetime | int) -> int:
        """Parses various timestamp representations into UTC epoch milliseconds."""
        if isinstance(value, int):
            # Already integer ms or seconds
            if value < 10_000_000_000:
                return value * 1000
            return value
        if isinstance(value, str):
            ts = pd.to_datetime(value, utc=True)
            return int(ts.timestamp() * 1000)
        if isinstance(value, (pd.Timestamp, datetime)):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return int(value.timestamp() * 1000)
        raise TypeError(f"Cannot parse timestamp of type {type(value)}: {value}")

    def load_from_csv(self, file_path: str | Path) -> pd.DataFrame:
        """Loads historical OHLCV data from a local CSV file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file does not exist at: {path}")

        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows from CSV: {path.name}")
        return df

    def _atomic_save_csv(self, df: pd.DataFrame, file_path: str | Path) -> Path:
        """Atomically saves OHLCV DataFrame to CSV via temporary file and atomic rename."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out_df = df.reset_index() if isinstance(df.index, pd.DatetimeIndex) else df

        # Unique temporary filename within same folder to guarantee atomic replace
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
        logger.info(f"Atomically saved {len(out_df)} rows to CSV: {path}")
        return path

    def save_to_csv(self, df: pd.DataFrame, file_path: str | Path) -> Path:
        """Saves OHLCV DataFrame to CSV atomically."""
        return self._atomic_save_csv(df, file_path)

    def save_metadata(self, data_file_path: str | Path, meta_dict: dict[str, Any]) -> Path:
        """Saves a JSON metadata sidecar next to the dataset file with SHA-256 and provenance."""
        data_path = Path(data_file_path)
        meta_path = data_path.with_name(f"{data_path.name}.meta.json")
        out_meta = dict(meta_dict)

        if data_path.exists():
            out_meta["sha256"] = compute_file_checksum(data_path)
            out_meta["file_size_bytes"] = data_path.stat().st_size

        meta_temp = meta_path.parent / f".{meta_path.name}.tmp_{int(time.time() * 1000)}_{os.getpid()}"
        try:
            with open(meta_temp, "w", encoding="utf-8") as f:
                json.dump(out_meta, f, indent=2, default=str)
            os.replace(meta_temp, meta_path)
        finally:
            if meta_temp.exists():
                try:
                    meta_temp.unlink()
                except OSError:
                    pass
        logger.info(f"Saved dataset metadata sidecar: {meta_path}")
        return meta_path

    def load_metadata(self, data_file_path: str | Path) -> dict[str, Any]:
        """Loads JSON metadata sidecar for a dataset file if present."""
        data_path = Path(data_file_path)
        meta_path = data_path.with_name(f"{data_path.name}.meta.json")
        if not meta_path.exists():
            return {}
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def fetch_binance_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 1000,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> pd.DataFrame:
        """Fetches OHLCV kline data from Binance public spot REST endpoints with failover and backoff."""
        params: dict[str, str | int] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if start_time_ms is not None:
            params["startTime"] = start_time_ms
        if end_time_ms is not None:
            params["endTime"] = end_time_ms

        for attempt in range(1, self.max_retries + 1):
            url = self.BINANCE_KLINES_URLS[(attempt - 1) % len(self.BINANCE_KLINES_URLS)]
            try:
                time.sleep(self.rate_limit_delay)
                response = requests.get(
                    url,
                    params=params,
                    timeout=self.request_timeout,
                    headers={"User-Agent": "CryptoPredictionAI/1.0"},
                )
                if response.status_code == 200:
                    raw_data = response.json()
                    if not raw_data:
                        logger.warning(f"Empty response returned for {symbol} ({interval}) from {url}.")
                        return pd.DataFrame()

                    rows = []
                    for k in raw_data:
                        rows.append(
                            {
                                "open_time": pd.to_datetime(k[0], unit="ms", utc=True),
                                "open": float(k[1]),
                                "high": float(k[2]),
                                "low": float(k[3]),
                                "close": float(k[4]),
                                "volume": float(k[5]),
                            }
                        )
                    df = pd.DataFrame(rows)
                    logger.info(f"Successfully fetched {len(df)} bars for {symbol} ({interval}) from Binance.")
                    return df
                elif response.status_code in (418, 429):
                    retry_after = int(response.headers.get("Retry-After", 2**attempt))
                    logger.warning(
                        f"Attempt {attempt}/{self.max_retries}: HTTP {response.status_code} rate limit hit from {url}. Backing off {retry_after}s..."
                    )
                    time.sleep(retry_after)
                elif response.status_code >= 500:
                    backoff = min(2**attempt, 30)
                    logger.warning(
                        f"Attempt {attempt}/{self.max_retries}: HTTP {response.status_code} server error from {url}. Backing off {backoff}s..."
                    )
                    time.sleep(backoff)
                else:
                    logger.warning(
                        f"Attempt {attempt}/{self.max_retries}: {url} returned HTTP {response.status_code}: {response.text[:100]}"
                    )
            except Exception as exc:
                backoff = min(2**attempt, 15)
                logger.warning(
                    f"Attempt {attempt}/{self.max_retries} with {url} failed with error: {exc}. Retrying in {backoff}s..."
                )
                time.sleep(backoff)

        raise ConnectionError(f"Failed to fetch market data from Binance API after {self.max_retries} attempts.")

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
        """Fetches historical OHLCV data over a date range via forward pagination.

        Supports resumable collection and incremental updates against existing raw files,
        atomic file writes, and generation of SHA-256 provenance metadata sidecars.
        """
        if market_type.lower() != "spot":
            raise ValueError(
                f"Unsupported market_type '{market_type}'. Only 'spot' is supported to prevent mixing spot and futures datasets."
            )

        start_ms = self.parse_to_epoch_ms(start_time) if start_time is not None else None
        end_ms = self.parse_to_epoch_ms(end_time) if end_time is not None else None

        raw_file = self.raw_dir / f"{symbol}_{interval}.csv"
        existing_df = pd.DataFrame()

        # If resumable and raw file exists, inspect existing coverage
        if resumable and raw_file.exists():
            try:
                cached_df = self.load_from_csv(raw_file)
                if not cached_df.empty and "open_time" in cached_df.columns:
                    cached_df["open_time"] = pd.to_datetime(cached_df["open_time"], utc=True)
                    cached_df = (
                        cached_df.sort_values(by="open_time")
                        .drop_duplicates(subset=["open_time"])
                        .reset_index(drop=True)
                    )
                    min_ms = int(cached_df["open_time"].min().timestamp() * 1000)
                    max_ms = int(cached_df["open_time"].max().timestamp() * 1000)

                    # Check if cached data covers or extends the requested range
                    if start_ms is None or start_ms >= min_ms:
                        if end_ms is not None and max_ms >= end_ms:
                            logger.info(
                                f"Existing raw cache for {symbol} covers up to {cached_df['open_time'].max()} "
                                f"(>= {pd.to_datetime(end_ms, unit='ms', utc=True)}). Returning cached range."
                            )
                            filtered = cached_df
                            if start_ms is not None:
                                filtered = filtered[
                                    filtered["open_time"] >= pd.to_datetime(start_ms, unit="ms", utc=True)
                                ]
                            if end_ms is not None:
                                filtered = filtered[
                                    filtered["open_time"] <= pd.to_datetime(end_ms, unit="ms", utc=True)
                                ]
                            filtered = filtered.reset_index(drop=True)
                            filtered.attrs["data_source"] = "cached_raw_csv"
                            filtered.attrs["is_synthetic"] = False
                            filtered.attrs["market_type"] = "spot"
                            return filtered
                        else:
                            # Forward resume: start fetching right after max cached bar
                            logger.info(
                                f"Resuming incremental fetch for {symbol} from {cached_df['open_time'].max()}..."
                            )
                            start_ms = max_ms + 1
                            existing_df = cached_df
                    else:
                        logger.info(
                            f"Requested start {pd.to_datetime(start_ms, unit='ms', utc=True)} is earlier than cache minimum. Fetching full range."
                        )
                        existing_df = cached_df
            except Exception as e:
                logger.warning(f"Failed to inspect existing raw cache ({e}). Proceeding with fresh fetch.")
                existing_df = pd.DataFrame()

        # Defaults
        if start_ms is None:
            # Default to 3 years ago if start is unspecified
            start_ms = int(pd.Timestamp("2022-01-01 00:00:00", tz="UTC").timestamp() * 1000)

        now_utc = pd.Timestamp.now(tz="UTC")
        if end_ms is None:
            end_ms = int(now_utc.timestamp() * 1000)

        interval_delta = self.get_interval_delta(interval)
        interval_ms = int(interval_delta.total_seconds() * 1000)
        all_chunks = []
        current_start_ms = start_ms

        logger.info(
            f"Beginning forward pagination for {symbol} ({interval}) from "
            f"{pd.to_datetime(current_start_ms, unit='ms', utc=True)} to {pd.to_datetime(end_ms, unit='ms', utc=True)}..."
        )

        while current_start_ms < end_ms:
            chunk = self.fetch_binance_klines(
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time_ms=current_start_ms,
                end_time_ms=end_ms,
            )
            if chunk.empty:
                logger.info(
                    f"No more candles returned for {symbol} at {pd.to_datetime(current_start_ms, unit='ms', utc=True)}."
                )
                break

            all_chunks.append(chunk)
            last_open = chunk["open_time"].max()
            last_open_ms = int(last_open.timestamp() * 1000)

            # Move forward past the last fetched bar
            next_start_ms = last_open_ms + interval_ms
            if next_start_ms <= current_start_ms:
                next_start_ms = current_start_ms + 1

            current_start_ms = next_start_ms

            if len(chunk) < 1000:
                logger.info(f"Reached latest available candle for {symbol} ({last_open}).")
                break

        new_df = pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()

        # Merge new chunks with existing cache if incremental
        if not existing_df.empty:
            if not new_df.empty:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = existing_df
        else:
            combined_df = new_df

        if combined_df.empty:
            raise RuntimeError(
                f"No market data could be retrieved for {symbol} ({interval}) between {start_time} and {end_time}."
            )

        # Deduplicate and sort chronologically
        combined_df = (
            combined_df.drop_duplicates(subset=["open_time"]).sort_values(by="open_time").reset_index(drop=True)
        )

        # Exclude incomplete / unclosed current bar
        if exclude_unclosed_bar and not combined_df.empty:
            if combined_df["open_time"].iloc[-1] + interval_delta > now_utc:
                logger.info(
                    f"Excluding current unclosed bar ({combined_df['open_time'].iloc[-1]} UTC) to prevent partial candle distortion."
                )
                combined_df = combined_df.iloc[:-1].reset_index(drop=True)

        combined_df.attrs["data_source"] = "binance_spot_live"
        combined_df.attrs["is_synthetic"] = False
        combined_df.attrs["market_type"] = "spot"

        if save_to_raw:
            self._atomic_save_csv(combined_df, raw_file)
            meta = {
                "file_name": raw_file.name,
                "symbol": symbol.upper(),
                "interval": interval,
                "market_type": "spot",
                "total_bars": len(combined_df),
                "first_bar_utc": str(combined_df["open_time"].min()),
                "last_bar_utc": str(combined_df["open_time"].max()),
                "retrieved_at_utc": str(pd.Timestamp.now(tz="UTC")),
                "data_source": "binance_spot_live",
                "status": "raw",
            }
            self.save_metadata(raw_file, meta)

        return combined_df

    def fetch_historical_bars(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        total_bars: int = 1500,
        fallback_to_sample: bool = True,
        exclude_unclosed_bar: bool = True,
        market_type: str = "spot",
    ) -> pd.DataFrame:
        """Fetches a sequence of historical bars, paginating API requests backwards if total_bars > 1000,
        excluding incomplete current candles if requested, and falling back safely if offline.
        """
        if market_type.lower() != "spot":
            raise ValueError(
                f"Unsupported market_type '{market_type}'. Only 'spot' is supported to prevent mixing spot and futures datasets."
            )

        try:
            logger.info(f"Attempting live fetch for {symbol} ({interval}, total_bars={total_bars})...")
            all_chunks = []
            remaining = total_bars
            current_end_time_ms: int | None = None

            while remaining > 0:
                batch_limit = min(remaining, 1000)
                chunk = self.fetch_binance_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=batch_limit,
                    end_time_ms=current_end_time_ms,
                )
                if chunk.empty:
                    break
                all_chunks.append(chunk)
                remaining -= len(chunk)
                earliest_open_time = chunk["open_time"].min()
                current_end_time_ms = int(earliest_open_time.timestamp() * 1000) - 1
                if len(chunk) < batch_limit:
                    break

            if all_chunks:
                df = pd.concat(all_chunks, ignore_index=True)
                df = df.drop_duplicates(subset=["open_time"]).sort_values(by="open_time").reset_index(drop=True)

                if exclude_unclosed_bar and not df.empty:
                    now_utc = pd.Timestamp.now(tz="UTC")
                    delta = self.get_interval_delta(interval)
                    if df["open_time"].iloc[-1] + delta > now_utc:
                        logger.info(
                            f"Excluding current unclosed bar ({df['open_time'].iloc[-1]} UTC) to prevent partial candle distortion."
                        )
                        df = df.iloc[:-1].reset_index(drop=True)

                if len(df) > total_bars:
                    df = df.iloc[-total_bars:].reset_index(drop=True)

                df.attrs["data_source"] = "binance_spot_live"
                df.attrs["is_synthetic"] = False
                df.attrs["market_type"] = "spot"
                logger.info(
                    f"[DATA SOURCE: binance_spot_live] Acquired {len(df)} bars for {symbol} ({interval}) | "
                    f"Range: {df['open_time'].min()} to {df['open_time'].max()}"
                )

                cache_file = self.raw_dir / f"{symbol}_{interval}.csv"
                # If a larger cache already exists, merge new bars into it without truncating historical depth
                if cache_file.exists():
                    try:
                        existing_cache = self.load_from_csv(cache_file)
                        if "open_time" in existing_cache.columns:
                            existing_cache["open_time"] = pd.to_datetime(existing_cache["open_time"], utc=True)
                            combined_cache = (
                                pd.concat([existing_cache, df], ignore_index=True)
                                .drop_duplicates(subset=["open_time"])
                                .sort_values(by="open_time")
                                .reset_index(drop=True)
                            )
                            self.save_to_csv(combined_cache, cache_file)
                            meta = {
                                "file_name": cache_file.name,
                                "symbol": symbol.upper(),
                                "interval": interval,
                                "market_type": "spot",
                                "total_bars": len(combined_cache),
                                "first_bar_utc": str(combined_cache["open_time"].min()),
                                "last_bar_utc": str(combined_cache["open_time"].max()),
                                "retrieved_at_utc": str(pd.Timestamp.now(tz="UTC")),
                                "data_source": "binance_spot_live",
                                "status": "raw",
                            }
                            self.save_metadata(cache_file, meta)
                        else:
                            self.save_to_csv(df, cache_file)
                    except Exception:
                        self.save_to_csv(df, cache_file)
                else:
                    self.save_to_csv(df, cache_file)
                    meta = {
                        "file_name": cache_file.name,
                        "symbol": symbol.upper(),
                        "interval": interval,
                        "market_type": "spot",
                        "total_bars": len(df),
                        "first_bar_utc": str(df["open_time"].min()),
                        "last_bar_utc": str(df["open_time"].max()),
                        "retrieved_at_utc": str(pd.Timestamp.now(tz="UTC")),
                        "data_source": "binance_spot_live",
                        "status": "raw",
                    }
                    self.save_metadata(cache_file, meta)
                return df
        except Exception as e:
            logger.warning(f"Live fetch unavailable ({e}). Proceeding to fallback options.")

        # Priority 1 Fallback: Cached real raw data
        raw_file = self.raw_dir / f"{symbol}_{interval}.csv"
        if raw_file.exists():
            df_raw = self.load_from_csv(raw_file)
            df_raw.attrs["data_source"] = "cached_raw_csv"
            df_raw.attrs["is_synthetic"] = False
            df_raw.attrs["market_type"] = "spot"
            logger.warning(
                f"[DATA SOURCE: cached_raw_csv] Using cached real market data from {raw_file} ({len(df_raw)} bars)."
            )
            return df_raw

        # Priority 2 Fallback: Verified static sample data
        sample_file = self.sample_dir / f"{symbol}_{interval}.csv"
        if sample_file.exists():
            df_sample = self.load_from_csv(sample_file)
            df_sample.attrs["data_source"] = "local_sample_csv"
            df_sample.attrs["is_synthetic"] = False
            df_sample.attrs["market_type"] = "spot"
            logger.warning(
                f"[DATA SOURCE: local_sample_csv] Using static verified sample from {sample_file} ({len(df_sample)} bars)."
            )
            return df_sample

        # Priority 3 Fallback: Deterministic synthetic data (strictly isolated)
        if fallback_to_sample:
            logger.warning(
                f"[DATA SOURCE: synthetic] Generating deterministic synthetic data for {symbol} ({total_bars} bars). "
                "NOTICE: Synthetic data cannot be used to substantiate real market trading performance."
            )
            df_synth = self.generate_synthetic_ohlcv(symbol=symbol, n_bars=total_bars, interval=interval)
            df_synth.attrs["data_source"] = "synthetic"
            df_synth.attrs["is_synthetic"] = True
            df_synth.attrs["market_type"] = "synthetic"
            synth_file = self.synthetic_dir / f"{symbol}_{interval}_synthetic.csv"
            self.save_to_csv(df_synth, synth_file)
            return df_synth

        raise RuntimeError(f"Could not acquire market data for {symbol} ({interval}).")

    @staticmethod
    def generate_synthetic_ohlcv(
        symbol: str = "BTCUSDT",
        n_bars: int = 1500,
        interval: str = "1h",
        start_date: str = "2024-01-01 00:00:00",
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generates deterministic, statistically realistic cryptocurrency OHLCV data."""
        rng = np.random.default_rng(seed)

        symbol_params = {
            "BTCUSDT": {"init_price": 42000.0, "annual_vol": 0.55, "base_volume": 1200.0},
            "ETHUSDT": {"init_price": 2400.0, "annual_vol": 0.70, "base_volume": 15000.0},
            "SOLUSDT": {"init_price": 100.0, "annual_vol": 0.90, "base_volume": 85000.0},
        }
        params = symbol_params.get(symbol.upper(), {"init_price": 1000.0, "annual_vol": 0.65, "base_volume": 5000.0})

        dt = 1.0 / 8760.0
        hourly_vol = params["annual_vol"] * np.sqrt(dt)
        mu = 0.05 * dt

        vol_state = np.ones(n_bars)
        shocks = rng.standard_normal(n_bars)
        for t in range(1, n_bars):
            vol_state[t] = 0.92 * vol_state[t - 1] + 0.08 * (1.0 + 0.5 * abs(shocks[t - 1]))

        returns = mu + (hourly_vol * vol_state) * shocks
        log_prices = np.log(params["init_price"]) + np.cumsum(returns)
        closes = np.exp(log_prices)

        opens = np.empty(n_bars)
        highs = np.empty(n_bars)
        lows = np.empty(n_bars)
        volumes = np.empty(n_bars)

        opens[0] = params["init_price"]
        for t in range(n_bars):
            if t > 0:
                opens[t] = closes[t - 1] * (1.0 + rng.normal(0, 0.0005))

            bar_vol = abs(returns[t]) + rng.exponential(scale=hourly_vol * 0.5)
            spread_high = abs(rng.normal(0, bar_vol))
            spread_low = abs(rng.normal(0, bar_vol))

            higher = max(opens[t], closes[t])
            lower = min(opens[t], closes[t])

            highs[t] = higher * (1.0 + spread_high)
            lows[t] = lower * (1.0 - spread_low)

            vol_multiplier = 1.0 + 15.0 * bar_vol
            volumes[t] = params["base_volume"] * vol_multiplier * rng.lognormal(0, 0.3)

        timestamps = pd.date_range(start=start_date, periods=n_bars, freq=interval, tz="UTC")

        df = pd.DataFrame(
            {
                "open_time": timestamps,
                "open": np.round(opens, 2),
                "high": np.round(highs, 2),
                "low": np.round(lows, 2),
                "close": np.round(closes, 2),
                "volume": np.round(volumes, 4),
            }
        )
        return df
