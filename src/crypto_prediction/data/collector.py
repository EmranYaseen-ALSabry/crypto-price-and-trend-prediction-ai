"""Market data collector supporting Binance REST API, offline CSVs, and synthetic fallback."""

from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

from crypto_prediction.utils.logger import get_logger

logger = get_logger("data_collector")


class DataCollector:
    """Collects OHLCV market data via REST API or local CSVs, with deterministic synthetic fallback."""

    BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

    def __init__(
        self,
        base_dir: str | Path = "data",
        request_timeout: int = 10,
        rate_limit_delay: float = 0.25,
        max_retries: int = 3,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.sample_dir = self.base_dir / "sample"
        self.synthetic_dir = self.base_dir / "synthetic"
        self.request_timeout = request_timeout
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.synthetic_dir.mkdir(parents=True, exist_ok=True)

    def load_from_csv(self, file_path: str | Path) -> pd.DataFrame:
        """Loads historical OHLCV data from a local CSV file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file does not exist at: {path}")

        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows from CSV: {path.name}")
        return df

    def save_to_csv(self, df: pd.DataFrame, file_path: str | Path) -> Path:
        """Saves OHLCV DataFrame to CSV."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # If open_time is in index, reset index before saving
        out_df = df.reset_index() if isinstance(df.index, pd.DatetimeIndex) else df
        out_df.to_csv(path, index=False)
        logger.info(f"Saved {len(out_df)} rows to CSV: {path}")
        return path

    def fetch_binance_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 1000,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> pd.DataFrame:
        """Fetches OHLCV kline data from Binance public REST endpoint."""
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
            try:
                time.sleep(self.rate_limit_delay)
                response = requests.get(
                    self.BINANCE_KLINES_URL,
                    params=params,
                    timeout=self.request_timeout,
                    headers={"User-Agent": "CryptoPredictionAI/1.0"},
                )
                if response.status_code == 200:
                    raw_data = response.json()
                    if not raw_data:
                        logger.warning(f"Empty response returned for {symbol} ({interval}).")
                        return pd.DataFrame()

                    # Binance kline payload structure:
                    # [0: open_time, 1: open, 2: high, 3: low, 4: close, 5: volume, 6: close_time, ...]
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
                else:
                    logger.warning(
                        f"Attempt {attempt}/{self.max_retries}: Binance API returned HTTP {response.status_code}."
                    )
            except Exception as exc:
                logger.warning(f"Attempt {attempt}/{self.max_retries} failed with error: {exc}")

        raise ConnectionError(f"Failed to fetch market data from Binance API after {self.max_retries} attempts.")

    def fetch_historical_bars(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        total_bars: int = 1500,
        fallback_to_sample: bool = True,
        exclude_unclosed_bar: bool = True,
    ) -> pd.DataFrame:
        """Fetches a sequence of historical bars, paginating API requests if total_bars > 1000,
        excluding incomplete current candles if requested, and falling back safely if offline.
        """
        try:
            # Attempt live API fetch with pagination
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
                    break  # Reached end of historical data

            if all_chunks:
                df = pd.concat(all_chunks, ignore_index=True)
                df = df.drop_duplicates(subset=["open_time"]).sort_values(by="open_time").reset_index(drop=True)

                # Check and exclude incomplete / unclosed current bar
                if exclude_unclosed_bar and not df.empty:
                    now_utc = pd.Timestamp.now(tz="UTC")
                    interval_deltas = {
                        "1m": pd.Timedelta(minutes=1),
                        "5m": pd.Timedelta(minutes=5),
                        "15m": pd.Timedelta(minutes=15),
                        "30m": pd.Timedelta(minutes=30),
                        "1h": pd.Timedelta(hours=1),
                        "4h": pd.Timedelta(hours=4),
                        "1d": pd.Timedelta(days=1),
                    }
                    delta = interval_deltas.get(interval, pd.Timedelta(hours=1))
                    if df["open_time"].iloc[-1] + delta > now_utc:
                        logger.info(
                            f"Excluding current unclosed bar ({df['open_time'].iloc[-1]} UTC) to prevent partial candle distortion."
                        )
                        df = df.iloc[:-1].reset_index(drop=True)

                if len(df) > total_bars:
                    df = df.iloc[-total_bars:].reset_index(drop=True)

                df.attrs["data_source"] = "binance_live"
                df.attrs["is_synthetic"] = False
                logger.info(
                    f"[DATA SOURCE: binance_live] Acquired {len(df)} bars for {symbol} ({interval}) | "
                    f"Range: {df['open_time'].min()} to {df['open_time'].max()}"
                )

                # Cache to raw directory
                cache_file = self.raw_dir / f"{symbol}_{interval}.csv"
                self.save_to_csv(df, cache_file)
                return df
        except Exception as e:
            logger.warning(f"Live fetch unavailable ({e}). Proceeding to fallback options.")

        # Priority 1 Fallback: Cached real raw data
        raw_file = self.raw_dir / f"{symbol}_{interval}.csv"
        if raw_file.exists():
            df_raw = self.load_from_csv(raw_file)
            df_raw.attrs["data_source"] = "cached_raw_csv"
            df_raw.attrs["is_synthetic"] = False
            logger.warning(f"[DATA SOURCE: cached_raw_csv] Using cached real market data from {raw_file} ({len(df_raw)} bars).")
            return df_raw

        # Priority 2 Fallback: Verified static sample data
        sample_file = self.sample_dir / f"{symbol}_{interval}.csv"
        if sample_file.exists():
            df_sample = self.load_from_csv(sample_file)
            df_sample.attrs["data_source"] = "local_sample_csv"
            df_sample.attrs["is_synthetic"] = False
            logger.warning(f"[DATA SOURCE: local_sample_csv] Using static verified sample from {sample_file} ({len(df_sample)} bars).")
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
        """Generates deterministic, statistically realistic cryptocurrency OHLCV data.

        Features:
        - Geometric Brownian motion with volatility clustering (GARCH-like).
        - Symbol-specific baseline price and volatility regime.
        - Realistic intra-bar spread (High >= Max(Open, Close), Low <= Min(Open, Close)).
        - Correlated trading volume and price volatility.

        Args:
            symbol: Trading symbol (e.g. BTCUSDT, ETHUSDT, SOLUSDT).
            n_bars: Number of hourly bars.
            interval: Frequency string ('1h', '4h', etc.).
            start_date: Timestamp origin.
            seed: Integer random seed for determinism.

        Returns:
            DataFrame with columns ['open_time', 'open', 'high', 'low', 'close', 'volume'].
        """
        rng = np.random.default_rng(seed)

        # Baseline parameters per symbol
        symbol_params = {
            "BTCUSDT": {"init_price": 42000.0, "annual_vol": 0.55, "base_volume": 1200.0},
            "ETHUSDT": {"init_price": 2400.0, "annual_vol": 0.70, "base_volume": 15000.0},
            "SOLUSDT": {"init_price": 100.0, "annual_vol": 0.90, "base_volume": 85000.0},
        }
        params = symbol_params.get(symbol.upper(), {"init_price": 1000.0, "annual_vol": 0.65, "base_volume": 5000.0})

        # Hourly drift and base volatility (8760 hours/year)
        dt = 1.0 / 8760.0
        hourly_vol = params["annual_vol"] * np.sqrt(dt)
        mu = 0.05 * dt  # slight positive drift

        # Volatility clustering (autoregressive volatility state)
        vol_state = np.ones(n_bars)
        shocks = rng.standard_normal(n_bars)
        for t in range(1, n_bars):
            # AR(1) log-volatility process
            vol_state[t] = 0.92 * vol_state[t - 1] + 0.08 * (1.0 + 0.5 * abs(shocks[t - 1]))

        # Returns with stochastic volatility
        returns = mu + (hourly_vol * vol_state) * shocks

        # Construct close price path
        log_prices = np.log(params["init_price"]) + np.cumsum(returns)
        closes = np.exp(log_prices)

        # Build realistic Open, High, Low, Volume
        opens = np.empty(n_bars)
        highs = np.empty(n_bars)
        lows = np.empty(n_bars)
        volumes = np.empty(n_bars)

        opens[0] = params["init_price"]
        for t in range(n_bars):
            if t > 0:
                # Open is close of previous bar plus small micro-gap
                opens[t] = closes[t - 1] * (1.0 + rng.normal(0, 0.0005))

            # Intra-bar volatility
            bar_vol = abs(returns[t]) + rng.exponential(scale=hourly_vol * 0.5)
            spread_high = abs(rng.normal(0, bar_vol))
            spread_low = abs(rng.normal(0, bar_vol))

            higher = max(opens[t], closes[t])
            lower = min(opens[t], closes[t])

            highs[t] = higher * (1.0 + spread_high)
            lows[t] = lower * (1.0 - spread_low)

            # Volume is positively correlated with price volatility
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
