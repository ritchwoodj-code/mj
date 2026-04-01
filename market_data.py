"""Market data service - fetches real-time futures data from free sources.

Automatically falls back to realistic simulation mode if Yahoo Finance
is unreachable (e.g. behind a proxy/firewall).
"""

import time
import threading
import logging
import math
import random
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np

from config import INSTRUMENTS, TIMEFRAMES

logger = logging.getLogger(__name__)


def _test_yahoo_connection():
    """Check if we can reach Yahoo Finance."""
    try:
        import yfinance as yf
        ticker = yf.Ticker("GC=F")
        info = ticker.fast_info
        # Try to access any attribute
        _ = info.get("lastPrice") or info.get("last_price")
        df = ticker.history(period="1d", interval="1m")
        if df.empty:
            return False
        return True
    except Exception as e:
        logger.warning(f"Yahoo Finance unavailable: {e}")
        return False


class SimulatedPriceEngine:
    """Generates realistic futures price movements for demo mode."""

    def __init__(self):
        self._prices = {
            "GOLD": 3010.50,  # Realistic gold futures price
            "NQ": 19850.25,   # Realistic NQ futures price
        }
        self._volatility = {
            "GOLD": 0.8,   # ~$0.80 per tick noise
            "NQ": 5.0,     # ~$5 per tick noise
        }
        self._trend = {"GOLD": 0, "NQ": 0}
        self._trend_duration = {"GOLD": 0, "NQ": 0}
        self._volume_base = {"GOLD": 1200, "NQ": 3500}

    def next_price(self, instrument):
        """Generate next realistic price tick."""
        vol = self._volatility[instrument]
        price = self._prices[instrument]

        # Occasionally change trend (mean-reverting random walk with drift)
        self._trend_duration[instrument] -= 1
        if self._trend_duration[instrument] <= 0:
            # New trend: slight bias up or down
            self._trend[instrument] = random.gauss(0, vol * 0.3)
            self._trend_duration[instrument] = random.randint(10, 120)

        # Price movement = trend + noise
        noise = random.gauss(0, vol)
        move = self._trend[instrument] * 0.1 + noise

        # Add time-of-day volatility spikes (simulate session opens)
        hour = datetime.utcnow().hour
        if hour in [13, 14, 18, 19]:  # Market opens in UTC
            move *= 1.5

        new_price = price + move

        # Apply tick size constraints
        tick = INSTRUMENTS[instrument]["tick_size"]
        new_price = round(round(new_price / tick) * tick, 2)

        self._prices[instrument] = new_price

        # Simulate volume
        vol_noise = random.gauss(1.0, 0.3)
        volume = max(1, int(self._volume_base[instrument] * abs(vol_noise)))

        spread = tick * random.randint(1, 3)

        return {
            "price": new_price,
            "bid": round(new_price - spread / 2, 2),
            "ask": round(new_price + spread / 2, 2),
            "volume": volume,
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": INSTRUMENTS[instrument]["symbol"],
            "instrument": instrument,
        }

    def generate_historical(self, instrument, num_candles, interval_minutes):
        """Generate realistic historical OHLCV data."""
        # Start from a realistic base and build up to current price
        base_prices = {"GOLD": 3010.50, "NQ": 19850.25}
        price = base_prices.get(instrument, self._prices[instrument])
        vol = self._volatility[instrument]
        records = []

        # Walk backwards to create history
        base_time = datetime.utcnow() - timedelta(minutes=interval_minutes * num_candles)

        # Scale volatility by timeframe but keep it reasonable
        tf_vol = vol * math.sqrt(min(interval_minutes, 60)) * 0.5

        trend_bias = random.gauss(0, tf_vol * 0.02)

        for i in range(num_candles):
            ts = base_time + timedelta(minutes=interval_minutes * i)

            # Generate OHLCV
            open_p = price
            moves = [random.gauss(trend_bias, tf_vol) for _ in range(4)]
            intra_prices = [open_p + sum(moves[:j+1]) for j in range(4)]
            close_p = intra_prices[-1]
            high_p = max(open_p, close_p, max(intra_prices)) + abs(random.gauss(0, tf_vol * 0.3))
            low_p = min(open_p, close_p, min(intra_prices)) - abs(random.gauss(0, tf_vol * 0.3))

            tick = INSTRUMENTS[instrument]["tick_size"]
            open_p = round(round(open_p / tick) * tick, 2)
            high_p = round(round(high_p / tick) * tick, 2)
            low_p = round(round(low_p / tick) * tick, 2)
            close_p = round(round(close_p / tick) * tick, 2)

            vol_mult = random.gauss(1.0, 0.4)
            volume = max(100, int(self._volume_base[instrument] * abs(vol_mult) * (interval_minutes / 5)))

            records.append({
                "timestamp": ts,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": volume,
            })

            price = close_p
            trend_bias += random.gauss(0, tf_vol * 0.005)
            # Mean revert toward base price to prevent drift
            base = base_prices.get(instrument, 3000)
            drift = (base - price) * 0.0001
            trend_bias += drift

        self._prices[instrument] = price

        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        return df


class MarketDataService:
    """Fetches and caches real-time market data for Gold and NQ futures.

    Automatically uses Yahoo Finance when available, falls back to
    realistic simulation for demo/testing.
    """

    def __init__(self):
        self._cache = {}
        self._live_prices = {}
        self._lock = threading.Lock()
        self._running = False
        self._callbacks = []
        self._use_live = False
        self._sim_engine = SimulatedPriceEngine()

    def register_callback(self, fn):
        self._callbacks.append(fn)

    def _notify(self, instrument, price_data):
        for cb in self._callbacks:
            try:
                cb(instrument, price_data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def start(self):
        """Start background threads for price updates."""
        self._running = True

        # Test if Yahoo Finance is reachable
        logger.info("Testing Yahoo Finance connection...")
        self._use_live = _test_yahoo_connection()

        if self._use_live:
            logger.info("Yahoo Finance connected - using LIVE market data.")
            for inst_key, inst_cfg in INSTRUMENTS.items():
                t = threading.Thread(
                    target=self._poll_live_price,
                    args=(inst_key, inst_cfg["symbol"]),
                    daemon=True,
                )
                t.start()
        else:
            logger.info("Yahoo Finance unavailable - using SIMULATED market data.")
            logger.info("Prices will move realistically for demo/testing purposes.")
            # Generate initial historical data
            self._generate_sim_historical()
            # Start simulated price feed
            for inst_key in INSTRUMENTS:
                t = threading.Thread(
                    target=self._poll_sim_price,
                    args=(inst_key,),
                    daemon=True,
                )
                t.start()

        # Historical data refresh
        t2 = threading.Thread(target=self._refresh_historical_loop, daemon=True)
        t2.start()

        logger.info("MarketDataService started.")

    def stop(self):
        self._running = False

    # --- Live Yahoo Finance ---

    def _poll_live_price(self, inst_key, symbol):
        """Poll live price every second using yfinance."""
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        while self._running:
            try:
                info = ticker.fast_info
                price = info.get("lastPrice") or info.get("last_price")
                if price is None:
                    df = ticker.history(period="1d", interval="1m")
                    if not df.empty:
                        price = float(df["Close"].iloc[-1])

                if price is not None:
                    bid = info.get("bid", price - 0.10)
                    ask = info.get("ask", price + 0.10)
                    vol = info.get("lastVolume") or info.get("last_volume", 0)

                    price_data = {
                        "price": round(float(price), 2),
                        "bid": round(float(bid), 2) if bid else round(float(price) - 0.10, 2),
                        "ask": round(float(ask), 2) if ask else round(float(price) + 0.10, 2),
                        "volume": int(vol) if vol else 0,
                        "timestamp": datetime.utcnow().isoformat(),
                        "symbol": symbol,
                        "instrument": inst_key,
                    }

                    with self._lock:
                        self._live_prices[inst_key] = price_data

                    self._notify(inst_key, price_data)

            except Exception as e:
                logger.warning(f"Price poll error for {inst_key}: {e}")

            time.sleep(1)

    # --- Simulated Price Feed ---

    def _poll_sim_price(self, inst_key):
        """Generate simulated price ticks every second."""
        while self._running:
            try:
                price_data = self._sim_engine.next_price(inst_key)

                with self._lock:
                    self._live_prices[inst_key] = price_data

                    # Update the latest candle in 1m cache
                    if inst_key in self._cache and "1m" in self._cache[inst_key]:
                        df = self._cache[inst_key]["1m"]
                        if not df.empty:
                            df.iloc[-1, df.columns.get_loc("close")] = price_data["price"]
                            df.iloc[-1, df.columns.get_loc("high")] = max(
                                df.iloc[-1]["high"], price_data["price"])
                            df.iloc[-1, df.columns.get_loc("low")] = min(
                                df.iloc[-1]["low"], price_data["price"])
                            df.iloc[-1, df.columns.get_loc("volume")] += price_data["volume"]

                self._notify(inst_key, price_data)

            except Exception as e:
                logger.warning(f"Sim price error for {inst_key}: {e}")

            time.sleep(1)

    def _generate_sim_historical(self):
        """Generate simulated historical data for all timeframes."""
        tf_candles = {
            "1m": (500, 1),
            "5m": (500, 5),
            "15m": (400, 15),
            "30m": (300, 30),
            "1h": (300, 60),
            "4h": (200, 240),
            "1d": (500, 1440),
            "1w": (200, 10080),
        }

        for inst_key in INSTRUMENTS:
            inst_cache = {}
            for tf_key, (num_candles, interval_min) in tf_candles.items():
                df = self._sim_engine.generate_historical(inst_key, num_candles, interval_min)
                inst_cache[tf_key] = df

            with self._lock:
                self._cache[inst_key] = inst_cache

        logger.info("Simulated historical data generated for all instruments.")

    # --- Historical Refresh ---

    def _refresh_historical_loop(self):
        while self._running:
            if self._use_live:
                self._fetch_live_historical()
            else:
                # In sim mode, add new candles periodically
                self._update_sim_candles()
            time.sleep(60)

    def _fetch_live_historical(self):
        """Fetch historical data from Yahoo Finance."""
        import yfinance as yf
        for inst_key, inst_cfg in INSTRUMENTS.items():
            symbol = inst_cfg["symbol"]
            ticker = yf.Ticker(symbol)
            inst_cache = {}

            for tf_key, tf_cfg in TIMEFRAMES.items():
                try:
                    df = ticker.history(
                        period=tf_cfg["period"],
                        interval=tf_cfg["interval"],
                    )
                    if df.empty:
                        continue

                    if tf_key == "4h":
                        df = df.resample("4h").agg({
                            "Open": "first",
                            "High": "max",
                            "Low": "min",
                            "Close": "last",
                            "Volume": "sum",
                        }).dropna()

                    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                    df.columns = ["open", "high", "low", "close", "volume"]
                    inst_cache[tf_key] = df

                except Exception as e:
                    logger.warning(f"Historical fetch error {inst_key}/{tf_key}: {e}")

            with self._lock:
                self._cache[inst_key] = inst_cache

        logger.info("Historical data refreshed for all instruments.")

    def _update_sim_candles(self):
        """Add new simulated candles to maintain fresh data."""
        for inst_key in INSTRUMENTS:
            live = self._live_prices.get(inst_key)
            if not live:
                continue

            price = live["price"]
            with self._lock:
                if inst_key not in self._cache:
                    continue
                # Add a new 1m candle
                if "1m" in self._cache[inst_key]:
                    df = self._cache[inst_key]["1m"]
                    new_ts = df.index[-1] + timedelta(minutes=1) if not df.empty else datetime.utcnow()
                    new_row = pd.DataFrame({
                        "open": [price],
                        "high": [price],
                        "low": [price],
                        "close": [price],
                        "volume": [0],
                    }, index=[new_ts])
                    self._cache[inst_key]["1m"] = pd.concat([df.iloc[-499:], new_row])

    # --- Public API ---

    def get_live_price(self, instrument):
        with self._lock:
            return self._live_prices.get(instrument)

    def get_all_live_prices(self):
        with self._lock:
            return dict(self._live_prices)

    def get_historical(self, instrument, timeframe):
        with self._lock:
            inst_data = self._cache.get(instrument, {})
            df = inst_data.get(timeframe)
            if df is not None:
                return df.copy()
            return pd.DataFrame()

    def get_all_timeframes(self, instrument):
        with self._lock:
            inst_data = self._cache.get(instrument, {})
            return {tf: df.copy() for tf, df in inst_data.items()}

    def is_live(self):
        """Return whether using live or simulated data."""
        return self._use_live
