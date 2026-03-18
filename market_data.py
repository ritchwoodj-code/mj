"""Market data service - fetches real-time futures data from free sources."""

import time
import threading
import logging
from datetime import datetime, timedelta
from collections import defaultdict

import yfinance as yf
import pandas as pd
import numpy as np
import requests

from config import INSTRUMENTS, TIMEFRAMES

logger = logging.getLogger(__name__)


class MarketDataService:
    """Fetches and caches real-time market data for Gold and NQ futures."""

    def __init__(self):
        self._cache = {}           # {instrument: {timeframe: DataFrame}}
        self._live_prices = {}     # {instrument: {price, bid, ask, volume, timestamp}}
        self._lock = threading.Lock()
        self._running = False
        self._callbacks = []

    def register_callback(self, fn):
        """Register a function to call when new price data arrives."""
        self._callbacks.append(fn)

    def _notify(self, instrument, price_data):
        for cb in self._callbacks:
            try:
                cb(instrument, price_data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def start(self):
        """Start background threads for live price polling."""
        self._running = True
        for inst_key, inst_cfg in INSTRUMENTS.items():
            t = threading.Thread(
                target=self._poll_live_price,
                args=(inst_key, inst_cfg["symbol"]),
                daemon=True,
            )
            t.start()
        # Historical data refresh every 60s
        t2 = threading.Thread(target=self._refresh_historical_loop, daemon=True)
        t2.start()
        logger.info("MarketDataService started.")

    def stop(self):
        self._running = False

    def _poll_live_price(self, inst_key, symbol):
        """Poll live price every second using yfinance fast_info."""
        ticker = yf.Ticker(symbol)
        while self._running:
            try:
                info = ticker.fast_info
                price = info.get("lastPrice") or info.get("last_price")
                if price is None:
                    # Fallback: fetch 1m data for latest close
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

    def _refresh_historical_loop(self):
        """Refresh historical candle data periodically."""
        while self._running:
            self.fetch_all_historical()
            time.sleep(60)

    def fetch_all_historical(self):
        """Fetch historical OHLCV data for all instruments and timeframes."""
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

                    # For 4h timeframe, resample from 1h
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

    def get_live_price(self, instrument):
        """Get the latest live price for an instrument."""
        with self._lock:
            return self._live_prices.get(instrument)

    def get_all_live_prices(self):
        """Get all live prices."""
        with self._lock:
            return dict(self._live_prices)

    def get_historical(self, instrument, timeframe):
        """Get historical OHLCV DataFrame for instrument/timeframe."""
        with self._lock:
            inst_data = self._cache.get(instrument, {})
            df = inst_data.get(timeframe)
            if df is not None:
                return df.copy()
            return pd.DataFrame()

    def get_all_timeframes(self, instrument):
        """Get all available timeframe data for an instrument."""
        with self._lock:
            inst_data = self._cache.get(instrument, {})
            return {tf: df.copy() for tf, df in inst_data.items()}
