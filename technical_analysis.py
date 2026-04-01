"""Technical analysis engine - computes indicators and detects patterns."""

import numpy as np
import pandas as pd
from ta.trend import (
    EMAIndicator, SMAIndicator, MACD, ADXIndicator,
    IchimokuIndicator, VortexIndicator
)
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import (
    VolumeWeightedAveragePrice, OnBalanceVolumeIndicator,
    MFIIndicator
)


class TechnicalAnalyzer:
    """Computes a full suite of technical indicators and pattern detection."""

    def analyze(self, df):
        """Run full technical analysis on an OHLCV DataFrame.

        Returns a dict with all indicators and pattern signals.
        """
        if df.empty or len(df) < 20:
            return {}

        result = {}
        result["indicators"] = self._compute_indicators(df)
        result["patterns"] = self._detect_patterns(df)
        result["structure"] = self._analyze_market_structure(df)
        result["support_resistance"] = self._find_support_resistance(df)
        result["trend"] = self._determine_trend(df, result["indicators"])
        result["momentum"] = self._assess_momentum(result["indicators"])
        result["volatility"] = self._assess_volatility(df, result["indicators"])
        return result

    def _compute_indicators(self, df):
        """Compute all technical indicators."""
        ind = {}
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Moving Averages
        for period in [9, 20, 50, 100, 200]:
            if len(df) >= period:
                ind[f"ema_{period}"] = EMAIndicator(close, window=period).ema_indicator().iloc[-1]
                ind[f"sma_{period}"] = SMAIndicator(close, window=period).sma_indicator().iloc[-1]

        # RSI
        if len(df) >= 14:
            rsi = RSIIndicator(close, window=14)
            ind["rsi_14"] = rsi.rsi().iloc[-1]

        # MACD
        if len(df) >= 26:
            macd = MACD(close)
            ind["macd_line"] = macd.macd().iloc[-1]
            ind["macd_signal"] = macd.macd_signal().iloc[-1]
            ind["macd_histogram"] = macd.macd_diff().iloc[-1]

        # Bollinger Bands
        if len(df) >= 20:
            bb = BollingerBands(close)
            ind["bb_upper"] = bb.bollinger_hband().iloc[-1]
            ind["bb_middle"] = bb.bollinger_mavg().iloc[-1]
            ind["bb_lower"] = bb.bollinger_lband().iloc[-1]
            ind["bb_width"] = bb.bollinger_wband().iloc[-1]

        # ATR
        if len(df) >= 14:
            atr = AverageTrueRange(high, low, close, window=14)
            ind["atr_14"] = atr.average_true_range().iloc[-1]

        # Stochastic
        if len(df) >= 14:
            stoch = StochasticOscillator(high, low, close)
            ind["stoch_k"] = stoch.stoch().iloc[-1]
            ind["stoch_d"] = stoch.stoch_signal().iloc[-1]

        # ADX
        if len(df) >= 14:
            adx = ADXIndicator(high, low, close)
            ind["adx"] = adx.adx().iloc[-1]
            ind["adx_pos"] = adx.adx_pos().iloc[-1]
            ind["adx_neg"] = adx.adx_neg().iloc[-1]

        # VWAP (intraday only)
        if len(df) >= 2 and volume.sum() > 0:
            try:
                vwap = VolumeWeightedAveragePrice(high, low, close, volume)
                ind["vwap"] = vwap.volume_weighted_average_price().iloc[-1]
            except Exception:
                ind["vwap"] = close.iloc[-1]

        # OBV
        if len(df) >= 2:
            obv = OnBalanceVolumeIndicator(close, volume)
            obv_series = obv.on_balance_volume()
            ind["obv"] = obv_series.iloc[-1]
            ind["obv_slope"] = (obv_series.iloc[-1] - obv_series.iloc[-5]) if len(obv_series) >= 5 else 0

        # MFI
        if len(df) >= 14 and volume.sum() > 0:
            mfi = MFIIndicator(high, low, close, volume)
            ind["mfi"] = mfi.money_flow_index().iloc[-1]

        # Williams %R
        if len(df) >= 14:
            wr = WilliamsRIndicator(high, low, close)
            ind["williams_r"] = wr.williams_r().iloc[-1]

        # Keltner Channel
        if len(df) >= 20:
            kc = KeltnerChannel(high, low, close)
            ind["kc_upper"] = kc.keltner_channel_hband().iloc[-1]
            ind["kc_lower"] = kc.keltner_channel_lband().iloc[-1]

        return ind

    def _detect_patterns(self, df):
        """Detect candlestick and price patterns."""
        patterns = []
        if len(df) < 5:
            return patterns

        o = df["open"].values
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values

        # Engulfing patterns
        if c[-1] > o[-1] and c[-2] < o[-2]:
            if c[-1] > o[-2] and o[-1] < c[-2]:
                patterns.append({"name": "Bullish Engulfing", "bias": "bullish", "strength": 75})
        if c[-1] < o[-1] and c[-2] > o[-2]:
            if c[-1] < o[-2] and o[-1] > c[-2]:
                patterns.append({"name": "Bearish Engulfing", "bias": "bearish", "strength": 75})

        # Doji
        body = abs(c[-1] - o[-1])
        total_range = h[-1] - l[-1]
        if total_range > 0 and body / total_range < 0.1:
            patterns.append({"name": "Doji", "bias": "neutral", "strength": 50})

        # Hammer / Shooting Star
        if total_range > 0:
            upper_wick = h[-1] - max(o[-1], c[-1])
            lower_wick = min(o[-1], c[-1]) - l[-1]
            if lower_wick > 2 * body and upper_wick < body:
                patterns.append({"name": "Hammer", "bias": "bullish", "strength": 70})
            if upper_wick > 2 * body and lower_wick < body:
                patterns.append({"name": "Shooting Star", "bias": "bearish", "strength": 70})

        # Morning/Evening Star (3-candle)
        if len(df) >= 3:
            body_3 = abs(c[-3] - o[-3])
            body_2 = abs(c[-2] - o[-2])
            body_1 = abs(c[-1] - o[-1])
            if (c[-3] < o[-3] and body_2 < body_3 * 0.3 and c[-1] > o[-1]
                    and c[-1] > (o[-3] + c[-3]) / 2):
                patterns.append({"name": "Morning Star", "bias": "bullish", "strength": 80})
            if (c[-3] > o[-3] and body_2 < body_3 * 0.3 and c[-1] < o[-1]
                    and c[-1] < (o[-3] + c[-3]) / 2):
                patterns.append({"name": "Evening Star", "bias": "bearish", "strength": 80})

        # Three White Soldiers / Three Black Crows
        if len(df) >= 3:
            if all(c[-i] > o[-i] for i in range(1, 4)) and c[-1] > c[-2] > c[-3]:
                patterns.append({"name": "Three White Soldiers", "bias": "bullish", "strength": 85})
            if all(c[-i] < o[-i] for i in range(1, 4)) and c[-1] < c[-2] < c[-3]:
                patterns.append({"name": "Three Black Crows", "bias": "bearish", "strength": 85})

        # Double top/bottom detection (simplified)
        if len(df) >= 20:
            recent_highs = h[-20:]
            recent_lows = l[-20:]
            max_h = np.max(recent_highs)
            min_l = np.min(recent_lows)
            range_pct = (max_h - min_l) / max_h * 100 if max_h > 0 else 0

            # Find peaks close to each other
            peak_indices = []
            for i in range(2, len(recent_highs) - 2):
                if recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i+1]:
                    if recent_highs[i] > recent_highs[i-2] and recent_highs[i] > recent_highs[i+2]:
                        peak_indices.append(i)
            if len(peak_indices) >= 2:
                p1, p2 = recent_highs[peak_indices[-2]], recent_highs[peak_indices[-1]]
                if abs(p1 - p2) / p1 < 0.003:  # Within 0.3%
                    patterns.append({"name": "Double Top", "bias": "bearish", "strength": 80})

            # Find troughs
            trough_indices = []
            for i in range(2, len(recent_lows) - 2):
                if recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i+1]:
                    if recent_lows[i] < recent_lows[i-2] and recent_lows[i] < recent_lows[i+2]:
                        trough_indices.append(i)
            if len(trough_indices) >= 2:
                t1, t2 = recent_lows[trough_indices[-2]], recent_lows[trough_indices[-1]]
                if abs(t1 - t2) / t1 < 0.003:
                    patterns.append({"name": "Double Bottom", "bias": "bullish", "strength": 80})

        return patterns

    def _analyze_market_structure(self, df):
        """Analyze higher highs, higher lows, etc."""
        if len(df) < 10:
            return {"type": "unknown"}

        h = df["high"].values[-20:] if len(df) >= 20 else df["high"].values
        l = df["low"].values[-20:] if len(df) >= 20 else df["low"].values

        # Find swing points
        swing_highs = []
        swing_lows = []
        for i in range(2, len(h) - 2):
            if h[i] > h[i-1] and h[i] > h[i+1] and h[i] > h[i-2] and h[i] > h[i+2]:
                swing_highs.append(h[i])
            if l[i] < l[i-1] and l[i] < l[i+1] and l[i] < l[i-2] and l[i] < l[i+2]:
                swing_lows.append(l[i])

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {"type": "ranging", "swing_highs": swing_highs, "swing_lows": swing_lows}

        hh = swing_highs[-1] > swing_highs[-2]  # Higher high
        hl = swing_lows[-1] > swing_lows[-2]     # Higher low
        lh = swing_highs[-1] < swing_highs[-2]   # Lower high
        ll = swing_lows[-1] < swing_lows[-2]     # Lower low

        if hh and hl:
            structure = "uptrend"
        elif lh and ll:
            structure = "downtrend"
        elif hh and ll:
            structure = "expansion"
        elif lh and hl:
            structure = "compression"
        else:
            structure = "ranging"

        return {
            "type": structure,
            "swing_highs": swing_highs[-3:],
            "swing_lows": swing_lows[-3:],
            "higher_high": hh,
            "higher_low": hl,
            "lower_high": lh,
            "lower_low": ll,
        }

    def _find_support_resistance(self, df, num_levels=5):
        """Find key support and resistance levels."""
        if len(df) < 20:
            return {"support": [], "resistance": []}

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        current = close[-1]

        # Use pivot points from swing highs/lows
        all_pivots = []
        for i in range(2, len(high) - 2):
            if high[i] > high[i-1] and high[i] > high[i+1]:
                all_pivots.append(high[i])
            if low[i] < low[i-1] and low[i] < low[i+1]:
                all_pivots.append(low[i])

        if not all_pivots:
            return {"support": [current * 0.99], "resistance": [current * 1.01]}

        # Cluster nearby levels
        all_pivots.sort()
        clusters = []
        cluster = [all_pivots[0]]
        threshold = current * 0.002  # 0.2% clustering

        for p in all_pivots[1:]:
            if p - cluster[-1] < threshold:
                cluster.append(p)
            else:
                clusters.append(np.mean(cluster))
                cluster = [p]
        clusters.append(np.mean(cluster))

        support = sorted([c for c in clusters if c < current], reverse=True)[:num_levels]
        resistance = sorted([c for c in clusters if c > current])[:num_levels]

        return {
            "support": [round(s, 2) for s in support],
            "resistance": [round(r, 2) for r in resistance],
        }

    def _determine_trend(self, df, indicators):
        """Determine the current trend direction and strength."""
        scores = []
        close = df["close"].iloc[-1]

        # EMA alignment
        emas = []
        for p in [9, 20, 50, 100, 200]:
            key = f"ema_{p}"
            if key in indicators:
                emas.append(indicators[key])

        if len(emas) >= 3:
            if close > emas[0] > emas[1] > emas[2]:
                scores.append(90)  # Strong uptrend
            elif close < emas[0] < emas[1] < emas[2]:
                scores.append(10)  # Strong downtrend
            elif close > emas[1]:
                scores.append(65)
            else:
                scores.append(35)

        # MACD
        if "macd_histogram" in indicators:
            hist = indicators["macd_histogram"]
            if hist > 0:
                scores.append(70)
            else:
                scores.append(30)

        # ADX direction
        if "adx" in indicators and "adx_pos" in indicators and "adx_neg" in indicators:
            if indicators["adx"] > 25:
                if indicators["adx_pos"] > indicators["adx_neg"]:
                    scores.append(80)
                else:
                    scores.append(20)
            else:
                scores.append(50)

        avg_score = np.mean(scores) if scores else 50

        if avg_score > 70:
            direction = "bullish"
        elif avg_score < 30:
            direction = "bearish"
        else:
            direction = "neutral"

        return {
            "direction": direction,
            "score": round(avg_score, 1),
            "strength": "strong" if abs(avg_score - 50) > 30 else "moderate" if abs(avg_score - 50) > 15 else "weak",
        }

    def _assess_momentum(self, indicators):
        """Assess current momentum conditions."""
        signals = []

        if "rsi_14" in indicators:
            rsi = indicators["rsi_14"]
            if rsi > 70:
                signals.append({"indicator": "RSI", "state": "overbought", "value": round(rsi, 1)})
            elif rsi < 30:
                signals.append({"indicator": "RSI", "state": "oversold", "value": round(rsi, 1)})
            else:
                signals.append({"indicator": "RSI", "state": "neutral", "value": round(rsi, 1)})

        if "stoch_k" in indicators:
            k = indicators["stoch_k"]
            if k > 80:
                signals.append({"indicator": "Stoch", "state": "overbought", "value": round(k, 1)})
            elif k < 20:
                signals.append({"indicator": "Stoch", "state": "oversold", "value": round(k, 1)})

        if "mfi" in indicators:
            mfi = indicators["mfi"]
            if mfi > 80:
                signals.append({"indicator": "MFI", "state": "overbought", "value": round(mfi, 1)})
            elif mfi < 20:
                signals.append({"indicator": "MFI", "state": "oversold", "value": round(mfi, 1)})

        return signals

    def _assess_volatility(self, df, indicators):
        """Assess current volatility conditions."""
        result = {}

        if "atr_14" in indicators:
            atr = indicators["atr_14"]
            close = df["close"].iloc[-1]
            atr_pct = (atr / close) * 100
            result["atr"] = round(atr, 2)
            result["atr_pct"] = round(atr_pct, 3)
            if atr_pct > 2:
                result["state"] = "high"
            elif atr_pct > 1:
                result["state"] = "normal"
            else:
                result["state"] = "low"

        if "bb_width" in indicators:
            result["bb_width"] = round(indicators["bb_width"], 4)
            if indicators["bb_width"] < 0.02:
                result["squeeze"] = True
            else:
                result["squeeze"] = False

        return result
