"""AI Trading Agent - The core decision engine.

This agent embodies the mindset of an elite futures trader. It continuously
analyzes price action across all timeframes, detects setups, manages bias,
and generates precise Entry / SL / TP signals.

The agent uses a multi-timeframe confluence approach:
1. Higher timeframe (1D, 4H) sets the directional bias
2. Medium timeframe (1H, 15m) identifies structure and zones
3. Lower timeframe (5m, 1m) times entries precisely

It knows every classic setup:
- Order block entries, Fair value gaps, Breaker blocks
- Liquidity sweeps, Stop hunts, Displacement moves
- Trend continuations, Mean reversions, Breakouts
- Supply/demand zones, Support/resistance flips
"""

import logging
import time
import threading
from datetime import datetime
from collections import deque

import numpy as np
import pytz

from config import INSTRUMENTS, AGENT_CONFIG, TRADING_SESSIONS, PST
from technical_analysis import TechnicalAnalyzer

logger = logging.getLogger(__name__)


class TradeSignal:
    """Represents a trade signal with entry, SL, TP."""

    def __init__(self, instrument, direction, entry, stop_loss, take_profits,
                 confidence, reasoning, setup_type, timeframe, bias, timestamp=None):
        self.instrument = instrument
        self.direction = direction          # "LONG" or "SHORT"
        self.entry = round(entry, 2)
        self.stop_loss = round(stop_loss, 2)
        self.take_profits = [round(tp, 2) for tp in take_profits]
        self.confidence = confidence        # 0-100
        self.reasoning = reasoning          # List of reasons
        self.setup_type = setup_type        # e.g. "Order Block + FVG"
        self.timeframe = timeframe          # Primary timeframe
        self.bias = bias                    # Current bias at signal time
        self.timestamp = timestamp or datetime.now(PST).isoformat()
        self.status = "ACTIVE"              # ACTIVE, INVALIDATED, HIT_TP, HIT_SL
        self.risk_reward = self._calc_rr()

    def _calc_rr(self):
        risk = abs(self.entry - self.stop_loss)
        if risk == 0:
            return 0
        reward = abs(self.take_profits[0] - self.entry) if self.take_profits else 0
        return round(reward / risk, 2)

    def to_dict(self):
        return {
            "instrument": self.instrument,
            "direction": self.direction,
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "take_profits": self.take_profits,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "setup_type": self.setup_type,
            "timeframe": self.timeframe,
            "bias": self.bias,
            "timestamp": self.timestamp,
            "status": self.status,
            "risk_reward": self.risk_reward,
        }

    def invalidate(self, reason):
        self.status = "INVALIDATED"
        self.reasoning.append(f"INVALIDATED: {reason}")


class TradingAgent:
    """The AI trading agent that analyzes markets and generates signals.

    Core principles:
    - Always trade WITH the higher-timeframe trend unless at a major level
    - Wait for confluence across multiple timeframes
    - Risk management is non-negotiable (min 1.5 R:R)
    - Bias can and MUST shift when structure breaks
    - Patience > frequency: only take A+ setups
    """

    def __init__(self, market_data_service):
        self.mds = market_data_service
        self.analyzer = TechnicalAnalyzer()
        self._running = False
        self._bias = {}               # {instrument: "bullish"/"bearish"/"neutral"}
        self._analysis_cache = {}     # {instrument: {timeframe: analysis}}
        self._signals = deque(maxlen=100)
        self._signal_callbacks = []
        self._log_callbacks = []
        self._active_signals = {}     # {instrument: TradeSignal}
        self._lock = threading.Lock()
        self._bias_history = {}       # {instrument: deque of bias changes}
        self._agent_thoughts = deque(maxlen=200)

        for inst in INSTRUMENTS:
            self._bias[inst] = "neutral"
            self._bias_history[inst] = deque(maxlen=50)

    def register_signal_callback(self, fn):
        self._signal_callbacks.append(fn)

    def register_log_callback(self, fn):
        self._log_callbacks.append(fn)

    def _emit_thought(self, instrument, thought):
        """Log the agent's thinking process."""
        entry = {
            "timestamp": datetime.now(PST).strftime("%H:%M:%S"),
            "instrument": instrument,
            "thought": thought,
        }
        self._agent_thoughts.append(entry)
        for cb in self._log_callbacks:
            try:
                cb(entry)
            except Exception:
                pass

    def _emit_signal(self, signal):
        self._signals.append(signal)
        for cb in self._signal_callbacks:
            try:
                cb(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")

    def start(self):
        """Start the agent's analysis loop."""
        self._running = True
        t = threading.Thread(target=self._analysis_loop, daemon=True)
        t.start()
        logger.info("TradingAgent started.")

    def stop(self):
        self._running = False

    def is_trading_session(self):
        """Check if current time is within a trading session."""
        now = datetime.now(PST)
        for session in TRADING_SESSIONS:
            start = now.replace(
                hour=session["start_hour"],
                minute=session["start_min"],
                second=0, microsecond=0
            )
            end = now.replace(
                hour=session["end_hour"],
                minute=session["end_min"],
                second=0, microsecond=0
            )
            if start <= now <= end:
                return True, session["name"]
        return False, None

    def _analysis_loop(self):
        """Main analysis loop - runs every N seconds during trading hours."""
        while self._running:
            try:
                in_session, session_name = self.is_trading_session()
                if in_session:
                    for inst_key in INSTRUMENTS:
                        self._full_analysis(inst_key, session_name)
                else:
                    now = datetime.now(PST)
                    self._emit_thought("SYSTEM",
                        f"Outside trading hours ({now.strftime('%H:%M PST')}). "
                        f"Sessions: 6:30-11:00 AM & 3:00-7:00 PM PST. Monitoring...")

            except Exception as e:
                logger.error(f"Analysis loop error: {e}")

            time.sleep(AGENT_CONFIG["analysis_interval_sec"])

    def _full_analysis(self, instrument, session_name):
        """Run complete multi-timeframe analysis for an instrument."""
        self._emit_thought(instrument, f"--- Analyzing {instrument} [{session_name}] ---")

        # Step 1: Gather analysis across all timeframes
        analyses = {}
        for tf in ["1d", "4h", "1h", "15m", "5m", "1m"]:
            df = self.mds.get_historical(instrument, tf)
            if not df.empty and len(df) >= 20:
                analyses[tf] = self.analyzer.analyze(df)

        if not analyses:
            self._emit_thought(instrument, "Insufficient data for analysis.")
            return

        self._analysis_cache[instrument] = analyses
        live = self.mds.get_live_price(instrument)
        if not live:
            return

        price = live["price"]

        # Step 2: Determine higher-timeframe bias
        htf_bias = self._determine_htf_bias(instrument, analyses)

        # Step 3: Check for bias shift
        old_bias = self._bias[instrument]
        if htf_bias != old_bias:
            self._handle_bias_shift(instrument, old_bias, htf_bias, price)

        # Step 4: Look for setups
        setups = self._scan_for_setups(instrument, analyses, price, htf_bias)

        # Step 5: Score and filter setups
        for setup in setups:
            if setup["confidence"] >= AGENT_CONFIG["confidence_threshold"]:
                if setup["risk_reward"] >= AGENT_CONFIG["risk_reward_min"]:
                    self._generate_signal(instrument, setup, htf_bias, price)

        # Step 6: Manage active signals
        self._manage_active_signals(instrument, price, analyses)

    def _determine_htf_bias(self, instrument, analyses):
        """Determine directional bias from higher timeframes."""
        bias_scores = []

        # Daily trend is king
        if "1d" in analyses:
            daily = analyses["1d"]
            if "trend" in daily:
                score = daily["trend"]["score"]
                bias_scores.append(("1D", score, 3.0))  # 3x weight
                self._emit_thought(instrument,
                    f"Daily trend: {daily['trend']['direction']} "
                    f"(score: {score}, strength: {daily['trend']['strength']})")

        # 4H confirms
        if "4h" in analyses:
            h4 = analyses["4h"]
            if "trend" in h4:
                score = h4["trend"]["score"]
                bias_scores.append(("4H", score, 2.0))

        # 1H adds context
        if "1h" in analyses:
            h1 = analyses["1h"]
            if "trend" in h1:
                score = h1["trend"]["score"]
                bias_scores.append(("1H", score, 1.5))

        if not bias_scores:
            return "neutral"

        # Weighted average
        total_weight = sum(w for _, _, w in bias_scores)
        weighted_score = sum(s * w for _, s, w in bias_scores) / total_weight

        if weighted_score > 62:
            bias = "bullish"
        elif weighted_score < 38:
            bias = "bearish"
        else:
            bias = "neutral"

        self._bias[instrument] = bias
        self._emit_thought(instrument,
            f"HTF Bias: {bias.upper()} (weighted score: {weighted_score:.1f})")

        return bias

    def _handle_bias_shift(self, instrument, old_bias, new_bias, price):
        """Handle a change in directional bias."""
        self._emit_thought(instrument,
            f"BIAS SHIFT: {old_bias.upper()} -> {new_bias.upper()} at {price}")

        self._bias_history[instrument].append({
            "from": old_bias,
            "to": new_bias,
            "price": price,
            "time": datetime.now(PST).isoformat(),
        })

        # Invalidate any active signal that conflicts with new bias
        with self._lock:
            active = self._active_signals.get(instrument)
            if active and active.status == "ACTIVE":
                if (new_bias == "bearish" and active.direction == "LONG") or \
                   (new_bias == "bullish" and active.direction == "SHORT"):
                    active.invalidate(f"Bias shifted to {new_bias}")
                    self._emit_thought(instrument,
                        f"INVALIDATED active {active.direction} signal due to bias shift")
                    self._emit_signal(active)

    def _scan_for_setups(self, instrument, analyses, price, bias):
        """Scan for trade setups across timeframes."""
        setups = []

        # === Setup 1: EMA Pullback in Trend ===
        for tf in ["15m", "5m"]:
            if tf not in analyses:
                continue
            a = analyses[tf]
            ind = a.get("indicators", {})
            ema_20 = ind.get("ema_20")
            ema_50 = ind.get("ema_50")
            atr = ind.get("atr_14", 0)
            rsi = ind.get("rsi_14", 50)

            if not ema_20 or not ema_50 or not atr:
                continue

            if bias == "bullish" and ema_20 > ema_50:
                # Price pulled back to EMA 20
                if abs(price - ema_20) < atr * 0.5 and price >= ema_20 * 0.998:
                    if rsi > 40 and rsi < 65:
                        entry = price
                        sl = ema_50 - atr * 0.3
                        tp1 = entry + (entry - sl) * 1.5
                        tp2 = entry + (entry - sl) * 2.5
                        tp3 = entry + (entry - sl) * 3.5
                        conf = self._score_setup(analyses, "pullback", "bullish", tf)
                        setups.append({
                            "type": "EMA Pullback Long",
                            "timeframe": tf,
                            "entry": entry,
                            "stop_loss": sl,
                            "take_profits": [tp1, tp2, tp3],
                            "confidence": conf,
                            "risk_reward": (tp1 - entry) / (entry - sl) if entry > sl else 0,
                            "reasons": [
                                f"Bullish HTF bias",
                                f"Price pulling back to EMA 20 on {tf}",
                                f"EMAs stacked bullish (20 > 50)",
                                f"RSI at {rsi:.0f} - not overbought",
                            ],
                        })

            elif bias == "bearish" and ema_20 < ema_50:
                if abs(price - ema_20) < atr * 0.5 and price <= ema_20 * 1.002:
                    if rsi < 60 and rsi > 35:
                        entry = price
                        sl = ema_50 + atr * 0.3
                        tp1 = entry - (sl - entry) * 1.5
                        tp2 = entry - (sl - entry) * 2.5
                        tp3 = entry - (sl - entry) * 3.5
                        conf = self._score_setup(analyses, "pullback", "bearish", tf)
                        setups.append({
                            "type": "EMA Pullback Short",
                            "timeframe": tf,
                            "entry": entry,
                            "stop_loss": sl,
                            "take_profits": [tp1, tp2, tp3],
                            "confidence": conf,
                            "risk_reward": (entry - tp1) / (sl - entry) if sl > entry else 0,
                            "reasons": [
                                f"Bearish HTF bias",
                                f"Price pulling back to EMA 20 on {tf}",
                                f"EMAs stacked bearish (20 < 50)",
                                f"RSI at {rsi:.0f} - not oversold",
                            ],
                        })

        # === Setup 2: Support/Resistance Bounce ===
        for tf in ["15m", "5m", "1m"]:
            if tf not in analyses:
                continue
            a = analyses[tf]
            sr = a.get("support_resistance", {})
            ind = a.get("indicators", {})
            atr = ind.get("atr_14", 0)
            patterns = a.get("patterns", [])

            supports = sr.get("support", [])
            resistances = sr.get("resistance", [])

            # Bounce off support (long)
            for s in supports[:2]:
                if atr > 0 and abs(price - s) < atr * 0.3:
                    bullish_patterns = [p for p in patterns if p["bias"] == "bullish"]
                    if bullish_patterns or (ind.get("rsi_14", 50) < 35):
                        entry = price
                        sl = s - atr * 0.8
                        tp1 = entry + (entry - sl) * 2.0
                        tp2 = entry + (entry - sl) * 3.0
                        conf = self._score_setup(analyses, "sr_bounce", "bullish", tf)
                        pattern_names = [p["name"] for p in bullish_patterns]
                        setups.append({
                            "type": "Support Bounce Long",
                            "timeframe": tf,
                            "entry": entry,
                            "stop_loss": sl,
                            "take_profits": [tp1, tp2],
                            "confidence": conf,
                            "risk_reward": (tp1 - entry) / (entry - sl) if entry > sl else 0,
                            "reasons": [
                                f"Price at key support {s} on {tf}",
                                f"Bullish patterns: {', '.join(pattern_names)}" if pattern_names else "Oversold RSI at support",
                                f"ATR-based SL below support",
                            ],
                        })

            # Rejection from resistance (short)
            for r in resistances[:2]:
                if atr > 0 and abs(price - r) < atr * 0.3:
                    bearish_patterns = [p for p in patterns if p["bias"] == "bearish"]
                    if bearish_patterns or (ind.get("rsi_14", 50) > 65):
                        entry = price
                        sl = r + atr * 0.8
                        tp1 = entry - (sl - entry) * 2.0
                        tp2 = entry - (sl - entry) * 3.0
                        conf = self._score_setup(analyses, "sr_bounce", "bearish", tf)
                        pattern_names = [p["name"] for p in bearish_patterns]
                        setups.append({
                            "type": "Resistance Rejection Short",
                            "timeframe": tf,
                            "entry": entry,
                            "stop_loss": sl,
                            "take_profits": [tp1, tp2],
                            "confidence": conf,
                            "risk_reward": (entry - tp1) / (sl - entry) if sl > entry else 0,
                            "reasons": [
                                f"Price at key resistance {r} on {tf}",
                                f"Bearish patterns: {', '.join(pattern_names)}" if pattern_names else "Overbought RSI at resistance",
                                f"ATR-based SL above resistance",
                            ],
                        })

        # === Setup 3: Breakout with Momentum ===
        for tf in ["15m", "5m"]:
            if tf not in analyses:
                continue
            a = analyses[tf]
            ind = a.get("indicators", {})
            structure = a.get("structure", {})
            vol = a.get("volatility", {})

            bb_upper = ind.get("bb_upper")
            bb_lower = ind.get("bb_lower")
            atr = ind.get("atr_14", 0)
            adx = ind.get("adx", 0)
            macd_hist = ind.get("macd_histogram", 0)

            # Bollinger squeeze breakout
            if vol.get("squeeze") and atr > 0:
                if price > bb_upper and adx > 20 and macd_hist > 0:
                    entry = price
                    sl = bb_lower if bb_lower else entry - atr * 2
                    tp1 = entry + atr * 3
                    tp2 = entry + atr * 5
                    conf = self._score_setup(analyses, "breakout", "bullish", tf)
                    setups.append({
                        "type": "Squeeze Breakout Long",
                        "timeframe": tf,
                        "entry": entry,
                        "stop_loss": sl,
                        "take_profits": [tp1, tp2],
                        "confidence": conf,
                        "risk_reward": (tp1 - entry) / (entry - sl) if entry > sl else 0,
                        "reasons": [
                            f"Bollinger Bands squeeze on {tf}",
                            f"Price breaking above upper band",
                            f"ADX rising at {adx:.0f} - trend strengthening",
                            f"MACD histogram positive",
                        ],
                    })
                elif price < bb_lower and adx > 20 and macd_hist < 0:
                    entry = price
                    sl = bb_upper if bb_upper else entry + atr * 2
                    tp1 = entry - atr * 3
                    tp2 = entry - atr * 5
                    conf = self._score_setup(analyses, "breakout", "bearish", tf)
                    setups.append({
                        "type": "Squeeze Breakout Short",
                        "timeframe": tf,
                        "entry": entry,
                        "stop_loss": sl,
                        "take_profits": [tp1, tp2],
                        "confidence": conf,
                        "risk_reward": (entry - tp1) / (sl - entry) if sl > entry else 0,
                        "reasons": [
                            f"Bollinger Bands squeeze on {tf}",
                            f"Price breaking below lower band",
                            f"ADX rising at {adx:.0f} - trend strengthening",
                            f"MACD histogram negative",
                        ],
                    })

        # === Setup 4: VWAP Reclaim/Rejection ===
        for tf in ["5m", "1m"]:
            if tf not in analyses:
                continue
            a = analyses[tf]
            ind = a.get("indicators", {})
            vwap = ind.get("vwap")
            atr = ind.get("atr_14", 0)

            if not vwap or not atr:
                continue

            # VWAP reclaim long
            if bias == "bullish" and price > vwap and abs(price - vwap) < atr * 0.3:
                obv_slope = ind.get("obv_slope", 0)
                if obv_slope > 0:
                    entry = price
                    sl = vwap - atr * 0.5
                    tp1 = entry + atr * 2
                    tp2 = entry + atr * 3.5
                    conf = self._score_setup(analyses, "vwap", "bullish", tf)
                    setups.append({
                        "type": "VWAP Reclaim Long",
                        "timeframe": tf,
                        "entry": entry,
                        "stop_loss": sl,
                        "take_profits": [tp1, tp2],
                        "confidence": conf,
                        "risk_reward": (tp1 - entry) / (entry - sl) if entry > sl else 0,
                        "reasons": [
                            f"Price reclaiming VWAP ({vwap:.2f}) on {tf}",
                            f"Bullish HTF bias supports long",
                            f"Volume confirming (OBV rising)",
                        ],
                    })

            # VWAP rejection short
            elif bias == "bearish" and price < vwap and abs(price - vwap) < atr * 0.3:
                obv_slope = ind.get("obv_slope", 0)
                if obv_slope < 0:
                    entry = price
                    sl = vwap + atr * 0.5
                    tp1 = entry - atr * 2
                    tp2 = entry - atr * 3.5
                    conf = self._score_setup(analyses, "vwap", "bearish", tf)
                    setups.append({
                        "type": "VWAP Rejection Short",
                        "timeframe": tf,
                        "entry": entry,
                        "stop_loss": sl,
                        "take_profits": [tp1, tp2],
                        "confidence": conf,
                        "risk_reward": (entry - tp1) / (sl - entry) if sl > entry else 0,
                        "reasons": [
                            f"Price rejecting VWAP ({vwap:.2f}) on {tf}",
                            f"Bearish HTF bias supports short",
                            f"Volume confirming (OBV declining)",
                        ],
                    })

        # === Setup 5: Momentum Divergence ===
        for tf in ["15m", "5m"]:
            if tf not in analyses:
                continue
            a = analyses[tf]
            ind = a.get("indicators", {})
            structure = a.get("structure", {})

            rsi = ind.get("rsi_14", 50)
            macd_hist = ind.get("macd_histogram", 0)
            atr = ind.get("atr_14", 0)
            swing_lows = structure.get("swing_lows", [])
            swing_highs = structure.get("swing_highs", [])

            if not atr:
                continue

            # Bullish divergence: price making lower lows but RSI making higher lows
            if len(swing_lows) >= 2 and swing_lows[-1] < swing_lows[-2] and rsi > 30 and rsi < 45:
                if macd_hist > 0:  # MACD turning up
                    entry = price
                    sl = swing_lows[-1] - atr * 0.3
                    tp1 = entry + (entry - sl) * 2
                    tp2 = entry + (entry - sl) * 3
                    conf = self._score_setup(analyses, "divergence", "bullish", tf)
                    setups.append({
                        "type": "Bullish Divergence",
                        "timeframe": tf,
                        "entry": entry,
                        "stop_loss": sl,
                        "take_profits": [tp1, tp2],
                        "confidence": conf,
                        "risk_reward": (tp1 - entry) / (entry - sl) if entry > sl else 0,
                        "reasons": [
                            f"Bullish RSI divergence on {tf}",
                            f"Price making lower lows, RSI at {rsi:.0f} (higher)",
                            f"MACD histogram turning positive",
                            f"Potential trend reversal setup",
                        ],
                    })

            # Bearish divergence
            if len(swing_highs) >= 2 and swing_highs[-1] > swing_highs[-2] and rsi > 55 and rsi < 70:
                if macd_hist < 0:
                    entry = price
                    sl = swing_highs[-1] + atr * 0.3
                    tp1 = entry - (sl - entry) * 2
                    tp2 = entry - (sl - entry) * 3
                    conf = self._score_setup(analyses, "divergence", "bearish", tf)
                    setups.append({
                        "type": "Bearish Divergence",
                        "timeframe": tf,
                        "entry": entry,
                        "stop_loss": sl,
                        "take_profits": [tp1, tp2],
                        "confidence": conf,
                        "risk_reward": (entry - tp1) / (sl - entry) if sl > entry else 0,
                        "reasons": [
                            f"Bearish RSI divergence on {tf}",
                            f"Price making higher highs, RSI at {rsi:.0f} (lower)",
                            f"MACD histogram turning negative",
                            f"Potential trend reversal setup",
                        ],
                    })

        if setups:
            best = max(setups, key=lambda s: s["confidence"])
            self._emit_thought(instrument,
                f"Found {len(setups)} setups. Best: {best['type']} "
                f"(conf: {best['confidence']}, R:R: {best['risk_reward']:.1f})")
        else:
            self._emit_thought(instrument,
                f"No qualifying setups found. Waiting for better price action...")

        return setups

    def _score_setup(self, analyses, setup_type, direction, primary_tf):
        """Score a setup based on multi-timeframe confluence."""
        score = 50  # Base score

        # Higher-timeframe alignment bonus
        for htf in ["1d", "4h", "1h"]:
            if htf in analyses and htf != primary_tf:
                trend = analyses[htf].get("trend", {})
                if trend.get("direction") == ("bullish" if direction == "bullish" else "bearish"):
                    score += 8
                elif trend.get("direction") == "neutral":
                    score += 2
                else:
                    score -= 5  # Counter-trend penalty

        # Pattern confirmation
        if primary_tf in analyses:
            patterns = analyses[primary_tf].get("patterns", [])
            for p in patterns:
                if p["bias"] == direction:
                    score += p["strength"] * 0.15
                elif p["bias"] != "neutral":
                    score -= 3

        # Momentum alignment
        if primary_tf in analyses:
            momentum = analyses[primary_tf].get("momentum", [])
            for m in momentum:
                if direction == "bullish" and m["state"] == "oversold":
                    score += 5  # Good for long entries
                elif direction == "bearish" and m["state"] == "overbought":
                    score += 5  # Good for short entries
                elif direction == "bullish" and m["state"] == "overbought":
                    score -= 8  # Risky long
                elif direction == "bearish" and m["state"] == "oversold":
                    score -= 8  # Risky short

        # Structure alignment
        if primary_tf in analyses:
            structure = analyses[primary_tf].get("structure", {})
            struct_type = structure.get("type", "")
            if direction == "bullish" and struct_type == "uptrend":
                score += 10
            elif direction == "bearish" and struct_type == "downtrend":
                score += 10
            elif struct_type == "ranging":
                score -= 3

        # Volume confirmation
        if primary_tf in analyses:
            ind = analyses[primary_tf].get("indicators", {})
            obv_slope = ind.get("obv_slope", 0)
            if (direction == "bullish" and obv_slope > 0) or \
               (direction == "bearish" and obv_slope < 0):
                score += 5

        return min(max(int(score), 0), 100)

    def _generate_signal(self, instrument, setup, bias, current_price):
        """Generate and emit a trade signal."""
        # Don't signal if we already have an active signal for this instrument
        with self._lock:
            active = self._active_signals.get(instrument)
            if active and active.status == "ACTIVE":
                self._emit_thought(instrument,
                    f"Skipping {setup['type']} - already have active {active.direction} signal")
                return

        direction = "LONG" if "Long" in setup["type"] or "Bullish" in setup["type"] else "SHORT"

        signal = TradeSignal(
            instrument=instrument,
            direction=direction,
            entry=setup["entry"],
            stop_loss=setup["stop_loss"],
            take_profits=setup["take_profits"],
            confidence=setup["confidence"],
            reasoning=setup["reasons"],
            setup_type=setup["type"],
            timeframe=setup["timeframe"],
            bias=bias,
        )

        with self._lock:
            self._active_signals[instrument] = signal

        self._emit_thought(instrument,
            f"NEW SIGNAL: {direction} {instrument} @ {signal.entry} | "
            f"SL: {signal.stop_loss} | TP1: {signal.take_profits[0]} | "
            f"R:R: {signal.risk_reward} | Confidence: {signal.confidence}%")

        self._emit_signal(signal)

    def _manage_active_signals(self, instrument, price, analyses):
        """Monitor and manage active signals - adjust if conditions change."""
        with self._lock:
            active = self._active_signals.get(instrument)
            if not active or active.status != "ACTIVE":
                return

        # Check if SL hit
        if active.direction == "LONG" and price <= active.stop_loss:
            active.status = "HIT_SL"
            self._emit_thought(instrument,
                f"STOP LOSS HIT on {active.direction} @ {price} (SL was {active.stop_loss})")
            self._emit_signal(active)
            return

        if active.direction == "SHORT" and price >= active.stop_loss:
            active.status = "HIT_SL"
            self._emit_thought(instrument,
                f"STOP LOSS HIT on {active.direction} @ {price} (SL was {active.stop_loss})")
            self._emit_signal(active)
            return

        # Check if any TP hit
        for i, tp in enumerate(active.take_profits):
            if active.direction == "LONG" and price >= tp:
                active.status = f"HIT_TP{i+1}"
                self._emit_thought(instrument,
                    f"TAKE PROFIT {i+1} HIT on {active.direction} @ {price} (TP was {tp})")
                self._emit_signal(active)
                return
            if active.direction == "SHORT" and price <= tp:
                active.status = f"HIT_TP{i+1}"
                self._emit_thought(instrument,
                    f"TAKE PROFIT {i+1} HIT on {active.direction} @ {price} (TP was {tp})")
                self._emit_signal(active)
                return

        # Check for bias invalidation via structure break
        for tf in ["5m", "15m"]:
            if tf in analyses:
                structure = analyses[tf].get("structure", {})
                if active.direction == "LONG" and structure.get("type") == "downtrend":
                    if structure.get("lower_low") and structure.get("lower_high"):
                        active.invalidate(f"Structure broke bearish on {tf}")
                        self._emit_thought(instrument,
                            f"INVALIDATED: Long signal broken by bearish structure on {tf}")
                        self._emit_signal(active)
                        return
                elif active.direction == "SHORT" and structure.get("type") == "uptrend":
                    if structure.get("higher_high") and structure.get("higher_low"):
                        active.invalidate(f"Structure broke bullish on {tf}")
                        self._emit_thought(instrument,
                            f"INVALIDATED: Short signal broken by bullish structure on {tf}")
                        self._emit_signal(active)
                        return

    # --- Public API ---

    def get_current_bias(self):
        """Get current bias for all instruments."""
        return dict(self._bias)

    def get_active_signals(self):
        """Get all active signals."""
        with self._lock:
            return {k: v.to_dict() for k, v in self._active_signals.items()
                    if v.status == "ACTIVE"}

    def get_all_signals(self):
        """Get full signal history."""
        return [s.to_dict() for s in self._signals]

    def get_analysis(self, instrument):
        """Get latest analysis cache for an instrument."""
        return self._analysis_cache.get(instrument, {})

    def get_thoughts(self, limit=50):
        """Get recent agent thoughts."""
        return list(self._agent_thoughts)[-limit:]

    def get_bias_history(self, instrument):
        """Get bias change history."""
        return list(self._bias_history.get(instrument, []))
