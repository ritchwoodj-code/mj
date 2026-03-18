"""Configuration for the AI Trading Agent application."""

import pytz

# Trading instruments
INSTRUMENTS = {
    "GOLD": {
        "symbol": "GC=F",          # Gold Futures (CME)
        "display_name": "Gold Futures",
        "tick_size": 0.10,
        "point_value": 100,        # $100 per point
    },
    "NQ": {
        "symbol": "NQ=F",          # Nasdaq 100 E-mini Futures
        "display_name": "NQ Futures",
        "tick_size": 0.25,
        "point_value": 20,         # $20 per point
    },
}

# Timeframes for analysis
TIMEFRAMES = {
    "1m": {"period": "1d", "interval": "1m"},
    "5m": {"period": "5d", "interval": "5m"},
    "15m": {"period": "1mo", "interval": "15m"},
    "30m": {"period": "1mo", "interval": "30m"},
    "1h": {"period": "3mo", "interval": "1h"},
    "4h": {"period": "6mo", "interval": "1h"},   # aggregate from 1h
    "1d": {"period": "2y", "interval": "1d"},
    "1w": {"period": "5y", "interval": "1wk"},
}

# Trading session windows (PST)
PST = pytz.timezone("America/Los_Angeles")

TRADING_SESSIONS = [
    {"name": "Morning Session", "start_hour": 6, "start_min": 30, "end_hour": 11, "end_min": 0},
    {"name": "Afternoon Session", "start_hour": 15, "start_min": 0, "end_hour": 19, "end_min": 0},
]

# Agent configuration
AGENT_CONFIG = {
    "risk_reward_min": 1.5,        # Minimum R:R ratio
    "max_risk_pct": 1.0,           # Max risk per trade as % of account
    "confidence_threshold": 70,     # Minimum confidence score to signal
    "bias_shift_threshold": 60,     # Score below which bias flips
    "lookback_candles": 200,        # Candles to analyze for patterns
    "update_interval_sec": 1,       # How often to refresh data
    "analysis_interval_sec": 5,     # How often agent re-analyzes
}
