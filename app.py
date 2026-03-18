"""Flask application - serves the trading dashboard and WebSocket updates."""

import logging
import json
from datetime import datetime

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

from config import INSTRUMENTS, TIMEFRAMES, TRADING_SESSIONS, PST, AGENT_CONFIG
from market_data import MarketDataService
from trading_agent import TradingAgent

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = "trading-agent-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Initialize services
mds = MarketDataService()
agent = TradingAgent(mds)


# --- WebSocket Callbacks ---

def on_price_update(instrument, price_data):
    """Push live price to all connected clients."""
    socketio.emit("price_update", price_data)


def on_signal(signal):
    """Push trade signal to all connected clients."""
    socketio.emit("trade_signal", signal.to_dict() if hasattr(signal, "to_dict") else signal)


def on_agent_thought(thought):
    """Push agent thinking to all connected clients."""
    socketio.emit("agent_thought", thought)


mds.register_callback(on_price_update)
agent.register_signal_callback(on_signal)
agent.register_log_callback(on_agent_thought)


# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/instruments")
def api_instruments():
    return jsonify(INSTRUMENTS)


@app.route("/api/prices")
def api_prices():
    return jsonify(mds.get_all_live_prices())


@app.route("/api/historical/<instrument>/<timeframe>")
def api_historical(instrument, timeframe):
    df = mds.get_historical(instrument, timeframe)
    if df.empty:
        return jsonify([])
    records = []
    for idx, row in df.iterrows():
        records.append({
            "time": int(idx.timestamp()) if hasattr(idx, "timestamp") else str(idx),
            "open": round(row["open"], 2),
            "high": round(row["high"], 2),
            "low": round(row["low"], 2),
            "close": round(row["close"], 2),
            "volume": int(row["volume"]),
        })
    return jsonify(records)


@app.route("/api/analysis/<instrument>")
def api_analysis(instrument):
    analysis = agent.get_analysis(instrument)
    # Convert non-serializable types
    result = {}
    for tf, data in analysis.items():
        tf_result = {}
        for key, val in data.items():
            if isinstance(val, dict):
                tf_result[key] = {k: _serialize(v) for k, v in val.items()}
            elif isinstance(val, list):
                tf_result[key] = [_serialize(v) for v in val]
            else:
                tf_result[key] = _serialize(val)
        result[tf] = tf_result
    return jsonify(result)


@app.route("/api/signals")
def api_signals():
    return jsonify(agent.get_all_signals())


@app.route("/api/active-signals")
def api_active_signals():
    return jsonify(agent.get_active_signals())


@app.route("/api/bias")
def api_bias():
    return jsonify(agent.get_current_bias())


@app.route("/api/thoughts")
def api_thoughts():
    return jsonify(agent.get_thoughts(100))


@app.route("/api/session-status")
def api_session_status():
    in_session, name = agent.is_trading_session()
    now = datetime.now(PST)
    return jsonify({
        "in_session": in_session,
        "session_name": name,
        "current_time_pst": now.strftime("%H:%M:%S PST"),
        "sessions": TRADING_SESSIONS,
    })


def _serialize(val):
    """Make values JSON-serializable."""
    if hasattr(val, "item"):  # numpy scalar
        return val.item()
    if isinstance(val, (list, tuple)):
        return [_serialize(v) for v in val]
    if isinstance(val, dict):
        return {k: _serialize(v) for k, v in val.items()}
    return val


# --- Socket Events ---

@socketio.on("connect")
def handle_connect():
    logger.info("Client connected")
    # Send initial data
    socketio.emit("initial_data", {
        "prices": mds.get_all_live_prices(),
        "bias": agent.get_current_bias(),
        "active_signals": agent.get_active_signals(),
        "session": {
            "in_session": agent.is_trading_session()[0],
            "session_name": agent.is_trading_session()[1],
        },
    })


@socketio.on("disconnect")
def handle_disconnect():
    logger.info("Client disconnected")


# --- Start Services ---

def start_services():
    """Initialize and start market data and agent services."""
    logger.info("Starting market data service...")
    mds.start()
    logger.info("Starting trading agent...")
    agent.start()


if __name__ == "__main__":
    start_services()
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
