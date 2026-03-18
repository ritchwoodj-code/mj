#!/usr/bin/env python3
"""Entry point for the AI Futures Trading Agent application.

Usage:
    python run.py

The app will start on http://localhost:5000
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if __name__ == "__main__":
    from app import app, socketio, start_services

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║         AI FUTURES TRADING AGENT                     ║
    ║                                                      ║
    ║  Instruments: Gold (GC=F) | NQ (NQ=F)               ║
    ║  Sessions:    6:30-11:00 AM PST                      ║
    ║               3:00-7:00 PM PST                       ║
    ║                                                      ║
    ║  Data Source:  Yahoo Finance (Free)                   ║
    ║  Dashboard:    http://localhost:5000                  ║
    ║                                                      ║
    ║  WARNING: This is for educational purposes only.     ║
    ║  Do NOT trade real money based on these signals.     ║
    ╚══════════════════════════════════════════════════════╝
    """)

    start_services()
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
