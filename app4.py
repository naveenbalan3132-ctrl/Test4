# Intraday NSE Live Screener (India)
# Streamlit-ready | For GitHub + Streamlit Cloud
# NOTE: NSE blocks automated scraping. This app uses a workaround header + session.
# If NSE blocks your IP, the data may fail. Use responsibly.

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="NSE Intraday Screener", layout="wide")

# ---------------------- NSE LIVE DATA FETCH ----------------------
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9"
}

session = requests.Session()
session.headers.update(headers)

NSE_CHART_URL = "https://www.nseindia.com/api/chart-databyindex?index={symbol}"

# ---------------------- INDICATORS ----------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=(period - 1), min_periods=period).mean()
    ma_down = down.ewm(com=(period - 1), min_periods=period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

# ---------------------- FETCH CHART CANDLES ----------------------
def get_nse_intraday(symbol):
    try:
        session.get("https://www.nseindia.com", timeout=5)
        url = NSE_CHART_URL.format(symbol=symbol)
        r = session.get(url, timeout=10)
        data = r.json()

        candles = data.get("grapthData", [])
        df = pd.DataFrame(candles, columns=["time", "price"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df = df.set_index("time")
        df["close"] = df["price"]
        df.drop(columns=["price"], inplace=True)
        df["open"] = df["close"].shift(1)
        df["high"] = df["close"].rolling(3).max()
        df["low"] = df["close"].rolling(3).min()
        df["volume"] = 1
        return df.dropna()
    except:
        return None

# ---------------------- BUY / SELL SIGNAL ----------------------
def generate_signal(df):
    if df is None or df.empty:
        return "NO DATA", []

    df["EMA9"] = ema(df["close"], 9)
    df["EMA21"] = ema(df["close"], 21)
    df["RSI14"] = rsi(df["close"], 14)

    reasons = []

    if df["EMA9"].iloc[-1] > df["EMA21"].iloc[-1] and df["RSI14"].iloc[-1] < 70:
        reasons.append("Bullish EMA crossover")
        reasons.append("RSI not overbought")
        return "BUY", reasons

    if df["EMA9"].iloc[-1] < df["EMA21"].iloc[-1] and df["RSI14"].iloc[-1] > 30:
        reasons.append("Bearish EMA crossover")
        reasons.append("RSI not oversold")
        return "SELL", reasons

    return "HOLD", ["No strong signal"]

# ---------------------- STREAMLIT UI ----------------------
st.title("📈 NSE Intraday Screener — Live Data")

stocks = [
    "NIFTY 50", "NIFTY BANK", "RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN", "KOTAKBANK",
    "ICICIBANK", "TATAMOTORS", "LT", "ITC", "MARUTI", "HCLTECH", "WIPRO", "AXISBANK"
]

symbol = st.selectbox("Select Stock / Index", stocks)

st.info("Fetching live candles may take 3–5 seconds…")

df = get_nse_intraday(symbol)

if df is None:
    st.error("❌ Failed to load NSE live data. Try again or change stock.")
    st.stop()

st.subheader("📊 Intraday Chart (Last 200 points)")
st.line_chart(df["close"].tail(200))

signal, reasons = generate_signal(df)

st.subheader(f"Signal: {signal}")
for r in reasons:
    st.write("-", r)

st.subheader("Raw Data (Last 50 rows)")
st.dataframe(df.tail(50))

st.caption("This intraday screener is for educational purposes only. Not financial advice.")
