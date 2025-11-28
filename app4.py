import streamlit as st
import pandas as pd
import yfinance as yf
import datetime

st.set_page_config(page_title="Intraday Screener — India", layout="wide")

st.title("📈 Intraday Stock Screener — India (Live Data)")
st.write("Live NSE intraday data using yfinance (5-minute interval)")

# ---- STOCK LIST ----
def get_nifty50_list():
    return [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "SBIN.NS", "LT.NS", "KOTAKBANK.NS", "AXISBANK.NS", "ITC.NS",
        "HINDUNILVR.NS", "BHARTIARTL.NS", "HCLTECH.NS", "MARUTI.NS",
        "ASIANPAINT.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS",
        "ONGC.NS", "WIPRO.NS"
    ]

# ---- DATA FETCH ----
def get_intraday(symbol):
    try:
        df = yf.download(symbol, interval="5m", period="1d")
        if df.empty:
            return None
        df.reset_index(inplace=True)
        df.rename(columns={
            "Datetime": "timestamp",
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume"
        }, inplace=True)
        return df
    except:
        return None

# ---- SIGNAL ENGINE ----
def get_signal(df):
    df["SMA5"] = df["close"].rolling(5).mean()
    df["SMA20"] = df["close"].rolling(20).mean()

    if df["SMA5"].iloc[-1] > df["SMA20"].iloc[-1]:
        return "BUY 📗"
    elif df["SMA5"].iloc[-1] < df["SMA20"].iloc[-1]:
        return "SELL 📕"
    else:
        return "NEUTRAL ⚪"

# ---- UI ----
st.sidebar.header("Settings")
selected = st.sidebar.multiselect(
    "Select Stocks", get_nifty50_list(), default=["RELIANCE.NS", "TCS.NS"]
)

if not selected:
    st.warning("Select at least one stock.")
    st.stop()

results = []

for sym in selected:
    df = get_intraday(sym)
    if df is None:
        results.append((sym, "No Data", "-"))
        continue

    signal = get_signal(df)
    last_price = round(df["close"].iloc[-1], 2)

    results.append((sym, last_price, signal))

# ---- DISPLAY ----
st.subheader("Live Intraday Signals (5-minute updates)")
res_df = pd.DataFrame(results, columns=["Symbol", "Last Price", "Signal"])
st.dataframe(res_df, use_container_width=True)

st.caption("Data source: Yahoo Finance (NSE intraday 5m interval)")
