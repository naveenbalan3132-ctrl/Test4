# Intraday Stock Screener (India) — Streamlit App
# File: intraday_screener_app.py
# Purpose: Screen suitable Indian stocks for intraday trading and provide buy/sell indications.
# Usage: Push this file to your GitHub repo and deploy on Streamlit Cloud (https://streamlit.io)

# Dependencies: streamlit, pandas, numpy, requests

# Plotly optional (Streamlit Cloud may not have it installed)
try:
    import plotly.graph_objects as go
except Exception:
    go = None

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="Intraday Screener (India)", layout="wide")

# -------------------------- Utilities: Indicators --------------------------

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

def atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# -------------------------- Data sources --------------------------

# Basic NIFTY50 constituents (hardcoded for reliability). You can replace/update with live list.
NIFTY50 = [
    'RELIANCE','TCS','HDFCBANK','HINDUNILVR','INFY','ICICIBANK','HDFC','KOTAKBANK','SBIN','BHARTIARTL',
    'ITC','LT','AXISBANK','MARUTI','ASIANPAINT','HCLTECH','BAJFINANCE','BAJAJ-AUTO','SUNPHARMA','ULTRACEMCO',
    'TITAN','NTPC','POWERGRID','DIVISLAB','DRREDDY','ONGC','BPCL','GRASIM','HINDALCO','EICHERMOT',
    'TECHM','TATASTEEL','COALINDIA','BPCL','WIPRO','ADANIENT','ADANIPORTS','INDUSINDBK','BRITANNIA','JSWSTEEL',
    'HEROMOTOCO','M&M','TATASTEEL','SHREECEM','CIPLA','GAIL','TATAMOTORS','SBILIFE','HDFCLIFE','VEDL'
]
# de-duplicate and upper-case
NIFTY50 = sorted(list({s.upper() for s in NIFTY50}))

# AlphaVantage intraday endpoint (free tier has rate limits). If you have an API key, use that.
ALPHA_URL = 'https://www.alphavantage.co/query'

# NSE quote endpoint (experimental; may be blocked by NSE when automated). Use responsibly.
NSE_QUOTE_URL = 'https://www.nseindia.com/api/quote-equity?symbol={}'

# -------------------------- Core: fetch intraday candles --------------------------

def fetch_intraday_alpha(symbol, api_key, interval='5min'):
    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': symbol,
        'interval': interval,
        'outputsize': 'compact',
        'apikey': api_key
    }
    r = requests.get(ALPHA_URL, params=params, timeout=15)
    data = r.json()
    key = f"Time Series ({interval})"
    if key not in data:
        return None
    ts = data[key]
    df = pd.DataFrame.from_dict(ts, orient='index')
    df = df.rename(columns=lambda s: s.split(' ')[1])
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.astype(float)
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    return df

def fetch_intraday_nse(symbol):
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    try:
        session.get('https://www.nseindia.com', headers=headers, timeout=10)
        resp = session.get(NSE_QUOTE_URL.format(symbol), headers=headers, timeout=10)
        data = resp.json()
        # NSE API returns lastPrice and other fields; constructing candles may be limited. We'll try to extract 1-min OHLC if available under 'tradedSeries' or 'priceInfo'
        price_info = data.get('priceInfo', {})
        # If no intraday OHLC available, return None
        return None
    except Exception:
        return None

# -------------------------- Scoring & Signals --------------------------

def score_stock(df):
    # df must be intraday candles with index increasing
    out = {'score': 0, 'reasons': []}
    if df is None or df.empty or len(df) < 10:
        out['reasons'].append('Insufficient data')
        return out
    close = df['close']
    vol = df['volume']
    # EMAs
    e9 = ema(close, 9)
    e21 = ema(close, 21)
    # RSI
    r = rsi(close, 14)
    # ATR
    a = atr(df, 14)
    # volume spike: current vol vs avg
    vol_avg = vol[-20:].mean()
    vol_now = vol.iloc[-1]
    # rules
    if e9.iloc[-1] > e21.iloc[-1]:
        out['score'] += 2
        out['reasons'].append('Short-term bullish (EMA9>EMA21)')
    else:
        out['reasons'].append('Short-term bearish or neutral')
    if r.iloc[-1] < 30:
        out['score'] += 1
        out['reasons'].append('RSI indicates oversold (potential long)')
    elif r.iloc[-1] > 70:
        out['score'] -= 1
        out['reasons'].append('RSI indicates overbought (caution long)')
    # volume spike
    if vol_now > vol_avg * 1.8:
        out['score'] += 2
        out['reasons'].append('Strong volume spike')
    # volatility check
    if a.iloc[-1] < (close.iloc[-1] * 0.005):
        out['reasons'].append('Low intraday volatility')
    else:
        out['score'] += 1
        out['reasons'].append('Healthy intraday volatility')
    out['score'] = max(-3, min(10, out['score']))
    return out


def generate_intraday_signal(df):
    """Simple intraday buy/sell indication based on EMA crossover + RSI + volume"""
    if df is None or df.empty or len(df) < 10:
        return 'HOLD', ['Insufficient data']
    close = df['close']
    e9 = ema(close, 9)
    e21 = ema(close, 21)
    r = rsi(close, 14)
    vol = df['volume']
    vol_avg = vol[-20:].mean()
    reasons = []
    if e9.iloc[-1] > e21.iloc[-1] and r.iloc[-1] < 70 and vol.iloc[-1] > vol_avg*1.2:
        reasons.append('Bullish EMA crossover + decent volume + RSI not overbought')
        return 'BUY', reasons
    if e9.iloc[-1] < e21.iloc[-1] and r.iloc[-1] > 30 and vol.iloc[-1] > vol_avg*1.2:
        reasons.append('Bearish EMA crossover + volume spike')
        return 'SELL', reasons
    reasons.append('No clear intraday setup')
    return 'HOLD', reasons

# -------------------------- Streamlit UI --------------------------

st.title('📊 Intraday Screener — Indian Stocks')

with st.sidebar:
    st.header('Data Source & Settings')
    data_source = st.selectbox('Data source', ['AlphaVantage (recommended, needs API key)', 'NSE (experimental)'])
    av_key = st.text_input('AlphaVantage API Key (leave blank to skip)', type='password')
    interval = st.selectbox('Candle interval', ['1min', '5min', '15min'])
    top_n = st.slider('Top N stocks to screen', 5, 50, 15)
    st.markdown('---')
    st.markdown('**How it works**: For each candidate stock we fetch intraday candles, compute EMA9/EMA21, RSI(14), ATR and a volume spike. A simple rule set assigns a score and a BUY/SELL/HOLD suggestion. This is an educational demo; do not treat as financial advice.')

# Candidate universe selection: default to NIFTY50 but allow custom input
st.subheader('Candidate universe')
use_custom = st.checkbox('Provide custom stock list (comma separated)', value=False)
if use_custom:
    custom_txt = st.text_input('Tickers (e.g., RELIANCE, TCS, INFY)')
    universe = [s.strip().upper() for s in custom_txt.split(',') if s.strip()]
    if not universe:
        st.warning('Enter at least one ticker')
        st.stop()
else:
    universe = NIFTY50

st.write(f'Screening {len(universe)} stocks — showing top {top_n} by score')

# Run screening
results = []

progress = st.progress(0)
for i, sym in enumerate(universe):
    # fetch data
    df = None
    if data_source.startswith('Alpha') and av_key:
        try:
            df = fetch_intraday_alpha(sym + '.NS' if not sym.endswith('.NS') else sym, av_key, interval=interval)
        except Exception:
            df = None
    elif data_source.startswith('NSE'):
        try:
            df = fetch_intraday_nse(sym)
        except Exception:
            df = None
    # perform scoring & signal
    s = score_stock(df)
    sig, reasons = generate_intraday_signal(df)
    results.append({
        'symbol': sym,
        'score': s['score'],
        'sig': sig,
        'reasons': '; '.join(s['reasons'][:3]),
        'last_close': df['close'].iloc[-1] if (df is not None and not df.empty) else np.nan
    })
    progress.progress((i + 1) / len(universe))

res_df = pd.DataFrame(results).sort_values(['score', 'last_close'], ascending=[False, False]).head(top_n)

# Display results
st.subheader('Top candidates')
st.dataframe(res_df.reset_index(drop=True))

# Interactive inspect
st.subheader('Inspect a ticker')
sel = st.selectbox('Choose ticker', res_df['symbol'].tolist())

# fetch latest for selected
selected_df = None
if data_source.startswith('Alpha') and av_key:
    selected_df = fetch_intraday_alpha(sel + '.NS' if not sel.endswith('.NS') else sel, av_key, interval=interval)
elif data_source.startswith('NSE'):
    selected_df = fetch_intraday_nse(sel)

if selected_df is None or selected_df.empty:
    st.warning('No intraday data available for this ticker (try AlphaVantage API key or pick another).')
else:
    st.write('Latest candles (last 50)')
    st.dataframe(selected_df.tail(50))
    # compute indicators and show
    selected_df['EMA9'] = ema(selected_df['close'], 9)
    selected_df['EMA21'] = ema(selected_df['close'], 21)
    selected_df['RSI14'] = rsi(selected_df['close'], 14)
    selected_df['ATR14'] = atr(selected_df)
    st.line_chart(selected_df[['close', 'EMA9', 'EMA21']].tail(200))
    st.line_chart(selected_df['RSI14'].tail(200))
    signal, reasons = generate_intraday_signal(selected_df)
    st.markdown(f'### Signal: **{signal}**')
    st.write('Reasons:')
    for r in reasons:
        st.write('- ' + r)

st.markdown('---')
st.caption('This app is a demo for educational purposes only. Intraday trading is risky and requires risk management. Always paper-trade and validate before using real capital.')
