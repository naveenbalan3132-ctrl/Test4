# Nifty50 Options — IV & Buy/Sell Signals Streamlit App
# File: nifty50_options_iv_app.py
# Single-file Streamlit app. Push to GitHub and deploy on https://streamlit.io/
# Dependencies: streamlit, requests, pandas, numpy, scipy, yfinance

import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.optimize import brentq
import yfinance as yf

st.set_page_config(page_title="Nifty50 Options — IV & Signals", layout="wide")

# ------------------------- Utility: Black-Scholes (for index options) -------------------------
def bs_price(call_put_flag, S, K, T, r, sigma):
    # call_put_flag: 'c' or 'p'
    if T <= 0:
        # option expired
        if call_put_flag == 'c':
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if call_put_flag == 'c':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price


def implied_vol(option_price, call_put_flag, S, K, T, r):
    # Solve for sigma using Brent's method
    if option_price <= 0:
        return 0.0
    try:
        f = lambda sigma: bs_price(call_put_flag, S, K, T, r, sigma) - option_price
        impv = brentq(f, 1e-6, 5.0, maxiter=200)
        return impv
    except Exception:
        return np.nan

# ------------------------- Fetch option chain from NSE (public API) -------------------------

def fetch_nse_option_chain(symbol='NIFTY'):
    # Use the NSE option chain endpoint
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    session = requests.Session()
    # NSE expects desktop user-agent and cookies; we'll set reasonable headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36',
        'Accept-Language': 'en,hi;q=0.9',
        'Accept': 'application/json, text/plain, */*'
    }
    try:
        # initial request to get cookies
        session.get('https://www.nseindia.com', headers=headers, timeout=10)
        resp = session.get(url, headers=headers, timeout=10)
        data = resp.json()
        return data
    except Exception as e:
        st.error(f"Failed to fetch option chain from NSE: {e}")
        return None

# ------------------------- Compute IVs and present option chain as DataFrame -------------------------

def parse_option_chain(data):
    records = []
    if not data:
        return pd.DataFrame()
    underlying = data.get('records', {}).get('underlyingValue')
    expiry_dates = sorted(list({item['expiryDate'] for item in data.get('records', {}).get('data', [])}))
    all_rows = data.get('records', {}).get('data', [])
    r = 0.06  # assume 6% risk-free annual
    today = datetime.now()
    for row in all_rows:
        # CE & PE might be present
        expiry = row.get('expiryDate')
        strike = float(row.get('strikePrice'))
        # CE data
        ce = row.get('CE')
        if ce:
            option_price = ce.get('lastPrice')
            T = (datetime.strptime(expiry, '%d-%b-%Y') - today).days / 365.0
            iv = implied_vol(option_price, 'c', underlying, strike, max(T, 1e-6), r)
            records.append({
                'expiry': expiry, 'strike': strike, 'type': 'CE', 'lastPrice': option_price,
                'iv_calc': iv, 'openInterest': ce.get('openInterest'), 'changeinOI': ce.get('changeinOpenInterest')
            })
        pe = row.get('PE')
        if pe:
            option_price = pe.get('lastPrice')
            T = (datetime.strptime(expiry, '%d-%b-%Y') - today).days / 365.0
            iv = implied_vol(option_price, 'p', underlying, strike, max(T, 1e-6), r)
            records.append({
                'expiry': expiry, 'strike': strike, 'type': 'PE', 'lastPrice': option_price,
                'iv_calc': iv, 'openInterest': pe.get('openInterest'), 'changeinOI': pe.get('changeinOpenInterest')
            })
    df = pd.DataFrame(records)
    if not df.empty:
        df.sort_values(['expiry', 'strike', 'type'], inplace=True)
    return df, underlying, expiry_dates

# ------------------------- IV Rank calculation -------------------------

def iv_rank(iv_series, window=60):
    # IV Rank = (current_iv - min_iv) / (max_iv - min_iv)
    if len(iv_series.dropna()) < 10:
        return np.nan
    min_iv = iv_series.min()
    max_iv = iv_series.max()
    cur = iv_series.iloc[-1]
    if max_iv - min_iv == 0:
        return 0.0
    return (cur - min_iv) / (max_iv - min_iv) * 100

# ------------------------- Simple buy/sell rules -------------------------

def generate_signal(df_underlying_hist, iv_rank_pct, sma_short=9, sma_long=21):
    # Use SMA crossover for directional bias
    close = df_underlying_hist['Close']
    sma_s = close.rolling(sma_short).mean()
    sma_l = close.rolling(sma_long).mean()
    if len(close) < sma_long:
        trend = 'neutral'
    else:
        trend = 'bull' if sma_s.iloc[-1] > sma_l.iloc[-1] else 'bear'
    # IV rules
    signal = 'HOLD'
    reason = []
    if iv_rank_pct < 30 and trend == 'bull':
        signal = 'BUY_CALL'
        reason.append('Low IV (good to buy options) + bullish trend')
    elif iv_rank_pct < 30 and trend == 'bear':
        signal = 'BUY_PUT'
        reason.append('Low IV + bearish trend')
    elif iv_rank_pct > 70:
        signal = 'SELL_PREMIUM'
        reason.append('High IV (consider selling premium/credit spreads)')
    else:
        reason.append('No clear setup based on IV and SMA')
    return signal, reason, trend

# ------------------------- Streamlit UI -------------------------

st.title('Nifty50 Options — Implied Vol & Buy/Sell Signals')

with st.sidebar:
    st.header('Settings')
    symbol = st.text_input('Index symbol', value='NIFTY')
    fetch_button = st.button('Fetch latest option chain')
    r_input = st.number_input('Assumed annual risk-free rate (%)', value=6.0, step=0.1)

if 'data' not in st.session_state:
    st.session_state['data'] = None

if fetch_button:
    st.session_state['data'] = fetch_nse_option_chain(symbol)

if st.session_state['data'] is None:
    st.info('Click "Fetch latest option chain" to load data from NSE (requires internet).')
    st.stop()

raw_data = st.session_state['data']

# parse
with st.spinner('Parsing option chain and computing implied vols...'):
    df_chain, underlying_value, expiry_dates = parse_option_chain(raw_data)

if df_chain.empty:
    st.error('No option data available.')
    st.stop()

col1, col2 = st.columns([2, 1])

with col2:
    st.metric('Underlying (Nifty) value', f"{underlying_value:.2f}")
    st.write('Available expiries')
    expiry = st.selectbox('Choose expiry', expiry_dates)
    st.write('Sampled as of:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# show option chain for selected expiry
df_exp = df_chain[df_chain['expiry'] == expiry].copy()
# pivot for display
pivot = df_exp.pivot_table(index='strike', columns='type', values=['lastPrice', 'iv_calc', 'openInterest'])

with col1:
    st.subheader('Option chain (selected expiry)')
    st.dataframe(pivot.fillna('-').head(200))

# compute IV time series approximation: we'll approximate IV history by computing IV for ATM over past 60 days using historical option prices is hard
# Instead compute IV of ATM today and get historical underlying to compute implied vol percentile using historical realized vol as proxy

# Fetch underlying historical from yfinance
with st.spinner('Fetching underlying history for IV rank and signals...'):
    ticker_map = {'NIFTY': '^NSEI'}
    ticker = ticker_map.get(symbol.upper(), '^NSEI')
    hist = yf.download(ticker, period='120d', interval='1d', progress=False)

if hist.empty:
    st.error('Failed to fetch underlying history via yfinance. Signals unavailable.')
    st.stop()

# compute simple realized vol (annualized) as proxy for IV history
hist['returns'] = hist['Close'].pct_change()
window = 30
hist['rv_30'] = hist['returns'].rolling(window).std() * np.sqrt(252)
# Build a synthetic IV series by taking today's ATM option implied vol across strike range median
atm_strike = df_exp['strike'].iloc[(df_exp['strike'] - underlying_value).abs().argsort()[:1]].values[0]
atm_row_call = df_exp[(df_exp['strike'] == atm_strike) & (df_exp['type'] == 'CE')]
atm_row_put = df_exp[(df_exp['strike'] == atm_strike) & (df_exp['type'] == 'PE')]
# pick available
atm_iv = np.nan
if not atm_row_call.empty:
    atm_iv = atm_row_call['iv_calc'].values[0]
elif not atm_row_put.empty:
    atm_iv = atm_row_put['iv_calc'].values[0]

# combine realized vol history with current ATM IV to compute a simple IV rank
# create a synthetic series: use realized vol history and append current iv
iv_series = hist['rv_30'].dropna().copy()
if not np.isnan(atm_iv):
    iv_series = iv_series.append(pd.Series([atm_iv], index=[hist.index[-1] + pd.Timedelta(days=1)]))
iv_rank_pct = iv_rank(iv_series)

# compute signal
signal, reasons, trend = generate_signal(hist, iv_rank_pct if not np.isnan(iv_rank_pct) else 50)

st.markdown('## Signals & IV Summary')
col_a, col_b, col_c = st.columns(3)
col_a.metric('ATM implied vol (today)', f"{atm_iv:.2%}" if not np.isnan(atm_iv) else 'N/A')
col_b.metric('IV Rank (approx)', f"{iv_rank_pct:.1f}%" if not np.isnan(iv_rank_pct) else 'N/A')
col_c.metric('Trend (SMA)', trend)

st.markdown('### Trading Signal')
if signal == 'BUY_CALL':
    st.success('BUY CALL')
elif signal == 'BUY_PUT':
    st.success('BUY PUT')
elif signal == 'SELL_PREMIUM':
    st.warning('SELL PREMIUM / CONSIDER CREDIT SPREADS')
else:
    st.info('HOLD — No clear trade signal')

st.write('Reasons:')
for r in reasons:
    st.write('- ' + r)

# Allow user to pick a strike and show Greeks-like info and IV
st.markdown('## Analyze specific strike')
strike_sel = st.number_input('Strike to analyze', value=int(atm_strike), step=50)
option_type = st.selectbox('Option type', ['CE', 'PE'])
row = df_exp[(df_exp['strike'] == float(strike_sel)) & (df_exp['type'] == option_type)]
if row.empty:
    st.warning('Strike/type not available for selected expiry.')
else:
    row = row.iloc[0]
    st.write('Last Price:', row['lastPrice'])
    st.write('Computed IV:', row['iv_calc'])
    st.write('Open Interest:', row['openInterest'])

st.markdown('---')
st.caption('This app is a demonstration. IV calculations use basic Black-Scholes and the NSE public option chain. Always validate live trading data and consider commissions, margin and execution risk.')

# End of file
