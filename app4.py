import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Indian Intraday Stock Scanner", layout="wide")

# ---- NSE LIVE DATA ----
def get_live_stock(symbol):
    url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
    }
    session = requests.Session()
    response = session.get(url, headers=headers)
    data = response.json()
    return data


# ---- Intraday Logic ----
def calculate_signals(df):
    df["SMA5"] = df["lastPrice"].rolling(5).mean()
    df["SMA20"] = df["lastPrice"].rolling(20).mean()

    df["Signal"] = ""
    for i in range(len(df)):
        if df["SMA5"][i] > df["SMA20"][i]:
            df["Signal"][i] = "BUY"
        else:
            df["Signal"][i] = "SELL"
    return df


# ---- UI ----
st.title("📈 Indian Stock Market – Intraday Algo Scanner (LIVE)")
st.write("Real-time signals using SMA crossover")

symbol = st.text_input("Enter NSE Stock Symbol (Example: TCS, INFY, RELIANCE)", "TCS")

if st.button("Fetch Live Data"):
    try:
        raw = get_live_stock(symbol.upper())

        prices = raw["priceInfo"]
        last_price = prices["lastPrice"]
        open_price = prices["open"]

        st.metric("Live Price", last_price)
        st.metric("Open Price", open_price)

        hist = raw["preOpenMarket"]["preopen"]
        df = pd.DataFrame(hist)
        df.rename(columns={"price": "lastPrice"}, inplace=True)

        df = calculate_signals(df)

        st.subheader("Intraday Buy/Sell Signal")
        st.dataframe(df[["lastPrice", "SMA5", "SMA20", "Signal"]])

        # ---- Chart ----
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df["lastPrice"], mode="lines", name="Price"))
        fig.add_trace(go.Scatter(y=df["SMA5"], mode="lines", name="SMA5"))
        fig.add_trace(go.Scatter(y=df["SMA20"], mode="lines", name="SMA20"))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error("Error fetching data. NSE may have blocked too many requests.")
        st.write(e)
