import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@st.cache_data
def get_data(ticker, period="6mo", interval="1d"):
    data = yf.download(ticker, period=period, interval=interval)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def moving_average_crossover(data):
    data["SMA20"] = data["Close"].rolling(20).mean()
    data["SMA50"] = data["Close"].rolling(50).mean()
    data["MA_Signal"] = 0
    data.loc[
        (data["SMA20"].shift(1) <= data["SMA50"].shift(1))
        & (data["SMA20"] > data["SMA50"]),
        "MA_Signal"
    ] = 1
    data.loc[
        (data["SMA20"].shift(1) >= data["SMA50"].shift(1))
        & (data["SMA20"] < data["SMA50"]),
        "MA_Signal"
    ] = -1
    return data

def compute_rsi(data, window=14):
    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))
    data["RSI_Signal"] = 0
    data.loc[
        (data["RSI"].shift(1) >= 30)
        & (data["RSI"] < 30),
        "RSI_Signal"
    ] = 1
    data.loc[
        (data["RSI"].shift(1) <= 70)
        & (data["RSI"] > 70),
        "RSI_Signal"
    ] = -1
    return data

def bollinger_bands(data):
    data["BB_SMA20"] = data["Close"].rolling(20).mean()
    data["BB_STD20"] = data["Close"].rolling(20).std()
    data["Upper"] = (data["BB_SMA20"] + 2 * data["BB_STD20"])
    data["Lower"] = (data["BB_SMA20"] - 2 * data["BB_STD20"])
    data["BB_Signal"] = 0
    data.loc[
        (data["Close"].shift(1) >= data["Lower"].shift(1))
        & (data["Close"] < data["Lower"]),
        "BB_Signal"
    ] = 1
    data.loc[
        (data["Close"].shift(1) <= data["Upper"].shift(1))
        & (data["Close"] > data["Upper"]),
        "BB_Signal"
    ] = -1
    return data

def macd(data):
    data["EMA12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA26"] = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = (data["EMA12"] - data["EMA26"])
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Signal"] = 0
    data.loc[
        (data["MACD"].shift(1)
         <= data["Signal_Line"].shift(1))
        & (data["MACD"] > data["Signal_Line"]),
        "MACD_Signal"
    ] = 1
    data.loc[
        (data["MACD"].shift(1)
         >= data["Signal_Line"].shift(1))
        & (data["MACD"] < data["Signal_Line"]),
        "MACD_Signal"
    ] = -1
    return data

def add_signal_markers(fig, data, signal_col):
    buys = data[data[signal_col] == 1]
    sells = data[data[signal_col] == -1]
    fig.add_trace(
        go.Scatter(
            x=buys.index,
            y=buys["Close"],
            mode="markers",
            marker=dict(
                color="green",
                size=10,
                symbol="triangle-up"
            ),
            name="Buy"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sells.index,
            y=sells["Close"],
            mode="markers",
            marker=dict(
                color="red",
                size=10,
                symbol="triangle-down"
            ),
            name="Sell"
        )
    )

def create_ma_chart(data):
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["SMA20"],
            name="SMA20"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["SMA50"],
            name="SMA50"
        )
    )
    add_signal_markers(
        fig,
        data,
        "MA_Signal"
    )
    fig.update_layout(
        title="Moving Average Crossover",
        xaxis_rangeslider_visible=False,
        height=500
    )
    return fig

def create_rsi_chart(data):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3]
    )
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"]
        ),
        row=1,
        col=1
    )
    buys = data[data["RSI_Signal"] == 1]
    sells = data[data["RSI_Signal"] == -1]
    fig.add_trace(
        go.Scatter(
            x=buys.index,
            y=buys["Close"],
            mode="markers",
            marker=dict(color="green", size=10),
            name="Buy"
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=sells.index,
            y=sells["Close"],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Sell"
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["RSI"],
            name="RSI"
        ),
        row=2,
        col=1
    )
    fig.add_hline(
        y=70,
        line_dash="dash",
        row=2,
        col=1
    )
    fig.add_hline(
        y=30,
        line_dash="dash",
        row=2,
        col=1
    )
    fig.update_layout(
        title="RSI",
        height=600,
        xaxis_rangeslider_visible=False
    )
    return fig

def create_bb_chart(data):
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"]
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Upper"],
            name="Upper Band"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Lower"],
            name="Lower Band"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["BB_SMA20"],
            name="Middle Band"
        )
    )
    add_signal_markers(
        fig,
        data,
        "BB_Signal"
    )
    fig.update_layout(
        title="Bollinger Bands",
        height=500,
        xaxis_rangeslider_visible=False
    )
    return fig

def create_macd_chart(data):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3]
    )
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"]
        ),
        row=1,
        col=1
    )
    buys = data[data["MACD_Signal"] == 1]
    sells = data[data["MACD_Signal"] == -1]
    fig.add_trace(
        go.Scatter(
            x=buys.index,
            y=buys["Close"],
            mode="markers",
            marker=dict(color="green", size=10),
            name="Buy"
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=sells.index,
            y=sells["Close"],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Sell"
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MACD"],
            name="MACD"
        ),
        row=2,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Signal_Line"],
            name="Signal Line"
        ),
        row=2,
        col=1
    )
    fig.update_layout(
        title="MACD",
        height=600,
        xaxis_rangeslider_visible=False
    )
    return fig

with st.sidebar:
    st.title("Stock Trading Signals")
    ticker = st.text_input("Ticker", "AAPL")
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"])
    interval = st.selectbox("Interval", ["1d", "1h"])
    run = st.button("Run Analysis")

if run:
    data = get_data(ticker, period, interval)
    if data.empty:
        st.error("No data returned.")
        st.stop()
    try:
        info = yf.Ticker(ticker)
        st.title(info.info["shortName"])
    except:
        st.title(ticker)
    ma_data = moving_average_crossover(data.copy())
    rsi_data = compute_rsi(data.copy())
    bb_data = bollinger_bands(data.copy())
    macd_data = macd(data.copy())

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_ma_chart(ma_data), use_container_width=True)
    with col2:
        st.plotly_chart(create_rsi_chart(rsi_data), use_container_width=True)
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(create_bb_chart(bb_data), use_container_width=True)
    with col4:
        st.plotly_chart(create_macd_chart(macd_data), use_container_width=True)