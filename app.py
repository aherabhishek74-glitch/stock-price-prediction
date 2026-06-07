"""
=============================================================
 Stock Price Prediction — LSTM Model
 File : app.py
 Desc : Streamlit web dashboard (run: streamlit run app.py)
=============================================================
"""

import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")

# ─── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Prediction — LSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
  .metric-box {
    background: #f0f4ff;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
  }
  .metric-box h4 { margin: 0; font-size: 13px; color: #555; }
  .metric-box p  { margin: 4px 0 0; font-size: 24px; font-weight: 700; color: #1a1a2e; }
  .signal-buy  { color: #2e7d32; font-weight: 700; }
  .signal-sell { color: #c62828; font-weight: 700; }
  .signal-hold { color: #e65100; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/3/37/Arc_de_Triomphe_de_l%27Etoile_designed_by_Jean_Chalgrin_1806.jpg",
    width=50,
)  # placeholder — replace with your college logo path
st.sidebar.title("📈 Stock Predictor")
st.sidebar.markdown("**LSTM-based Deep Learning**")
st.sidebar.markdown("---")

TICKER_MAP = {
    "Apple (AAPL)":     "AAPL",
    "Google (GOOGL)":   "GOOGL",
    "Microsoft (MSFT)": "MSFT",
    "Tesla (TSLA)":     "TSLA",
    "Amazon (AMZN)":    "AMZN",
    "Meta (META)":      "META",
    "NVIDIA (NVDA)":    "NVDA",
    "Infosys (INFY)":   "INFY",
    "TCS (TCS.NS)":     "TCS.NS",
    "Reliance (RELIANCE.NS)": "RELIANCE.NS",
}

selected_name = st.sidebar.selectbox("Select Stock", list(TICKER_MAP.keys()))
TICKER        = TICKER_MAP[selected_name]
PERIOD        = st.sidebar.selectbox("Data Period", ["1y", "2y", "3y", "5y"], index=1)
LOOK_BACK     = st.sidebar.slider("Look-back Window (days)", 30, 120, 60, 10)
EPOCHS        = st.sidebar.slider("Training Epochs", 10, 100, 50, 10)
BATCH_SIZE    = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
FORECAST_DAYS = st.sidebar.slider("Forecast Days", 7, 60, 30, 7)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Architecture**")
st.sidebar.code(
    "Input → LSTM(128) → Dropout\n"
    "     → LSTM(64)  → Dropout\n"
    "     → LSTM(32)  → Dropout\n"
    "     → Dense(25) → Dense(1)",
    language="text",
)
st.sidebar.markdown("---")
st.sidebar.caption("Last Year Project · B.E. Computer Engineering")


# ─── Helpers ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, progress=False)
    df.dropna(inplace=True)
    # Technical indicators
    close = df["Close"]
    df["SMA_20"]      = close.rolling(20).mean()
    df["SMA_50"]      = close.rolling(50).mean()
    df["EMA_12"]      = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"]      = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    delta             = close.diff()
    gain              = delta.clip(lower=0).rolling(14).mean()
    loss              = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"]         = 100 - (100 / (1 + gain / loss))
    df["BB_upper"]    = close.rolling(20).mean() + 2 * close.rolling(20).std()
    df["BB_lower"]    = close.rolling(20).mean() - 2 * close.rolling(20).std()
    df["Daily_Return"]= close.pct_change()
    df.dropna(inplace=True)
    return df


def build_sequences(df, look_back, train_split=0.80):
    prices = df[["Close"]].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)
    X, y   = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back : i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X).reshape(-1, look_back, 1), np.array(y)
    split  = int(len(X) * train_split)
    return X[:split], X[split:], y[:split], y[split:], scaler


def build_model(look_back):
    m = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(128, return_sequences=True), Dropout(0.2), BatchNormalization(),
        LSTM(64,  return_sequences=True), Dropout(0.2), BatchNormalization(),
        LSTM(32,  return_sequences=False), Dropout(0.2), BatchNormalization(),
        Dense(25, activation="relu"), Dense(1),
    ])
    m.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
    return m


# ─── Main page ────────────────────────────────────────────
st.title("📈 Stock Price Prediction using LSTM")
st.markdown(
    f"**Ticker:** `{TICKER}` &nbsp;|&nbsp; "
    f"**Period:** {PERIOD} &nbsp;|&nbsp; "
    f"**Look-back:** {LOOK_BACK} days &nbsp;|&nbsp; "
    f"**Forecast:** {FORECAST_DAYS} days"
)
st.markdown("---")

# ── Load data ─────────────────────────────────────────────
with st.spinner(f"Fetching {TICKER} data from Yahoo Finance ..."):
    df = fetch_data(TICKER, PERIOD)

current_price = float(df["Close"].iloc[-1])
prev_price    = float(df["Close"].iloc[-2])
day_chg       = (current_price - prev_price) / prev_price * 100
period_high   = float(df["Close"].max())
period_low    = float(df["Close"].min())
avg_vol       = int(df["Volume"].mean())

# ── Summary metrics row ───────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Price",  f"${current_price:.2f}", f"{day_chg:+.2f}% today")
c2.metric("Period High",    f"${period_high:.2f}")
c3.metric("Period Low",     f"${period_low:.2f}")
c4.metric("Avg Daily Volume", f"{avg_vol:,}")

# ── EDA Charts ───────────────────────────────────────────
st.subheader("📊 Exploratory Data Analysis")
tab1, tab2, tab3 = st.tabs(["Price & Moving Averages", "Volume", "RSI / MACD"])

with tab1:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df["Close"],   label="Close",  linewidth=1.2)
    ax.plot(df.index, df["SMA_20"],  label="SMA 20", linestyle="--", linewidth=1.0)
    ax.plot(df.index, df["SMA_50"],  label="SMA 50", linestyle="--", linewidth=1.0)
    ax.fill_between(df.index, df["BB_upper"], df["BB_lower"], alpha=0.1, label="Bollinger Bands")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)

with tab2:
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.bar(df.index, df["Volume"], width=1, alpha=0.7, color="#aec7e8")
    ax.set_ylabel("Volume"); ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)

with tab3:
    fig, axes = plt.subplots(2, 1, figsize=(14, 5))
    axes[0].plot(df.index, df["RSI"], color="#d62728", linewidth=1.0)
    axes[0].axhline(70, color="red",   linestyle="--", alpha=0.6)
    axes[0].axhline(30, color="green", linestyle="--", alpha=0.6)
    axes[0].set_title("RSI (14-day)"); axes[0].grid(alpha=0.3)
    axes[1].plot(df.index, df["MACD"],   label="MACD",   linewidth=1.0)
    axes[1].plot(df.index, df["EMA_26"], label="Signal", linewidth=1.0, linestyle="--")
    axes[1].set_title("MACD"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ── Train ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("🧠 Train LSTM Model")

if st.button("🚀 Train & Predict", type="primary"):
    with st.spinner("Preprocessing data ..."):
        X_tr, X_te, y_tr, y_te, scaler = build_sequences(df, LOOK_BACK)

    model = build_model(LOOK_BACK)

    progress_bar = st.progress(0, text="Training epoch 0 / …")
    loss_history, val_loss_history = [], []

    class StreamlitCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            loss_history.append(logs.get("loss", 0))
            val_loss_history.append(logs.get("val_loss", 0))
            pct = int((epoch + 1) / EPOCHS * 100)
            progress_bar.progress(
                pct,
                text=f"Epoch {epoch+1}/{EPOCHS} — loss: {logs.get('loss',0):.5f} — val_loss: {logs.get('val_loss',0):.5f}",
            )

    with st.spinner("Training LSTM ..."):
        model.fit(
            X_tr, y_tr,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            validation_split=0.1, verbose=0,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5),
                StreamlitCallback(),
            ],
        )
    progress_bar.progress(100, text="✅ Training complete!")
    st.success("Model trained successfully!")

    # Predictions
    pred_train = scaler.inverse_transform(model.predict(X_tr, verbose=0))
    pred_test  = scaler.inverse_transform(model.predict(X_te, verbose=0))
    true_train = scaler.inverse_transform(y_tr.reshape(-1, 1))
    true_test  = scaler.inverse_transform(y_te.reshape(-1, 1))

    rmse = math.sqrt(mean_squared_error(true_test, pred_test))
    mae  = mean_absolute_error(true_test, pred_test)
    r2   = r2_score(true_test, pred_test)
    mape = np.mean(np.abs((true_test - pred_test) / (true_test + 1e-8))) * 100
    acc  = max(0, 100 - mape)

    # Metrics
    st.markdown("---")
    st.subheader("📐 Model Performance Metrics")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("RMSE",     f"${rmse:.2f}")
    mc2.metric("MAE",      f"${mae:.2f}")
    mc3.metric("R² Score", f"{r2:.4f}")
    mc4.metric("Accuracy", f"{acc:.1f}%")

    # Loss curve
    st.subheader("📉 Training Loss Curve")
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(loss_history,     label="Train Loss", linewidth=1.5)
    ax.plot(val_loss_history, label="Val Loss",   linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)

    # Prediction chart
    st.subheader("📈 Actual vs Predicted — Test Set")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(true_test,  label="Actual",    linewidth=1.5, color="#1f77b4")
    ax.plot(pred_test,  label="Predicted", linewidth=1.2, color="#ff7f0e", linestyle="--")
    ax.set_ylabel("Price (USD)"); ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)

    # Future forecast
    st.subheader(f"🔮 {FORECAST_DAYS}-Day Future Forecast")
    seq = X_te[-1, :, 0].copy()
    forecasts = []
    for _ in range(FORECAST_DAYS):
        p = model.predict(seq.reshape(1, LOOK_BACK, 1), verbose=0)[0, 0]
        forecasts.append(p)
        seq = np.roll(seq, -1); seq[-1] = p
    forecast_prices = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))

    last_date    = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1),
                                  periods=FORECAST_DAYS)

    fig, ax = plt.subplots(figsize=(14, 4))
    hist_window = 60
    ax.plot(df.index[-hist_window:], df["Close"].values[-hist_window:],
            label="Historical", color="#1f77b4", linewidth=1.5)
    ax.plot(future_dates, forecast_prices, label="Forecast",
            color="#2ca02c", linewidth=1.5, marker="o", markersize=4)
    ax.axvline(x=last_date, color="gray", linestyle=":", linewidth=1)
    ax.set_ylabel("Price (USD)"); ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    st.pyplot(fig, use_container_width=True)

    # Forecast table
    st.subheader("📋 Forecast Table")
    forecast_df = pd.DataFrame({
        "Date":           future_dates.strftime("%Y-%m-%d"),
        "Forecast Price": [f"${p:.2f}" for p in forecast_prices.flatten()],
        "Change from Today": [
            f"{(p - current_price)/current_price*100:+.2f}%"
            for p in forecast_prices.flatten()
        ],
    })
    st.dataframe(forecast_df, use_container_width=True)

    final_price = forecast_prices[-1, 0]
    chg_pct     = (final_price - current_price) / current_price * 100
    signal = "BUY 🟢" if chg_pct > 2 else "SELL 🔴" if chg_pct < -2 else "HOLD 🟡"
    st.markdown(
        f"### Trading Signal: **{signal}**  "
        f"&nbsp; Expected {FORECAST_DAYS}-day change: **{chg_pct:+.2f}%**"
    )

else:
    st.info("⬆️ Click **Train & Predict** in the sidebar to start model training.")

# ─── Footer ───────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Stock Price Prediction using LSTM Neural Networks · "
    "Last Year B.E. Project · Data source: Yahoo Finance · "
    "⚠️ For educational purposes only — not financial advice."
)
