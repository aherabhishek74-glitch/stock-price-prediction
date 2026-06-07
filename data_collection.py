"""
=============================================================
 Stock Price Prediction — LSTM Model
 File : data_collection.py
 Desc : Download historical stock data and preprocess it
=============================================================
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────
TICKER      = "AAPL"          # Stock symbol
START_DATE  = "2018-01-01"    # Training data start
END_DATE    = "2024-01-01"    # Training data end
LOOK_BACK   = 60              # Days of history fed to LSTM
TRAIN_SPLIT = 0.80            # 80% train, 20% test
DATA_DIR    = "data"
# ──────────────────────────────────────────────────────────

os.makedirs(DATA_DIR, exist_ok=True)


# ── 1. Download ────────────────────────────────────────────
def download_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance."""
    print(f"[INFO] Downloading {ticker} from {start} to {end} ...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.dropna(inplace=True)
    path = os.path.join(DATA_DIR, f"{ticker}_raw.csv")
    df.to_csv(path)
    print(f"[INFO] Saved raw data → {path}  ({len(df)} rows)")
    return df


# ── 2. Feature Engineering ─────────────────────────────────
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA, EMA, RSI, Bollinger Bands, and MACD columns."""
    df = df.copy()
    close = df["Close"]

    # Simple / Exponential Moving Averages
    df["SMA_20"]  = close.rolling(window=20).mean()
    df["SMA_50"]  = close.rolling(window=50).mean()
    df["EMA_12"]  = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"]  = close.ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"]   = df["EMA_12"] - df["EMA_26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # RSI (14-day)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20-day)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_upper"] = sma20 + 2 * std20
    df["BB_lower"] = sma20 - 2 * std20

    # Daily return & volume change
    df["Daily_Return"]  = close.pct_change()
    df["Volume_Change"] = df["Volume"].pct_change()

    df.dropna(inplace=True)
    return df


# ── 3. Normalise & Sequence ────────────────────────────────
def prepare_sequences(
    df: pd.DataFrame,
    look_back: int = LOOK_BACK,
    train_split: float = TRAIN_SPLIT,
    feature_col: str = "Close",
) -> dict:
    """
    Scale data, build sliding-window sequences, split train/test.
    Returns a dict with arrays and the fitted scaler.
    """
    prices = df[[feature_col]].values

    scaler  = MinMaxScaler(feature_range=(0, 1))
    scaled  = scaler.fit_transform(prices)

    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back : i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)

    # Reshape X → (samples, time-steps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    split = int(len(X) * train_split)
    return {
        "X_train": X[:split],
        "X_test":  X[split:],
        "y_train": y[:split],
        "y_test":  y[split:],
        "scaler":  scaler,
        "df":      df,
    }


# ── 4. Visualise Raw Data ──────────────────────────────────
def plot_raw_data(df: pd.DataFrame, ticker: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f"{ticker} — Historical Overview", fontsize=15, fontweight="bold")

    # Close price + MAs
    axes[0].plot(df.index, df["Close"],  label="Close",  linewidth=1.2, color="#1f77b4")
    axes[0].plot(df.index, df["SMA_20"], label="SMA 20", linewidth=1.0, color="#ff7f0e", linestyle="--")
    axes[0].plot(df.index, df["SMA_50"], label="SMA 50", linewidth=1.0, color="#2ca02c", linestyle="--")
    axes[0].fill_between(df.index, df["BB_upper"], df["BB_lower"], alpha=0.1, color="gray", label="Bollinger")
    axes[0].set_title("Closing Price with Moving Averages & Bollinger Bands")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    # Volume
    axes[1].bar(df.index, df["Volume"], color="#aec7e8", alpha=0.7)
    axes[1].set_title("Trading Volume"); axes[1].grid(alpha=0.3)

    # RSI
    axes[2].plot(df.index, df["RSI"], color="#d62728", linewidth=1.0)
    axes[2].axhline(70, color="red",   linestyle="--", alpha=0.7, label="Overbought (70)")
    axes[2].axhline(30, color="green", linestyle="--", alpha=0.7, label="Oversold (30)")
    axes[2].set_title("RSI (14-day)")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join("outputs", f"{ticker}_eda.png")
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"[INFO] EDA chart saved → {out}")
    plt.show()


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    df_raw = download_stock_data(TICKER, START_DATE, END_DATE)
    df     = add_technical_indicators(df_raw)

    path = os.path.join(DATA_DIR, f"{TICKER}_features.csv")
    df.to_csv(path)
    print(f"[INFO] Feature-enriched data saved → {path}")

    plot_raw_data(df, TICKER)

    data = prepare_sequences(df)
    print(f"\n[INFO] Sequence shapes:")
    print(f"       X_train : {data['X_train'].shape}")
    print(f"       X_test  : {data['X_test'].shape}")
    print(f"       y_train : {data['y_train'].shape}")
    print(f"       y_test  : {data['y_test'].shape}")
    print("\n[OK] data_collection.py complete — run model_training.py next.")
