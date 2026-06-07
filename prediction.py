"""
=============================================================
 Stock Price Prediction — LSTM Model
 File : prediction.py
 Desc : Load saved model → predict on new data → visualise
=============================================================
Usage:
    python prediction.py --ticker AAPL --days 30
=============================================================
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from data_collection import (
    add_technical_indicators, prepare_sequences,
    LOOK_BACK, MODEL_DIR, OUTPUT_DIR
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Load the saved model ───────────────────────────────────
def load_model(ticker: str) -> tf.keras.Model:
    path = os.path.join(MODEL_DIR, f"{ticker}_best.keras")
    if not os.path.exists(path):
        path = os.path.join(MODEL_DIR, f"{ticker}_final.keras")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved model found for {ticker}. Run model_training.py first."
        )
    print(f"[INFO] Loading model from {path}")
    return tf.keras.models.load_model(path)


# ── Fetch fresh data for prediction ───────────────────────
def get_recent_data(ticker: str, period: str = "2y") -> tuple:
    """
    Download the last `period` of data, engineer features,
    scale and build sequences for prediction.
    Returns (model_input, scaler, df, dates).
    """
    print(f"[INFO] Fetching recent data for {ticker} ...")
    df_raw = yf.download(ticker, period=period, progress=False)
    df_raw.dropna(inplace=True)
    df = add_technical_indicators(df_raw)

    close = df[["Close"]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close)

    # Build all available sequences
    X = []
    for i in range(LOOK_BACK, len(scaled)):
        X.append(scaled[i - LOOK_BACK : i, 0])
    X = np.array(X).reshape(-1, LOOK_BACK, 1)

    dates = df.index[LOOK_BACK:]
    return X, scaler, df, dates


# ── Predict & inverse-transform ───────────────────────────
def predict(model, X: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    pred_scaled = model.predict(X, verbose=0)
    return scaler.inverse_transform(pred_scaled)


# ── Future walk-forward forecast ──────────────────────────
def walk_forward_forecast(
    model, last_sequence: np.ndarray,
    scaler: MinMaxScaler, n_days: int = 30
) -> tuple:
    """
    Given the last known LOOK_BACK sequence, predict n_days ahead.
    Returns (forecast_prices, confidence_lower, confidence_upper).
    """
    seq         = last_sequence.copy()
    predictions = []
    MC_RUNS     = 20          # Monte Carlo dropout passes for uncertainty

    for _ in range(n_days):
        mc_preds = []
        for _ in range(MC_RUNS):
            p = model(seq.reshape(1, LOOK_BACK, 1), training=True).numpy()[0, 0]
            mc_preds.append(p)
        mean_pred = np.mean(mc_preds)
        std_pred  = np.std(mc_preds)
        predictions.append((mean_pred, std_pred))
        seq = np.roll(seq, -1)
        seq[-1] = mean_pred

    means = np.array([p[0] for p in predictions]).reshape(-1, 1)
    stds  = np.array([p[1] for p in predictions]).reshape(-1, 1)

    forecast   = scaler.inverse_transform(means)
    lower      = scaler.inverse_transform(means - 2 * stds)
    upper      = scaler.inverse_transform(means + 2 * stds)
    return forecast, lower, upper


# ── Visualise ─────────────────────────────────────────────
def plot_prediction(
    df: pd.DataFrame,
    dates,
    actual: np.ndarray,
    predicted: np.ndarray,
    forecast: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    ticker: str,
    n_forecast: int,
) -> None:
    # Generate future trading dates
    last_date    = pd.Timestamp(dates[-1])
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1),
                                  periods=n_forecast)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(f"{ticker} — LSTM Prediction & {n_forecast}-Day Forecast",
                 fontsize=14, fontweight="bold")

    # ── Chart 1: Full history + predicted + forecast ──────
    ax = axes[0]
    ax.plot(dates[-180:], actual[-180:],
            label="Actual Close", color="#1f77b4", linewidth=1.5)
    ax.plot(dates[-180:], predicted[-180:],
            label="LSTM Predicted", color="#ff7f0e", linewidth=1.2, linestyle="--")
    ax.plot(future_dates, forecast,
            label=f"{n_forecast}-Day Forecast", color="#2ca02c", linewidth=1.5)
    ax.fill_between(future_dates, lower.flatten(), upper.flatten(),
                    alpha=0.20, color="#2ca02c", label="95% Confidence Interval")
    ax.axvline(x=last_date, color="gray", linestyle=":", linewidth=1, label="Today")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.set_title("Last 180 Days + Forecast")
    ax.set_ylabel("Price (USD)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Chart 2: Forecast close-up ─────────────────────────
    ax = axes[1]
    ax.plot(future_dates, forecast, color="#2ca02c", linewidth=2, marker="o",
            markersize=4, label="Forecast")
    ax.fill_between(future_dates, lower.flatten(), upper.flatten(),
                    alpha=0.25, color="#2ca02c", label="95% CI")
    for i, (d, p) in enumerate(zip(future_dates[::5], forecast[::5])):
        ax.annotate(f"${p[0]:.1f}", (d, p[0]),
                    textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.set_title(f"{n_forecast}-Day Forecast Close-Up")
    ax.set_ylabel("Price (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{ticker}_forecast.png")
    plt.savefig(out, dpi=150)
    print(f"[INFO] Forecast chart saved → {out}")
    plt.show()


# ── Summary Table ─────────────────────────────────────────
def print_forecast_table(forecast: np.ndarray, lower: np.ndarray,
                          upper: np.ndarray, n_days: int, ticker: str) -> None:
    print(f"\n{'='*55}")
    print(f"  {ticker}  —  {n_days}-Day Price Forecast")
    print(f"{'='*55}")
    print(f"  {'Day':<6} {'Forecast':>12} {'Lower 95%':>12} {'Upper 95%':>12}")
    print(f"  {'-'*42}")
    for i in range(n_days):
        print(f"  {i+1:<6} ${forecast[i,0]:>10.2f} ${lower[i,0]:>10.2f} ${upper[i,0]:>10.2f}")
    print(f"{'='*55}")
    chg = (forecast[-1, 0] - forecast[0, 0]) / forecast[0, 0] * 100
    direction = "📈 BULLISH" if chg > 0 else "📉 BEARISH"
    print(f"\n  Expected change over {n_days} days: {chg:+.2f}%  {direction}")


# ── CLI entry-point ───────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stock Price Prediction CLI")
    parser.add_argument("--ticker", default="AAPL",  help="Stock ticker (default: AAPL)")
    parser.add_argument("--days",   default=30, type=int, help="Days to forecast (default: 30)")
    args = parser.parse_args()

    ticker   = args.ticker.upper()
    n_days   = args.days

    model = load_model(ticker)
    X, scaler, df, dates = get_recent_data(ticker)

    actual    = scaler.inverse_transform(
        np.array([X[i, -1, 0] for i in range(len(X))]).reshape(-1, 1)
    )
    predicted = predict(model, X, scaler)

    last_seq            = X[-1, :, 0]
    forecast, lo, hi    = walk_forward_forecast(model, last_seq, scaler, n_days)

    plot_prediction(df, dates, actual, predicted, forecast, lo, hi, ticker, n_days)
    print_forecast_table(forecast, lo, hi, n_days, ticker)


if __name__ == "__main__":
    main()
