"""
=============================================================
 Stock Price Prediction — LSTM Model
 File : model_training.py
 Desc : Build, train, and save the LSTM neural network
=============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import warnings
warnings.filterwarnings("ignore")

# Import preprocessing helpers from data_collection.py
from data_collection import (
    download_stock_data, add_technical_indicators, prepare_sequences,
    TICKER, START_DATE, END_DATE, LOOK_BACK, TRAIN_SPLIT
)

# ─── Hyper-parameters ──────────────────────────────────────
LSTM_UNITS   = [128, 64, 32]   # Units in each LSTM layer
DROPOUT_RATE = 0.20
EPOCHS       = 100
BATCH_SIZE   = 32
LEARNING_RATE= 0.001
MODEL_DIR    = "models"
OUTPUT_DIR   = "outputs"
# ──────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Build Model ─────────────────────────────────────────
def build_lstm_model(look_back: int = LOOK_BACK) -> tf.keras.Model:
    """
    Stacked LSTM with:
      • 3 LSTM layers (128 → 64 → 32 units)
      • Dropout (0.2) after each LSTM
      • BatchNormalization for faster convergence
      • Dense output layer (1 neuron — next-day close price)
    """
    model = Sequential([
        Input(shape=(look_back, 1)),

        LSTM(LSTM_UNITS[0], return_sequences=True),
        Dropout(DROPOUT_RATE),
        BatchNormalization(),

        LSTM(LSTM_UNITS[1], return_sequences=True),
        Dropout(DROPOUT_RATE),
        BatchNormalization(),

        LSTM(LSTM_UNITS[2], return_sequences=False),
        Dropout(DROPOUT_RATE),
        BatchNormalization(),

        Dense(25, activation="relu"),
        Dense(1),                       # Linear output — regression
    ], name="LSTM_StockPredictor")

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    model.summary()
    return model


# ── 2. Callbacks ───────────────────────────────────────────
def get_callbacks(ticker: str) -> list:
    return [
        EarlyStopping(
            monitor="val_loss", patience=15,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f"{ticker}_best.keras"),
            monitor="val_loss", save_best_only=True, verbose=0
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=7, min_lr=1e-6, verbose=1
        ),
        TensorBoard(log_dir=os.path.join(MODEL_DIR, "logs"), histogram_freq=0),
    ]


# ── 3. Train ───────────────────────────────────────────────
def train_model(model, data: dict, ticker: str) -> tf.keras.callbacks.History:
    print(f"\n[INFO] Training LSTM for {ticker} ...")
    history = model.fit(
        data["X_train"], data["y_train"],
        validation_split=0.10,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(ticker),
        verbose=1,
    )
    # Save final weights
    model.save(os.path.join(MODEL_DIR, f"{ticker}_final.keras"))
    print(f"[INFO] Model saved → {MODEL_DIR}/{ticker}_final.keras")
    return history


# ── 4. Evaluate ────────────────────────────────────────────
def evaluate_model(model, data: dict, ticker: str) -> dict:
    """Inverse-transform predictions and compute regression metrics."""
    scaler = data["scaler"]

    pred_train_scaled = model.predict(data["X_train"], verbose=0)
    pred_test_scaled  = model.predict(data["X_test"],  verbose=0)

    pred_train = scaler.inverse_transform(pred_train_scaled)
    pred_test  = scaler.inverse_transform(pred_test_scaled)
    true_train = scaler.inverse_transform(data["y_train"].reshape(-1, 1))
    true_test  = scaler.inverse_transform(data["y_test"].reshape(-1, 1))

    rmse_train = math.sqrt(mean_squared_error(true_train, pred_train))
    rmse_test  = math.sqrt(mean_squared_error(true_test,  pred_test))
    mae_train  = mean_absolute_error(true_train, pred_train)
    mae_test   = mean_absolute_error(true_test,  pred_test)
    r2_train   = r2_score(true_train, pred_train)
    r2_test    = r2_score(true_test,  pred_test)

    metrics = {
        "RMSE_train": round(rmse_train, 4),
        "RMSE_test":  round(rmse_test,  4),
        "MAE_train":  round(mae_train,  4),
        "MAE_test":   round(mae_test,   4),
        "R2_train":   round(r2_train,   4),
        "R2_test":    round(r2_test,    4),
    }

    print("\n" + "=" * 40)
    print("  MODEL EVALUATION METRICS")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"  {k:<15}: {v}")
    print("=" * 40)

    return {
        **metrics,
        "pred_train": pred_train,
        "pred_test":  pred_test,
        "true_train": true_train,
        "true_test":  true_test,
    }


# ── 5. Plot Results ────────────────────────────────────────
def plot_results(history, results: dict, data: dict, ticker: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{ticker} — LSTM Stock Price Prediction Results",
                 fontsize=14, fontweight="bold")

    # (a) Training & Validation Loss
    ax = axes[0, 0]
    ax.plot(history.history["loss"],     label="Train Loss",  color="#1f77b4")
    ax.plot(history.history["val_loss"], label="Val Loss",    color="#d62728", linestyle="--")
    ax.set_title("Training & Validation Loss (MSE)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # (b) Train set: actual vs predicted
    ax = axes[0, 1]
    ax.plot(results["true_train"], label="Actual",    color="#1f77b4", linewidth=1.0)
    ax.plot(results["pred_train"], label="Predicted", color="#ff7f0e", linewidth=1.0, linestyle="--")
    ax.set_title(f"Train Set — RMSE ${results['RMSE_train']:.2f}  R²={results['R2_train']:.4f}")
    ax.legend(); ax.grid(alpha=0.3)

    # (c) Test set: actual vs predicted
    ax = axes[1, 0]
    ax.plot(results["true_test"], label="Actual",    color="#1f77b4", linewidth=1.2)
    ax.plot(results["pred_test"], label="Predicted", color="#2ca02c", linewidth=1.2, linestyle="--")
    ax.set_title(f"Test Set  — RMSE ${results['RMSE_test']:.2f}  R²={results['R2_test']:.4f}")
    ax.legend(); ax.grid(alpha=0.3)

    # (d) Scatter: actual vs predicted (test)
    ax = axes[1, 1]
    ax.scatter(results["true_test"], results["pred_test"],
               alpha=0.4, s=10, color="#9467bd")
    mn = min(results["true_test"].min(), results["pred_test"].min())
    mx = max(results["true_test"].max(), results["pred_test"].max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_title("Actual vs Predicted (Test Set)")
    ax.set_xlabel("Actual Price ($)"); ax.set_ylabel("Predicted Price ($)")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{ticker}_results.png")
    plt.savefig(out, dpi=150)
    print(f"[INFO] Results chart saved → {out}")
    plt.show()


# ── 6. Forecast next N days ────────────────────────────────
def forecast_future(model, data: dict, n_days: int = 30) -> np.ndarray:
    """Walk-forward forecast for the next n_days trading days."""
    scaler      = data["scaler"]
    last_seq    = data["X_test"][-1].copy()         # shape (look_back, 1)
    predictions = []

    for _ in range(n_days):
        pred_scaled = model.predict(last_seq.reshape(1, LOOK_BACK, 1), verbose=0)
        predictions.append(pred_scaled[0, 0])
        last_seq = np.roll(last_seq, -1, axis=0)
        last_seq[-1, 0] = pred_scaled[0, 0]

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Data
    df_raw = download_stock_data(TICKER, START_DATE, END_DATE)
    df     = add_technical_indicators(df_raw)
    data   = prepare_sequences(df)

    # 2. Model
    model   = build_lstm_model(LOOK_BACK)
    history = train_model(model, data, TICKER)

    # 3. Evaluate
    results = evaluate_model(model, data, TICKER)
    plot_results(history, results, data, TICKER)

    # 4. Future forecast
    future_prices = forecast_future(model, data, n_days=30)
    print(f"\n[INFO] 30-day price forecast for {TICKER}:")
    for i, p in enumerate(future_prices.flatten(), 1):
        print(f"  Day {i:>2}: ${p:.2f}")

    print("\n[OK] model_training.py complete — run prediction.py for live prediction.")
