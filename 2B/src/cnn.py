"""
1D CNN model for traffic flow prediction (third ML technique).

Architecture: two Conv1d blocks with BatchNorm → Flatten → FC head.
Unlike the previous AdaptiveAvgPool approach (which collapsed the sequence
to a single mean value and destroyed temporal order), this version flattens
the full feature map so the FC layers can still distinguish *which* timestep
each feature came from.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRAIN_PATH = "data/train.csv"
VAL_PATH   = "data/val.csv"
TEST_PATH  = "data/test.csv"
MODEL_PATH = "models/cnn_best.pth"

LAGS       = 12      # 12 × 15 min = 3-hour lookback (same as LSTM/GRU)
BATCH_SIZE = 32
EPOCHS     = 100
PATIENCE   = 15      # increased from 10 — CNN val loss is noisier, needs more room
LR         = 0.0005  # halved from 0.001 — prevents the optimizer from overshooting
                     # the narrow loss basin that caused the flat training curve


# ─────────────────────────────────────────────
# STEP 1: Load data  (unchanged)
# ─────────────────────────────────────────────
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    val   = pd.read_csv(VAL_PATH)
    test  = pd.read_csv(TEST_PATH)
    for df in [train, val, test]:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"Train: {len(train):,} rows | Val: {len(val):,} rows | Test: {len(test):,} rows")
    print(f"Sites: {train['scats_id'].nunique()}")
    print(f"Flow range: {train['flow_15min'].min()} – {train['flow_15min'].max()}")
    return train, val, test


# ─────────────────────────────────────────────
# STEP 2: Normalise  (unchanged)
# ─────────────────────────────────────────────
def normalise(train, val, test):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = train.copy()
    val   = val.copy()
    test  = test.copy()
    train["flow_scaled"] = scaler.fit_transform(train[["flow_15min"]])
    val["flow_scaled"]   = scaler.transform(val[["flow_15min"]])
    test["flow_scaled"]  = scaler.transform(test[["flow_15min"]])
    print(f"Scaled range — min: {train['flow_scaled'].min():.2f}  max: {train['flow_scaled'].max():.2f}")
    
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    return train, val, test, scaler


# ─────────────────────────────────────────────
# STEP 3: Build sliding windows  (unchanged)
# ─────────────────────────────────────────────
def build_sequences(df, lags=LAGS):
    X, y = [], []
    for _, site_row in df.groupby("scats_id"):
        site_row = site_row.sort_values("timestamp")
        flow = site_row["flow_scaled"].values
        for i in range(lags, len(flow)):
            X.append(flow[i - lags : i])
            y.append(flow[i])
    X = np.array(X).reshape(-1, lags, 1)
    y = np.array(y)
    print(f"X shape: {X.shape}  |  y shape: {y.shape}")
    return X, y


# ─────────────────────────────────────────────
# STEP 4: Model  ← KEY CHANGES HERE
# ─────────────────────────────────────────────
class CNNModel(nn.Module):
    """
    Two-block 1D CNN with BatchNorm, followed by a two-layer FC head.

    Why these changes vs the original:

    1. Replaced AdaptiveAvgPool1d(1) with Flatten().
       AdaptiveAvgPool collapsed the 12-timestep sequence into a single
       averaged value, destroying temporal ordering entirely.  Flatten
       keeps all 64 channels × 12 positions = 768 values, so the FC head
       can still distinguish features at different time positions.

    2. Added BatchNorm1d after each Conv layer.
       Traffic flow has high variance across sites and hours.  BatchNorm
       normalises each channel's activations per mini-batch, which
       stabilises gradients and is the main reason train loss was flat
       (exploding/vanishing gradients in the conv stack).

    3. Kept kernel_size=3 with padding=1 so sequence length is preserved
       through both conv layers (same padding convention as LSTM/GRU use
       implicitly via recurrent connections).
    """
    def __init__(self, lags=LAGS):
        super().__init__()

        # ── Convolutional feature extractor ──
        # Input:  (batch, 1, 12)   [channels-first after transpose]
        # Output: (batch, 64, 12)  [same spatial length due to padding=1]
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=1,  out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),   # normalise across the 32 feature maps
            nn.ReLU(),

            # Block 2
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Flatten ALL channels × timesteps — preserves temporal position info
            # Output: (batch, 64 * lags) = (batch, 768)
            nn.Flatten(),
        )

        # ── Fully-connected prediction head ──
        self.fc = nn.Sequential(
            nn.Linear(64 * lags, 64),
            nn.ReLU(),
            nn.Dropout(0.2),      # same dropout rate as LSTM/GRU for fair comparison
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x arrives as (batch, seq=12, features=1)
        # Conv1d expects (batch, channels, seq) so transpose first
        x = x.transpose(1, 2)          # → (batch, 1, 12)
        x = self.conv(x)               # → (batch, 768)
        return self.fc(x)              # → (batch, 1)


# ─────────────────────────────────────────────
# STEP 5: Train  (unchanged except model instantiation)
# ─────────────────────────────────────────────
def train_model(X_train, y_train, X_val, y_val):
    os.makedirs("models", exist_ok=True)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t   = torch.FloatTensor(X_val)
    y_val_t   = torch.FloatTensor(y_val).unsqueeze(1)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model     = CNNModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss    = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    print("\nTraining 1D CNN...")
    for epoch in range(EPOCHS):

        # ── Training phase ──
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(loader))

        # ── Validation phase ──
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1:3d}/{EPOCHS} — "
                f"train: {train_losses[-1]:.6f}  val: {val_loss:.6f}"
            )

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(
                    f"\n  Early stopping at epoch {epoch+1} "
                    f"— best val: {best_val_loss:.6f}"
                )
                break

    print(f"  Model saved to {MODEL_PATH}")

    pd.DataFrame({
        "epoch":      range(1, len(train_losses) + 1),
        "train_loss": train_losses,
        "val_loss":   val_losses,
    }).to_csv(MODEL_PATH.replace(".pth", "_losses.csv"), index=False)

    return train_losses, val_losses


# ─────────────────────────────────────────────
# STEP 6: Evaluate  (unchanged)
# ─────────────────────────────────────────────
def evaluate(X_test, y_test, scaler):
    model = CNNModel()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test)).numpy()

    preds_real  = scaler.inverse_transform(preds)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae  = mean_absolute_error(y_test_real, preds_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, preds_real))
    mask = y_test_real.flatten() > 10
    mape = (
        np.mean(np.abs((y_test_real[mask] - preds_real[mask]) / y_test_real[mask])) * 100
        if mask.any()
        else float("nan")
    )

    print(f"\nCNN Test Results:")
    print(f"  MAE:  {mae:.2f}  vehicles/15min")
    print(f"  RMSE: {rmse:.2f} vehicles/15min")
    print(f"  MAPE: {mape:.2f}%")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train, val, test            = load_data()
    train, val, test, scaler    = normalise(train, val, test)
    X_train, y_train            = build_sequences(train)
    X_val,   y_val              = build_sequences(val)
    X_test,  y_test             = build_sequences(test)
    train_model(X_train, y_train, X_val, y_val)
    evaluate(X_test, y_test, scaler)