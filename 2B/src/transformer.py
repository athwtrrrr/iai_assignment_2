import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_PATH = "data/train.csv"
VAL_PATH   = "data/val.csv"
TEST_PATH  = "data/test.csv"

# Sequence / training hyper-parameters (match lstm.py / gru.py for fair comparison)
LAGS       = 12      # 12 × 15 min = 3-hour lookback
BATCH_SIZE = 32
EPOCHS     = 100
PATIENCE   = 10
LR         = 0.001

# Transformer-specific hyper-parameters
D_MODEL    = 64      # must be divisible by NHEAD
NHEAD      = 4       # 64 / 4 = 16 per head — works fine
NUM_LAYERS = 2       # TransformerEncoder depth
DIM_FF     = 128     # feedforward layer width inside each encoder block
DROPOUT    = 0.1     # applied in pos-encoding, encoder layers, and output head


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1  Load pre-split data
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """Load train / val / test CSVs produced by data_processing.py."""
    train = pd.read_csv(TRAIN_PATH)
    val   = pd.read_csv(VAL_PATH)
    test  = pd.read_csv(TEST_PATH)

    for df in [train, val, test]:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(f"Train: {len(train):,} rows | Val: {len(val):,} rows | Test: {len(test):,} rows")
    print(f"Sites: {train['scats_id'].nunique()}")
    print(f"Flow  range  : {train['flow_15min'].min():.0f} – {train['flow_15min'].max():.0f} veh/15min")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2  Global normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalise(train, val, test):
    """
    Fit MinMaxScaler on training data, apply to val/test.
    A single global scaler is used (consistent with lstm.py and gru.py).
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    train["flow_scaled"] = scaler.fit_transform(train[["flow_15min"]])
    val["flow_scaled"]   = scaler.transform(val[["flow_15min"]])
    test["flow_scaled"]  = scaler.transform(test[["flow_15min"]])
    print(f"Scaled range — min: {train['flow_scaled'].min():.3f}  "
          f"max: {train['flow_scaled'].max():.3f}")
    return train, val, test, scaler


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3  Sliding-window sequence builder
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(df, lags=LAGS):
    """
    Create supervised (X, y) pairs for one site's subset DataFrame.
    Rows must be sorted by timestamp before calling this function.

    Returns
    -------
    X : np.ndarray  shape (n_samples, lags, 1)   ← (batch, seq, features)
    y : np.ndarray  shape (n_samples,)
    """
    X, y = [], []

    for site_id, site_rows in df.groupby("scats_id"):
        site_rows = site_rows.sort_values("timestamp")
        flow = site_rows["flow_scaled"].values

        for i in range(lags, len(flow)):
            X.append(flow[i - lags : i])
            y.append(flow[i])

    if len(X) == 0:
        return np.empty((0, lags, 1), dtype=np.float32), np.empty(0, dtype=np.float32)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    X = X.reshape(X.shape[0], X.shape[1], 1)   # (N, lags, 1)
    print(f"X shape: {X.shape}  |  y shape: {y.shape}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4  Transformer model definition
# ─────────────────────────────────────────────────────────────────────────────

class _PositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding (Vaswani et al., 2017).

    Adds a fixed position signal to each token embedding so the self-attention
    mechanism can distinguish temporal position within the lookback window.

    Because traffic intervals are evenly spaced (15 min), sinusoidal encoding
    works well without needing learnable position embeddings.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe       = torch.zeros(max_len, d_model)                         # (L, d)
        position = torch.arange(0, max_len).unsqueeze(1).float()         # (L, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                             # (1, L, d)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Time-series Transformer for univariate traffic volume forecasting.

    Architecture
    ------------
    Linear projection  →  Positional Encoding  →  TransformerEncoder (N layers)
    →  global average pooling  →  Dropout  →  FC output

    Why global average pooling instead of [CLS] token?
    ---------------------------------------------------
    All 12 timesteps carry predictive signal for short-term traffic.
    Averaging prevents the network from over-relying on a single position
    and produces more stable gradients than taking only the last token.

    Input  : (batch, seq_len=12, input_size=1)
    Output : (batch, 1)
    """

    def __init__(
        self,
        input_size: int   = 1,
        d_model:    int   = D_MODEL,
        nhead:      int   = NHEAD,
        num_layers: int   = NUM_LAYERS,
        dim_ff:     int   = DIM_FF,
        dropout:    float = DROPOUT,
    ):
        super().__init__()

        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead}). "
            f"Choose nhead ∈ {{1, 2, 4, 8, 16, 32, 64}} for d_model=64."
        )

        # 1. Project raw feature dimension (1) up to d_model so positional
        #    encoding and multi-head attention operate in a richer space.
        self.input_projection = nn.Linear(input_size, d_model)

        # 2. Positional encoding — sinusoidal, fixed (not learned)
        self.pos_encoding = _PositionalEncoding(d_model, dropout)

        # 3. Stacked TransformerEncoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_ff,
            dropout         = dropout,
            batch_first     = True,   # input shape: (batch, seq, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 4. Output head
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1)
        x = self.input_projection(x)       # → (batch, seq_len, d_model)
        x = self.pos_encoding(x)           # → (batch, seq_len, d_model)
        x = self.transformer_encoder(x)    # → (batch, seq_len, d_model)
        x = x.mean(dim=1)                  # global avg pool → (batch, d_model)
        x = self.dropout(x)
        return self.fc(x)                  # → (batch, 1)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5  Per-site training with early stopping
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, X_val, y_val, site_id):
    """
    Train one TransformerModel for a single SCATS site.

    Key difference vs LSTM/GRU
    --------------------------
    gradient clipping (clip_grad_norm_ max_norm=1.0) is applied before every
    optimiser step.  Transformers without gradient clipping can diverge abruptly
    when a rare high-flow spike produces a large attention weight update.

    Returns
    -------
    train_losses, val_losses : List[float]
    """
    os.makedirs("models", exist_ok=True)
    model_path = f"models/transformer_site_{site_id}.pth"

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)   # (N, 1)
    X_val_t   = torch.FloatTensor(X_val)
    y_val_t   = torch.FloatTensor(y_val).unsqueeze(1)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size = BATCH_SIZE,
        shuffle    = True,
    )

    model     = TransformerModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss    = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    print(f"\nTraining Transformer for site {site_id}…")

    for epoch in range(EPOCHS):
        # ── Training phase ──────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss   = criterion(output, y_batch)
            loss.backward()
            # Gradient clipping — critical for Transformer stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(loader)
        train_losses.append(avg_train_loss)

        # ── Validation phase ─────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"  Site {site_id} | Epoch {epoch+1:3d}/{EPOCHS} — "
                f"train loss: {avg_train_loss:.6f}  val loss: {val_loss:.6f}"
            )

        # ── Early stopping ───────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(
                    f"  Site {site_id} early stopping at epoch {epoch + 1} — "
                    f"best val loss: {best_val_loss:.6f}"
                )
                break

    # Save loss history for plotting
    loss_df  = pd.DataFrame({
        "epoch":      list(range(1, len(train_losses) + 1)),
        "train_loss": train_losses,
        "val_loss":   val_losses,
    })
    csv_path = model_path.replace(".pth", "_losses.csv")
    loss_df.to_csv(csv_path, index=False)
    print(f"  Model saved       → {model_path}")
    print(f"  Loss history CSV  → {csv_path}")

    return train_losses, val_losses


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6  Evaluation — MAE, RMSE, MAPE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(X_test, y_test, scaler, site_id):
    """
    Load the saved best model for *site_id* and compute test-set metrics in the
    original (veh/15min) unit.

    MAPE is computed only on intervals where actual flow > 10 veh/15min to
    avoid division-near-zero inflation (consistent with lstm.py / gru.py).
    """
    model_path = f"models/transformer_site_{site_id}.pth"
    model      = TransformerModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test)).numpy()   # (N, 1)

    preds_real  = scaler.inverse_transform(preds)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae  = mean_absolute_error(y_test_real, preds_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, preds_real))
    mask = y_test_real.flatten() > 10
    mape = (
        np.mean(
            np.abs((y_test_real[mask] - preds_real[mask]) / y_test_real[mask])
        ) * 100
        if np.any(mask) else float("nan")
    )

    print(
        f"  Site {site_id} | "
        f"MAE: {mae:.2f}  RMSE: {rmse:.2f}  MAPE: {mape:.2f}%"
    )
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}, preds_real, y_test_real


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7  Plotting — same three plots as lstm.py
# ─────────────────────────────────────────────────────────────────────────────

def plot_predictions(site_id, test_site, preds_real, y_test_real,
                     lags=LAGS, save_dir="plots"):
    """
    Save two diagnostic plots per site:
      1. Time-series overlay  (actual vs predicted over the test period)
      2. Scatter plot         (predicted vs actual with perfect-fit diagonal)
    """
    os.makedirs(save_dir, exist_ok=True)

    test_sorted = test_site.sort_values("timestamp")
    timestamps  = test_sorted["timestamp"].iloc[lags:].reset_index(drop=True)

    n = min(len(timestamps), len(preds_real))
    results_df = pd.DataFrame({
        "timestamp": timestamps[:n],
        "actual":    y_test_real.flatten()[:n],
        "predicted": preds_real.flatten()[:n],
    })

    # ── Plot 1: Time-series ──────────────────────────────────────────────
    plt.figure(figsize=(12, 5))
    plt.plot(results_df["timestamp"], results_df["actual"],
             label="Actual",    alpha=0.75, linewidth=1.0)
    plt.plot(results_df["timestamp"], results_df["predicted"],
             label="Predicted", alpha=0.75, linewidth=1.0, linestyle="--")
    plt.title(f"Site {site_id} — Actual vs Predicted (Transformer, Test Set)")
    plt.xlabel("Time");  plt.ylabel("Flow (vehicles / 15 min)")
    plt.legend();        plt.grid(True, alpha=0.3);  plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/transformer_site_{site_id}_timeseries.png", dpi=150)
    plt.close()

    # ── Plot 2: Scatter ──────────────────────────────────────────────────
    plt.figure(figsize=(6, 6))
    plt.scatter(results_df["actual"], results_df["predicted"],
                alpha=0.45, s=10)
    lo = min(results_df["actual"].min(), results_df["predicted"].min())
    hi = max(results_df["actual"].max(), results_df["predicted"].max())
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit (y = x)")
    plt.title(f"Site {site_id} — Scatter Plot (Transformer)")
    plt.xlabel("Actual Flow (veh / 15 min)")
    plt.ylabel("Predicted Flow (veh / 15 min)")
    plt.legend();  plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/transformer_site_{site_id}_scatter.png", dpi=150)
    plt.close()

    print(f"  Plots saved for site {site_id} in '{save_dir}/'")


def plot_loss_curve(site_id, loss_csv_path, save_dir="plots"):
    """Save train-vs-val loss curve from the saved CSV."""
    if not os.path.exists(loss_csv_path):
        return
    os.makedirs(save_dir, exist_ok=True)

    loss_df = pd.read_csv(loss_csv_path)
    plt.figure(figsize=(8, 4))
    plt.plot(loss_df["epoch"], loss_df["train_loss"],
             label="Train Loss", linewidth=1.5)
    plt.plot(loss_df["epoch"], loss_df["val_loss"],
             label="Validation Loss", linewidth=1.5, linestyle="--")
    plt.title(f"Site {site_id} — Transformer Training History")
    plt.xlabel("Epoch");  plt.ylabel("MSE Loss")
    plt.legend();         plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"{save_dir}/transformer_site_{site_id}_loss_curve.png", dpi=150
    )
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN  per-site loop
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Load data
    train, val, test = load_data()

    # 2. Normalise
    train, val, test, scaler = normalise(train, val, test)

    site_ids = sorted(train["scats_id"].unique())
    print(f"\nFound {len(site_ids)} unique sites. "
          f"Training one Transformer per site.\n")

    all_metrics = []

    for site_id in site_ids:
        print(f"\n{'>'*4} Site {site_id}")

        # Subset data for this site
        train_site = train[train["scats_id"] == site_id].copy()
        val_site   = val[val["scats_id"]   == site_id].copy()
        test_site  = test[test["scats_id"] == site_id].copy()

        # Build sequences
        X_tr, y_tr = build_sequences(train_site)
        X_vl, y_vl = build_sequences(val_site)
        X_ts, y_ts = build_sequences(test_site)

        if len(X_tr) == 0 or len(X_vl) == 0 or len(X_ts) == 0:
            print(f"  Skipping site {site_id} — insufficient data for sequences.")
            continue

        # Train
        train_model(X_tr, y_tr, X_vl, y_vl, site_id)

        # Evaluate
        metrics, preds_real, y_test_real = evaluate(X_ts, y_ts, scaler, site_id)
        metrics["site_id"] = site_id
        all_metrics.append(metrics)

        # Plots
        plot_predictions(site_id, test_site, preds_real, y_test_real, save_dir="plots/transformer")
        loss_csv = f"models/transformer_site_{site_id}_losses.csv"
        if os.path.exists(loss_csv):
            plot_loss_curve(site_id, loss_csv, save_dir="plots/transformer")

    # Summary
    if all_metrics:
        results_df = pd.DataFrame(all_metrics)
        out_path = "models/transformer_summary.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\n{'='*50}")
        print("Transformer — Summary across all sites")
        print(f"{'='*50}")
        print(results_df[["site_id", "MAE", "RMSE", "MAPE"]].to_string(index=False))
        print(f"\nMeans:\n{results_df[['MAE','RMSE','MAPE']].mean().round(4)}")
        print(f"\nSummary saved → {out_path}")
    else:
        print("\nNo sites were processed — check your data/ directory.")
