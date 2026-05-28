import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRAIN_PATH  = "data/train.csv"
VAL_PATH    = "data/val.csv"
TEST_PATH   = "data/test.csv"
LAGS        = 12      # 12 x 15min = 3 hours
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
BATCH_SIZE  = 32
EPOCHS      = 100
PATIENCE    = 10
LR          = 0.001


# ─────────────────────────────────────────────
# STEP 1: Load data
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
# STEP 2: Normalise (global scaler)
# ─────────────────────────────────────────────
def normalise(train, val, test):
    scaler = MinMaxScaler(feature_range=(0, 1))

    train["flow_scaled"] = scaler.fit_transform(train[["flow_15min"]])
    val["flow_scaled"]   = scaler.transform(val[["flow_15min"]])
    test["flow_scaled"]  = scaler.transform(test[["flow_15min"]])

    print(f"Scaled range — min: {train['flow_scaled'].min():.2f}  max: {train['flow_scaled'].max():.2f}")
    return train, val, test, scaler


# ─────────────────────────────────────────────
# STEP 3: Build sliding windows
# ─────────────────────────────────────────────
def build_sequences(df, lags=LAGS):
    X, y = [], []

    for site_id, site_row in df.groupby("scats_id"):
        site_row = site_row.sort_values("timestamp")
        flow     = site_row["flow_scaled"].values

        for i in range(lags, len(flow)):
            X.append(flow[i - lags : i])
            y.append(flow[i])

    if len(X) == 0:
        return np.empty((0, lags, 1)), np.empty(0)

    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)   # (N, lags, 1)

    print(f"X shape: {X.shape}  |  y shape: {y.shape}")
    return X, y


# ─────────────────────────────────────────────
# STEP 4: Model
# ─────────────────────────────────────────────
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super(GRUModel, self).__init__()

        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.2,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, timesteps, features)
        out, _ = self.gru(x)
        out    = out[:, -1, :]   # take last timestep
        return self.fc(out)


# ─────────────────────────────────────────────
# STEP 5: Train (per site)
# ─────────────────────────────────────────────
def train_model(X_train, y_train, X_val, y_val, site_id):
    os.makedirs("models", exist_ok=True)
    model_path = f"models/gru_site_{site_id}.pth"

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t   = torch.FloatTensor(X_val)
    y_val_t   = torch.FloatTensor(y_val).unsqueeze(1)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size = BATCH_SIZE,
        shuffle    = True,
    )

    model     = GRUModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss    = float("inf")
    patience_counter = 0
    train_losses     = []
    val_losses       = []

    print(f"\nTraining GRU for site {site_id}...")

    for epoch in range(EPOCHS):
        # ── Training phase ──────────────────────────────────────────────
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss   = criterion(output, y_batch)
            loss.backward()
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
            print(f"  Site {site_id} | Epoch {epoch+1:3d}/{EPOCHS} — "
                  f"train loss: {avg_train_loss:.6f}  val loss: {val_loss:.6f}")

        # ── Early stopping ───────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Site {site_id} early stopping at epoch {epoch+1} — "
                      f"best val loss: {best_val_loss:.6f}")
                break

    # Save loss history CSV
    loss_df  = pd.DataFrame({
        "epoch":      list(range(1, len(train_losses) + 1)),
        "train_loss": train_losses,
        "val_loss":   val_losses,
    })
    csv_path = model_path.replace(".pth", "_losses.csv")
    loss_df.to_csv(csv_path, index=False)
    print(f"  Model saved to {model_path}")
    print(f"  Loss history saved to {csv_path}")

    return train_losses, val_losses


# ─────────────────────────────────────────────
# STEP 6: Evaluate (returns metrics + predictions)
# ─────────────────────────────────────────────
def evaluate(X_test, y_test, scaler, site_id):
    model_path = f"models/gru_site_{site_id}.pth"
    model      = GRUModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
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
        if np.any(mask) else float("nan")
    )

    print(f"  Site {site_id} | MAE: {mae:.2f}  RMSE: {rmse:.2f}  MAPE: {mape:.2f}%")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}, preds_real, y_test_real


# ─────────────────────────────────────────────
# STEP 7: Plotting functions
# ─────────────────────────────────────────────
def plot_predictions(site_id, test_site, preds_real, y_test_real,
                     lags=LAGS, save_dir="plots"):
    """Time-series overlay and scatter plot for one site."""
    os.makedirs(save_dir, exist_ok=True)

    test_sorted = test_site.sort_values("timestamp")
    timestamps  = test_sorted["timestamp"].iloc[lags:].reset_index(drop=True)

    n = min(len(timestamps), len(preds_real))
    results_df = pd.DataFrame({
        "timestamp": timestamps[:n],
        "actual":    y_test_real.flatten()[:n],
        "predicted": preds_real.flatten()[:n],
    })

    # ── Time-series plot ─────────────────────────────────────────────────
    plt.figure(figsize=(12, 5))
    plt.plot(results_df["timestamp"], results_df["actual"],
             label="Actual",    alpha=0.7, linewidth=1)
    plt.plot(results_df["timestamp"], results_df["predicted"],
             label="Predicted", alpha=0.7, linewidth=1)
    plt.title(f"Site {site_id} – Actual vs Predicted Traffic Flow (GRU, Test Set)")
    plt.xlabel("Time");  plt.ylabel("Flow (vehicles/15min)")
    plt.legend();        plt.grid(True, alpha=0.3);  plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/gru_site_{site_id}_timeseries.png", dpi=150)
    plt.close()

    # ── Scatter plot ─────────────────────────────────────────────────────
    plt.figure(figsize=(6, 6))
    plt.scatter(results_df["actual"], results_df["predicted"], alpha=0.5, s=10)
    max_val = max(results_df["actual"].max(), results_df["predicted"].max())
    min_val = min(results_df["actual"].min(), results_df["predicted"].min())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect fit")
    plt.title(f"Site {site_id} – Predicted vs Actual (GRU)")
    plt.xlabel("Actual Flow");   plt.ylabel("Predicted Flow")
    plt.legend();                plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/gru_site_{site_id}_scatter.png", dpi=150)
    plt.close()

    print(f"  Plots saved for site {site_id} in '{save_dir}/'")


def plot_loss_curve(site_id, loss_csv_path, save_dir="plots"):
    """Train vs validation loss curve for one site."""
    if not os.path.exists(loss_csv_path):
        return
    os.makedirs(save_dir, exist_ok=True)

    loss_df = pd.read_csv(loss_csv_path)
    plt.figure(figsize=(8, 4))
    plt.plot(loss_df["epoch"], loss_df["train_loss"], label="Train Loss")
    plt.plot(loss_df["epoch"], loss_df["val_loss"],   label="Validation Loss")
    plt.title(f"Site {site_id} – GRU Training History")
    plt.xlabel("Epoch");  plt.ylabel("Loss (MSE)")
    plt.legend();         plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/gru_site_{site_id}_loss_curve.png", dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# MAIN – per-site training, evaluation, plotting
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train, val, test         = load_data()
    train, val, test, scaler = normalise(train, val, test)

    site_ids = sorted(train["scats_id"].unique())
    print(f"\nFound {len(site_ids)} unique sites. Training one GRU per site.\n")

    all_metrics = []

    for site_id in site_ids:
        print(f"\n>>> Processing site {site_id}")

        train_site = train[train["scats_id"] == site_id].copy()
        val_site   = val[val["scats_id"]     == site_id].copy()
        test_site  = test[test["scats_id"]   == site_id].copy()

        X_tr, y_tr = build_sequences(train_site, lags=LAGS)
        X_vl, y_vl = build_sequences(val_site,   lags=LAGS)
        X_ts, y_ts = build_sequences(test_site,  lags=LAGS)

        if len(X_tr) == 0 or len(X_vl) == 0 or len(X_ts) == 0:
            print(f"  Skipping site {site_id} – insufficient data for sequences")
            continue

        # Train
        train_model(X_tr, y_tr, X_vl, y_vl, site_id)

        # Evaluate
        metrics, preds_real, y_test_real = evaluate(X_ts, y_ts, scaler, site_id)
        metrics["site_id"] = site_id
        all_metrics.append(metrics)

        # Plot predictions
        plot_predictions(site_id, test_site, preds_real, y_test_real, lags=LAGS, save_dir="plots/gru")

        # Plot loss curve
        loss_csv = f"models/gru_site_{site_id}_losses.csv"
        if os.path.exists(loss_csv):
            plot_loss_curve(site_id, loss_csv, save_dir="plots/gru")

    # Save summary
    if all_metrics:
        results_df = pd.DataFrame(all_metrics)
        results_df.to_csv("models/gru_per_site_summary.csv", index=False)
        print("\n\n===== GRU Per-site Summary =====")
        print(results_df[["site_id", "MAE", "RMSE", "MAPE"]].to_string(index=False))
        print(f"\nMeans:\n{results_df[['MAE','RMSE','MAPE']].mean().round(4)}")
    else:
        print("\nNo site had enough data to train a model.")
