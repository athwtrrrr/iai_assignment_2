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
TRAIN_PATH  = "data/train.csv"
VAL_PATH    = "data/val.csv"
TEST_PATH   = "data/test.csv"
MODEL_PATH  = "models/gru_best.pth"
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

    # CSV saves datetime as string, convert back
    for df in [train, val, test]:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(f"Train: {len(train):,} rows | Val: {len(val):,} rows | Test: {len(test):,} rows")
    print(f"Sites: {train['scats_id'].nunique()}")
    print(f"Flow range: {train['flow_15min'].min()} – {train['flow_15min'].max()}")
    return train, val, test
# ─────────────────────────────────────────────
# STEP 2: Normalise
# ─────────────────────────────────────────────
def normalise(train, val, test):
    scaler = MinMaxScaler(feature_range=(0, 1))

    train["flow_scaled"] = scaler.fit_transform(train[["flow_15min"]])
    val["flow_scaled"]   = scaler.transform(val[["flow_15min"]])
    test["flow_scaled"]  = scaler.transform(test[["flow_15min"]])

    print(f"Scaled range — min: {train['flow_scaled'].min():.2f}  max: {train['flow_scaled'].max():.2f}")
    
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    return train, val, test, scaler
# ─────────────────────────────────────────────
# STEP 3: Build sliding windows
# ─────────────────────────────────────────────
def build_sequences(df, lags=LAGS):
    
    X, y = [], [] # Input, output

    for site_id, site_row in df.groupby("scats_id"):
        site_row = site_row.sort_values("timestamp")  # Sort rows in time order
        flow  = site_row["flow_scaled"].values # Extract flow values from sites'rows

        for i in range(lags, len(flow)):
            X.append(flow[i - lags:i])   # List of input windows (arrays)
            y.append(flow[i])           # List of target values

    X = np.array(X) # 2D (number of input windows, lags)
    y = np.array(y)

    # Reshape X to 3D: (samples, timesteps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)

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
            dropout     = 0.2
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape:   (batch, timesteps, features)
        out, _ = self.gru(x)
        # out shape: (batch, timesteps, hidden_size)
        # Take only the last timestep's output
        out = out[:, -1, :]
        # out shape: (batch, hidden_size)
        out = self.fc(out)
        # out shape: (batch, 1)
        return out
# ─────────────────────────────────────────────
# STEP 5: Train
# ─────────────────────────────────────────────
def train_model(X_train, y_train, X_val, y_val):
    os.makedirs("models", exist_ok=True)

    # Convert numpy arrays to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)  # shape: (samples, 1)
    X_val_t   = torch.FloatTensor(X_val)
    y_val_t   = torch.FloatTensor(y_val).unsqueeze(1)

    # DataLoader batches the data and shuffles each epoch
    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size = BATCH_SIZE,
        shuffle    = True
    )

    model     = GRUModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss    = float("inf")
    patience_counter = 0
    train_losses     = []
    val_losses       = []

    print("\nTraining GRU...")
    for epoch in range(EPOCHS):

        # ── Training phase ──
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()           # clear gradients from last step
            output = model(X_batch)         # forward pass
            loss   = criterion(output, y_batch)  # compute loss
            loss.backward()                 # backward pass — compute gradients
            optimizer.step()               # update weights
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(loader)
        train_losses.append(avg_train_loss)

        # ── Validation phase ──
        model.eval()
        with torch.no_grad():   # no gradient computation needed for validation
            val_loss = criterion(model(X_val_t), y_val_t).item()
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} — train loss: {avg_train_loss:.6f}  val loss: {val_loss:.6f}")

        # ── Early stopping ──
        # Save model when val loss improves, stop when it stops improving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1} — best val loss: {best_val_loss:.6f}")
                break

    print(f"  Model saved to {MODEL_PATH}")

    loss_df = pd.DataFrame({
        "epoch": list(range(1, len(train_losses) + 1)),
        "train_loss": train_losses,
        "val_loss": val_losses
    })
    csv_path = MODEL_PATH.replace(".pth", "_losses.csv")
    loss_df.to_csv(csv_path, index=False)
    print(f"  Loss history saved to {csv_path}")
    
    return train_losses, val_losses
# ─────────────────────────────────────────────
# STEP 6: Evaluate
# ─────────────────────────────────────────────
def evaluate(X_test, y_test, scaler):
    # Load best saved model
    model = GRUModel()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test)).numpy()

    # Convert 0-1 predictions back to real vehicle counts
    preds_real  = scaler.inverse_transform(preds)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae  = mean_absolute_error(y_test_real, preds_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, preds_real))
    mask = y_test_real.flatten() > 10   # only include rows where actual flow > 10
    mape = np.mean(
    np.abs((y_test_real[mask] - preds_real[mask]) / y_test_real[mask])
) * 100

    print(f"\nGRU Test Results:")
    print(f"  MAE:  {mae:.2f}  vehicles/15min")
    print(f"  RMSE: {rmse:.2f} vehicles/15min")
    print(f"  MAPE: {mape:.2f}%")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Run full pipeline
    train, val, test            = load_data()
    train, val, test, scaler    = normalise(train, val, test)
    X_train, y_train            = build_sequences(train)
    X_val,   y_val              = build_sequences(val)
    X_test,  y_test             = build_sequences(test)
    train_losses, val_losses    = train_model(X_train, y_train, X_val, y_val)
    metrics                     = evaluate(X_test, y_test, scaler)