"""
1D CNN model for traffic flow prediction (third ML technique).
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

TRAIN_PATH = "data/train.csv"
VAL_PATH = "data/val.csv"
TEST_PATH = "data/test.csv"
MODEL_PATH = "models/cnn_best.pth"
LAGS = 12
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 10
LR = 0.001


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)
    test = pd.read_csv(TEST_PATH)
    for df in [train, val, test]:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return train, val, test


def normalise(train, val, test):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = train.copy()
    val = val.copy()
    test = test.copy()
    train["flow_scaled"] = scaler.fit_transform(train[["flow_15min"]])
    val["flow_scaled"] = scaler.transform(val[["flow_15min"]])
    test["flow_scaled"] = scaler.transform(test[["flow_15min"]])
    return train, val, test, scaler


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
    return X, y


class CNNModel(nn.Module):
    def __init__(self, lags=LAGS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq, 1) -> (batch, 1, seq)
        x = x.transpose(1, 2)
        x = self.conv(x)
        return self.fc(x)


def train_model(X_train, y_train, X_val, y_val):
    os.makedirs("models", exist_ok=True)
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True
    )
    model = CNNModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    print("\nTraining 1D CNN...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(loader))

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1:3d}/{EPOCHS} — train: {train_losses[-1]:.6f}  val: {val_loss:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1} — best val: {best_val_loss:.6f}")
                break

    pd.DataFrame(
        {"epoch": range(1, len(train_losses) + 1), "train_loss": train_losses, "val_loss": val_losses}
    ).to_csv(MODEL_PATH.replace(".pth", "_losses.csv"), index=False)
    return train_losses, val_losses


def evaluate(X_test, y_test, scaler):
    model = CNNModel()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test)).numpy()
    preds_real = scaler.inverse_transform(preds)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
    mae = mean_absolute_error(y_test_real, preds_real)
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


if __name__ == "__main__":
    train, val, test = load_data()
    train, val, test, scaler = normalise(train, val, test)
    X_train, y_train = build_sequences(train)
    X_val, y_val = build_sequences(val)
    X_test, y_test = build_sequences(test)
    train_model(X_train, y_train, X_val, y_val)
    evaluate(X_test, y_test, scaler)
