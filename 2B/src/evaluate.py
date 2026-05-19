"""
Compare LSTM, GRU, and 1D CNN on the held-out test set.
Saves metrics and comparison plots for the report.
"""
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from lstm import LSTMModel, build_sequences as lstm_seq, LAGS, load_data, normalise
from gru import GRUModel
from cnn import CNNModel

MODELS = {
    "lstm": (LSTMModel, "models/lstm_best.pth"),
    "gru": (GRUModel, "models/gru_best.pth"),
    "cnn": (CNNModel, "models/cnn_best.pth"),
}


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true.flatten() > 10
    mape = (
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        if mask.any()
        else float("nan")
    )
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def evaluate_all():
    os.makedirs("models", exist_ok=True)
    train, val, test = load_data()
    train, val, test, scaler = normalise(train, val, test)
    X_test, y_test = lstm_seq(test)

    y_true = scaler.inverse_transform(y_test.reshape(-1, 1))
    rows = []

    for name, (cls, path) in MODELS.items():
        if not os.path.exists(path):
            print(f"  Skipping {name.upper()} — train with {name}.py first")
            continue
        model = cls()
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_test)).numpy()
        preds_real = scaler.inverse_transform(preds)
        m = metrics(y_true, preds_real)
        m["model"] = name.upper()
        rows.append(m)
        print(f"{name.upper()}: MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  MAPE={m['MAPE']:.2f}%")

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df.to_csv("models/model_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, ["MAE", "RMSE", "MAPE"]):
        ax.bar(df["model"], df[metric], color=["#4C72B0", "#55A868", "#C44E52"])
        ax.set_title(metric)
        ax.set_ylabel("vehicles/15min" if metric != "MAPE" else "%")
    plt.tight_layout()
    plt.savefig("models/model_comparison.png", dpi=150)
    print("\nSaved models/model_comparison.csv and models/model_comparison.png")
    return df


if __name__ == "__main__":
    evaluate_all()
