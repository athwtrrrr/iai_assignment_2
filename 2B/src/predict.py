import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from lstm import LSTMModel
from gru import GRUModel
from cnn import CNNModel


LAGS = 12
MODEL_PATHS = {
    "lstm": "models/lstm_best.pth",
    "gru": "models/gru_best.pth",
    "cnn": "models/cnn_best.pth",
}

# ─────────────────────────────────────────────
# Load historical data (needed to build input sequence)
# ─────────────────────────────────────────────
# Load once at module level — not inside the function
# so it doesn't reload from disk on every prediction call
_history = None

def _get_history():
    """
    Load all available flow data (train + val + test).
    Used to look up the last 12 readings before a given timestamp.
    """
    global _history
    if _history is None:
        train = pd.read_csv("data/train.csv")
        val   = pd.read_csv("data/val.csv")
        test  = pd.read_csv("data/test.csv")
        _history = pd.concat([train, val, test])
        _history["timestamp"] = pd.to_datetime(_history["timestamp"])
    return _history


# ─────────────────────────────────────────────
# Fit scaler (must match exactly what was used in training)
# ─────────────────────────────────────────────
_scaler = None

def _get_scaler():
    """
    Refit the scaler on training data.
    Must produce identical scaling to what lstm.py used during training.
    """
    global _scaler
    if _scaler is None:
        train = pd.read_csv("data/train.csv")
        _scaler = MinMaxScaler(feature_range=(0, 1))
        _scaler.fit(train[["flow_15min"]])
    return _scaler


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
_models = {}

def _get_model(model_name):
    """
    Load and cache model. Only loads from disk once per model type.
    """
    if model_name not in _models:
        factories = {"lstm": LSTMModel, "gru": GRUModel, "cnn": CNNModel}
        if model_name not in factories:
            raise ValueError(f"Unknown model: {model_name}. Choose lstm, gru, or cnn.")
        m = factories[model_name]()

        m.load_state_dict(torch.load(MODEL_PATHS[model_name], weights_only=True))
        m.eval()
        _models[model_name] = m

    return _models[model_name]


# ─────────────────────────────────────────────
# Main prediction function
# ─────────────────────────────────────────────
def predict_flow(site_id, timestamp, model="lstm"):
    """
    Predict traffic flow for a given site and timestamp.

    Arguments:
        site_id   : int or str  — SCATS site number e.g. 2000
        timestamp : str or datetime — e.g. "2006-10-15 08:00"
        model     : str — "lstm", "gru", or "cnn"

    Returns:
        float — predicted flow in vehicles per 15 minutes
    """
    site_id   = int(site_id)
    timestamp = pd.Timestamp(timestamp)
    scaler    = _get_scaler()
    history   = _get_history()

    # ── Get last 12 readings before the requested timestamp ──
    site_history = (
        history[
            (history["scats_id"]  == site_id) &
            (history["timestamp"] <  timestamp)
        ]
        .sort_values("timestamp")
        .tail(LAGS)   # last 12 readings
    )

    if len(site_history) < LAGS:
        raise ValueError(
            f"Not enough history for site {site_id} before {timestamp}. "
            f"Need {LAGS} readings, only found {len(site_history)}."
        )

    # ── Scale the 12 input values ──
    flow_values = site_history["flow_15min"].values.reshape(-1, 1)
    flow_df     = pd.DataFrame(flow_values, columns=["flow_15min"])
    flow_scaled = scaler.transform(flow_df).flatten()

    # ── Build input tensor: shape (1, 12, 1) ──
    # 1 = one prediction at a time
    # 12 = LAGS timesteps
    # 1 = one feature (flow)
    X = torch.FloatTensor(flow_scaled).unsqueeze(0).unsqueeze(2)
    # unsqueeze(0) adds batch dimension:   (12,) → (1, 12)
    # unsqueeze(2) adds feature dimension: (1, 12) → (1, 12, 1)

    # ── Run model ──
    model_obj = _get_model(model)
    with torch.no_grad():
        prediction_scaled = model_obj(X).item()

    # ── Inverse transform back to real vehicles/15min ──
    pred_df     = pd.DataFrame([[prediction_scaled]], columns=["flow_15min"])
    prediction  = scaler.inverse_transform(pred_df)[0][0]

    return float(prediction)


# ─────────────────────────────────────────────
# Test it manually
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Quick test — predict flow for site 2000 at 8am on Oct 27
    flow = predict_flow(
        site_id   = 3812,
        timestamp = "2006-10-26 01:00",
        model     = "lstm"
    )
    print(f"LSTM predicted: {flow:.1f} vehicles/15min")

    # Test GRU too
    flow_gru = predict_flow(
        site_id   = 3812,
        timestamp = "2006-10-26 01:00",
        model     = "gru"
    )
    print(f"GRU predicted:  {flow_gru:.1f} vehicles/15min")
