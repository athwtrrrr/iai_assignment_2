import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from lstm        import LSTMModel
from gru         import GRUModel
from transformer import TransformerModel


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LAGS = 12

# Per-site model path templates
MODEL_PATH_TEMPLATES = {
    "lstm":        "models/lstm_site_{site_id}.pth",
    "gru":         "models/gru_site_{site_id}.pth",
    "transformer": "models/transformer_site_{site_id}.pth",
}


# ─────────────────────────────────────────────
# Load historical data
# ─────────────────────────────────────────────
_history = None

def _get_history():
    """
    Load all available flow data (train + val + test).
    Used to look up the last 12 readings before a given timestamp.
    Cached after first call.
    """
    global _history
    if _history is None:
        train = pd.read_csv("data/train.csv")
        val   = pd.read_csv("data/val.csv")
        test  = pd.read_csv("data/test.csv")
        _history = pd.concat([train, val, test], ignore_index=True)
        _history["timestamp"] = pd.to_datetime(_history["timestamp"])
    return _history


# ─────────────────────────────────────────────
# Scaler (must match training)
# ─────────────────────────────────────────────
_scaler = None

def _get_scaler():
    """
    Refit the global scaler on training data.
    Must match exactly what lstm.py / gru.py / transformer.py used.
    """
    global _scaler
    if _scaler is None:
        train    = pd.read_csv("data/train.csv")
        _scaler  = MinMaxScaler(feature_range=(0, 1))
        _scaler.fit(train[["flow_15min"]])
    return _scaler


# ─────────────────────────────────────────────
# Per-site model loader with cache
# ─────────────────────────────────────────────
_models = {}

def _get_model(model_name: str, site_id: int):
    """
    Load and cache the trained model for a specific site.

    Looks for: models/{model_name}_site_{site_id}.pth
    Raises FileNotFoundError with a helpful message if not found.

    Cache key is (model_name, site_id) so different sites get
    different model instances.
    """
    cache_key = f"{model_name}_{site_id}"

    if cache_key not in _models:
        model_name_lower = model_name.lower()

        if model_name_lower not in MODEL_PATH_TEMPLATES:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from: {list(MODEL_PATH_TEMPLATES.keys())}"
            )

        path = MODEL_PATH_TEMPLATES[model_name_lower].format(site_id=site_id)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model weights not found: {path}\n"
                f"  → Run '{model_name_lower}.py' first to train the model."
            )

        # Instantiate correct architecture
        if model_name_lower == "lstm":
            m = LSTMModel()
        elif model_name_lower == "gru":
            m = GRUModel()
        elif model_name_lower == "transformer":
            m = TransformerModel()

        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        _models[cache_key] = m

    return _models[cache_key]


# ─────────────────────────────────────────────
# Main prediction function
# ─────────────────────────────────────────────
def predict_flow(site_id, timestamp, model="lstm"):
    """
    Predict traffic flow for a given SCATS site at a given timestamp.

    Arguments
    ---------
    site_id   : int or str  — SCATS site number, e.g. 2000
    timestamp : str or datetime — e.g. "2006-10-27 08:00"
    model     : str — "lstm" | "gru" | "transformer"

    Returns
    -------
    float — predicted flow in vehicles per 15 minutes

    How it works
    ------------
    1. Look up the 12 most-recent flow readings before `timestamp`
       for this site from the historical dataset.
    2. Normalise those 12 values with the global scaler.
    3. Run the per-site model to get a scaled prediction.
    4. Inverse-transform back to vehicle count.
    """
    site_id   = int(site_id)
    timestamp = pd.Timestamp(timestamp)
    scaler    = _get_scaler()
    history   = _get_history()

    # ── Get last LAGS readings before the requested timestamp ────────────
    site_history = (
        history[
            (history["scats_id"]  == site_id) &
            (history["timestamp"] <  timestamp)
        ]
        .sort_values("timestamp")
        .tail(LAGS)
    )

    if len(site_history) < LAGS:
        raise ValueError(
            f"Not enough history for site {site_id} before {timestamp}. "
            f"Need {LAGS} readings, only found {len(site_history)}."
        )

    # ── Scale the LAGS input values ──────────────────────────────────────
    flow_values = site_history["flow_15min"].values.reshape(-1, 1)
    flow_scaled = scaler.transform(flow_values).flatten()

    # ── Build input tensor: (1, LAGS, 1) ────────────────────────────────
    X = torch.FloatTensor(flow_scaled).unsqueeze(0).unsqueeze(2)
    # unsqueeze(0) → (1, 12)   [batch dimension]
    # unsqueeze(2) → (1, 12, 1) [feature dimension]

    # ── Run per-site model ───────────────────────────────────────────────
    model_obj         = _get_model(model, site_id)
    with torch.no_grad():
        prediction_scaled = model_obj(X).item()

    # ── Inverse-transform to real vehicle count ──────────────────────────
    prediction = scaler.inverse_transform([[prediction_scaled]])[0][0]

    return float(prediction)


# ─────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    TEST_SITE      = 2000
    TEST_TIMESTAMP = "2006-10-27 08:00"

    for model_name in ["lstm", "gru", "transformer"]:
        try:
            flow = predict_flow(
                site_id   = TEST_SITE,
                timestamp = TEST_TIMESTAMP,
                model     = model_name,
            )
            print(f"{model_name.upper():>12} predicted: {flow:.1f} vehicles/15min")
        except FileNotFoundError as e:
            print(f"{model_name.upper():>12} — model not found: {e}")
