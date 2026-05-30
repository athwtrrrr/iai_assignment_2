# config.py
import os
import yaml

# Look for config.yaml in the project root (parent directory of src/)
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

def load_config():
    """Load YAML config; return dict with built‑in defaults if file missing."""
    defaults = {
        "data": {
            "train_end_date": "2006-10-22",
            "val_end_date": "2006-10-27"
        },
        "models": {
            "lags": 12,
            "batch_size": 32,
            "epochs": 100,
            "patience": 10,
            "learning_rate": 0.001,
            "lstm": {
                "hidden_size": 64,
                "num_layers": 2
            },
            "gru": {
                "hidden_size": 64,
                "num_layers": 2
            },
            "transformer": {
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 128,
                "dropout": 0.1
            }
        },
        "routing": {
            "max_neighbours": 3,
            "k_paths": 5,
            "fallback_flow": 300.0
        },
        "travel_time": {
            "speed_limit_kmh": 60,
            "capacity_flow_vph": 1500,
            "capacity_speed_kmh": 32,
            "intersection_delay_sec": 30
        },
        "graph": {
            "lat_offset": -0.0011,
            "lon_offset": 0.0010
        },
        "gui": {
            "default_zoom": 14,
            "window_width": 1380,
            "window_height": 820,
            "left_panel_width": 370,
            "route_colors": ["red", "blue", "green", "purple", "orange"],
            "route_hex": {
                "red": "#ff1900",
                "blue": "#56b3f1",
                "green": "#00c351",
                "purple": "#984fb7",
                "orange": "#ff973c"
            },
            "map_tiles": "OpenStreetMap"
        }
    }
    if not os.path.exists(CONFIG_PATH):
        print(f"[Config] Warning: {CONFIG_PATH} not found. Using built‑in defaults.")
        return defaults
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    # Merge missing keys (keep defaults for anything not in YAML)
    for key, val in defaults.items():
        cfg.setdefault(key, val)
    return cfg

cfg = load_config()