# Assignment 2B — Traffic-Based Route Guidance System (TBRGS)

## Setup

```bash
ccdd iai_assignment_2
python3 -m venv .venv
source .venv/bin/activate
pip install -r 2B/requirements.txt
```

Place raw data under `2B/src/data/raw/`:
- `Scats Data October 2006.xls`
- (optional) `Traffic_Count_Locations_with_LONG_LAT.csv`

## Workflow

From `2B/src`:

```bash
# 1. Process raw SCATS data
python data_processing.py

# 2. Train models
python lstm.py
python gru.py
python cnn.py

# 3. Compare models
python evaluate.py

# 4. Find routes (CLI)
python tbrgs.py -o 2000 -d 3002 -t "2006-10-27 08:00" -m lstm

# 5. GUI
python gui.py
```

## Configuration

Defaults are in `2B/config.yaml` (origin, destination, timestamp, model, top-k).

## Components

| Module | Purpose |
|--------|---------|
| `data_processing.py` | Extract & split SCATS October 2006 data |
| `lstm.py`, `gru.py`, `cnn.py` | Three ML predictors |
| `evaluate.py` | Test-set comparison + plots |
| `predict.py` | Runtime flow prediction |
| `travel_time.py` | Flow → speed → travel time |
| `graph.py` | Boroondara road network (NetworkX) |
| `route_search.py` | Top-k paths + Part A A* integration |
| `tbrgs.py` | CLI |
| `gui.py` | Tkinter GUI |
