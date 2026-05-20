import os
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# Load raw data 
# ─────────────────────────────────────────────
XLS_PATH = "data/raw/Scats Data October 2006.xls"

def load_raw_data(xls_path):

    # Use header=None to read all rows, then assign column names manually
    df_raw = pd.read_excel(xls_path, sheet_name="Data", header=None, engine="xlrd")
    
    # Row 1 contains column names
    col_names = df_raw.iloc[1, :].tolist()
    # Data starts from row 2
    df = df_raw.iloc[2:].copy()
    df.columns = col_names
    
    print(f"  Raw rows loaded: {len(df)}")
    print(f"  Unique SCATS sites: {df['SCATS Number'].nunique()}")
    return df

# ─────────────────────────────────────────────
# Extract site metadata (coords, names)
# ─────────────────────────────────────────────
def extract_site_info(df):

    # Change 0 in long, lat to NaN
    df["NB_LATITUDE"]  = pd.to_numeric(df["NB_LATITUDE"],  errors="coerce").replace(0, np.nan)
    df["NB_LONGITUDE"] = pd.to_numeric(df["NB_LONGITUDE"], errors="coerce").replace(0, np.nan)

    # Remove NaN value rows, get average long, lat, first road name for each site
    site_info = (
        df.dropna(subset=["NB_LATITUDE", "NB_LONGITUDE"])
        .groupby("SCATS Number", as_index=False)
        .agg(lat=("NB_LATITUDE", "mean"),
             lon=("NB_LONGITUDE", "mean"),
             location=("Location", "first"))
    )
    site_info = site_info.rename(columns={"SCATS Number": "scats_id"})
    print(f"  Sites with valid coordinates: {len(site_info)}")
    return site_info

# ─────────────────────────────────────────────
# Melt wide -> long, aggregate directions
# ─────────────────────────────────────────────
V_COLS = [f"V{i:02d}" for i in range(96)]  # V00..V95 = 96 x 15-min slots

def build_long_format(df):

    # Remove stray time keep date only
    df["date"] = pd.to_datetime(df["Date"]).dt.normalize()

    # Melt: one row per site x date x interval
    id_cols = ["SCATS Number", "date"]
    df_long = df[id_cols + V_COLS].melt(
        id_vars=id_cols,
        value_vars=V_COLS,
        var_name="interval",
        value_name="flow_15min"
    )

    # Convert interval label (V00..V95) to actual timestamp
    df_long["interval_num"] = df_long["interval"].str[1:].astype(int)
    df_long["timestamp"] = df_long["date"] + pd.to_timedelta(df_long["interval_num"] * 15, unit="min")

    # Sum all detector directions per site per 15-min slot
    df_agg = (
        df_long
        .groupby(["SCATS Number", "timestamp"], as_index=False)["flow_15min"]
        .sum()
    )

    df_agg = df_agg.rename(columns={"SCATS Number": "scats_id"})
    print(f"  Long format rows: {len(df_agg)}")
    return df_agg


# ─────────────────────────────────────────────
# Merge coords, add time features
# ─────────────────────────────────────────────
def enrich_and_clean(df_long, site_info):
    df = df_long.merge(site_info[["scats_id", "lat", "lon", "location"]], on="scats_id", how="left")

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"]  = df["day_of_week"] >= 5

    df["hour_bucket"] = df["timestamp"].dt.floor("h")
    df_hourly = (
        df.groupby(["scats_id", "hour_bucket", "lat", "lon", "location"], as_index=False)
        .agg(flow_hourly=("flow_15min", "sum"),
             day_of_week=("day_of_week", "first"),
             is_weekend=("is_weekend", "first"))
        .rename(columns={"hour_bucket": "timestamp"})
    )
    df_hourly["hour_of_day"] = df_hourly["timestamp"].dt.hour

    
    df = df.drop(columns=["hour_bucket"])

    return df, df_hourly

# ─────────────────────────────────────────────
# STEP 5: Train / test split
# ─────────────────────────────────────────────
TRAIN_END = "2006-10-22"   # Oct 1-21  = train (21 days, 70%)
VAL_END   = "2006-10-27"   # Oct 22-26 = val   (5 days,  16%)
                            # Oct 27-31 = test  (5 days,  16%)

def split_data(df, timestamp_col="timestamp"):
    train_end = pd.Timestamp(TRAIN_END)
    val_end   = pd.Timestamp(VAL_END)

    train = df[df[timestamp_col] <  train_end].copy()
    val   = df[(df[timestamp_col] >= train_end) & (df[timestamp_col] < val_end)].copy()
    test  = df[df[timestamp_col] >= val_end].copy()

    print(f"  Train: {len(train):,} rows ({train[timestamp_col].min().date()} to {train[timestamp_col].max().date()})")
    print(f"  Val:   {len(val):,} rows ({val[timestamp_col].min().date()} to {val[timestamp_col].max().date()})")
    print(f"  Test:  {len(test):,} rows ({test[timestamp_col].min().date()} to {test[timestamp_col].max().date()})")

    return train, val, test

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    df_raw              = load_raw_data(XLS_PATH)
    site_info           = extract_site_info(df_raw)
    df_long             = build_long_format(df_raw)
    df_clean, df_hourly = enrich_and_clean(df_long, site_info)

    print("\nSplitting 15-min data...")
    train_15, val_15, test_15 = split_data(df_clean, "timestamp")

    print("\nSplitting hourly data...")
    train_h, val_h, test_h = split_data(df_hourly, "timestamp")

    # Save all outputs
    site_info.to_csv( "data/site_info.csv",       index=False)
    df_clean.to_csv(  "data/scats_clean.csv",      index=False)
    df_hourly.to_csv( "data/scats_hourly.csv",     index=False)
    train_15.to_csv(  "data/train.csv",            index=False)
    val_15.to_csv(    "data/val.csv",              index=False)
    test_15.to_csv(   "data/test.csv",             index=False)
    train_h.to_csv(   "data/train_hourly.csv",     index=False)
    val_h.to_csv(     "data/val_hourly.csv",       index=False)
    test_h.to_csv(    "data/test_hourly.csv",      index=False)

    print(f"""
Files saved to data/
  site_info.csv       - {len(site_info)} sites with coordinates
  scats_clean.csv     - {len(df_clean):,} rows (15-min, all sites)
  scats_hourly.csv    - {len(df_hourly):,} rows (hourly, all sites)
  train.csv           - {len(train_15):,} rows (Oct 1-21)
  val.csv             - {len(val_15):,} rows (Oct 22-26)
  test.csv            - {len(test_15):,} rows (Oct 27-31)
  train_hourly.csv    - {len(train_h):,} rows (Oct 1-21, hourly)
  val_hourly.csv      - {len(val_h):,} rows (Oct 22-26, hourly)
  test_hourly.csv     - {len(test_h):,} rows (Oct 27-31, hourly)
    """)