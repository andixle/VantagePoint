"""VLR ingest stub (sample only).

`load_player_maps()` returns per-map player stats from sample CSV.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def load_player_maps() -> pd.DataFrame:
    fp = DATA_DIR / "sample_player_maps.csv"
    df = pd.read_csv(fp, parse_dates=['date'])
    return df

def load_map_pool() -> pd.DataFrame:
    fp = DATA_DIR / "sample_map_pool.csv"
    return pd.read_csv(fp)
