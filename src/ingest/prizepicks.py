"""PrizePicks ingest stub.

Replace `load_current_offers()` with a real fetcher if/when you have legitimate access.
For now, we load sample_offers from data/sample_offers.csv
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def load_current_offers() -> pd.DataFrame:
    fp = DATA_DIR / "sample_offers.csv"
    df = pd.read_csv(fp)
    # normalize types
    df['offer_time'] = pd.to_datetime(df['offer_time'], utc=True)
    return df
