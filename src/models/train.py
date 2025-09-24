from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from pathlib import Path

from src.ingest.prizepicks import load_current_offers
from src.ingest.vlr import load_player_maps, load_map_pool
from src.features.engine import last15_features, head_to_head_over_rate, map_mixture_expectation

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SILVER_DIR = DATA_DIR / "silver"
SILVER_DIR.mkdir(parents=True, exist_ok=True)

def build_training_frame() -> pd.DataFrame:
    offers = load_current_offers()
    pmap = load_player_maps()
    mpool = load_map_pool()
    rows = []
    for _, off in offers.iterrows():
        player = off['player']
        opponent = off['opponent']
        line = float(off['line'])
        # last-15
        l15 = last15_features(pmap, player)
        # h2h (against opponent in sample set)
        h2h = head_to_head_over_rate(pmap, player, opponent, line)
        # map mixture using player's team row from mpool (if any)
        mp_row = mpool[(mpool['match_id'] == off['series_id']) & (mpool['team'] == off['team'])]
        mix = map_mixture_expectation(pmap, player, mp_row.squeeze() if not mp_row.empty else pd.Series({}))
        rows.append({
            'player': player,
            'opponent': opponent,
            'line': line,
            'l15_mean': l15['mean'],
            'l15_std': l15['std'],
            'h2h_over_rate': h2h['h2h_over_rate'],
            'map_mix_mu': mix.get('map_mixture_mu', np.nan),
            # label: in demo, synthesize with l15_mean vs line (do NOT do this in real life)
            'y_over': 1 if (pd.notna(l15['mean']) and l15['mean'] > line) else 0
        })
    return pd.DataFrame(rows)

def train_baseline():
    df = build_training_frame().dropna(subset=['l15_mean','line'])
    if len(df) < 2:
        print("Not enough data to train. Add more sample rows.")
        return
    X = df[['l15_mean','l15_std','map_mix_mu']].fillna(df[['l15_mean','l15_std','map_mix_mu']].mean())
    y = df['y_over'].astype(int)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    # crude eval on the same data (for demo only)
    p = model.predict_proba(X)[:,1]
    print("LogLoss:", log_loss(y, p))
    print("Brier:", brier_score_loss(y, p))
    out = {'coef': model.coef_.tolist(), 'intercept': model.intercept_.tolist()}
    (SILVER_DIR / "baseline_model.json").write_text(json.dumps(out, indent=2))
    (SILVER_DIR / "training_frame.csv").write_text(df.to_csv(index=False))
    print("Saved model to", SILVER_DIR / "baseline_model.json")

if __name__ == "__main__":
    train_baseline()
