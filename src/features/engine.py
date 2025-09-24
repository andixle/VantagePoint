from __future__ import annotations
import pandas as pd
import numpy as np

def last15_features(player_maps: pd.DataFrame, player: str, stat: str = 'kills') -> dict:
    dfp = player_maps[player_maps['player'] == player].copy()
    dfp = dfp.sort_values('date', ascending=False).head(15)
    if dfp.empty:
        return {'count': 0, 'mean': np.nan, 'median': np.nan, 'std': np.nan, 'over_rate': np.nan}
    out = {
        'count': len(dfp),
        'mean': dfp[stat].mean(),
        'median': dfp[stat].median(),
        'std': dfp[stat].std(ddof=1) if len(dfp) > 1 else 0.0,
    }
    return out

def head_to_head_over_rate(player_maps: pd.DataFrame, player: str, opponent: str, line: float, stat: str = 'kills', window: int = 10) -> dict:
    dfp = player_maps[(player_maps['player'] == player) & (player_maps['opponent'] == opponent)].copy()
    dfp = dfp.sort_values('date', ascending=False).head(window)
    if dfp.empty:
        return {'h2h_count': 0, 'h2h_over_rate': np.nan}
    over = (dfp[stat] > line).mean()
    return {'h2h_count': len(dfp), 'h2h_over_rate': over}

def map_mixture_expectation(player_maps: pd.DataFrame, player: str, map_pool_row: pd.Series, stat: str = 'kills') -> dict:
    # crude per-map means * map probabilities
    results = {}
    exp_val = 0.0
    total_p = 0.0
    for map_name, p in map_pool_row.items():
        if map_name in ('match_id', 'team'): 
            continue
        try:
            p = float(p)
        except Exception:
            continue
        if p <= 0: 
            continue
        m = player_maps[(player_maps['player']==player) & (player_maps['map_name']==map_name)]
        mu = m[stat].mean() if len(m) else np.nan
        results[map_name] = {'p': p, 'mu': mu}
        if not np.isnan(mu):
            exp_val += p * mu
            total_p += p
    if total_p == 0:
        return {'map_mixture_mu': np.nan, 'per_map': results}
    return {'map_mixture_mu': exp_val, 'per_map': results}
