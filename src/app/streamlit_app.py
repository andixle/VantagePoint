import streamlit as st
import pandas as pd
from pathlib import Path

from src.ingest.prizepicks import load_current_offers
from src.ingest.vlr import load_player_maps, load_map_pool
from src.features.engine import last15_features, head_to_head_over_rate, map_mixture_expectation

st.set_page_config(page_title="VALORANT PrizePicks — MVP", layout="centered")

offers = load_current_offers()
pmap = load_player_maps()
mpool = load_map_pool()

st.title("VALORANT PrizePicks — Bet Card (Demo)")

offer = st.selectbox("Select an offer", offers['offer_id'].tolist())
off = offers[offers['offer_id']==offer].iloc[0]

st.subheader(f"{off['player']} — {off['stat_type'].title()} line {off['line']} vs {off['opponent']} ({off['team']})")

l15 = last15_features(pmap, off['player'])
h2h = head_to_head_over_rate(pmap, off['player'], off['opponent'], float(off['line']))
mp_row = mpool[(mpool['match_id'] == off['series_id']) & (mpool['team'] == off['team'])]
mix = map_mixture_expectation(pmap, off['player'], mp_row.squeeze() if not mp_row.empty else pd.Series({}))

c1, c2 = st.columns(2)
with c1:
    st.metric("Last-15 mean", f"{l15['mean']:.1f}" if pd.notna(l15['mean']) else "—")
    st.metric("Last-15 std", f"{l15['std']:.1f}" if pd.notna(l15['std']) else "—")
with c2:
    st.metric("H2H count", h2h['h2h_count'])
    st.metric("H2H over-rate", f"{(h2h['h2h_over_rate']*100):.0f}%" if pd.notna(h2h['h2h_over_rate']) else "—")

st.write("### Map mixture (demo)")
if 'per_map' in mix and len(mix['per_map']):
    df_map = pd.DataFrame([{'map': k, 'P(map)': v['p'], 'per-map μ (kills)': v['mu']} for k,v in mix['per_map'].items()])
    st.dataframe(df_map.sort_values('P(map)', ascending=False).reset_index(drop=True))
    st.info(f"Mixture expected kills: {mix['map_mixture_mu']:.2f}" if pd.notna(mix.get('map_mixture_mu')) else "Mixture μ unavailable")
else:
    st.write("No map pool row for this match/team in sample.")

st.caption("Demo uses synthetic sample data. Replace CSVs in /data with your own scraped data.")
