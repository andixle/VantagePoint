from __future__ import annotations
import time
from typing import Dict, Any, List, Optional
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

PP_BASE = "https://api.prizepicks.com"
# You can set this to None and filter by league name later if IDs shift.
VALORANT_LEAGUE_IDS = {  # Keep flexible; add/remove as you learn IDs.
    # Example known IDs (may change). Leave empty set to fetch all then filter by name.
}
# How many records per page; PP often caps around 500.
PER_PAGE = 500
TIMEOUT = 15

# Light UA helps some CDNs; do not hammer.
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (educational DS project; contact: youremail@example.com)"
}

def _get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def _paginate(path: str, base_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Paginate PrizePicks JSON: returns concatenated `data` arrays."""
    page = 1
    out: List[Dict[str, Any]] = []
    while True:
        params = dict(base_params)
        params["page"] = page
        j = _get(f"{PP_BASE}/{path}", params)
        data = j.get("data", [])
        if not data:
            break
        out.extend(data)
        # prizepicks pagination style varies; stop if fewer than per_page
        if len(data) < base_params.get("per_page", PER_PAGE):
            break
        page += 1
        time.sleep(0.8)  # gentle pause
    return out

def _index_included(included: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx = {}
    for obj in included or []:
        key = f"{obj.get('type')}:{obj.get('id')}"
        idx[key] = obj
    return idx

def fetch_raw_projections(league_id: Optional[int] = None, single_stat: bool = True) -> Dict[str, Any]:
    """
    Returns the full JSON (data + included) for projections (all pages).
    If league_id is None, fetches all leagues; you'll filter by league name downstream.
    """
    params = {"per_page": PER_PAGE}
    if single_stat:
        params["single_stat"] = "true"
    if league_id is not None:
        params["league_id"] = league_id
    # First page to capture `included`
    first = _get(f"{PP_BASE}/projections", params)
    included = first.get("included", [])
    data = first.get("data", [])
    # Additional pages
    more = _paginate("projections", params)
    if more:
        data = more  # already includes page 1; we re-used paginate for all
        # some responses don’t repeat `included` every page; keep the first
    return {"data": data, "included": included}

def _is_valorant_league(obj: Dict[str, Any]) -> bool:
    name = obj.get("attributes", {}).get("name", "").lower()
    return "valorant" in name or name.strip().upper() in {"VAL", "VALORANT"}

def _extract_offer_rows(j: Dict[str, Any]) -> pd.DataFrame:
    data = j.get("data", [])
    incl = _index_included(j.get("included", []))

    rows = []
    for d in data:
        attrs = d.get("attributes", {})
        rel = d.get("relationships", {})

        # Identify league via relationships -> league -> data -> id
        league_rel = rel.get("league", {}).get("data")
        league_ok = True
        if league_rel:
            league_obj = incl.get(f"leagues:{league_rel.get('id')}")
            if league_obj:
                league_ok = _is_valorant_league(league_obj)
        # If we specified VALORANT ids elsewhere, you could intersect here.

        if not league_ok:
            continue

        # Player
        player_id = rel.get("new_player", {}).get("data", {}) or rel.get("player", {}).get("data", {})
        player_name = None
        team_abbrev = None
        if player_id:
            pobj = incl.get(f"new_players:{player_id.get('id')}") or incl.get(f"players:{player_id.get('id')}")
            if pobj:
                pattr = pobj.get("attributes", {})
                player_name = pattr.get("name") or pattr.get("display_name")
                team_abbrev = pattr.get("team", {}).get("abbr") if isinstance(pattr.get("team"), dict) else pattr.get("team")

        # Opponent & game
        opponent = attrs.get("opponent", None)  # sometimes null; PP doesn’t always fill for esports pre-slate
        game_rel = rel.get("game", {}).get("data", {})
        series_id = game_rel.get("id")

        # Stat line
        stat_type = attrs.get("stat_type", attrs.get("stat_name", ""))  # e.g., "kills", "fantasy_score"
        line = attrs.get("line_score") or attrs.get("line") or attrs.get("value")

        # Timing
        offer_time = attrs.get("board_time") or attrs.get("start_time") or attrs.get("updated_at")
        if offer_time:
            try:
                offer_time = datetime.fromisoformat(offer_time.replace("Z", "+00:00"))
            except Exception:
                offer_time = None

        # Map scope ( PrizePicks may distinguish per map vs per match in stat_type names or via markets )
        map_scope = "per-map" if isinstance(stat_type, str) and "map" in stat_type.lower() else "per-series"

        if player_name and stat_type and line is not None:
            rows.append({
                "offer_id": d.get("id"),
                "player": player_name,
                "stat_type": str(stat_type).lower(),
                "line": float(line),
                "team": team_abbrev,
                "opponent": opponent,
                "series_id": series_id,
                "map_scope": map_scope,
                "offer_time": offer_time if isinstance(offer_time, datetime) else None,
            })

    return pd.DataFrame(rows)

def load_current_offers_valorant() -> pd.DataFrame:
    """
    Public function: fetch current PrizePicks VALORANT projections -> normalized offers DF.
    """
    if VALORANT_LEAGUE_IDS:
        # Try each known league id (collect all)
        frames = []
        for lid in VALORANT_LEAGUE_IDS:
            j = fetch_raw_projections(league_id=lid)
            frames.append(_extract_offer_rows(j))
            time.sleep(0.8)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        # Fetch all and filter by 'valorant' via included league names
        j = fetch_raw_projections(league_id=None)
        df = _extract_offer_rows(j)
    # Basic de-dupe and sort
    if not df.empty:
        df = df.drop_duplicates(subset=["offer_id"]).sort_values("offer_time", na_position="last", ascending=False)
    return df.reset_index(drop=True)

if __name__ == "__main__":
    df = load_current_offers_valorant()
    print(df.head(20).to_string(index=False))
