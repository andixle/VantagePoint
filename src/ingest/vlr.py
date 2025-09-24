from __future__ import annotations
import os, re, time, json, hashlib, argparse
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import requests
import pandas as pd
from bs4 import BeautifulSoup

# Optional normalization helpers
try:
    from src.utils.aliases import normalize_offer_fields
except Exception:  # pragma: no cover
    def normalize_offer_fields(**kwargs):
        return kwargs

API_BASE = "https://vlrggapi.vercel.app"  # unofficial community API
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Educational DS project; contact: your-email@example.com)"
}

# ------------- tiny disk cache for scraper -------------
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "vlr_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(url: str) -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
    return os.path.join(CACHE_DIR, f"{h}.html")

def fetch_html(url: str, use_cache: bool = True, sleep_sec: float = 0.8) -> str:
    cp = _cache_path(url)
    if use_cache and os.path.exists(cp):
        with open(cp, "r", encoding="utf-8") as f:
            return f.read()
    time.sleep(sleep_sec)
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    html = r.text
    if use_cache:
        with open(cp, "w", encoding="utf-8") as f:
            f.write(html)
    return html

# ------------- API helpers (best-effort) -------------
def _api_get(path: str, params: Optional[dict] = None) -> dict:
    url = f"{API_BASE.rstrip('/')}/{path.lstrip('/')}"
    r = requests.get(url, params=params or {}, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()

def _extract_match_id_from_url(url_or_id: str) -> str:
    m = re.search(r"/(\d+)", url_or_id)
    return m.group(1) if m else url_or_id.strip()

def api_fetch_match(match_id: str) -> Tuple[dict, pd.DataFrame, pd.DataFrame, List[dict]]:
    """
    Try a few plausible endpoints from vlrggapi.
    Returns (meta, maps_df, player_maps_df, vetoes_list).
    If anything critical is missing, raise to trigger scraper fallback.
    """
    # Common endpoints seen in the wild (subject to change)
    tried = []
    j = None
    for path, params in [
        ("match", {"id": match_id}),
        ("matches", {"id": match_id}),
        (f"match/{match_id}", None),
    ]:
        tried.append((path, params))
        try:
            j = _api_get(path, params)
            break
        except Exception:
            j = None
    if j is None or not isinstance(j, dict):
        raise RuntimeError(f"API miss for match {match_id}; tried: {tried}")

    # Heuristically locate fields
    meta_raw = j.get("data") or j.get("match") or j
    if not isinstance(meta_raw, dict):
        raise RuntimeError("API match payload not a dict")

    # Meta fields (best effort)
    meta = {
        "team_a": meta_raw.get("team1") or meta_raw.get("team_a") or meta_raw.get("teamA"),
        "team_b": meta_raw.get("team2") or meta_raw.get("team_b") or meta_raw.get("teamB"),
        "event": meta_raw.get("event") or meta_raw.get("tournament"),
        "bo_format": meta_raw.get("format") or meta_raw.get("bo"),
        "date": None,
    }
    # date could be 'epoch' or 'date' string
    epoch = meta_raw.get("epoch") or meta_raw.get("timestamp")
    if epoch:
        try:
            meta["date"] = datetime.fromtimestamp(int(epoch), tz=timezone.utc).isoformat()
        except Exception:
            pass
    if not meta["date"]:
        meta["date"] = meta_raw.get("date")

    # Maps & per-map stats (if provided)
    maps_df = pd.DataFrame(columns=["match_id","map_num","map_name","winner","score","picked_by","first_pick_order"])
    pmaps_df = pd.DataFrame(columns=[
        "match_id","map_num","date","player","team","opponent","map_name","agent",
        "kills","deaths","assists","ACS","rounds_played"
    ])
    vetoes_list: List[dict] = []

    # API may expose a `maps` array with basic info
    maps_raw = meta_raw.get("maps") or meta_raw.get("map_list") or []
    if isinstance(maps_raw, list) and maps_raw:
        rows = []
        for i, mobj in enumerate(maps_raw, 1):
            rows.append({
                "match_id": match_id,
                "map_num": i,
                "map_name": mobj.get("map") or mobj.get("name"),
                "winner": mobj.get("winner"),
                "score": mobj.get("score"),
                "picked_by": mobj.get("picked_by") or mobj.get("pick"),
                "first_pick_order": None,
            })
            # Per-map player stats (rare in API; often absent)
            for side in ("team1_players", "team2_players", "players"):
                players = mobj.get(side) or []
                for prow in players:
                    pa = normalize_offer_fields(
                        player = prow.get("name") or prow.get("player"),
                        team   = prow.get("team"),
                        opponent = meta["team_b"] if (prow.get("team") == meta["team_a"]) else meta["team_a"],
                        map_name = mobj.get("map") or mobj.get("name"),
                        agent  = prow.get("agent"),
                        stat_type = "kills",
                    )
                    pmaps_df = pd.concat([pmaps_df, pd.DataFrame([{
                        "match_id": match_id,
                        "map_num": i,
                        "date": meta["date"],
                        "player": pa.get("player"),
                        "team": pa.get("team_name") or prow.get("team"),
                        "opponent": pa.get("opponent_name"),
                        "map_name": pa.get("map_name"),
                        "agent": pa.get("agent"),
                        "kills": prow.get("kills"),
                        "deaths": prow.get("deaths"),
                        "assists": prow.get("assists"),
                        "ACS": prow.get("acs"),
                        "rounds_played": prow.get("rounds"),
                    }])], ignore_index=True)
        maps_df = pd.DataFrame(rows)

    # Vetoes / notes if provided
    vetoes = meta_raw.get("vetoes") or meta_raw.get("notes") or []
    if isinstance(vetoes, list):
        vetoes_list = [{"note": str(v)} for v in vetoes]

    # sanity: require at least teams to consider API “good”
    if not meta["team_a"] or not meta["team_b"]:
        raise RuntimeError("API missing teams; falling back to scraper.")

    return meta, maps_df, pmaps_df, vetoes_list

# ------------- Scraper fallback (HTML) -------------
def _parse_match_header(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    out = {"team_a": None, "team_b": None, "event": None, "bo_format": None, "date": None}
    teams = soup.select("div.match-header .match-header-link .wf-title-med")
    if len(teams) >= 2:
        out["team_a"] = teams[0].get_text(strip=True)
        out["team_b"] = teams[1].get_text(strip=True)
    ev = soup.select_one(".match-header .match-header-event a")
    if ev: out["event"] = ev.get_text(strip=True)
    bo = soup.select_one(".match-header .match-header-vs-note")
    if bo: out["bo_format"] = bo.get_text(strip=True)
    dt_node = soup.select_one(".match-header .moment-tz-convert")
    if dt_node and dt_node.has_attr("data-epoch"):
        try:
            ts = int(dt_node["data-epoch"])
            out["date"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except Exception:
            pass
    if not out["date"]:
        dt_txt = soup.select_one(".match-header .match-header-date")
        if dt_txt: out["date"] = dt_txt.get_text(" ", strip=True)
    return out

def _parse_vetoes(soup: BeautifulSoup) -> List[Dict[str, str]]:
    out = []
    sections = soup.select(".match-veto-box, .match-header-note")
    for sec in sections:
        text = sec.get_text(" ", strip=True)
        if any(k in text.lower() for k in ["ban", "pick", "decider"]):
            out.append({"note": text})
    return out

def _parse_maps_and_players(soup: BeautifulSoup, team_a: str, team_b: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    map_rows: List[Dict] = []
    player_rows: List[Dict] = []

    maps = soup.select(".vm-stats-game")
    map_num = 0
    for m in maps:
        map_num += 1
        map_name = None
        name_el = m.select_one(".map")
        if name_el:
            map_name = name_el.get_text(strip=True)

        picked_by = None
        pick_el = m.select_one(".mod-picked, .picked, .note")
        if pick_el:
            txt = pick_el.get_text(" ", strip=True)
            mm = re.search(r"picked by\s+([A-Za-z0-9\s]+)", txt, flags=re.I)
            if mm: picked_by = mm.group(1).strip()

        header_txt = m.get_text(" ", strip=True)
        score = None; winner = None
        ms = re.search(r"(\b\d{1,2}\b)\s*-\s*(\b\d{1,2}\b)", header_txt)
        if ms:
            s_a, s_b = int(ms.group(1)), int(ms.group(2))
            score = f"{s_a}-{s_b}"
            winner = team_a if s_a > s_b else team_b

        # Player tables (2 per map typically)
        tables = m.select(".vm-stats-game .wf-table-inset, .vm-stats-game .wf-table")
        if not tables:
            tables = m.select(".wf-table-inset, .wf-table")
        team_i = 0
        for tbl in tables:
            team_i += 1
            heading = tbl.find_previous("div", class_="vm-stats-game-header")
            team_name = None
            if heading:
                bits = heading.get_text(" ", strip=True)
                if team_a and team_a.lower() in bits.lower(): team_name = team_a
                elif team_b and team_b.lower() in bits.lower(): team_name = team_b
            if not team_name:
                team_name = team_a if team_i == 1 else team_b

            trs = tbl.select("tr")
            for tr in trs:
                cols = [td.get_text(" ", strip=True) for td in tr.select("td")]
                if len(cols) < 3: continue
                player = tr.select_one("td a, td span")
                player_name = player.get_text(strip=True) if player else cols[0]
                agent = None
                img = tr.select_one("img")
                if img and img.has_attr("alt"):
                    agent = img["alt"].strip()
                kda = None
                for c in cols:
                    if re.search(r"\b\d+\s*/\s*\d+\s*/\s*\d+\b", c):
                        kda = c; break
                kills = deaths = assists = None
                if kda:
                    k, d, a = re.findall(r"\d+", kda)[:3]
                    kills, deaths, assists = int(k), int(d), int(a)
                acs = None
                for c in cols:
                    if re.match(r"^\d{2,3}$", c):
                        acs = int(c); break

                pa = normalize_offer_fields(
                    player=player_name, team=team_name,
                    opponent=(team_b if team_name == team_a else team_a),
                    map_name=map_name, agent=agent, stat_type="kills"
                )
                player_rows.append({
                    "match_id": None,
                    "map_num": map_num,
                    "date": None,
                    "player": pa.get("player") or player_name,
                    "team": pa.get("team_name") or team_name,
                    "opponent": pa.get("opponent_name") or (team_b if team_name == team_a else team_a),
                    "map_name": pa.get("map_name") or map_name,
                    "agent": pa.get("agent") or agent,
                    "kills": kills, "deaths": deaths, "assists": assists,
                    "ACS": acs, "rounds_played": None,
                })

        map_rows.append({
            "match_id": None, "map_num": map_num, "map_name": map_name,
            "winner": winner, "score": score, "picked_by": picked_by, "first_pick_order": None
        })

    df_maps = pd.DataFrame(map_rows)
    df_pmaps = pd.DataFrame(player_rows)
    return df_maps, df_pmaps

def scrape_match(match_url: str) -> Tuple[dict, pd.DataFrame, pd.DataFrame, List[dict]]:
    html = fetch_html(match_url)
    soup = BeautifulSoup(html, "lxml")
    meta = _parse_match_header(soup)
    team_a, team_b = meta.get("team_a") or "", meta.get("team_b") or ""
    df_maps, df_pmaps = _parse_maps_and_players(soup, team_a, team_b)
    vetoes = _parse_vetoes(soup)
    match_id = _extract_match_id_from_url(match_url)
    for df in (df_maps, df_pmaps):
        if not df.empty: df["match_id"] = match_id
    if not df_pmaps.empty: df_pmaps["date"] = meta.get("date")
    return meta, df_maps, df_pmaps, vetoes

# ------------- Public API (with adapter) -------------
def parse_match(match_url_or_id: str) -> Tuple[dict, pd.DataFrame, pd.DataFrame, List[dict]]:
    """
    API-first: try vlrggapi by match_id; if incomplete -> scrape vlr.gg page.
    """
    match_id = _extract_match_id_from_url(match_url_or_id)
    # Try API
    try:
        meta, maps_df, pmaps_df, vetoes = api_fetch_match(match_id)
        # If maps or pmaps look empty AND we have a URL, consider falling back
        if maps_df.empty or pmaps_df.empty:
            # Try to scrape for richer granularity if a full URL was provided
            if match_url_or_id.startswith("http"):
                meta_s, maps_s, pmaps_s, vetoes_s = scrape_match(match_url_or_id)
                # Merge: fill blanks from scraper
                if maps_df.empty: maps_df = maps_s
                if pmaps_df.empty: pmaps_df = pmaps_s
                # prefer API meta if present; else use scraper meta
                for k, v in meta_s.items():
                    if not meta.get(k): meta[k] = v
                vetoes = vetoes or vetoes_s
        return meta, maps_df, pmaps_df, vetoes
    except Exception:
        # Fall back to scraper (needs URL)
        if not match_url_or_id.startswith("http"):
            # Construct a canonical VLR URL if only id is given
            match_url = f"https://www.vlr.gg/{match_id}/match"
        else:
            match_url = match_url_or_id
        return scrape_match(match_url)

def scrape_matches(match_urls: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Given match URLs (or ids), return (matches_df, maps_df, player_maps_df).
    Uses API-first per match with scraper fallback.
    """
    matches_meta, all_maps, all_pmaps = [], [], []
    for u in match_urls:
        try:
            meta, df_maps, df_pmaps, vetoes = parse_match(u)
        except Exception as e:
            print(f"[WARN] Failed to parse {u}: {e}")
            continue
        match_id = _extract_match_id_from_url(u)
        matches_meta.append({
            "match_id": match_id,
            "event": meta.get("event"),
            "date": meta.get("date"),
            "bo_format": meta.get("bo_format"),
            "team_a": meta.get("team_a"),
            "team_b": meta.get("team_b"),
            "vetoes_json": json.dumps(vetoes, ensure_ascii=False),
            "src_url": u if str(u).startswith("http") else f"https://www.vlr.gg/{match_id}/match",
        })
        if not df_maps.empty: all_maps.append(df_maps)
        if not df_pmaps.empty: all_pmaps.append(df_pmaps)

    matches_df = pd.DataFrame(matches_meta)
    maps_df = pd.concat(all_maps, ignore_index=True) if all_maps else pd.DataFrame(
        columns=["match_id","map_num","map_name","winner","score","picked_by","first_pick_order"]
    )
    player_maps_df = pd.concat(all_pmaps, ignore_index=True) if all_pmaps else pd.DataFrame(
        columns=["match_id","map_num","date","player","team","opponent","map_name","agent","kills","deaths","assists","ACS","rounds_played"]
    )
    return matches_df, maps_df, player_maps_df

# Back-compat stubs (so other parts of the project don't break)
def load_player_maps() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["match_id","map_num","date","player","team","opponent","map_name","agent","kills","deaths","assists","ACS","rounds_played"]
    )

def load_map_pool() -> pd.DataFrame:
    return pd.DataFrame(columns=["match_id","team","Ascent","Bind","Lotus","Haven","Icebox","Split","Sunset","Breeze"])

# ------------- CLI -------------
def _cli():
    ap = argparse.ArgumentParser(description="VLR ingest (API-first with scraper fallback)")
    ap.add_argument("--match", help="VLR match URL or numeric id", default=None)
    ap.add_argument("--batch", nargs="+", help="Space-separated list of match URLs/ids", default=None)
    args = ap.parse_args()

    if args.match:
        meta, df_maps, df_pmaps, vetoes = parse_match(args.match)
        print("META:", json.dumps(meta, indent=2, ensure_ascii=False))
        print("\nMAPS:\n", (df_maps.to_string(index=False) if not df_maps.empty else "<empty>"))
        print("\nPLAYER MAPS:\n", (df_pmaps.head(30).to_string(index=False) if not df_pmaps.empty else "<empty>"))
        print("\nVETOES:", json.dumps(vetoes, indent=2, ensure_ascii=False))
        return

    if args.batch:
        matches_df, maps_df, pmaps_df = scrape_matches(args.batch)
        print("MATCHES:\n", (matches_df.to_string(index=False) if not matches_df.empty else "<empty>"))
        print("\nMAPS:\n", (maps_df.head(30).to_string(index=False) if not maps_df.empty else "<empty>"))
        print("\nPLAYER MAPS:\n", (pmaps_df.head(30).to_string(index=False) if not pmaps_df.empty else "<empty>"))

if __name__ == "__main__":
    _cli()
