import os
import datetime as dt
from typing import List, Optional, Dict, Any
import pandas as pd
from pybaseball import schedule_and_record, batting_stats, pitching_stats

# Optional MLB StatsAPI for schedule/probables and player-team mapping
try:
    import statsapi  # MLB-StatsAPI
except Exception:  # pragma: no cover
    statsapi = None

DATA_DIR = os.path.abspath("data")
SNAPSHOT_DIR = os.path.abspath("data_snapshots")
os.makedirs(DATA_DIR, exist_ok=True)

# --- Load snapshots (season-to-date aggregates) ---
def load_snapshots(snapshot_date: Optional[str] = None):
    if snapshot_date is None:
        snapshot_date = dt.date.today().isoformat()
    batter_path = os.path.join(SNAPSHOT_DIR, f"batter_{snapshot_date}.csv")
    pitcher_path = os.path.join(SNAPSHOT_DIR, f"pitcher_{snapshot_date}.csv")
    bat = pd.read_csv(batter_path)
    pit = pd.read_csv(pitcher_path)
    # normalize column names
    bat.columns = [c.strip().lower().replace(' ', '_').replace('%','percent') for c in bat.columns]
    pit.columns = [c.strip().lower().replace(' ', '_').replace('%','percent') for c in pit.columns]
    return bat, pit

# Helper to coerce percentage-like strings and numeric columns

def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace('%','', regex=False).str.replace('"','', regex=False), errors='coerce')


def _standardize_name_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Handle both 'last_name, first_name' and normalized 'last_name,_first_name'
    for cand in ["last_name, first_name", "last_name,_first_name", "player_name", "name"]:
        if cand in df.columns:
            if cand != "name":
                df = df.rename(columns={cand: "name"})
            break
    return df

# --- Simple features for per-game batter and pitcher ---

def make_batter_features(bat_snap: pd.DataFrame) -> pd.DataFrame:
    bat_snap = _standardize_name_cols(bat_snap.copy())
    use_cols = [
        "name", "player_id", "year", "pa", "k_percent", "bb_percent", "woba", "xwoba",
        "sweet_spot_percent", "barrel_batted_rate", "hard_hit_percent", "whiff_percent", "swing_percent"
    ]
    cols = [c for c in use_cols if c in bat_snap.columns]
    X = bat_snap[cols].copy()
    # Coerce numeric fields
    for c in ["pa","k_percent","bb_percent","sweet_spot_percent","barrel_batted_rate","hard_hit_percent","whiff_percent","swing_percent","woba","xwoba"]:
        if c in X.columns:
            X[c] = _to_float_series(X[c])
    # Scale percent features from 0-100 to 0-1 to match training
    for c in ["k_percent","bb_percent","sweet_spot_percent","barrel_batted_rate","hard_hit_percent","whiff_percent","swing_percent"]:
        if c in X.columns:
            X[c] = X[c] / 100.0
    return X


def make_pitcher_features(pit_snap: pd.DataFrame) -> pd.DataFrame:
    pit_snap = _standardize_name_cols(pit_snap.copy())
    use_cols = [
        "name", "player_id", "year", "pa", "k_percent", "bb_percent", "woba", "xwoba",
        "sweet_spot_percent", "barrel_batted_rate", "hard_hit_percent", "whiff_percent", "swing_percent"
    ]
    cols = [c for c in use_cols if c in pit_snap.columns]
    X = pit_snap[cols].copy()
    for c in ["pa","k_percent","bb_percent","sweet_spot_percent","barrel_batted_rate","hard_hit_percent","whiff_percent","swing_percent","woba","xwoba"]:
        if c in X.columns:
            X[c] = _to_float_series(X[c])
    for c in ["k_percent","bb_percent","sweet_spot_percent","barrel_batted_rate","hard_hit_percent","whiff_percent","swing_percent"]:
        if c in X.columns:
            X[c] = X[c] / 100.0
    return X

# --- Schedule / probables / park factors helpers ---

def _to_mmddyyyy(date_str: str) -> str:
    try:
        d = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        d = dt.date.today()
    return d.strftime("%m/%d/%Y")


def get_schedule_with_probables(date_str: Optional[str]) -> pd.DataFrame:
    if statsapi is None:
        return pd.DataFrame()
    date_str = date_str or dt.date.today().isoformat()
    sched = statsapi.schedule(date=_to_mmddyyyy(date_str))
    if not sched:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for g in sched:
        rows.append({
            "game_pk": g.get("game_id") or g.get("game_pk") or g.get("gamePk"),
            "date": g.get("game_date") or g.get("gameDate"),
            "home_id": g.get("home_id"),
            "home": g.get("home_name"),
            "away_id": g.get("away_id"),
            "away": g.get("away_name"),
            "venue_id": g.get("venue_id"),
            "venue": g.get("venue_name"),
            "home_probable_id": g.get("home_probable_pitcher_id"),
            "home_probable": g.get("home_probable_pitcher"),
            "away_probable_id": g.get("away_probable_pitcher_id"),
            "away_probable": g.get("away_probable_pitcher"),
        })
    df = pd.DataFrame(rows)
    # Normalize types
    for c in ["game_pk","home_id","away_id","venue_id","home_probable_id","away_probable_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
    return df


def get_player_team_map(player_ids: List[int]) -> pd.DataFrame:
    if statsapi is None or not player_ids:
        return pd.DataFrame(columns=["player_id","team_id","team_name","team_abbrev"])  
    # StatsAPI allows batching up to ~100 ids
    out_rows: List[Dict[str, Any]] = []
    ids = [int(x) for x in pd.Series(player_ids).dropna().unique().tolist() if pd.notna(x)]
    for i in range(0, len(ids), 80):
        chunk = ids[i:i+80]
        try:
            data = statsapi.get('people', {'personIds': ','.join(str(x) for x in chunk), 'hydrate': 'team'})
            people = data.get('people', []) if isinstance(data, dict) else []
        except Exception:
            people = []
        for p in people:
            pid = p.get('id')
            team = p.get('currentTeam') or {}
            out_rows.append({
                'player_id': pid,
                'team_id': (team or {}).get('id'),
                'team_name': (team or {}).get('name'),
                'team_abbrev': (team or {}).get('abbreviation'),
            })
    return pd.DataFrame(out_rows)


def load_park_factors() -> pd.DataFrame:
    """Load optional park factors from data/park_factors.csv.
    Expected columns: venue (name) or venue_id, park_hr_factor, park_hit_factor.
    Defaults to 1.0 if not found.
    """
    pf_path = os.path.join(DATA_DIR, 'park_factors.csv')
    if not os.path.exists(pf_path):
        return pd.DataFrame(columns=["venue","venue_id","park_hr_factor","park_hit_factor"])
    pf = pd.read_csv(pf_path)
    pf.columns = [c.strip().lower() for c in pf.columns]
    for c in ["park_hr_factor","park_hit_factor"]:
        if c in pf.columns:
            pf[c] = pd.to_numeric(pf[c], errors='coerce')
    return pf


def build_batter_context(bat_df: pd.DataFrame, pit_df: pd.DataFrame, date_str: Optional[str]) -> pd.DataFrame:
    """Return a dataframe with opponent probable pitcher and park for each batter.
    Columns added: team_id, opp_pitcher_id, opp_pitcher_name, venue, park_* and opponent pitcher features prefixed with opp_.
    """
    if bat_df is None or bat_df.empty:
        return pd.DataFrame()
    # Map batter -> team
    team_map = get_player_team_map(bat_df.get("player_id", pd.Series(dtype=int)).tolist())
    if team_map.empty:
        return pd.DataFrame()
    # Normalize dtypes for merge keys
    for c in ["player_id", "team_id"]:
        if c in team_map.columns:
            team_map[c] = pd.to_numeric(team_map[c], errors='coerce').astype('Int64')
    # Schedule with probables
    sched = get_schedule_with_probables(date_str)
    if sched.empty:
        return pd.DataFrame()
    # Merge schedule rows twice to get each team perspective
    home_df = sched[["home_id","away_id","venue","venue_id","away_probable_id","away_probable"]].rename(columns={
        "home_id":"team_id","away_id":"opp_team_id","venue":"venue","venue_id":"venue_id","away_probable_id":"opp_pitcher_id","away_probable":"opp_pitcher_name"
    })
    away_df = sched[["away_id","home_id","venue","venue_id","home_probable_id","home_probable"]].rename(columns={
        "away_id":"team_id","home_id":"opp_team_id","venue":"venue","venue_id":"venue_id","home_probable_id":"opp_pitcher_id","home_probable":"opp_pitcher_name"
    })
    teams_sched = pd.concat([home_df, away_df], ignore_index=True)
    for c in ["team_id","opp_team_id","venue_id","opp_pitcher_id"]:
        if c in teams_sched.columns:
            teams_sched[c] = pd.to_numeric(teams_sched[c], errors='coerce').astype('Int64')
    # Batter team -> matchup row
    m = team_map.merge(teams_sched, on="team_id", how="left")
    # Attach to batters
    ctx = bat_df.merge(m, on="player_id", how="left")
    # Opponent pitcher features
    pitX = make_pitcher_features(pit_df)
    for c in ["player_id"]:
        if c in pitX.columns:
            pitX[c] = pd.to_numeric(pitX[c], errors='coerce').astype('Int64')
    opp = pitX.add_prefix("opp_")
    opp = opp.rename(columns={"opp_player_id":"opp_pitcher_id","opp_name":"opp_pitcher_name"})
    for c in ["opp_pitcher_id"]:
        if c in opp.columns:
            opp[c] = pd.to_numeric(opp[c], errors='coerce').astype('Int64')
    ctx = ctx.merge(opp, on="opp_pitcher_id", how="left")
    # Park factors
    pf = load_park_factors()
    if not pf.empty:
        # Try join by venue_id if present else by venue name (lower)
        if "venue_id" in ctx.columns and "venue_id" in pf.columns:
            # coerce venue_id types
            ctx["venue_id"] = pd.to_numeric(ctx["venue_id"], errors='coerce').astype('Int64')
            pf_cols = [c for c in ["venue_id","park_hr_factor","park_hit_factor"] if c in pf.columns]
            pf2 = pf[pf_cols].copy()
            if "venue_id" in pf2.columns:
                pf2["venue_id"] = pd.to_numeric(pf2["venue_id"], errors='coerce').astype('Int64')
            ctx = ctx.merge(pf2, on="venue_id", how="left")
        elif "venue" in ctx.columns and "venue" in pf.columns:
            ctx["venue_l"] = ctx["venue"].astype(str).str.lower()
            pf["venue_l"] = pf["venue"].astype(str).str.lower()
            ctx = ctx.merge(pf[[c for c in ["venue_l","park_hr_factor","park_hit_factor"] if c in pf.columns]], on="venue_l", how="left")
            ctx.drop(columns=["venue_l"], inplace=True, errors='ignore')
    # Defaults for missing
    if "park_hr_factor" not in ctx.columns:
        ctx["park_hr_factor"] = 1.0
    if "park_hit_factor" not in ctx.columns:
        ctx["park_hit_factor"] = 1.0
    return ctx

# Placeholder: join today schedule and build inference frame later


def load_training_frame(snapshot_date: Optional[str] = None):
    bat, pit = load_snapshots(snapshot_date)
    batX = make_batter_features(bat)
    pitX = make_pitcher_features(pit)
    return batX, pitX

def _extract_game_pk(g: Dict[str, Any]) -> Optional[int]:
    for k in ("game_pk", "gamePk", "game_pk", "gamePk", "game_id"):
        if k in g and g[k] is not None:
            try:
                s = str(g[k])
                # If it's like '2025_..._gameid', extract digits
                digits = ''.join(ch for ch in s if ch.isdigit())
                if digits:
                    return int(digits)
            except Exception:
                continue
    return None


def get_confirmed_starters(date_str: Optional[str]) -> pd.DataFrame:
    """Return confirmed starters (batting order) for the given date if available.
    Columns: player_id, game_pk, team_id, order (1-9).
    """
    if statsapi is None:
        return pd.DataFrame(columns=["player_id","game_pk","team_id","order"])
    date_str = date_str or dt.date.today().isoformat()
    sched = statsapi.schedule(date=_to_mmddyyyy(date_str)) or []
    rows: List[Dict[str, Any]] = []
    for g in sched:
        game_pk = _extract_game_pk(g)
        if not game_pk:
            continue
        try:
            bx = statsapi.boxscore_data(game_pk)
        except Exception:
            bx = None
        if not bx:
            continue
        for side in ["home", "away"]:
            side_data = bx.get(side) or {}
            batting = side_data.get("battingOrder") or side_data.get("batting_order")
            team_id = side_data.get("teamId") or side_data.get("team_id")
            # batting may be list of player ids or dash-separated strings
            order_list: List[Any] = []
            if isinstance(batting, list):
                order_list = batting
            elif isinstance(batting, str):
                order_list = [x.strip() for x in batting.split(",") if x.strip()]
            # Build rows
            for idx, pid in enumerate(order_list, start=1):
                try:
                    pid_int = int(str(pid))
                except Exception:
                    continue
                rows.append({
                    "player_id": pid_int,
                    "game_pk": int(game_pk),
                    "team_id": pd.to_numeric(team_id, errors='coerce'),
                    "order": idx,
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["player_id"] = pd.to_numeric(df["player_id"], errors='coerce').astype('Int64')
        if "team_id" in df.columns:
            df["team_id"] = pd.to_numeric(df["team_id"], errors='coerce').astype('Int64')
    return df

if __name__ == "__main__":
    batX, pitX = load_training_frame()
    print(batX.head())
    print(pitX.head())
