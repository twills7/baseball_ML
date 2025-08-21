import os
import datetime as dt
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from tqdm import tqdm
from pybaseball import statcast
from ml_models import train_baseline_classifier
from data_pipeline import load_park_factors

# Optional StatsAPI for venues
try:
    import statsapi
except Exception:  # pragma: no cover
    statsapi = None

MODELS_DIR = os.path.abspath("models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Helper to classify outcomes from statcast
HIT_EVENTS = {"single", "double", "triple", "home_run"}
HR_EVENT = "home_run"


def load_statcast_window(start: str, end: str) -> pd.DataFrame:
    print(f"Downloading Statcast {start} to {end} (this can take a while)...")
    # Enable caching to avoid re-downloading large queries
    try:
        from pybaseball import cache
        cache.enable()
    except Exception:
        pass
    df = statcast(start_dt=start, end_dt=end)
    # Ensure required columns exist
    needed = [
        "game_date", "game_pk", "batter", "player_name", "events", "description",
        "launch_speed", "launch_angle", "estimated_woba_using_speedangle", "type",
        "pitcher",
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    return df


def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Per batter per game labels
    g = df.groupby(["batter", "game_pk", "game_date", "player_name"], sort=False)
    had_hit = g["events"].apply(lambda s: int(any(str(x) in HIT_EVENTS for x in s)))
    had_hr = g["events"].apply(lambda s: int(any(str(x) == HR_EVENT for x in s)))

    # Opposing pitcher for this batter in this game: take pitcher in batter's first PA
    # Use groupby first() to avoid index/length mismatches
    first_p = g["pitcher"].first()

    out = pd.concat([
        had_hit.rename("label_hit"),
        had_hr.rename("label_hr"),
        first_p.rename("opp_pitcher_id"),
    ], axis=1).reset_index()
    return out


def is_swing(row) -> bool:
    desc = str(row.get("description", ""))
    return any(k in desc for k in ["swinging_strike", "foul", "foul_tip", "hit_into_play"]) or row.get("type") == "S"


def is_whiff(row) -> bool:
    desc = str(row.get("description", ""))
    return "swinging_strike" in desc


def is_ball_in_play(row) -> bool:
    desc = str(row.get("description", ""))
    return "hit_into_play" in desc


def is_barrel_like(row) -> bool:
    try:
        ev = float(row.get("launch_speed", np.nan))
        la = float(row.get("launch_angle", np.nan))
    except Exception:
        return False
    if np.isnan(ev) or np.isnan(la):
        return False
    return (ev >= 98.0) and (26 <= la <= 30)


def rolling_features(df: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    # Sort by batter and date
    df = df.sort_values(["batter", "game_date"]).copy()

    # Build per-batter game aggregates via pitch level flags
    df["swing"] = df.apply(is_swing, axis=1).astype(int)
    df["whiff"] = df.apply(is_whiff, axis=1).astype(int)
    df["bip"] = df.apply(is_ball_in_play, axis=1).astype(int)
    df["hard_hit"] = (df["launch_speed"].astype(float) >= 95.0).astype(int)
    df["barrel_like"] = df.apply(is_barrel_like, axis=1).astype(int)

    agg = df.groupby(["batter", "game_pk", "game_date", "player_name"]).agg(
        pa=("type", "count"),
        swings=("swing", "sum"),
        whiffs=("whiff", "sum"),
        bip=("bip", "sum"),
        hard_hits=("hard_hit", "sum"),
        barrels=("barrel_like", "sum"),
        walks=("events", lambda s: sum(str(x) in ["walk", "hit_by_pitch"] for x in s)),
        strikeouts=("events", lambda s: sum("strikeout" in str(x) for x in s)),
        hrs=("events", lambda s: sum(str(x) == HR_EVENT for x in s)),
        xwoba_sum=("estimated_woba_using_speedangle", "sum"),
        xwoba_cnt=("estimated_woba_using_speedangle", lambda s: s.notna().sum()),
    ).reset_index()

    agg["xwoba"] = agg["xwoba_sum"] / agg["xwoba_cnt"].replace({0: np.nan})

    def _calc_rolling(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("game_date").copy()
        g["date"] = pd.to_datetime(g["game_date"])
        feats = []
        for _, row in g.iterrows():
            cutoff = row["date"] - pd.Timedelta(days=window_days)
            hist = g[(g["date"] < row["date"]) & (g["date"] >= cutoff)]
            totals = hist[["pa", "swings", "whiffs", "bip", "hard_hits", "barrels", "walks", "strikeouts", "hrs"]].sum(min_count=1)
            xwoba = hist["xwoba"].mean()
            pa = float(totals.get("pa", 0.0) or 0.0)
            feats.append({
                "batter": row["batter"],
                "game_pk": row["game_pk"],
                "game_date": row["game_date"],
                "player_name": row["player_name"],
                "pa_rolling": pa,
                "k_percent": (float(totals.get("strikeouts", 0.0) or 0.0) / pa) if pa > 0 else np.nan,
                "bb_percent": (float(totals.get("walks", 0.0) or 0.0) / pa) if pa > 0 else np.nan,
                "whiff_percent": (float(totals.get("whiffs", 0.0) or 0.0) / float(totals.get("swings", 0.0) or 0.0)) if float(totals.get("swings", 0.0) or 0.0) > 0 else np.nan,
                "hard_hit_percent": (float(totals.get("hard_hits", 0.0) or 0.0) / float(totals.get("bip", 0.0) or 0.0)) if float(totals.get("bip", 0.0) or 0.0) > 0 else np.nan,
                "barrel_batted_rate": (float(totals.get("barrels", 0.0) or 0.0) / float(totals.get("bip", 0.0) or 0.0)) if float(totals.get("bip", 0.0) or 0.0) > 0 else np.nan,
                "xwoba": xwoba,
            })
        return pd.DataFrame(feats)

    feat_df = agg.groupby("batter", group_keys=False).apply(_calc_rolling).reset_index(drop=True)
    return feat_df


def pitcher_rolling_features(df: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    # Similar approach from the pitcher perspective
    df = df.sort_values(["pitcher", "game_date"]).copy()
    df["swing"] = df.apply(is_swing, axis=1).astype(int)
    df["whiff"] = df.apply(is_whiff, axis=1).astype(int)
    df["bip"] = df.apply(is_ball_in_play, axis=1).astype(int)
    df["barrel_like"] = df.apply(is_barrel_like, axis=1).astype(int)

    agg = df.groupby(["pitcher", "game_pk", "game_date"]).agg(
        bf=("type", "count"),  # batters faced proxy
        swings=("swing", "sum"),
        whiffs=("whiff", "sum"),
        bip=("bip", "sum"),
        barrels=("barrel_like", "sum"),
        walks=("events", lambda s: sum(str(x) in ["walk", "hit_by_pitch"] for x in s)),
        strikeouts=("events", lambda s: sum("strikeout" in str(x) for x in s)),
        xwoba_sum=("estimated_woba_using_speedangle", "sum"),
        xwoba_cnt=("estimated_woba_using_speedangle", lambda s: s.notna().sum()),
    ).reset_index()

    agg["xwoba"] = agg["xwoba_sum"] / agg["xwoba_cnt"].replace({0: np.nan})

    def _calc_rolling(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("game_date").copy()
        g["date"] = pd.to_datetime(g["game_date"])
        feats = []
        for _, row in g.iterrows():
            cutoff = row["date"] - pd.Timedelta(days=window_days)
            hist = g[(g["date"] < row["date"]) & (g["date"] >= cutoff)]
            totals = hist[["bf", "swings", "whiffs", "bip", "barrels", "walks", "strikeouts"]].sum(min_count=1)
            xwoba = hist["xwoba"].mean()
            bf = float(totals.get("bf", 0.0) or 0.0)
            feats.append({
                "pitcher": row["pitcher"],
                "game_pk": row["game_pk"],
                "game_date": row["game_date"],
                "opp_k_percent": (float(totals.get("strikeouts", 0.0) or 0.0) / bf) if bf > 0 else np.nan,
                "opp_xwoba": xwoba,
                "opp_barrel_batted_rate": (float(totals.get("barrels", 0.0) or 0.0) / float(totals.get("bip", 0.0) or 0.0)) if float(totals.get("bip", 0.0) or 0.0) > 0 else np.nan,
            })
        return pd.DataFrame(feats)

    feat_df = agg.groupby("pitcher", group_keys=False).apply(_calc_rolling).reset_index(drop=True)
    return feat_df


def get_game_venues_for_dates(dates: List[dt.date]) -> pd.DataFrame:
    if statsapi is None:
        return pd.DataFrame(columns=["game_pk", "venue", "venue_id"])
    rows: List[Dict[str, Any]] = []
    for d in sorted(set(dates)):
        try:
            sched = statsapi.schedule(date=d.strftime("%m/%d/%Y"))
        except Exception:
            sched = []
        for g in sched or []:
            game_pk = g.get("game_id") or g.get("game_pk") or g.get("gamePk")
            try:
                game_pk = int(''.join(ch for ch in str(game_pk) if ch.isdigit()))
            except Exception:
                game_pk = None
            rows.append({
                "game_pk": game_pk,
                "venue": g.get("venue_name"),
                "venue_id": g.get("venue_id"),
            })
    df = pd.DataFrame(rows).dropna(subset=["game_pk"]) if rows else pd.DataFrame(columns=["game_pk","venue","venue_id"])
    if not df.empty:
        df["game_pk"] = pd.to_numeric(df["game_pk"], errors='coerce').astype('Int64')
        if "venue_id" in df.columns:
            df["venue_id"] = pd.to_numeric(df["venue_id"], errors='coerce').astype('Int64')
    return df


def build_training(start: str, end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_statcast_window(start, end)
    labels = make_labels(raw)
    bat_feats = rolling_features(raw, window_days=30)

    # Pitcher rolling features
    pit_feats = pitcher_rolling_features(raw, window_days=30)

    # Map each batter-game to first opposing pitcher id already in labels as opp_pitcher_id
    # Join pitcher features by (pitcher, game_pk/date)
    # First coerce types
    labels["opp_pitcher_id"] = pd.to_numeric(labels["opp_pitcher_id"], errors='coerce').astype('Int64')
    pit_feats_ren = pit_feats.rename(columns={"pitcher": "opp_pitcher_id"})
    # Join on opp_pitcher_id and exact game_date (we want rolling as of that game)
    df = labels.merge(bat_feats, on=["batter", "game_pk", "game_date", "player_name"], how="inner")
    df = df.merge(pit_feats_ren, on=["opp_pitcher_id", "game_pk", "game_date"], how="left")

    # Attach park factors via venue mapping
    dates = pd.to_datetime(df["game_date"]).dt.date.tolist()
    venue_map = get_game_venues_for_dates(dates)
    df = df.merge(venue_map, on="game_pk", how="left")
    pf = load_park_factors()
    if not pf.empty:
        if "venue_id" in pf.columns and "venue_id" in df.columns:
            pf2 = pf[[c for c in ["venue_id", "park_hr_factor", "park_hit_factor"] if c in pf.columns]].copy()
            if "venue_id" in pf2.columns:
                pf2["venue_id"] = pd.to_numeric(pf2["venue_id"], errors='coerce').astype('Int64')
            df = df.merge(pf2, on="venue_id", how="left")
        elif "venue" in pf.columns and "venue" in df.columns:
            df["venue_l"] = df["venue"].astype(str).str.lower()
            pf["venue_l"] = pf["venue"].astype(str).str.lower()
            df = df.merge(pf[[c for c in ["venue_l", "park_hr_factor", "park_hit_factor"] if c in pf.columns]], on="venue_l", how="left")
            df.drop(columns=["venue_l"], inplace=True, errors='ignore')
    # Defaults if missing
    if "park_hr_factor" not in df.columns:
        df["park_hr_factor"] = 1.0
    if "park_hit_factor" not in df.columns:
        df["park_hit_factor"] = 1.0

    # Filter rows with some batter history
    df = df[df["pa_rolling"].fillna(0) > 10].copy()

    # Create task-specific frames
    hit_df = df.dropna(subset=["k_percent", "bb_percent", "xwoba"]).copy()
    hr_df = hit_df.copy()
    return hit_df, hr_df


def main():
    # Default: modest window to keep runtime reasonable for first run
    start = os.environ.get("TRAIN_START", f"{dt.date.today().year-1}-04-01")
    end = os.environ.get("TRAIN_END", f"{dt.date.today().year-1}-06-01")

    hit_df, hr_df = build_training(start, end)

    # Train models with context features included
    drop_cols = ("batter", "game_pk", "game_date", "player_name", "pa_rolling", "opp_pitcher_id", "venue", "venue_id")
    m_hit = train_baseline_classifier(
        hit_df.rename(columns={"label_hit": "label"}),
        "label",
        drop_cols,
        "batter_hit_prob",
        calib_method="isotonic",
        balance_classes=False,
    )

    drop_cols_hr = drop_cols + ("hr_per_pa",)  # ensure not using any leakage-like features
    m_hr = train_baseline_classifier(
        hr_df.rename(columns={"label_hr": "label"}),
        "label",
        drop_cols_hr,
        "batter_hr_prob",
        calib_method="sigmoid",
        balance_classes=True,
    )

    print("Saved models to:", MODELS_DIR)


if __name__ == "__main__":
    main()
