import os
import datetime as dt
import pandas as pd
import joblib
from data_pipeline import load_snapshots, make_batter_features, make_pitcher_features, build_batter_context, get_confirmed_starters
from ml_models import predict_proba
from train_team_model import predict_today as predict_teams_today

MODELS_DIR = os.path.abspath("models")
# Remove hardcoded feature list; rely on features saved with the model bundles for alignment


def load_models():
    # Return the saved bundles (dicts with model + feature_names)
    m_hit = joblib.load(os.path.join(MODELS_DIR, "batter_hit_prob.joblib"))
    m_hr = joblib.load(os.path.join(MODELS_DIR, "batter_hr_prob.joblib"))
    return m_hit, m_hr


def _apply_context_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight multiplicative adjustments using park and opponent pitcher.
    We cap adjustments to avoid extreme probabilities.
    """
    out = df.copy()
    # Anchors
    anchor_xwoba = 0.320
    anchor_barrel = 0.075
    anchor_k = 0.230

    # Baseline factors (Series)
    park_hr = pd.to_numeric(out.get("park_hr_factor"), errors='coerce') if "park_hr_factor" in out.columns else pd.Series(1.0, index=out.index)
    park_hit = pd.to_numeric(out.get("park_hit_factor"), errors='coerce') if "park_hit_factor" in out.columns else pd.Series(1.0, index=out.index)
    park_hr = park_hr.fillna(1.0)
    park_hit = park_hit.fillna(1.0)

    # Opponent pitcher quality proxies
    if "opp_xwoba" in out.columns:
        opp_xwoba = pd.to_numeric(out["opp_xwoba"], errors='coerce').fillna(anchor_xwoba)
    else:
        opp_xwoba = pd.Series(anchor_xwoba, index=out.index)
    if "opp_barrel_batted_rate" in out.columns:
        opp_barrel = pd.to_numeric(out["opp_barrel_batted_rate"], errors='coerce').fillna(anchor_barrel)
    else:
        opp_barrel = pd.Series(anchor_barrel, index=out.index)
    if "opp_k_percent" in out.columns:
        opp_k = pd.to_numeric(out["opp_k_percent"], errors='coerce').fillna(anchor_k)
    else:
        opp_k = pd.Series(anchor_k, index=out.index)

    # Build suppression multiplier around anchors
    pitcher_factor = (
        (anchor_xwoba / opp_xwoba.clip(lower=0.250, upper=0.400)).pow(0.5) *
        (anchor_barrel / opp_barrel.clip(lower=0.02, upper=0.15)).pow(0.3) *
        (anchor_k / opp_k.clip(lower=0.10, upper=0.35)).pow(0.2)
    )
    pitcher_factor = pitcher_factor.clip(lower=0.80, upper=1.20).fillna(1.0)

    # Apply to HR and Hit separately with park emphasis
    if "p_hr" in out.columns:
        adj_hr = out["p_hr"] * park_hr.clip(0.8, 1.2) * pitcher_factor
        out["p_hr_adj"] = adj_hr.clip(lower=0.001, upper=0.35)
    if "p_hit" in out.columns:
        adj_hit = out["p_hit"] * park_hit.clip(0.9, 1.15) * pitcher_factor.pow(0.5)
        out["p_hit_adj"] = adj_hit.clip(lower=0.05, upper=0.95)

    return out


def predict_batters(snapshot_date=None):
    bat_snap, pit_snap = load_snapshots(snapshot_date)
    batX = make_batter_features(bat_snap)
    # Build matchup context
    ctx = build_batter_context(bat_snap, pit_snap, snapshot_date)
    # Merge minimal context columns back to feature table
    if not ctx.empty:
        keep_cols = [c for c in [
            "player_id","team_id","venue","venue_id","opp_pitcher_id","opp_pitcher_name",
            "park_hr_factor","park_hit_factor","opp_xwoba","opp_barrel_batted_rate","opp_k_percent"
        ] if c in ctx.columns]
        batX = batX.merge(ctx[keep_cols], on="player_id", how="left")

    m_hit, m_hr = load_models()
    # Use centralized alignment helper to match training features
    batX_pred = batX.copy()
    batX_pred["p_hit"] = predict_proba(m_hit, batX_pred, drop_cols=())
    batX_pred["p_hr"] = predict_proba(m_hr, batX_pred, drop_cols=())

    # Apply adjustments if context is available
    batX_pred = _apply_context_adjustments(batX_pred)

    # Add lineup flag if confirmed
    starters = get_confirmed_starters(snapshot_date)
    if not starters.empty:
        starters["is_starter"] = True
        batX_pred = batX_pred.merge(starters[["player_id","is_starter"]], on="player_id", how="left")
        batX_pred["is_starter"] = (
            batX_pred["is_starter"].fillna(False).infer_objects(copy=False).astype(bool)
        )

    return batX_pred


def print_console_summary(bat_preds: pd.DataFrame, team_preds: pd.DataFrame):
    # Prefer starters for top picks when available
    preds = bat_preds.copy()
    if "is_starter" in preds.columns:
        starters_only = preds[preds["is_starter"] == True]
        if not starters_only.empty:
            preds = starters_only

    # Best HR
    hr_row = None
    col_hr = "p_hr_adj" if "p_hr_adj" in preds.columns else "p_hr"
    if not preds.empty and col_hr in preds.columns:
        hr_row = preds.loc[preds[col_hr].idxmax()].to_dict()
    # Best Hit
    hit_row = None
    col_hit = "p_hit_adj" if "p_hit_adj" in preds.columns else "p_hit"
    if not preds.empty and col_hit in preds.columns:
        hit_row = preds.loc[preds[col_hit].idxmax()].to_dict()

    # All projected winners (consider home and away)
    team_lines = []
    if team_preds is not None and not team_preds.empty and "p_home_win" in team_preds.columns:
        def pick(row):
            if row["p_home_win"] >= 0.5:
                return {
                    "team": row["home"],
                    "opp": row["away"],
                    "p": float(row["p_home_win"]),
                    "venue": "home",
                }
            else:
                return {
                    "team": row["away"],
                    "opp": row["home"],
                    "p": float(1 - row["p_home_win"]),
                    "venue": "away",
                }
        picks = team_preds.apply(pick, axis=1).tolist()
        picks.sort(key=lambda d: d["p"], reverse=True)
        for d in picks:
            team_lines.append(f"{d['team']} over {d['opp']} ({d['p']:.1%}, {d['venue']})")

    # Optional edges if odds were joined
    edge_lines = []
    if team_preds is not None and not team_preds.empty and "p_home_win" in team_preds.columns and (
        "implied_home_fair" in team_preds.columns or "implied_away_fair" in team_preds.columns
    ):
        def edge_pick(row):
            if row["p_home_win"] >= 0.5:
                model_p = float(row["p_home_win"])
                fair = row.get("implied_home_fair")
                team = row["home"]
                opp = row["away"]
                venue = "home"
            else:
                model_p = float(1 - row["p_home_win"])
                fair = row.get("implied_away_fair")
                team = row["away"]
                opp = row["home"]
                venue = "away"
            if pd.notna(fair):
                return {
                    "team": team,
                    "opp": opp,
                    "p": model_p,
                    "fair": float(fair),
                    "edge": model_p - float(fair),
                    "venue": venue,
                }
            return None
        edges = [e for e in team_preds.apply(edge_pick, axis=1).tolist() if e]
        # Filter for positive edges >= 3%
        edges = [e for e in edges if e["edge"] >= 0.03]
        edges.sort(key=lambda d: d["edge"], reverse=True)
        for e in edges[:5]:
            edge_lines.append(f"Edge: {e['team']} over {e['opp']} (+{e['edge']*100:.1f}%)")

    # Compose output lines
    lines = []
    if team_lines:
        lines.append("Projected winners:")
        lines.extend(team_lines)
    if edge_lines:
        lines.append("Top edges:")
        lines.extend(edge_lines)
    if hr_row is not None:
        name = hr_row.get("name") or f"Player {int(hr_row.get('player_id')) if pd.notna(hr_row.get('player_id')) else ''}"
        p = hr_row.get(col_hr, hr_row.get("p_hr"))
        lines.append(f"Best HR: {name} ({p:.1%})")
    if hit_row is not None:
        name = hit_row.get("name") or f"Player {int(hit_row.get('player_id')) if pd.notna(hit_row.get('player_id')) else ''}"
        p = hit_row.get(col_hit, hit_row.get("p_hit"))
        lines.append(f"Best Hit: {name} ({p:.1%})")

    if lines:
        print("\n".join(lines))


def main():
    today = dt.date.today().isoformat()
    out = predict_batters(today)
    # Save full set
    out_path = os.path.join("data", f"predictions_{today}.csv")
    out.to_csv(out_path, index=False)
    print("Wrote:", out_path)

    # Also save starters-only view if lineup data exists
    if "is_starter" in out.columns:
        starters_only = out[out["is_starter"] == True].copy()
        starters_path = os.path.join("data", f"predictions_{today}_starters.csv")
        starters_only.to_csv(starters_path, index=False)
        print("Wrote:", starters_path)

    # Team predictions and console summary
    try:
        team_preds = predict_teams_today()
    except Exception as e:
        print("Team predictions skipped:", e)
        team_preds = pd.DataFrame()

    print_console_summary(out, team_preds)


if __name__ == "__main__":
    main()
