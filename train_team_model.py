import os
import datetime as dt
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
import joblib
import statsapi
import requests  # Added for odds fetching
from dotenv import load_dotenv  # NEW

MODELS_DIR = os.path.abspath("models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Load .env for ODDS_API_KEY, etc.
load_dotenv()


def fetch_games(year: int) -> pd.DataFrame:
    start = f"{year}-03-15"
    end = f"{year}-11-15"
    games = statsapi.schedule(start_date=start, end_date=end)
    rows = []
    for g in games:
        if g.get('status') != 'Final':
            continue
        home = g.get('home_name')
        away = g.get('away_name')
        hs = g.get('home_score')
        as_ = g.get('away_score')
        date = pd.to_datetime(g.get('game_date')).date()
        if home is None or away is None or hs is None or as_ is None:
            continue
        rows.append({
            'date': date,
            'game_pk': g.get('game_id') or g.get('game_pk'),
            'home': home,
            'away': away,
            'home_score': hs,
            'away_score': as_,
        })
    return pd.DataFrame(rows)


def build_team_records(gdf: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    teams = pd.unique(gdf[['home','away']].values.ravel('K'))
    recs: Dict[str, pd.DataFrame] = {}
    for t in teams:
        # Extract games involving team t
        sub = gdf[(gdf['home'] == t) | (gdf['away'] == t)].copy()
        sub = sub.sort_values('date')
        # outcome as viewed from team's perspective
        sub['is_home'] = (sub['home'] == t).astype(int)
        sub['team_runs'] = np.where(sub['is_home']==1, sub['home_score'], sub['away_score'])
        sub['opp_runs'] = np.where(sub['is_home']==1, sub['away_score'], sub['home_score'])
        sub['win'] = (sub['team_runs'] > sub['opp_runs']).astype(int)
        sub['loss'] = 1 - sub['win']
        # pregame cumulative (shifted)
        sub['cum_wins'] = sub['win'].cumsum().shift(1).fillna(0)
        sub['cum_losses'] = sub['loss'].cumsum().shift(1).fillna(0)
        sub['games'] = sub['cum_wins'] + sub['cum_losses']
        sub['win_pct'] = np.where(sub['games']>0, sub['cum_wins']/sub['games'], 0.5)
        recs[t] = sub[['date','game_pk','is_home','win','loss','cum_wins','cum_losses','games','win_pct']]
    return recs


def make_training_frame(years: Tuple[int, int]) -> pd.DataFrame:
    start_year, end_year = years
    frames = []
    for y in range(start_year, end_year+1):
        g = fetch_games(y)
        if g.empty:
            continue
        recs = build_team_records(g)
        # Build per game row with pregame win pct features
        merged = []
        g_sorted = g.sort_values('date')
        for _, row in g_sorted.iterrows():
            h = row['home']
            a = row['away']
            # lookup pregame records
            hrec = recs[h]
            arec = recs[a]
            # find matching game row
            hrow = hrec[(hrec['date'] == row['date']) & (hrec['game_pk'] == row['game_pk'])]
            arow = arec[(arec['date'] == row['date']) & (arec['game_pk'] == row['game_pk'])]
            if hrow.empty or arow.empty:
                continue
            target = int(row['home_score'] > row['away_score'])
            merged.append({
                'date': row['date'],
                'home': h,
                'away': a,
                'home_wpct': float(hrow['win_pct'].iloc[0]),
                'away_wpct': float(arow['win_pct'].iloc[0]),
                'label_home_win': target,
            })
        if merged:
            frames.append(pd.DataFrame(merged))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def train_team_model(start_year: int = 2022, end_year: int = None):
    if end_year is None:
        end_year = dt.date.today().year - 1
    df = make_training_frame((start_year, end_year))
    if df.empty:
        raise RuntimeError("No training data collected for team model.")
    X = df[['home_wpct','away_wpct']]
    y = df['label_home_win']
    base = LogisticRegression(max_iter=200)
    model = CalibratedClassifierCV(base, cv=5, method='isotonic')
    model.fit(X, y)
    p = model.predict_proba(X)[:,1]
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = float('nan')
    brier = brier_score_loss(y, p)
    print(f"Team model - AUC: {auc:.3f}, Brier: {brier:.3f}, N={len(df)}")
    joblib.dump(model, os.path.join(MODELS_DIR, 'team_win_prob.joblib'))
    return model


# --- Odds utilities (optional, requires ODDS_API_KEY) ---

def _moneyline_to_prob(ml: float) -> float:
    try:
        ml = float(ml)
    except Exception:
        return np.nan
    if ml > 0:
        return 100.0 / (ml + 100.0)
    else:
        return (-ml) / ((-ml) + 100.0)


def _fair_probs(p_home_raw: float, p_away_raw: float) -> Tuple[float, float]:
    if not np.isfinite(p_home_raw) or not np.isfinite(p_away_raw):
        return np.nan, np.nan
    s = p_home_raw + p_away_raw
    if s <= 0:
        return np.nan, np.nan
    return p_home_raw / s, p_away_raw / s


def _fetch_odds_today(api_key: str, regions: str = "us", bookmaker_preference = ("draftkings","fanduel","betmgm")) -> pd.DataFrame:
    """Fetch today's MLB moneyline odds via The Odds API and return a DF with implied/fair probabilities.
    If unavailable or fails, returns empty DataFrame.
    """
    try:
        today = dt.date.today().isoformat()
        url = (
            "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
            f"?apiKey={api_key}&regions={regions}&markets=h2h&oddsFormat=american&dateFormat=iso"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        events = resp.json()
        rows = []
        for ev in events:
            home_team = ev.get("home_team")
            away_team = ev.get("away_team")
            books = ev.get("bookmakers", [])
            chosen = None
            # pick preferred bookmaker
            pref_map = {b.lower(): b for b in bookmaker_preference}
            for b in books:
                key = (b.get("key") or "").lower()
                if key in pref_map:
                    chosen = b
                    break
            if chosen is None and books:
                chosen = books[0]
            if not chosen:
                continue
            markets = chosen.get("markets", [])
            h2h = next((m for m in markets if m.get("key") == "h2h"), None)
            if not h2h:
                continue
            outcomes = h2h.get("outcomes", [])
            home_ml = None
            away_ml = None
            for o in outcomes:
                if o.get("name") == home_team:
                    home_ml = o.get("price")
                if o.get("name") == away_team:
                    away_ml = o.get("price")
            ph_raw = _moneyline_to_prob(home_ml)
            pa_raw = _moneyline_to_prob(away_ml)
            ph_fair, pa_fair = _fair_probs(ph_raw, pa_raw)
            rows.append({
                "home": home_team,
                "away": away_team,
                "book": chosen.get("title") or chosen.get("key"),
                "home_ml": home_ml,
                "away_ml": away_ml,
                "implied_home_raw": ph_raw,
                "implied_away_raw": pa_raw,
                "implied_home_fair": ph_fair,
                "implied_away_fair": pa_fair,
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


# --- Predict today's games ---

def predict_today(output_csv: str = None):
    model_path = os.path.join(MODELS_DIR, 'team_win_prob.joblib')
    if not os.path.exists(model_path):
        raise RuntimeError("Train team model first (run train_team_model.py)")
    model = joblib.load(model_path)

    today = dt.date.today()
    season_start = dt.date(today.year, 3, 15)
    yesterday = today - dt.timedelta(days=1)

    # Get all finished games up to yesterday to compute pregame win% today
    hist = statsapi.schedule(start_date=season_start.strftime('%Y-%m-%d'), end_date=yesterday.strftime('%Y-%m-%d'))
    hist_rows = []
    for g in hist:
        if g.get('status') != 'Final':
            continue
        hist_rows.append({
            'date': pd.to_datetime(g.get('game_date')).date(),
            'game_pk': g.get('game_id') or g.get('game_pk'),
            'home': g.get('home_name'),
            'away': g.get('away_name'),
            'home_score': g.get('home_score'),
            'away_score': g.get('away_score'),
        })
    hist_df = pd.DataFrame(hist_rows)
    if hist_df.empty:
        raise RuntimeError('No historical games found to compute records.')

    recs = build_team_records(hist_df)

    # Today's schedule
    todays = statsapi.schedule(start_date=today.strftime('%Y-%m-%d'), end_date=today.strftime('%Y-%m-%d'))
    rows = []
    for g in todays:
        if g.get('status') not in ('Scheduled','Pre-Game','Warmup'):
            continue
        h = g.get('home_name')
        a = g.get('away_name')
        hwp = recs.get(h, pd.DataFrame({'win_pct':[0.5]}))['win_pct'].iloc[-1] if h in recs and not recs[h].empty else 0.5
        awp = recs.get(a, pd.DataFrame({'win_pct':[0.5]}))['win_pct'].iloc[-1] if a in recs and not recs[a].empty else 0.5
        rows.append({'date': today, 'home': h, 'away': a, 'home_wpct': float(hwp), 'away_wpct': float(awp)})

    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        print('No games scheduled today.')
        return pd.DataFrame()

    pred_df['p_home_win'] = model.predict_proba(pred_df[['home_wpct','away_wpct']])[:,1]

    # Optional: join odds for comparison
    api_key = os.getenv('ODDS_API_KEY')
    if api_key:
        odds_df = _fetch_odds_today(api_key)
        if not odds_df.empty:
            merged = pred_df.merge(odds_df, on=['home','away'], how='left')
            if 'implied_home_fair' in merged.columns:
                merged['edge_home'] = merged['p_home_win'] - merged['implied_home_fair']
            pred_df = merged
        else:
            print('Odds fetch returned empty; skipping odds join.')
    else:
        # Silent if no key; keep original pred_df
        pass

    if output_csv is None:
        output_csv = os.path.join('data', f'team_preds_{today}.csv')
    os.makedirs('data', exist_ok=True)
    pred_df.to_csv(output_csv, index=False)
    print('Wrote:', output_csv)
    return pred_df


if __name__ == '__main__':
    # Train if desired, then predict today
    if os.environ.get('TRAIN', '0') == '1':
        train_team_model()
    predict_today()
