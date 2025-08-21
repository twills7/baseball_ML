# Baseball Predictor

End-to-end MLB ML pipeline that predicts per-game:

- Player hit probability
- Player home run probability
- Team win probability

Daily predictions join matchup context (probable pitcher, park factors) and confirmed lineups, and can compare team edges vs Vegas odds.

## Features

- Data: season-to-date batter/pitcher snapshots + Statcast (pybaseball) + MLB StatsAPI schedule/lineups
- Models: calibrated classifiers for batter hit/HR and team win
- Context: opponent pitcher stats, park HR/hit factors, confirmed starters
- Outputs: CSVs for all players and starters-only, plus team predictions with optional odds comparisons
- Console summary: projected winners, top edges (if odds set), top hit/HR players

## Setup

1. Create and activate a virtual environment (optional but recommended)
2. Install dependencies
3. Create a `.env` in the project root (do not commit secrets)

Install deps:

```bash
pip install -r requirements.txt
```

Environment variables (example):

```env
# Optional: market odds comparison (The Odds API)
ODDS_API_KEY="your_odds_api_key"

# Optional: future integrations
DISCORD_TOKEN="your_discord_bot_token"
OPENAI_API_KEY="your_openai_key"
GEMINI_API_KEY="your_gemini_key"
```

Note: `.env` and `data/` are already gitignored.

## Data files

- `data_snapshots/`: input snapshots (`batter_*.csv`, `pitcher_*.csv`)
- `data/park_factors.csv`: park HR/hit multipliers (editable)
- `models/`: trained models (`*.joblib`)
- `data/`: outputs (predictions and team preds CSVs)

## Train models

- Team model (historical schedule results → calibrated logistic regression):

```bash
TRAIN=1 python train_team_model.py
```

Or from Python:

```python
from train_team_model import train_team_model
import datetime as dt
train_team_model(2021, dt.date.today().year)
```

- Batter models (Statcast-based features; window configured in `train_batter_models.py`):

```bash
python train_batter_models.py
```

Models are saved under `models/`:

- `batter_hit_prob.joblib`
- `batter_hr_prob.joblib`
- `team_win_prob.joblib`

## Daily predictions

Run:

```bash
python predict_today.py
```

Writes:

- `data/predictions_{YYYY-MM-DD}.csv`
- `data/predictions_{YYYY-MM-DD}_starters.csv` (if lineups available)
- `data/team_preds_{YYYY-MM-DD}.csv`

Console prints projected winners, top edges (if odds enabled), and best HR and hit picks.

## Odds integration (optional)

Set `ODDS_API_KEY` in `.env` (The Odds API). Team predictions will include:

- `implied_home_fair`, `implied_away_fair`
- `edge_home` (model vs fair market)

The console will also print Top edges.

## Confirmed lineups

The pipeline fetches confirmed starters via MLB StatsAPI and flags `is_starter`. A starters-only CSV is also written.

## Notes on calibration

- Class imbalance handled; probabilities are calibrated (isotonic) for better realism, especially HRs.
- Light context-based adjustments applied post-prediction (park, pitcher) with caps to avoid extremes.

## Troubleshooting

- StatsAPI/pybaseball timeouts: re-run; caching is enabled for Statcast to speed retries.
- Missing odds/edges: ensure `ODDS_API_KEY` is present and valid; API may rate-limit.
- Pandas FutureWarnings are harmless; they do not affect outputs.

## Docker (optional)

Build and run:

```bash
docker build -t bettor-ai .
docker run --rm -v "$PWD/data:/app/data" --env-file .env bettor-ai python predict_today.py
```

This mounts `data/` for outputs and loads env vars for odds.