# FPL Points Prediction — Project Context

## Project overview

Build a neural network system that predicts Fantasy Premier League (FPL) weekly points for all ~600 players, then uses those predictions to automatically select and manage a squad across an entire season. The system is evaluated by simulating the full 2025-26 PL season and comparing the bot's total points against the live FPL global leaderboard.

This is a university Neural Networks course project built in Python with PyTorch.

---

## Pipeline architecture

The system has 6 phases:

### Phase 1: Data collection

**Training data (2016-17 to 2024-25):**
- Primary source: `vaastav/Fantasy-Premier-League` GitHub repo
  - URL pattern: `https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv`
  - Seasons: `2016-17`, `2017-18`, `2018-19`, `2019-20`, `2020-21`, `2021-22`, `2022-23`, `2023-24`, `2024-25`
  - Key columns: `total_points, minutes, goals_scored, assists, clean_sheets, bonus, bps, creativity, influence, threat, ict_index, value, transfers_in, transfers_out, selected, was_home, opponent_team, GW, name, position, team`
  - Also has per-player files at `data/{season}/players/{player_name}_{id}/gw.csv`
  - And `data/{season}/cleaned_players.csv` for season-level overview

- Enrichment source: Understat (xG/xA per match)
  - Scrape via Python (`understatapi` package or manual scraping — data is embedded as JSON in `<script>` tags on the page)
  - Provides per-shot xG, xA, xGChain, xGBuildup, shot locations
  - Available from 2014-15 onward
  - Player ID matching needed (name + team + season fuzzy matching against vaastav IDs)

**Test data (2025-26 season — live):**
- Official FPL API: `https://fantasy.premierleague.com/api/`
  - `bootstrap-static/` — all players (elements array), teams, gameweeks (events), with season totals, prices, form, ICT, xG/xA/xGI. Also has `average_entry_score` per gameweek in the events data.
  - `element-summary/{player_id}/` — per-match history for a specific player + upcoming fixtures
  - `fixtures/` — all fixtures with difficulty ratings
  - `event/{gw}/live/` — live/actual points for all players in a specific gameweek
  - `leagues-classic/314/standings/` — overall global leaderboard (paginated)
- The 2025-26 season ends May 24, 2026, so nearly all GWs are available for backtesting

### Phase 2: Feature engineering

For each player before each GW, compute features using ONLY data available before the GW deadline (no leakage). Feature groups:

1. **Player form**: Rolling averages over last 3/5/10 GWs of: total_points, goals_scored, assists, minutes, xG, xA, bonus, bps, ICT index
2. **Fixture context**: Opponent team strength (FDR rating from API), home/away flag, days since last match, double/blank gameweek indicator
3. **Team-level**: Team goals scored/conceded (rolling), league position, team Elo rating (can compute from results)
4. **Market signals**: Price (value), price change trend, ownership %, transfers in/out volume
5. **Player metadata**: Position (GK=1, DEF=2, MID=3, FWD=4), minutes trend (starting vs bench), season-long averages as anchoring features

### Phase 3: Neural network model

**Architecture — feedforward NN:**
- Input: 50-100 features per player per GW
- Hidden layers: 128 → 64 → 32 with ReLU activations, batch normalization, dropout (~0.2-0.3)
- Decomposed output heads (multi-task learning) — predict sub-events separately rather than raw total_points:
  - P(start) — will the player start? (binary classification)
  - P(goal | started) — probability of scoring (regression or Poisson)
  - P(assist | started) — probability of assist
  - P(clean sheet | team) — for DEF/GK (binary, team-level)
  - P(bonus) — expected bonus points
- Combine predictions into expected FPL points using the official scoring rules:
  - GK/DEF: 6 pts/goal, 1 pt/CS, 4 pts/pen save, etc.
  - MID: 5 pts/goal, 3 pts/assist
  - FWD: 4 pts/goal, 3 pts/assist
  - All: 2 pts for 60+ min, 1 pt for 1-59 min, -1 per 2 goals conceded (GK/DEF), bonus points

**Design decisions:**
- Consider training separate models per position (GK/DEF/MID/FWD) since scoring dynamics differ significantly
- Loss function: MSE for regression heads, BCE for binary heads, combined with task weights
- Optimizer: Adam with learning rate scheduling
- Alternative to explore: LSTM/GRU that takes a sequence of the last N gameweeks per player as input to model form trajectories

**Training setup:**
- Train on seasons 2016-17 through 2023-24
- Validation on 2024-25
- Test on 2025-26 (out-of-sample, live API)
- ~200k total training samples (600 players × 38 GWs × 9 seasons, minus players with 0 minutes)
- Trains in < 5 minutes on CPU, no GPU needed
- Hyperparameter search: learning rate, hidden sizes, dropout rate, rolling window lengths, sequence length (if LSTM)

### Phase 4: Squad optimizer

Given predicted expected points for all ~600 players, select the optimal 15-man squad. This is an Integer Linear Programming (ILP) problem.

**Use PuLP (Python ILP library).**

**Constraints:**
- Total squad cost ≤ £100m (prices from API, in 0.1m units)
- Exactly 2 GK, 5 DEF, 5 MID, 3 FWD
- Max 3 players from any single team
- Starting XI must be a valid formation: 1 GK + at least 3 DEF + at least 1 FWD, 11 total
- Bench order matters (auto-subs happen in order)

**Objective:** Maximize total expected points of starting XI + captain bonus (2x the captain's predicted points).

**Captain selection:** Pick the player with the highest predicted points in the starting XI. The captain's score is doubled, so this decision is critical — often worth more than transfers.

### Phase 5: Weekly simulation loop

For each GW in the 2025-26 season (GW1 through GW38):

1. **Compute features** for all players using only data available before this GW's deadline
2. **Run the NN model** to predict expected points for all players for the upcoming GW
3. **Decide transfers:**
   - 1 free transfer per week (rolls over up to max 2)
   - Each additional transfer costs -4 points
   - Only transfer if: `E[points_gained_over_remaining_horizon] > 4` for a hit
   - Need a threshold/policy to avoid chasing noise
4. **Re-optimize starting XI** from the 15-man squad (formation, bench order)
5. **Select captain** (highest predicted points in starting XI)
6. **Lock in the team**
7. **Score against actual results** — pull real points from the FPL API
8. **Update feature pipeline** with the new actual data for the next iteration

**Chip strategy (can start with simple heuristics):**
- Wildcard (2 per season): unlimited free transfers for one GW. Use when squad needs major overhaul (e.g., after international break, or when many high-value players emerge).
- Bench Boost: bench players score too. Use on a double gameweek (DGW) when bench has two-game players.
- Triple Captain: captain scores 3x instead of 2x. Use on a DGW for the single best fixture.
- Free Hit: make unlimited transfers for one GW only, then squad reverts. Use on a blank gameweek (BGW) with many postponed fixtures.

### Phase 6: Evaluation

**Model-level metrics:**
- MAE / RMSE of predicted vs actual weekly points
- R² score — variance explained
- Spearman rank correlation (ρ) per GW — do we correctly rank the top options?

**System-level metrics:**
- Total season points → map to overall rank via FPL API leaderboard
- Points vs global average per GW (from `events` data in `bootstrap-static`)
- % of GWs where bot beats the average
- Captain success rate — % of GWs where captain was the highest scorer in our starting XI
- Transfer ROI — net points gained/lost from transfers

**Baselines to beat:**
1. Rolling average heuristic: pick players with highest average points over last 5 GWs
2. Linear regression: same features, but linear model (isolates whether NN adds value)
3. Global average manager score: are we better than the crowd?

**Ablation studies:**
- Drop feature groups (no xG, no market signals, no team-level) to quantify contribution
- Feedforward NN vs LSTM/GRU comparison
- Decomposed heads vs direct total_points prediction
- Single model vs per-position models

---

## Technical stack

- Python 3.10+
- PyTorch (model training and inference)
- pandas (data loading, feature engineering)
- numpy
- PuLP (integer linear programming for squad optimization)
- requests (FPL API calls)
- understatapi or custom scraper (Understat xG data)
- scikit-learn (baselines, metrics, preprocessing)
- matplotlib/seaborn (visualization of results)

---

## Project structure (suggested)

```
fpl-predictor/
├── data/
│   ├── raw/                    # Downloaded CSVs from vaastav
│   ├── understat/              # Scraped xG data
│   ├── api/                    # Cached FPL API responses
│   └── processed/              # Feature-engineered datasets
├── src/
│   ├── data/
│   │   ├── collect.py          # Download vaastav data + scrape Understat
│   │   ├── api.py              # FPL API wrapper
│   │   └── features.py         # Feature engineering pipeline
│   ├── model/
│   │   ├── network.py          # NN architecture (PyTorch)
│   │   ├── train.py            # Training loop
│   │   └── predict.py          # Inference
│   ├── optimizer/
│   │   ├── squad.py            # ILP squad selection (PuLP)
│   │   ├── transfer.py         # Transfer decision logic
│   │   └── captain.py          # Captain selection
│   ├── simulation/
│   │   ├── engine.py           # Main GW-by-GW simulation loop
│   │   ├── chips.py            # Chip usage strategy
│   │   └── scorer.py           # Score a team against actual results
│   └── evaluation/
│       ├── metrics.py          # MAE, RMSE, Spearman, etc.
│       ├── baselines.py        # Rolling avg, linear regression baselines
│       └── analysis.py         # Ablation studies, visualizations
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_simulation_results.ipynb
├── results/                    # Saved model weights, simulation outputs
├── requirements.txt
└── README.md
```

---

## FPL scoring rules reference

| Event | GK | DEF | MID | FWD |
|-------|-----|-----|-----|-----|
| Playing 1-59 min | 1 | 1 | 1 | 1 |
| Playing 60+ min | 2 | 2 | 2 | 2 |
| Goal scored | 6 | 6 | 5 | 4 |
| Assist | 3 | 3 | 3 | 3 |
| Clean sheet | 4 | 4 | 1 | 0 |
| Every 3 saves (GK) | 1 | - | - | - |
| Penalty save | 5 | 5 | 5 | 5 |
| Penalty miss | -2 | -2 | -2 | -2 |
| Every 2 goals conceded | -1 | -1 | 0 | 0 |
| Yellow card | -1 | -1 | -1 | -1 |
| Red card | -3 | -3 | -3 | -3 |
| Own goal | -2 | -2 | -2 | -2 |
| Bonus (1-3 pts) | BPS-based | BPS-based | BPS-based | BPS-based |

Captain doubles all points. Triple Captain triples all points.

---

## Key risks and things to watch

- **Player ID matching across sources**: vaastav uses FPL element IDs, Understat has its own IDs, need fuzzy name+team matching
- **Promoted/relegated teams**: new teams each season have no historical data in the system — handle gracefully (league-average priors)
- **Mid-season transfers**: players who change clubs need their team feature updated
- **Double/blank gameweeks**: some GWs have teams playing 2 games or 0 games — affects feature computation and chip timing
- **Signal-to-noise ratio**: weekly FPL points are very noisy (~30-40% variance explained at best). Transfer logic should be conservative to avoid chasing noise with -4 hits
- **Data leakage**: the most critical thing to get right. Every feature must use only pre-deadline data. Be especially careful with rolling averages near season boundaries.
