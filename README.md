# FPL Points Prediction

University Neural Networks course project. We trained neural networks to predict weekly Fantasy Premier League points, fed those predictions into an Integer Linear Program that picks a squad and plans transfers, layered a chip strategy on top, and simulated the full 2025/26 season end-to-end.

## Headline result

On the just-completed 2025/26 season, our system scored **2,463 points** — roughly **rank 450** on the official FPL Overall leaderboard (top 0.005% of ~11 million human managers). The world's top-ranked manager finished on 2,582.

| Configuration | Net total | vs top human | Approx. global rank |
|---|---|---|---|
| No chips (baseline) | 2,384 | -198 | ~17,648 |
| + heuristic TC + BB | 2,440 | -142 | ~1,495 |
| **+ heuristic TC + BB + WC** | **2,463** | **-119** | **~450** |
| Oracle (perfect chip timing) | 2,522 | -60 | ~15 |

Full writeup in [FINAL_REPORT.docx](FINAL_REPORT.docx).

## Repo layout

```
FPL_project/
├── scripts/
│   ├── build_dataset.py            # Phase 1-2: historical pipeline
│   ├── build_live_dataset.py       # Phase 5: rebuild 2025/26 from FPL API
│   ├── squad_optimizer.py          # Phase 4: ILP — fresh / greedy / 4-GW horizon
│   ├── run_live_inference.py       # Run tuned decomposed model on live data
│   ├── run_live_backtest.py        # 2025/26 backtest (no chips)
│   ├── chip_strategy.py            # Phase 5: TC / BB / WC chip support
│   ├── run_live_chips.py           # 2025/26 backtest with chip strategy
│   ├── run_feature_ablations.py    # Phase 6: retrain with one group dropped
│   └── visualize_backtest.py       # Plots + per-GW season report
├── notebooks/
│   ├── 01_build_fpl_dataset.ipynb
│   ├── 02_phase3_baseline_model.ipynb
│   ├── 03_phase3_decomposed_heads.ipynb
│   └── 04_phase3_lstm.ipynb
├── data/processed/                 # generated CSVs (checked in)
├── results/                        # model weights, predictions, backtests, charts
├── FINAL_REPORT.docx
└── README.md
```

## How to reproduce, end to end

```bash
# 1. Build historical training dataset (2016-17 → 2024-25)
python scripts/build_dataset.py

# 2. Train models (Karmo's notebooks)
jupyter notebook notebooks/02_phase3_baseline_model.ipynb
jupyter notebook notebooks/03_phase3_decomposed_heads.ipynb   # tuned model
jupyter notebook notebooks/04_phase3_lstm.ipynb

# 3. Build live 2025/26 dataset from the FPL API (cached after first run)
python scripts/build_live_dataset.py

# 4. Run model inference on live data
python scripts/run_live_inference.py

# 5. Run the 2025/26 backtest with chip strategy
python scripts/run_live_chips.py

# 6. Feature-group ablation study
python scripts/run_feature_ablations.py
```

## Pipeline at a glance

### Phase 1-2 — data + features
`scripts/build_dataset.py` produces `data/processed/fpl_model_dataset.csv` (224k rows × 98 cols). One row per (player, gameweek). All features use only data available before the gameweek deadline (`.shift(1).rolling(...)` for every rolling stat).

Canonical IDs make historical ↔ live joining survive transfers and team renames:
- `player_id` — normalised name without numeric suffixes (bridges vaastav `Salah_233` with FPL API `Mohamed_Salah`)
- `team_name` — string club name, backfilled for 2016-17 → 2019-20 via `players_raw.csv` + master team list
- `opponent_team_name` — resolved from the season-scoped opponent ID
- `is_promoted_team` — cold-start signal for clubs that weren't in PL the prior season

Cross-season anchors `last_season_ppg` and `last_season_minutes_share` join on `player_id`, so they survive mid-season club changes.

### Phase 3 — model (Karmo's work)
Three architectures trained on 2016-2024, validated on 2024-25:
- Direct MLP (128 → 64 → 32)
- Decomposed multi-head with FPL-scoring recombination + hyperparameter sweep (best: `[256, 128, 64]`)
- 5-GW LSTM

Validation results on 2024-25 (Spearman ρ on played rows / all rows):

| Model | ρ_played | ρ_all | MAE_all |
|---|---|---|---|
| **mlp_decomposed_tuned** | **0.369** | **0.694** | **1.039** |
| mlp_points | 0.367 | 0.686 | 1.056 |
| mlp_lstm | 0.362 | 0.685 | 1.064 |
| roll5 baseline | 0.290 | 0.676 | 1.090 |
| ridge_linear | 0.347 | 0.644 | 1.115 |
| last_season_ppg | 0.226 | 0.326 | 1.553 |

### Phase 4 — squad optimizer
PuLP ILP encoding all FPL constraints: £100m budget, 15-man squad (2/5/5/3), max 3 per club, valid XI formation, captain doubling. Constants are 2025/26 rules: `MAX_BANKED_TRANSFERS=4` (5 usable in any GW), `HIT_COST=4`, `TRANSFERS_CAP_PER_GW=20`, `XI_MIN["MID"]=2`.

Three modes:
1. **Fresh squad** — no transfer constraint, picks the best XI each week (upper bound)
2. **Greedy single-GW** — realistic 2025/26 rules, no lookahead
3. **4-GW horizon** — multi-period ILP, plans transfers over a rolling window, commits only the first GW's decision

On the 2024-25 validation season the horizon mode scored 2,735 / 38 GW = 72.0 pts/GW (above the FPL global average of ~55-58).

### Phase 5 — live 2025/26 simulation
`build_live_dataset.py` calls the FPL API to pull per-(player, gameweek) history for all 841 players in 2025/26, then pushes the rows through the same feature pipeline as historical. `run_live_inference.py` loads the tuned decomposed model weights and produces predictions. `run_live_chips.py` runs the horizon backtest with chip strategy.

Chip strategy implemented: **Triple Captain**, **Bench Boost**, and **Wildcard** (Free Hit deferred). Deployment gameweeks are chosen by a greedy heuristic over the baseline backtest (highest predicted uplift per chip per season half), with stacking prevention. An oracle schedule using actual realised points is also computed as an upper bound.

### Phase 6 — feature ablation study
`run_feature_ablations.py` retrains the tuned decomposed model eight times, each time with one named feature group dropped. The headline finding: rolling player-form features dominate (Δρ_played = -0.029) — roughly 10× the impact of any other group.

## Known caveats

- **Live cold-start**: 359 of the 841 2025/26 players don't match historical via `player_id`. Most are genuinely new (promoted clubs Burnley/Leeds/Sunderland, summer signings, academy debuts); a smaller subset are name-normalisation failures that need a manual override pass.
- **DefCon excluded**: 2025/26 introduced defensive-contribution scoring (10 actions for DEF, 12 for MID/FWD → +2 pts). The model was trained on pre-2025/26 data so doesn't predict it; the FPL leaderboard reference total counts DefCon, so our absolute-points comparison is slightly pessimistic.
- **Single-run ablations**: training variance is ~±0.003-0.005 on ρ_played, so deltas below that should be read as noise.
- **Free Hit not implemented**: lower expected value than TC/BB/WC and more complex revert mechanics.
- **Per-position models** considered but skipped.
- **Price changes not modelled**: optimiser uses each GW's price uniformly, ignoring the 50% sell-on fee. Cumulative drift is ~£1-3m over a season.
