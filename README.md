# FPL Points Prediction

University NN course project: predict weekly Fantasy Premier League points for all ~600 players, then use those predictions to auto-manage a squad across the 2025-26 season.

## v1 dataset (ready for modelling)

Run `python scripts/build_dataset.py` to (re)generate. Outputs:

- `data/processed/fpl_model_dataset.csv` — historical training data, 224,143 rows × 98 cols, 2016-17 → 2024-25
- `data/processed/live_player_pool.csv` — 2025-26 player pool from the live FPL API, 840 rows × 33 cols

### Row grain
One row per (player, gameweek). Sorted by `(season, element, gw)`.

### Identity columns (canonical, stable across seasons)
| Column | Description |
|---|---|
| `player_id` | Normalized, numeric-suffix-stripped key derived from full name. Bridges historical ↔ live. |
| `name_key` | Looser normalized key (keeps digits) — used for Understat merge. |
| `team_name` | Club name as a string. Backfilled for 2016-17 → 2019-20 via `players_raw.csv` + master team list. |
| `opponent_team_name` | Resolved from the season-scoped `opponent_team` int ID. |
| `season`, `season_start` | e.g. "2024-25", 2024. |
| `is_promoted_team` | True if club wasn't in PL the prior season. NA for the earliest season. |

### Target columns
`total_points` (primary), plus components for decomposed heads: `minutes`, `goals_scored`, `assists`, `clean_sheets`, `goals_conceded`, `bonus`, `bps`.

### Leakage-safe features (use only pre-deadline data)
- **Player form**: `total_points_roll{3,5}`, `minutes_roll{3,5}`, `goals_scored_roll{3,5}`, `assists_roll{3,5}`, `expected_goals_roll{3,5}`, `expected_assists_roll{3,5}` — rolling means using `.shift(1)`.
- **Team form**: `team_goals_scored_gw_roll5`, `team_goals_conceded_gw_roll5`, `team_points_gw_roll5`.
- **Opponent strength**: `opponent_team_points_roll5`, `opponent_team_gc_roll5`.
- **Cross-season anchors**: `last_season_ppg`, `last_season_minutes_share` (joined by `player_id`; survives mid-season club changes).
- **Fixture context**: `was_home`, `rest_days`.
- **Market signals**: `value`, `selected`, `transfers_in`, `transfers_out`, `transfers_balance`.

### Known gaps (v1 → v2 backlog)
- **FDR**: no fixture difficulty rating yet — currently using rolling-points-as-strength proxy. FPL API exposes this; cheap follow-up.
- **Position**: column inconsistent across older seasons; backfill from `players_raw.csv` if training per-position models.
- **Double/blank GW flags**: not encoded; a few rows per season are affected.
- **Per-match Understat xG**: only season-aggregate xG/xA merged; per-match scraping pending.
- **Live cold-start**: 359/840 of the 2025-26 live pool don't match historical via `player_id` (most are genuinely new — promoted teams, summer signings, academy debuts; a smaller subset are name-normalization failures that need a manual override pass).

## Pipeline

See `fpl_project_context.md` (in parent dir) for the full 6-phase plan: data → features → NN → optimizer → simulation → evaluation.

## Layout

```
FPL_project/
├── scripts/
│   ├── build_dataset.py          # Phase 1-2 pipeline
│   ├── squad_optimizer.py        # Phase 4 ILP (single-GW, transfer-constrained, 4-GW horizon)
│   └── visualize_backtest.py     # Phase 4 plots + season report
├── notebooks/
│   ├── 01_build_fpl_dataset.ipynb
│   ├── 02_phase3_baseline_model.ipynb
│   └── 03_phase3_decomposed_heads.ipynb
├── data/processed/               # generated CSVs (checked in for collaboration)
├── results/                      # model weights, predictions, backtests, charts
└── README.md
```

## Phase 4 squad optimizer (PuLP ILP)

Run `python scripts/squad_optimizer.py` to backtest. Three optimizers, all using 2025/26 rules (max 5 banked transfers, -4 per hit, 1-3-DEF / 1-MID-2 / 1-FWD minimums):

| Mode | Season net (2024-25 val) | Per GW |
|---|---|---|
| Fresh squad (no transfer limit, upper bound) | 2,830 pts | 74.5 |
| **4-GW horizon (realistic)** | **2,735 pts** | **72.0** |
| Greedy single-GW (no lookahead) | 2,363 pts | 62.2 |

FPL global average ≈ 55-58/GW. Horizon optimizer recovers ~80% of the transfer friction over greedy.

Run `python scripts/visualize_backtest.py` for a per-GW season report ([results/phase4_season_report.md](results/phase4_season_report.md)), squad-tenure heatmap, and score timeline.
