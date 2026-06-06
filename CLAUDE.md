# Project context for Claude

University NN course project. Two contributors: **Karmo** (this repo's owner, Phase 3 model) and **Eric** (data + features + downstream phases). See `README.md` for the v1 dataset schema; this file is the working-notes layer.

## Phase status

- **Phase 1 (data collection)** — done. vaastav 2016-17 → 2024-25, live FPL API, Understat season-aggregate.
- **Phase 2 (feature engineering)** — v1 complete. Known v2 backlog listed in README "Known gaps".
- **Phase 3 (NN model)** — **Karmo's task.** v1 done: baseline MLP + decomposed multi-head, weights in `results/`. v2 backlog: hyperparam sweep, per-position models, LSTM/GRU variant.
- **Phase 4 (PuLP squad optimizer)** — done. Three modes in `scripts/squad_optimizer.py`: fresh-squad, transfer-constrained greedy, and 4-GW horizon. Backtest harness + visualizer in place.
- **Phase 5 (full season simulation)** — partially done. Backtest already walks GW1→GW38 with state. Still missing: chip strategy (2× wildcard / free hit / bench boost / triple captain per the 2025/26 rules) and live 2025/26 inference path.
- **Phase 6 (evaluation/baselines)** — partial. Model-level metrics done (MAE/RMSE/ρ vs ridge + heuristics). System-level done (fresh vs greedy vs horizon, FPL avg comparison). Missing: ablation studies, formal baseline writeup.

## Things to know before editing `scripts/build_dataset.py`

- **The `team` column in vaastav's `merged_gw.csv` is a STRING name in 2020-21+ and MISSING in 2016-17 → 2019-20.** Do NOT pipe it through `pd.to_numeric` — that silently coerces everything to NaN. The current loader handles both cases per-season; preserve that pattern.
- **`master_team_list.csv` lags by one season.** Current code falls back to per-season `teams.csv` (which exists 2020-21+) via `load_master_team_list(extra_seasons=...)`. Keep that fallback when extending.
- **Two name keys exist on purpose:**
  - `name_key` keeps digits — used for the Understat merge where vaastav's `Salah_233`-style suffix is a feature, not a bug
  - `player_id` strips digits — used for cross-season and historical↔live joins
  - Don't unify them.
- **All rolling features must use `.shift(1)` before `.rolling(...)`.** Otherwise the current GW's label leaks into its own features. The pattern is enforced in `add_leakage_safe_features`; copy it for any new rolling feature.
- **Cross-season anchors are joined by `player_id`, not by `(name, team)`.** This is intentional — it makes anchors survive mid-season club changes (verified case: Mbeumo, Brentford → Man Utd). Don't add team to the anchor merge key.

## Known data caveats

- **359/840 live 2025-26 players don't match any historical row by `player_id`.** Most are genuinely new (promoted clubs Burnley/Leeds/Sunderland, summer signings, academy debuts). A subset are name-normalization failures (`Ødegaard` vs `Odegaard`, `Son Heung-min` vs `Heung-Min_Son`, nicknames like `Rodri`). Deferred until end-to-end pipeline exists so we can prioritize by who actually shows up in squads.
- **Position column is inconsistent in older seasons.** Empty in 2016-17 sample rows. If training per-position models, either drop early seasons or backfill from `players_raw.csv`.
- **Understat enrichment is season-aggregate only.** Per-match xG/xA scraping is on the v2 backlog — biggest single signal upgrade still pending.
- **DGW/BGW (double/blank gameweeks)** are not flagged. A few rows per season are affected; can corrupt rolling-window features if not handled.

## Things to know before editing `scripts/squad_optimizer.py`

- **2025/26 rules, not historical.** Constants at the top: `MAX_BANKED_TRANSFERS=4` (=5 max usable in any GW), `HIT_COST=4`, `TRANSFERS_CAP_PER_GW=20`, `XI_MIN["MID"]=2`. Don't drift these back to historical values.
- **DGW duplicates must be aggregated before optimizing.** The historical CSV has multiple rows per (element, gw) when a team has two fixtures in one GW. `_aggregate_dgw_rows()` collapses them. Without it the ILP can pick the same player twice.
- **The horizon optimizer's banked-transfer LP variable has slack.** Internal `b[t]` values in `horizon_plan` may not be tight; use the post-hoc `banked_next` computed from the squad delta as the source of truth for state propagation.
- **Pool trimming to top-150 per GW + current squad** is a heuristic to keep the horizon ILP tractable (~3s/decision). Widening to 200 might unlock a few more pts; profile before changing.
- **Price changes are not modeled.** Player prices use the current GW's `value` uniformly; 50% sell-on fee is ignored. Cumulative drift is ~£1-3m over a season — fine for v1.

## Conventions

- Comments in `build_dataset.py` lean on explaining *why*, not *what*. Match that style.
- New features get added to the pipeline in `build_dataset` (the orchestrator), not in the notebook. The notebook is a thin demo.
- Output CSVs are checked into git for collaboration. Historical CSV is ~84MB — past GitHub's 50MB warning. If it grows past 100MB, switch to git-lfs.

## Work log

**2026-05-23 (Karmo)** — Phase 3 follow-up:
- Fixed `03_phase3_decomposed_heads.ipynb`: consistent metrics (`mae_all` / `spearman_all` + played slice), fair comparison table, FPL point recombination, search by `spearman_all`; last cell now writes `mlp_decomposed_tuned` to `results/phase3_model_comparison_with_decomposed.csv`.
- Added `04_phase3_lstm.ipynb` (5-GW LSTM + static context); outputs in `results/phase3_model_comparison_with_lstm.csv`, `phase3_lstm.pt`. LSTM ≈ basic MLP, slightly below tuned decomposed — no major gain.
- Decomposed hyperparam search saved to `results/phase3_decomposed_search.csv`. Current best: **mlp_decomposed_tuned** (ρ_all ~0.694). Per-position models and live 2025/26 inference still TODO.
