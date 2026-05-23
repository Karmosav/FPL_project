# Project context for Claude

University NN course project. Two contributors: **Karmo** (this repo's owner, Phase 3 model) and **Eric** (data + features + downstream phases). See `README.md` for the v1 dataset schema; this file is the working-notes layer.

## Phase status

- **Phase 1 (data collection)** — done. vaastav 2016-17 → 2024-25, live FPL API, Understat season-aggregate.
- **Phase 2 (feature engineering)** — v1 complete. Known v2 backlog listed in README "Known gaps".
- **Phase 3 (NN model)** — **Karmo's task.** Not started.
- **Phase 4 (PuLP squad optimizer)** — Eric's next step. Model-orthogonal: can be built with placeholder predictions (e.g. `last_season_ppg`) and swapped to real predictions when Phase 3 lands.
- **Phase 5 (weekly simulation loop)** — blocked on 4.
- **Phase 6 (evaluation/baselines)** — not started.

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

## Conventions

- Comments in `build_dataset.py` lean on explaining *why*, not *what*. Match that style.
- New features get added to the pipeline in `build_dataset` (the orchestrator), not in the notebook. The notebook is a thin demo.
- Output CSVs are checked into git for collaboration. Historical CSV is ~84MB — past GitHub's 50MB warning. If it grows past 100MB, switch to git-lfs.
