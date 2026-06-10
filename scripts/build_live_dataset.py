"""Build the 2025/26 live dataset by pulling from the FPL API and pushing
through the same feature pipeline used for historical data.

The model was trained on the schema produced by ``build_dataset.py`` (rolling
form features, anchors, opponent strength, etc.). For 2025/26 inference we
need to reproduce that schema row-for-row, but sourced from the live API
instead of vaastav's CSVs.

Run:
    python scripts/build_live_dataset.py [--no-cache]

Outputs:
    data/processed/fpl_model_dataset_2025_26.csv     — 2025/26 only, model-ready
    data/processed/_cache/live_history.json          — raw API cache (skip re-fetch)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_dataset import (  # noqa: E402
    DEFAULT_SEASONS,
    FPL_API_BASE,
    NUMERIC_COLUMNS,
    add_cross_season_anchors,
    add_leakage_safe_features,
    add_promoted_flag,
    attach_opponent_team_name,
    build_season_anchor_table,
    load_historical_fpl,
    load_master_team_list,
    make_name_key,
    make_player_id,
)


LIVE_SEASON = "2025-26"
LIVE_SEASON_START = 2025
POSITION_FROM_ELEMENT_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

CACHE_DIR = ROOT / "data" / "processed" / "_cache"
LIVE_HISTORY_CACHE = CACHE_DIR / "live_history.json"
LIVE_BOOTSTRAP_CACHE = CACHE_DIR / "live_bootstrap.json"


def _fetch_json(url, timeout=30):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_bootstrap(use_cache=True):
    if use_cache and LIVE_BOOTSTRAP_CACHE.exists():
        return json.loads(LIVE_BOOTSTRAP_CACHE.read_text())
    bs = _fetch_json(f"{FPL_API_BASE}/bootstrap-static/")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_BOOTSTRAP_CACHE.write_text(json.dumps(bs))
    return bs


def fetch_all_history(element_ids, use_cache=True, polite_delay=0.1, verbose=True):
    """Pull element-summary/{id}/ for each player; concatenate the `history` arrays.

    Caches results to disk so subsequent runs are instant.
    """
    if use_cache and LIVE_HISTORY_CACHE.exists():
        if verbose:
            print(f"Using cached history from {LIVE_HISTORY_CACHE}")
        return json.loads(LIVE_HISTORY_CACHE.read_text())

    all_history = []
    n = len(element_ids)
    for i, pid in enumerate(element_ids, start=1):
        try:
            payload = _fetch_json(f"{FPL_API_BASE}/element-summary/{int(pid)}/")
        except Exception as exc:
            if verbose:
                print(f"  ! element {pid}: {exc}")
            continue
        for row in payload.get("history", []):
            row["element"] = int(pid)
            all_history.append(row)
        if polite_delay:
            time.sleep(polite_delay)
        if verbose and i % 100 == 0:
            print(f"  fetched {i}/{n} players")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_HISTORY_CACHE.write_text(json.dumps(all_history))
    if verbose:
        print(f"Cached {len(all_history)} history rows to {LIVE_HISTORY_CACHE}")
    return all_history


def assemble_live_raw(use_cache=True, verbose=True) -> pd.DataFrame:
    """Produce a per-(player, gw) DataFrame for 2025/26 in the same column
    schema that ``load_historical_fpl`` returns, so the rest of the pipeline
    works unchanged.
    """
    bootstrap = fetch_bootstrap(use_cache=use_cache)
    teams = pd.DataFrame(bootstrap["teams"])[["id", "name", "short_name"]]
    teams = teams.rename(columns={"id": "team_id", "name": "team_name"})

    elements = pd.DataFrame(bootstrap["elements"])
    keep = ["id", "first_name", "second_name", "web_name", "team", "element_type"]
    elements = elements[keep].rename(columns={"id": "element", "team": "team_id"})
    elements["full_name"] = (
        elements["first_name"].astype(str) + "_" + elements["second_name"].astype(str)
    )
    elements["name"] = elements["full_name"]  # match vaastav's `name` column convention
    elements["name_key"] = elements["name"].map(make_name_key)
    elements["player_id"] = elements["name"].map(make_player_id)
    elements["position"] = elements["element_type"].map(POSITION_FROM_ELEMENT_TYPE)
    elements = elements.merge(teams, on="team_id", how="left")

    if verbose:
        print(f"Bootstrap: {len(elements)} players, {len(teams)} teams")

    history_rows = fetch_all_history(
        elements["element"].tolist(), use_cache=use_cache, verbose=verbose,
    )
    history = pd.DataFrame(history_rows)
    if history.empty:
        raise RuntimeError("Empty history — FPL API may be down or rate-limited.")

    history = history.rename(columns={"round": "gw"})
    history["season"] = LIVE_SEASON
    history["season_start"] = LIVE_SEASON_START
    history["kickoff_time"] = pd.to_datetime(
        history["kickoff_time"], errors="coerce", utc=True,
    )

    # Merge in player metadata. team_name is the per-player current club; the
    # API doesn't expose mid-season transfers in element-summary history, so
    # this matches their end-of-season team. Acceptable for v1.
    meta_cols = ["element", "name", "name_key", "player_id", "position",
                 "team_id", "team_name", "short_name"]
    history = history.merge(elements[meta_cols], on="element", how="left")

    # Coerce numeric columns (same logic as load_historical_fpl).
    for col in NUMERIC_COLUMNS:
        if col in history.columns:
            history[col] = pd.to_numeric(history[col], errors="coerce")
    for col in ["opponent_team", "gw", "element"]:
        if col in history.columns:
            history[col] = pd.to_numeric(history[col], errors="coerce")

    history = history.sort_values(["element", "gw"]).reset_index(drop=True)
    return history


def build_live_dataset(out_path=None, use_cache=True, verbose=True) -> pd.DataFrame:
    """Run the full pipeline: fetch live + reuse historical anchors +
    apply feature engineering. Returns the 2025/26 subset only."""
    out_path = Path(out_path or ROOT / "data" / "processed" / "fpl_model_dataset_2025_26.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("[1/5] Loading historical data (for anchors + promoted check)...")
    team_map = load_master_team_list(extra_seasons=DEFAULT_SEASONS)
    historical = load_historical_fpl(DEFAULT_SEASONS, team_map=team_map)
    historical = attach_opponent_team_name(historical, team_map)
    anchor_table = build_season_anchor_table(historical)

    if verbose:
        print("[2/5] Fetching live 2025/26 history from FPL API...")
    live = assemble_live_raw(use_cache=use_cache, verbose=verbose)

    if verbose:
        print("[3/5] Resolving opponent_team_name...")
    # team_map covers historical seasons; for 2025/26 we resolve opponent ids
    # via the live bootstrap teams table directly.
    bootstrap = fetch_bootstrap(use_cache=True)
    live_team_map = pd.DataFrame(bootstrap["teams"])[["id", "name"]].rename(
        columns={"id": "_team_id", "name": "opponent_team_name"},
    )
    live = live.merge(
        live_team_map, left_on="opponent_team", right_on="_team_id", how="left",
    ).drop(columns="_team_id")

    if verbose:
        print("[4/5] Applying cross-season anchors + promoted-team flag...")
    live = add_cross_season_anchors(live, anchor_table)

    # For the promoted-team flag we need 2024/25 team_name set as the "prior season".
    teams_2024_25 = set(historical.loc[historical["season"] == "2024-25", "team_name"].dropna())
    if not teams_2024_25:
        raise RuntimeError("No 2024-25 teams found — historical anchor reference missing.")
    live["is_promoted_team"] = (~live["team_name"].isin(teams_2024_25)).astype(object)

    if verbose:
        print("[5/5] Computing rolling features...")
    live = add_leakage_safe_features(live)

    live.to_csv(out_path, index=False)
    if verbose:
        print(f"Saved {len(live):,} rows × {live.shape[1]} cols → {out_path}")
        print(f"Players: {live['element'].nunique()}, GWs: {sorted(live['gw'].dropna().unique().tolist())}")
        promoted_clubs = live.loc[live["is_promoted_team"] == True, "team_name"].drop_duplicates().tolist()
        print(f"Promoted clubs flagged: {promoted_clubs}")
    return live


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force a fresh fetch from FPL API (default: use cache if available).",
    )
    args = parser.parse_args()
    build_live_dataset(use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
