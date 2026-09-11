"""Live weekly runner — predict the upcoming gameweek and recommend a squad.

Season-agnostic: the target season and gameweek are read from the FPL API
(`events` with `is_next`), so this script does not need editing between
seasons. Run it any time before a deadline.

Feature parity with training is guaranteed by construction: we build the
completed-gameweek history, append synthetic rows for the *upcoming*
gameweek (stats NaN, fixture/price/ownership known), then run the exact same
``add_leakage_safe_features`` used to build the training set, and finally keep
only the upcoming-gameweek rows. Nothing about the rolling-window semantics is
reimplemented here.

Usage:
    python scripts/predict_next_gw.py                 # auto-detect next GW
    python scripts/predict_next_gw.py --gw 5          # force a gameweek
    python scripts/predict_next_gw.py --no-cache      # refetch from the API
    python scripts/predict_next_gw.py --fresh-squad   # ignore saved squad state

Outputs:
    results/live/gw{N}_predictions.csv   — per-player predicted points
    results/live/gw{N}_squad.csv         — recommended 15 + XI + captain
    data/live_state/squad_state.json     — squad carried into the next GW
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_dataset import (  # noqa: E402
    FPL_API_BASE,
    NUMERIC_COLUMNS,
    add_cross_season_anchors,
    add_leakage_safe_features,
    build_season_anchor_table,
    make_name_key,
    make_player_id,
)
from run_live_inference import (  # noqa: E402
    POSITION_MAP,
    DecomposedFPLNet,
    build_feature_matrix,
    expected_fpl_points,
)
from squad_optimizer import (  # noqa: E402
    HIT_COST,
    HIT_MARGIN,
    MAX_BANKED_TRANSFERS,
    SQUAD_TOTAL,
    _aggregate_dgw_rows,
    optimize_squad,
    optimize_squad_horizon,
)

POSITION_FROM_ELEMENT_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
CKPT_PATH = ROOT / "results" / "phase3_decomposed_tuned.pt"
STATE_PATH = ROOT / "data" / "live_state" / "squad_state.json"
CACHE_DIR = ROOT / "data" / "live_state" / "_cache"
OUT_DIR = ROOT / "results" / "live"
PRED_COL = "pred_decomposed_tuned"


# --------------------------------------------------------------------------
# API helpers
# --------------------------------------------------------------------------

def _get(url, timeout=30):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_bootstrap(use_cache=True):
    cache = CACHE_DIR / "bootstrap.json"
    if use_cache and cache.exists():
        return json.loads(cache.read_text())
    data = _get(f"{FPL_API_BASE}/bootstrap-static/")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


def fetch_fixtures(use_cache=True):
    cache = CACHE_DIR / "fixtures.json"
    if use_cache and cache.exists():
        return json.loads(cache.read_text())
    data = _get(f"{FPL_API_BASE}/fixtures/")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


def fetch_history(element_ids, use_cache=True, verbose=True):
    """Completed-gameweek history for the current season. Empty before GW1."""
    cache = CACHE_DIR / "history.json"
    if use_cache and cache.exists():
        return json.loads(cache.read_text())
    rows = []
    n = len(element_ids)
    for i, pid in enumerate(element_ids, 1):
        try:
            payload = _get(f"{FPL_API_BASE}/element-summary/{int(pid)}/")
        except Exception as exc:
            if verbose:
                print(f"    ! element {pid}: {exc}")
            continue
        for r in payload.get("history", []):
            r["element"] = int(pid)
            rows.append(r)
        time.sleep(0.05)
        if verbose and i % 150 == 0:
            print(f"    fetched {i}/{n}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows))
    return rows


def detect_season_and_gw(bootstrap, forced_gw=None):
    """Derive season label + target gameweek from the API's own event list."""
    events = bootstrap["events"]
    first_deadline = pd.to_datetime(events[0]["deadline_time"], utc=True)
    # A PL season starting in August of year Y is labelled "Y-(Y+1)".
    start_year = first_deadline.year if first_deadline.month >= 7 else first_deadline.year - 1
    season = f"{start_year}-{str(start_year + 1)[-2:]}"

    if forced_gw is not None:
        return season, start_year, int(forced_gw)

    nxt = [e["id"] for e in events if e.get("is_next")]
    if nxt:
        return season, start_year, int(nxt[0])
    cur = [e["id"] for e in events if e.get("is_current")]
    if cur:
        return season, start_year, int(cur[0]) + 1
    unfinished = [e["id"] for e in events if not e.get("finished")]
    return season, start_year, int(unfinished[0]) if unfinished else 38


# --------------------------------------------------------------------------
# Frame assembly
# --------------------------------------------------------------------------

def build_player_meta(bootstrap):
    teams = pd.DataFrame(bootstrap["teams"])[["id", "name", "short_name"]]
    teams = teams.rename(columns={"id": "team_id", "name": "team_name"})

    el = pd.DataFrame(bootstrap["elements"])
    el = el[[
        "id", "first_name", "second_name", "web_name", "team", "element_type",
        "now_cost", "selected_by_percent", "transfers_in_event",
        "transfers_out_event", "status", "news", "chance_of_playing_next_round",
    ]].rename(columns={"id": "element", "team": "team_id"})

    el["name"] = el["first_name"].astype(str) + "_" + el["second_name"].astype(str)
    el["name_key"] = el["name"].map(make_name_key)
    el["player_id"] = el["name"].map(make_player_id)
    el["position"] = el["element_type"].map(POSITION_FROM_ELEMENT_TYPE)
    return el.merge(teams, on="team_id", how="left"), teams


def build_history_frame(history_rows, meta, season, start_year, target_gw):
    """Rows for gameweeks already played this season (may be empty at GW1)."""
    cols = ["season", "season_start", "element", "gw", "kickoff_time",
            "opponent_team", "was_home", "value", "selected", "transfers_in",
            "transfers_out", "transfers_balance", "total_points", "minutes",
            "goals_scored", "assists", "clean_sheets", "goals_conceded",
            "expected_goals", "expected_assists", "starts", "bonus"]
    if not history_rows:
        return pd.DataFrame(columns=cols)

    h = pd.DataFrame(history_rows).rename(columns={"round": "gw"})
    h = h[h["gw"] < target_gw]
    if h.empty:
        return pd.DataFrame(columns=cols)

    h["season"] = season
    h["season_start"] = start_year
    h["kickoff_time"] = pd.to_datetime(h["kickoff_time"], errors="coerce", utc=True)
    for c in NUMERIC_COLUMNS + ["opponent_team", "gw", "element", "starts", "bonus"]:
        if c in h.columns:
            h[c] = pd.to_numeric(h[c], errors="coerce")
    return h


def build_future_frame(fixtures, meta, teams, season, start_year, target_gws,
                       total_managers):
    """One row per (player, fixture) for every gameweek in the horizon.

    A double gameweek naturally yields two rows for the same player; callers
    aggregate them later. A blank gameweek yields no rows, so those players
    simply cannot be selected in that gameweek.
    """
    fx = pd.DataFrame(fixtures)
    fx = fx[fx["event"].isin(target_gws)]
    if fx.empty:
        raise RuntimeError(f"No fixtures found for gameweeks {target_gws}.")

    legs = []
    for _, f in fx.iterrows():
        legs.append({"team_id": int(f["team_h"]), "opponent_team": int(f["team_a"]),
                     "was_home": True, "kickoff_time": f.get("kickoff_time"),
                     "gw": int(f["event"]), "fdr": f.get("team_h_difficulty")})
        legs.append({"team_id": int(f["team_a"]), "opponent_team": int(f["team_h"]),
                     "was_home": False, "kickoff_time": f.get("kickoff_time"),
                     "gw": int(f["event"]), "fdr": f.get("team_a_difficulty")})
    legs = pd.DataFrame(legs)

    fut = meta.merge(legs, on="team_id", how="inner")
    fut["season"] = season
    fut["season_start"] = start_year
    fut["kickoff_time"] = pd.to_datetime(fut["kickoff_time"], errors="coerce", utc=True)

    # Known-at-deadline market signals. `selected` must be a RAW ownership
    # count to match the training distribution (median ~21k, max ~9.6M);
    # bootstrap-static only exposes a percentage, so scale by total managers.
    fut["value"] = pd.to_numeric(fut["now_cost"], errors="coerce")
    fut["selected"] = (
        pd.to_numeric(fut["selected_by_percent"], errors="coerce")
        / 100.0 * float(total_managers)
    )
    fut["transfers_in"] = pd.to_numeric(fut["transfers_in_event"], errors="coerce")
    fut["transfers_out"] = pd.to_numeric(fut["transfers_out_event"], errors="coerce")
    fut["transfers_balance"] = fut["transfers_in"] - fut["transfers_out"]

    # Outcome columns are unknown for a future fixture.
    for c in ["total_points", "minutes", "goals_scored", "assists", "clean_sheets",
              "goals_conceded", "expected_goals", "expected_assists", "starts", "bonus"]:
        fut[c] = np.nan
    return fut


# Rolling columns describing a player's (or their team's) own recent form.
# Beyond the first horizon gameweek these cannot legitimately advance, because
# the intervening results have not happened yet.
FORM_ROLL_COLS = [
    "total_points_roll3", "total_points_roll5",
    "minutes_roll3", "minutes_roll5",
    "goals_scored_roll3", "goals_scored_roll5",
    "assists_roll3", "assists_roll5",
    "expected_goals_roll3", "expected_goals_roll5",
    "expected_assists_roll3", "expected_assists_roll5",
    "team_goals_scored_gw_roll5", "team_goals_conceded_gw_roll5",
    "team_points_gw_roll5",
]


def freeze_form_across_horizon(frame, target_gw):
    """Hold form features constant at their first-horizon-gameweek values.

    ``add_leakage_safe_features`` computes rolling means with
    ``.shift(1).rolling(w)``. Applied to a run of future rows whose stats are
    all NaN, the window walks off the end of the observed data and the means
    decay to NaN — so GW+3 would look like a cold-start row while GW+1 had
    real form. That is an artifact of the window, not a fact about the player,
    and it would bias the optimizer toward near-term gameweeks.

    Form through the last completed gameweek is the best estimate available
    for every gameweek in the horizon, so broadcast it forward. Opponent
    strength is handled separately because it genuinely varies by fixture.
    """
    frame = frame.copy()
    base = (
        frame[frame["gw"] == target_gw]
        .drop_duplicates("element")
        .set_index("element")
    )
    if base.empty:
        return frame

    future = frame["gw"] > target_gw
    for c in [c for c in FORM_ROLL_COLS if c in frame.columns]:
        frame.loc[future, c] = frame.loc[future, "element"].map(base[c])

    # Opponent strength: a team's own rolling form as seen by whoever faces it.
    # Mirrors the mapping inside add_leakage_safe_features, but keyed on the
    # last observed form rather than a decayed future window.
    team_form = (
        base.reset_index()
        .groupby("team_name")[["team_points_gw_roll5", "team_goals_conceded_gw_roll5"]]
        .mean()
    )
    if "opponent_team_name" in frame.columns and not team_form.empty:
        frame.loc[future, "opponent_team_points_roll5"] = (
            frame.loc[future, "opponent_team_name"]
            .map(team_form["team_points_gw_roll5"])
        )
        frame.loc[future, "opponent_team_gc_roll5"] = (
            frame.loc[future, "opponent_team_name"]
            .map(team_form["team_goals_conceded_gw_roll5"])
        )
    return frame


def assemble_feature_frame(bootstrap, fixtures, history_rows, prev_dataset_path,
                           season, start_year, target_gw, horizon_gws):
    meta, teams = build_player_meta(bootstrap)

    hist = build_history_frame(history_rows, meta, season, start_year, target_gw)
    fut = build_future_frame(
        fixtures, meta, teams, season, start_year, horizon_gws,
        total_managers=bootstrap.get("total_players") or 1,
    )

    # Give history rows their player/team metadata so both halves share a schema.
    meta_cols = ["element", "name", "name_key", "player_id", "position",
                 "team_id", "team_name"]
    if not hist.empty:
        hist = hist.merge(meta[meta_cols], on="element", how="left")

    keep = [c for c in fut.columns if c in set(hist.columns) | set(fut.columns)]
    combined = pd.concat(
        [hist.reindex(columns=keep), fut.reindex(columns=keep)],
        ignore_index=True,
    )

    # Opponent club name, needed by the opponent-strength rolling features.
    opp = teams.rename(columns={"team_id": "_tid", "team_name": "opponent_team_name"})
    combined = combined.merge(
        opp[["_tid", "opponent_team_name"]],
        left_on="opponent_team", right_on="_tid", how="left",
    ).drop(columns=["_tid"])

    # Anchors come from the previous completed season.
    prev = pd.read_csv(prev_dataset_path, low_memory=False,
                       usecols=["season_start", "player_id", "minutes",
                                "total_points", "team_name"])
    anchors = build_season_anchor_table(prev)
    combined = add_cross_season_anchors(combined, anchors)

    prev_teams = set(prev["team_name"].dropna().unique())
    combined["is_promoted_team"] = (~combined["team_name"].isin(prev_teams)).astype(object)

    combined = combined.sort_values(["element", "gw"]).reset_index(drop=True)
    combined = add_leakage_safe_features(combined)
    combined = freeze_form_across_horizon(combined, target_gw)

    return combined[combined["gw"].isin(horizon_gws)].copy(), prev_teams


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def run_model(frame):
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = DecomposedFPLNet(in_dim=ckpt["in_dim"], layers=ckpt["layers"],
                             dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    df = frame.copy()
    df["is_promoted_team"] = df["is_promoted_team"].map(
        {True: True, False: False, "True": True, "False": False}
    )
    X = build_feature_matrix(df)
    mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
    Xs = (X - mean) / np.where(scale == 0, 1.0, scale)

    with torch.no_grad():
        out = model(torch.tensor(Xs, dtype=torch.float32))
    h = {k: v.cpu().numpy() for k, v in out.items()}
    pos = df["position"].map(POSITION_MAP).fillna(0).astype(int).to_numpy()
    df[PRED_COL] = expected_fpl_points(
        h["play"], h["sixty"], h["goal"], h["assist"], h["cs"], h["bonus"], h["gc"], pos,
    )

    # Availability gate: the API flags injuries and suspensions the model
    # cannot see. Zero them out so the optimizer never selects them.
    unavailable = df["status"].isin(["i", "s", "u", "n"])
    df.loc[unavailable, PRED_COL] = 0.0
    # Count distinct players, not rows — a horizon spans several gameweeks and
    # would otherwise multiply the figure by the horizon length.
    return df, int(df.loc[unavailable, "element"].nunique())


# --------------------------------------------------------------------------
# Chip strategy (live)
# --------------------------------------------------------------------------
#
# 2025/26 onward: two of every chip per season, one usable in each half.
# Wildcard and Free Hit cannot fire in GW1. Only one chip may fire per GW.
#
# A chip is scarce in a way a transfer is not: burning the first-half Triple
# Captain in GW4 means it is gone until GW20. So the bar for firing one is
# deliberately higher than "any positive uplift" — the same reasoning behind
# the conservative-hit guard, but stronger, because the opportunity cost is a
# whole half-season rather than 4 points.
#
# Base thresholds are calibrated against what chips actually returned in the
# 2025/26 backtest, where realised uplifts ranged ~1-28 points. The duds were
# the low ones (Bench Boost for +1, Triple Captain for +8); the ones worth
# having were 20+. Thresholds sit between those so we skip the duds without
# holding out for perfection.

SEASON_HALF_END_GW = 19
CHIP_NAMES = ["wildcard", "free_hit", "bench_boost", "triple_captain"]
CHIP_EARLIEST_GW = {
    "wildcard": 2, "free_hit": 2, "bench_boost": 1, "triple_captain": 1,
}
CHIP_BASE_THRESHOLD = {
    "triple_captain": 10.0,  # needs a captain predicted well above a normal week
    "bench_boost": 12.0,     # our bench is cheap fodder; only a DGW clears this
    "wildcard": 15.0,        # horizon-wide predicted gain over the normal plan
    "free_hit": 12.0,        # one-week gain, squad reverts afterwards
}
CHIP_LABELS = {
    "wildcard": "Wildcard", "free_hit": "Free Hit",
    "bench_boost": "Bench Boost", "triple_captain": "Triple Captain",
}


def season_half(gw: int) -> int:
    return 1 if gw <= SEASON_HALF_END_GW else 2


def available_chips(chips_used: dict, target_gw: int) -> list[str]:
    """Chips that may legally fire in ``target_gw``.

    ``chips_used`` maps chip name -> list of gameweeks it has already been
    played in. A chip is unavailable if it was already used in the same half.
    """
    half = season_half(target_gw)
    out = []
    for name in CHIP_NAMES:
        if target_gw < CHIP_EARLIEST_GW[name]:
            continue
        used = chips_used.get(name, []) or []
        if any(season_half(int(g)) == half for g in used):
            continue
        out.append(name)
    return out


def chip_threshold(name: str, target_gw: int, scale: float = 1.0) -> float:
    """Uplift a chip must clear to be worth firing now.

    Decays toward the end of each half: an unused chip expires at the half
    boundary, so with two gameweeks left a mediocre return beats letting it
    lapse. Full threshold with 8+ gameweeks of runway, floored at 30%.
    """
    half_end = SEASON_HALF_END_GW if target_gw <= SEASON_HALF_END_GW else 38
    weeks_left = max(0, half_end - target_gw)
    decay = max(0.3, min(1.0, weeks_left / 8.0))
    return CHIP_BASE_THRESHOLD[name] * decay * scale


def _xi_plus_captain(squad: pd.DataFrame, pred_col: str) -> float:
    """Predicted points for a squad's XI, counting the captain twice."""
    xi = squad[squad["in_xi"] == 1]
    cap = squad[squad["is_captain"] == 1]
    total = float(xi[pred_col].sum())
    if len(cap):
        total += float(cap[pred_col].iloc[0])
    return total


def evaluate_chips(
    candidates: list[str],
    pools: list[pd.DataFrame],
    prev_ids: set,
    banked: int,
    res_normal: dict,
    pred_col: str,
    hit_margin: float,
    candidate_pool_size: int = 150,
) -> dict:
    """Estimate the predicted uplift of firing each candidate chip this GW.

    Returns name -> {"uplift": float, "note": str}. Uplifts are in predicted
    points and are directly comparable to ``chip_threshold``.

    Triple Captain and Bench Boost are read straight off the already-chosen
    squad (they change scoring, not selection). Wildcard and Free Hit change
    which squad you would pick, so each requires its own optimizer run.
    """
    out: dict[str, dict] = {}
    squad = res_normal["squad"]

    if "triple_captain" in candidates:
        cap = squad[squad["is_captain"] == 1]
        uplift = float(cap[pred_col].iloc[0]) if len(cap) else 0.0
        name = cap["web_name"].iloc[0] if len(cap) else "?"
        out["triple_captain"] = {
            "uplift": uplift,
            "note": f"captain {name} predicted {uplift:.2f} (chip adds this again)",
        }

    if "bench_boost" in candidates:
        bench = squad[squad["in_xi"] == 0]
        uplift = float(bench[pred_col].sum())
        out["bench_boost"] = {
            "uplift": uplift,
            "note": f"bench of {len(bench)} predicted {uplift:.2f} combined",
        }

    if "wildcard" in candidates:
        # Unlimited free transfers this GW: give the optimizer a banked count
        # it cannot exhaust, and no hit penalty. Compare horizon objectives.
        try:
            res_wc = optimize_squad_horizon(
                pools, prev_ids, MAX_BANKED_TRANSFERS, pred_col=pred_col,
                candidate_pool_size=candidate_pool_size, hit_margin=0.0,
            )
            uplift = float(res_wc["objective"]) - float(res_normal["objective"])
            out["wildcard"] = {
                "uplift": uplift,
                "note": f"{res_wc['transfers_in']} transfers free vs "
                        f"{res_normal.get('transfers_in', 0)} normally",
                "result": res_wc,
            }
        except Exception as exc:
            out["wildcard"] = {"uplift": 0.0, "note": f"evaluation failed: {exc}"}

    if "free_hit" in candidates:
        # One-week unlimited rebuild; squad reverts next GW, so only the
        # target gameweek's score matters.
        try:
            res_fh = optimize_squad(pools[0], pred_col=pred_col)
            uplift = _xi_plus_captain(res_fh["squad"], pred_col) - _xi_plus_captain(squad, pred_col)
            out["free_hit"] = {
                "uplift": uplift,
                "note": "best possible one-week XI vs current plan",
                "result": res_fh,
            }
        except Exception as exc:
            out["free_hit"] = {"uplift": 0.0, "note": f"evaluation failed: {exc}"}

    return out


# --------------------------------------------------------------------------
# Squad state
# --------------------------------------------------------------------------

def load_state():
    if not STATE_PATH.exists():
        return None
    return json.loads(STATE_PATH.read_text())


def save_state(squad_ids, banked, gw, season, chips_used=None):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps({
        "season": season,
        "last_gw": gw,
        "squad_ids": sorted(int(x) for x in squad_ids),
        "banked_transfers": int(banked),
        "chips_used": chips_used or {},
    }, indent=2))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, default=None, help="Force a target gameweek.")
    ap.add_argument("--no-cache", action="store_true", help="Refetch from the API.")
    ap.add_argument("--fresh-squad", action="store_true",
                    help="Ignore saved state and pick a squad from scratch.")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--prev-dataset", type=str, default=None,
                    help="Processed CSV of the previous season (for anchors).")
    ap.add_argument("--no-chips", action="store_true",
                    help="Skip chip evaluation entirely.")
    ap.add_argument("--use-chip", type=str, default=None, choices=CHIP_NAMES,
                    help="Force a specific chip this gameweek, ignoring thresholds.")
    ap.add_argument("--chip-threshold-scale", type=float, default=1.0,
                    help="Scale all chip thresholds (>1 stricter, <1 more willing).")
    ap.add_argument(
        "--hit-margin", type=float, default=HIT_MARGIN,
        help=(
            f"Require predicted gain to clear HIT_COST ({HIT_COST}) by this much "
            f"before taking a paid transfer (default {HIT_MARGIN}, i.e. effective "
            f"break-even of {HIT_COST + HIT_MARGIN} pts). Pass 0 to disable the guard "
            "and match the raw-breakeven behaviour used in the historical backtests."
        ),
    )
    args = ap.parse_args()
    use_cache = not args.no_cache

    print("Fetching FPL API state...")
    bootstrap = fetch_bootstrap(use_cache)
    fixtures = fetch_fixtures(use_cache)
    season, start_year, target_gw = detect_season_and_gw(bootstrap, args.gw)

    deadline = next((e["deadline_time"] for e in bootstrap["events"]
                     if e["id"] == target_gw), "unknown")
    print(f"  Season {season} · target GW{target_gw} · deadline {deadline}")

    prev_dataset = Path(args.prev_dataset) if args.prev_dataset else (
        ROOT / "data" / "processed" / f"fpl_model_dataset_{start_year-1}_{str(start_year)[-2:]}.csv"
    )
    if not prev_dataset.exists():
        raise SystemExit(
            f"Previous-season dataset not found: {prev_dataset}\n"
            f"Anchors (last_season_ppg / minutes_share) come from it. "
            f"Build it first or pass --prev-dataset."
        )
    print(f"  Anchors from: {prev_dataset.name}")

    print("Fetching player history (completed gameweeks)...")
    element_ids = [e["id"] for e in bootstrap["elements"]]
    history_rows = fetch_history(element_ids, use_cache) if target_gw > 1 else []
    print(f"  {len(history_rows):,} history rows")

    scheduled = sorted({int(f["event"]) for f in fixtures
                        if f.get("event") is not None})
    horizon_gws = [g for g in scheduled if target_gw <= g < target_gw + args.horizon]
    print(f"  Planning horizon: GW{horizon_gws[0]}-GW{horizon_gws[-1]} "
          f"({len(horizon_gws)} gameweeks)")

    print("Assembling features...")
    frame, prev_teams = assemble_feature_frame(
        bootstrap, fixtures, history_rows, prev_dataset, season, start_year,
        target_gw, horizon_gws,
    )
    per_gw = frame.groupby("gw")["element"].nunique()
    tgt = frame[frame["gw"] == target_gw]
    n_dgw = int(tgt.groupby("element").size().gt(1).sum())
    print(f"  {len(frame):,} rows across the horizon "
          f"(GW{target_gw}: {tgt['element'].nunique()} players, {n_dgw} with a double gameweek)")
    print(f"  players per GW: {per_gw.to_dict()}")
    if target_gw == 1:
        print("  NOTE: GW1 — no in-season history yet, so rolling form features are")
        print("        empty. Predictions rest on last-season anchors, price, position")
        print("        and fixture. Treat GW1 output as a cold-start prior.")

    print("Running model...")
    preds, n_blocked = run_model(frame)
    print(f"  {n_blocked} players zeroed out as unavailable (injury/suspension)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = OUT_DIR / f"gw{target_gw}_predictions.csv"
    cols = ["element", "name", "web_name", "team_name", "position", "value",
            "opponent_team_name", "was_home", "status", PRED_COL]
    preds[preds["gw"] == target_gw][cols].sort_values(
        PRED_COL, ascending=False).to_csv(pred_path, index=False)
    horizon_path = OUT_DIR / f"gw{target_gw}_horizon_predictions.csv"
    preds[["gw"] + cols].sort_values(["gw", PRED_COL],
                                     ascending=[True, False]).to_csv(horizon_path, index=False)

    print()
    print(f"Top 10 predicted for GW{target_gw}:")
    tgt_preds = preds[preds["gw"] == target_gw]
    top = tgt_preds.nlargest(10, PRED_COL)[["web_name", "team_name", "position",
                                            "value", "opponent_team_name", PRED_COL]]
    top = top.assign(price=lambda d: (d["value"] / 10).map("{:.1f}".format))
    print(top[["web_name", "team_name", "position", "price",
               "opponent_team_name", PRED_COL]].to_string(index=False))
    print()

    # ---- optimize -------------------------------------------------------
    # One pool per horizon gameweek, double gameweeks aggregated so a player
    # cannot be selected twice within the same week.
    pools = []
    for g in horizon_gws:
        pg = _aggregate_dgw_rows(preds[preds["gw"] == g], pred_cols=[PRED_COL])
        pg = pg.dropna(subset=["value", "position", "team_name", PRED_COL])
        pools.append(pg)

    state = None if args.fresh_squad else load_state()
    if state and state.get("season") == season and state.get("squad_ids"):
        banked = int(state.get("banked_transfers", 0))
        prev_ids = set(state["squad_ids"])
        print(f"Planning transfers over GW{horizon_gws[0]}-GW{horizon_gws[-1]} "
              f"from saved squad (after GW{state.get('last_gw')}, {banked} banked)...")
        res = optimize_squad_horizon(
            pools, prev_ids, banked, pred_col=PRED_COL,
            hit_margin=args.hit_margin,
        )
        if args.hit_margin:
            print(f"  hit guard: requires predicted gain > {HIT_COST + args.hit_margin:.1f} pts "
                  f"per paid transfer (HIT_COST {HIT_COST} + margin {args.hit_margin})")
        plan = res.get("horizon_plan") or []
        if plan:
            print("  planned transfers per GW: "
                  + ", ".join(f"GW{horizon_gws[q['gw_index']]}={q['transfers']}"
                              f"{'(-' + str(q['paid'] * 4) + ')' if q['paid'] else ''}"
                              for q in plan))
    else:
        if state and state.get("season") != season:
            print("Saved squad is from a previous season — starting fresh.")
        print("Picking an initial squad from scratch...")
        res = optimize_squad(pools[0], pred_col=PRED_COL)
        res.update({"transfers_in": 0, "paid_transfers": 0, "hit_cost": 0,
                    "banked_next": 0})

    # ---- chip decision ---------------------------------------------------
    chips_used = dict((state or {}).get("chips_used") or {})
    chosen_chip = None
    chip_eval = {}

    if args.no_chips:
        print("Chip evaluation skipped (--no-chips).")
    elif not (state and state.get("squad_ids")) and args.use_chip is None:
        # Opening squad: chips make no sense before a squad exists.
        print("Chip evaluation skipped (no prior squad yet).")
    else:
        candidates = available_chips(chips_used, target_gw)
        if args.use_chip:
            if args.use_chip not in candidates:
                raise SystemExit(
                    f"--use-chip {args.use_chip} is not available in GW{target_gw} "
                    f"(available: {candidates or 'none'}). Already used this half, "
                    f"or too early in the season."
                )
            chosen_chip = args.use_chip
            chip_eval = evaluate_chips(
                [chosen_chip], pools, set(state["squad_ids"]),
                int(state.get("banked_transfers", 0)), res, PRED_COL,
                args.hit_margin,
            )
            print(f"Chip forced by --use-chip: {CHIP_LABELS[chosen_chip]}")
        elif not candidates:
            print(f"No chips available in GW{target_gw} "
                  f"(half {season_half(target_gw)} chips already used).")
        else:
            print(f"Evaluating chips available in GW{target_gw}: "
                  f"{', '.join(CHIP_LABELS[c] for c in candidates)}")
            chip_eval = evaluate_chips(
                candidates, pools, set(state["squad_ids"]),
                int(state.get("banked_transfers", 0)), res, PRED_COL,
                args.hit_margin,
            )
            best_name, best_margin = None, 0.0
            for name in candidates:
                info = chip_eval.get(name)
                if not info:
                    continue
                thr = chip_threshold(name, target_gw, args.chip_threshold_scale)
                over = info["uplift"] - thr
                verdict = "FIRE" if over > 0 else "hold"
                print(f"    {CHIP_LABELS[name]:<15} uplift {info['uplift']:>6.2f} "
                      f"vs threshold {thr:>5.2f}  -> {verdict}   ({info['note']})")
                if over > best_margin:
                    best_name, best_margin = name, over
            chosen_chip = best_name
            if chosen_chip is None:
                print("  -> no chip clears its threshold; saving them all.")

    # Apply the chosen chip. Wildcard and Free Hit replace the squad entirely;
    # Triple Captain and Bench Boost only change how the chosen squad scores.
    chip_note = ""
    if chosen_chip in ("wildcard", "free_hit"):
        alt = (chip_eval.get(chosen_chip) or {}).get("result")
        if alt is None:
            print(f"  !! {CHIP_LABELS[chosen_chip]} selected but its squad could not be "
                  f"rebuilt; falling back to the normal plan.")
            chosen_chip = None
        else:
            res = alt
            res.setdefault("transfers_in", 0)
            res["hit_cost"] = 0
            res["paid_transfers"] = 0
            if chosen_chip == "free_hit":
                # Squad reverts next GW, so state must keep the PRE-chip squad.
                chip_note = " (Free Hit — squad reverts after this GW)"
            else:
                chip_note = " (Wildcard — unlimited free transfers)"
    elif chosen_chip == "triple_captain":
        chip_note = " (Triple Captain — captain scores 3x)"
    elif chosen_chip == "bench_boost":
        chip_note = " (Bench Boost — bench points count)"

    squad = res["squad"]
    print()
    header = f"=== Recommended squad for GW{target_gw} ==="
    if chosen_chip:
        header = (f"=== Recommended squad for GW{target_gw} — "
                  f"{CHIP_LABELS[chosen_chip].upper()}{chip_note} ===")
    print(header)
    print(f"Cost £{res['cost']:.1f}m · formation {res['formation']} · "
          f"transfers {res.get('transfers_in', 0)} (hit -{res.get('hit_cost', 0)})")
    print()
    view = squad.assign(price=lambda d: (d["value"] / 10).map("{:.1f}".format))
    for label, sel in [("STARTING XI", view["in_xi"] == 1), ("BENCH", view["in_xi"] == 0)]:
        print(f"  {label}")
        for _, r in view[sel].iterrows():
            tag = ""
            if r["is_captain"]:
                tag = "  (C)"
            elif r["is_vice"]:
                tag = "  (V)"
            print(f"    {r['position']:<4} {r['web_name']:<20} {r['team_name']:<14} "
                  f"£{r['price']:>5}  {r[PRED_COL]:>5.2f}{tag}")
        print()

    squad_path = OUT_DIR / f"gw{target_gw}_squad.csv"
    squad.to_csv(squad_path, index=False)

    # Record the chip so it cannot be reused in this half.
    if chosen_chip:
        chips_used.setdefault(chosen_chip, [])
        if target_gw not in chips_used[chosen_chip]:
            chips_used[chosen_chip].append(int(target_gw))

    # Free Hit reverts: next gameweek starts from the squad we held BEFORE the
    # chip, not the one-week team it bought. Every other case carries forward
    # the squad actually fielded.
    if chosen_chip == "free_hit" and state and state.get("squad_ids"):
        carry_ids = state["squad_ids"]
        carry_banked = int(state.get("banked_transfers", 0))
        print("  note: Free Hit squad is for this GW only — saved state keeps the "
              "pre-chip squad for GW planning.")
    else:
        carry_ids = squad["element"]
        carry_banked = res.get("banked_next", 0)

    save_state(carry_ids, carry_banked, target_gw, season, chips_used=chips_used)

    remaining = available_chips(chips_used, min(target_gw + 1, 38))
    print(f"\nChips still available after this GW: "
          f"{', '.join(CHIP_LABELS[c] for c in remaining) if remaining else 'none this half'}")
    print(f"Saved:\n  {pred_path}\n  {squad_path}\n  {STATE_PATH}")


if __name__ == "__main__":
    main()
