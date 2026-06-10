"""Phase 5 — chip strategy layered on top of the squad optimizer.

2025/26 gives each manager 2 of each chip (one in each season half):
  - Triple Captain (TC): captain's points count 3x instead of 2x
  - Bench Boost (BB): bench points are added to total
  - Wildcard (WC): unlimited free transfers this GW (squad persists after)
  - Free Hit (FH): unlimited free transfers this GW; squad reverts next GW

Activation windows (from FPL API):
  - WC/FH: GW2-19 (first half), GW20-38 (second half)
  - BB/TC: GW1-19, GW20-38

This module exposes:
  - ``backtest_with_chips(...)`` — runs the horizon backtest with a given chip schedule
  - ``find_heuristic_chip_schedule(...)`` — picks chip-deployment GWs from PREDICTIONS
  - ``find_oracle_chip_schedule(...)`` — same but using ACTUALS (perfect-info ceiling)
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from squad_optimizer import (  # noqa: E402
    SQUAD_TOTAL,
    _aggregate_dgw_rows,
    apply_auto_subs,
    optimize_squad,
    optimize_squad_horizon,
    optimize_squad_with_transfers,
    score_squad_realistic,
)


SEASON_HALF = 19  # GW1-19 then GW20-38
WILDCARD_EARLIEST_GW = 2  # WC and FH can't fire in GW1
FREE_HIT_EARLIEST_GW = 2


# --- per-GW chip effects ----------------------------------------------------

def _captain_or_vice_actual(squad: pd.DataFrame, actual_col="total_points",
                            minutes_col="minutes") -> float:
    """Actual points for whoever ends up wearing the armband (captain if played,
    else vice-captain if played, else 0)."""
    cap = squad[squad["is_captain"] == 1].iloc[0]
    if (cap[minutes_col] or 0) > 0:
        return float(cap[actual_col]) if pd.notna(cap[actual_col]) else 0.0
    vice = squad[squad["is_vice"] == 1].iloc[0]
    if (vice[minutes_col] or 0) > 0:
        return float(vice[actual_col]) if pd.notna(vice[actual_col]) else 0.0
    return 0.0


def _bench_actual(squad: pd.DataFrame, actual_col="total_points") -> float:
    """Actual points from the 4 bench players (with realistic scoring rules — same
    as XI: 0 if didn't play, full points if did)."""
    bench = squad[squad["in_xi"] == 0]
    return float(bench[actual_col].fillna(0).sum())


# --- chip-aware backtest ----------------------------------------------------

def _is_chip_active(chip_schedule: dict | None, chip_name: str, gw: int) -> bool:
    if chip_schedule is None:
        return False
    return gw in (chip_schedule.get(chip_name) or [])


def backtest_with_chips(
    predictions_csv: Path,
    dataset_csv: Path,
    pred_col: str = "pred_decomposed_tuned",
    horizon: int = 4,
    chip_schedule: Optional[dict] = None,
    output_csv: Optional[Path] = None,
    candidate_pool_size: int = 150,
    verbose: bool = False,
) -> pd.DataFrame:
    """Walk the season GW1->GW38, optimizing each GW with chip effects applied.

    ``chip_schedule`` is a dict like:
        {"triple_captain": [gw_a, gw_b], "bench_boost": [...],
         "wildcard": [...], "free_hit": [...]}
    Each list has 0-2 entries (one per season half).
    """
    preds = pd.read_csv(predictions_csv)
    hist = pd.read_csv(
        dataset_csv, low_memory=False,
        usecols=["season", "element", "gw", "value", "position", "team_name", "minutes"],
    )
    df = preds.merge(hist, on=["season", "element", "gw"], how="left")
    df = _aggregate_dgw_rows(df, pred_cols=[pred_col, "total_points"])
    df = df.dropna(subset=["value", "position", "team_name", pred_col, "total_points"])

    gws = sorted(df["gw"].unique())
    gw_to_pool = {gw: df[df["gw"] == gw].reset_index(drop=True) for gw in gws}

    rows = []
    current_squad_ids: Optional[set] = None
    saved_squad_before_fh: Optional[set] = None
    saved_banked_before_fh: int = 0
    banked = 0

    for i, gw in enumerate(gws):
        is_tc = _is_chip_active(chip_schedule, "triple_captain", gw)
        is_bb = _is_chip_active(chip_schedule, "bench_boost", gw)
        is_wc = _is_chip_active(chip_schedule, "wildcard", gw)
        is_fh = _is_chip_active(chip_schedule, "free_hit", gw)

        horizon_pools = [gw_to_pool[g] for g in gws[i : i + horizon]]
        if len(horizon_pools) == 0 or len(horizon_pools[0]) < SQUAD_TOTAL:
            continue

        # --- pick squad / decide transfers --------------------------------
        if current_squad_ids is None:
            # GW1 — always fresh, no chips applicable to transfers
            res = optimize_squad(horizon_pools[0], pred_col=pred_col)
            res.update({"transfers_in": 0, "transfers_out": 0, "free_available": 1,
                        "paid_transfers": 0, "hit_cost": 0, "banked_next": 0,
                        "missing_from_pool": 0})
        elif is_fh:
            # Free Hit: pretend we have a fresh budget and pick the best single-GW squad.
            # The persistent squad reverts NEXT GW.
            saved_squad_before_fh = set(current_squad_ids)
            saved_banked_before_fh = banked
            res = optimize_squad(horizon_pools[0], pred_col=pred_col)
            res.update({"transfers_in": 0, "transfers_out": 0, "free_available": 0,
                        "paid_transfers": 0, "hit_cost": 0,
                        "banked_next": banked,  # banked preserved; only this GW is "free"
                        "missing_from_pool": 0})
        elif is_wc:
            # Wildcard: unlimited free transfers; squad persists after.
            # Simulate with effectively-unlimited banked transfers (no hit cost possible).
            res = optimize_squad_with_transfers(
                horizon_pools[0], current_squad_ids, banked_transfers=SQUAD_TOTAL,
                pred_col=pred_col,
            )
            # Banked reverts to 1 next GW (FPL gives back 1 free transfer the GW after a WC).
            res["banked_next"] = 0
            res["hit_cost"] = 0
            res["paid_transfers"] = 0
        else:
            res = optimize_squad_horizon(
                horizon_pools, current_squad_ids, banked, pred_col=pred_col,
                candidate_pool_size=candidate_pool_size,
            )

        # --- score against actuals + chip effects -------------------------
        realized = score_squad_realistic(res["squad"], "total_points")
        gw_score = realized["total"]

        tc_uplift = 0.0
        bb_uplift = 0.0
        if is_tc:
            tc_uplift = _captain_or_vice_actual(res["squad"])
            gw_score += tc_uplift
        if is_bb:
            bb_uplift = _bench_actual(res["squad"])
            gw_score += bb_uplift

        net = gw_score - res["hit_cost"]

        rows.append({
            "gw": int(gw),
            "gw_score_gross": realized["total"],
            "chip": ",".join([c for c, on in [
                ("TC", is_tc), ("BB", is_bb), ("WC", is_wc), ("FH", is_fh)
            ] if on]) or "",
            "tc_uplift": tc_uplift,
            "bb_uplift": bb_uplift,
            "hit_cost": res["hit_cost"],
            "gw_score_net": net,
            "transfers_in": res["transfers_in"],
            "free_available": res["free_available"],
            "paid_transfers": res["paid_transfers"],
            "banked_before": banked,
            "banked_after": res["banked_next"],
            "captain_picked": res["captain"]["name"],
            "captain_used": realized["captain_used"],
            "subs_applied": realized["subs_applied"],
            "formation": res["formation"],
            "cost_m": res["cost"],
        })

        # --- advance state -----------------------------------------------
        if is_fh:
            # Squad reverts to what it was before the FH GW.
            current_squad_ids = saved_squad_before_fh
            banked = saved_banked_before_fh
        else:
            current_squad_ids = set(res["squad"]["element"].astype(int))
            banked = res["banked_next"]

        if verbose:
            tag = f" [{rows[-1]['chip']}]" if rows[-1]["chip"] else ""
            print(f"  GW{int(gw):>2}: net={net:5.1f} (gross={realized['total']:.1f}, "
                  f"hit={res['hit_cost']}){tag}")

    out = pd.DataFrame(rows)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
    return out


# --- chip schedule search ---------------------------------------------------

def _best_gw_in_range(
    gw_to_uplift: dict, low: int, high: int, excluded: set[int],
) -> tuple[int | None, float]:
    """Pick the GW with the highest uplift in [low, high], skipping `excluded`."""
    candidates = [(g, u) for g, u in gw_to_uplift.items() if low <= g <= high and g not in excluded]
    if not candidates:
        return None, 0.0
    g, u = max(candidates, key=lambda x: x[1])
    return g, u


def find_chip_schedule_from_baseline(
    baseline_results: pd.DataFrame,
    baseline_timeline: pd.DataFrame,
    pool_df: pd.DataFrame | None = None,
    use_actuals: bool = False,
) -> dict:
    """Greedy chip search using either predicted or actual uplifts.

    Per-GW uplift definitions:
      - TC: captain's points (extra 1x)
      - BB: sum of bench players' points
      - WC: (best 15-man squad pred this GW) − (current squad pred this GW)
            — measures how much the current squad lags the fresh optimum
    Chips can't stack: each GW can only host one chip. We pick chips greedily
    in TC → BB → WC order, skipping GWs already claimed.

    Args:
      baseline_timeline: per-(gw, player) detail from collect_timeline_from_backtest
      pool_df: optional full pool DataFrame (preds + dataset merged + dedup'd).
               Required for wildcard uplift estimation. If None, WC is skipped.
      use_actuals: if True use realized points (oracle upper bound); else
                   predicted points (realistic bot decision).
    """
    points_col = "actual_points" if use_actuals else "pred_points"
    minutes_col = "minutes"

    tc_per_gw: dict[int, float] = {}
    bb_per_gw: dict[int, float] = {}
    for gw, group in baseline_timeline.groupby("gw"):
        cap_row = group[group["is_captain"] == 1]
        if len(cap_row) == 0:
            continue
        if use_actuals:
            cap_played = cap_row[minutes_col].iloc[0] > 0
            if cap_played:
                tc_val = float(cap_row[points_col].iloc[0])
            else:
                vice_row = group[group["is_vice"] == 1]
                if len(vice_row) and vice_row[minutes_col].iloc[0] > 0:
                    tc_val = float(vice_row[points_col].iloc[0])
                else:
                    tc_val = 0.0
        else:
            tc_val = float(cap_row[points_col].iloc[0])
        tc_per_gw[int(gw)] = tc_val

        bench = group[group["in_xi"] == 0]
        bb_per_gw[int(gw)] = float(bench[points_col].fillna(0).sum())

    # Wildcard uplift: at each GW, compare current XI's points to a fresh
    # top-11 XI from the pool. This is a rough but reasonable proxy.
    wc_per_gw: dict[int, float] = {}
    if pool_df is not None:
        owned_by_gw = {int(g): set(grp["element"]) for g, grp in baseline_timeline.groupby("gw")}
        pool_col = "total_points" if use_actuals else "pred_points"
        # baseline_timeline has predicted+actual points already; for "optimal
        # top-15" we need the FULL pool at each GW so we look at pool_df.
        pool_pts_col = "total_points" if use_actuals else None  # placeholder
        for gw, owned in owned_by_gw.items():
            gw_pool = pool_df[pool_df["gw"] == gw]
            if gw_pool.empty:
                continue
            # Top-15 by metric of choice, minus the current XI's same-metric points.
            metric_col = "total_points" if use_actuals else None
            # The pool's `pred_decomposed_tuned` is the prediction column when
            # not using actuals — caller knows the right column name.
            pred_col_candidates = [c for c in gw_pool.columns
                                   if c.startswith("pred_") and not c.endswith("_ridge")]
            if not use_actuals and not pred_col_candidates:
                continue
            metric = "total_points" if use_actuals else pred_col_candidates[0]
            current_squad_score = float(
                baseline_timeline.loc[baseline_timeline["gw"] == gw, points_col].nlargest(15).sum()
            )
            fresh_top15 = float(gw_pool.nlargest(15, metric)[metric].sum())
            wc_per_gw[int(gw)] = max(0.0, fresh_top15 - current_squad_score)

    used: set[int] = set()
    schedule: dict[str, list[int]] = {}

    # TC first (biggest single-GW spike; binds the cleanest)
    tc_first, _ = _best_gw_in_range(tc_per_gw, 1, SEASON_HALF, used)
    if tc_first is not None:
        used.add(tc_first)
    tc_second, _ = _best_gw_in_range(tc_per_gw, SEASON_HALF + 1, 38, used)
    if tc_second is not None:
        used.add(tc_second)
    schedule["triple_captain"] = [g for g in [tc_first, tc_second] if g is not None]

    # BB next
    bb_first, _ = _best_gw_in_range(bb_per_gw, 1, SEASON_HALF, used)
    if bb_first is not None:
        used.add(bb_first)
    bb_second, _ = _best_gw_in_range(bb_per_gw, SEASON_HALF + 1, 38, used)
    if bb_second is not None:
        used.add(bb_second)
    schedule["bench_boost"] = [g for g in [bb_first, bb_second] if g is not None]

    # WC if we have a pool
    if wc_per_gw:
        wc_first, _ = _best_gw_in_range(wc_per_gw, WILDCARD_EARLIEST_GW, SEASON_HALF, used)
        if wc_first is not None:
            used.add(wc_first)
        wc_second, _ = _best_gw_in_range(wc_per_gw, SEASON_HALF + 1, 38, used)
        if wc_second is not None:
            used.add(wc_second)
        schedule["wildcard"] = [g for g in [wc_first, wc_second] if g is not None]

    return schedule


def collect_timeline_from_backtest(
    predictions_csv: Path,
    dataset_csv: Path,
    pred_col: str,
    horizon: int = 4,
    candidate_pool_size: int = 150,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the baseline horizon backtest while capturing per-(gw, player) detail.

    Returns (gw_summary, timeline). The timeline has one row per (gw, player_in_squad)
    with their role and predicted + actual points — exactly what chip search needs.
    """
    preds = pd.read_csv(predictions_csv)
    hist = pd.read_csv(
        dataset_csv, low_memory=False,
        usecols=["season", "element", "gw", "value", "position", "team_name", "minutes"],
    )
    df = preds.merge(hist, on=["season", "element", "gw"], how="left")
    df = _aggregate_dgw_rows(df, pred_cols=[pred_col, "total_points"])
    df = df.dropna(subset=["value", "position", "team_name", pred_col, "total_points"])

    gws = sorted(df["gw"].unique())
    gw_to_pool = {gw: df[df["gw"] == gw].reset_index(drop=True) for gw in gws}

    summary_rows = []
    timeline_rows = []
    current_squad_ids: Optional[set] = None
    banked = 0

    for i, gw in enumerate(gws):
        horizon_pools = [gw_to_pool[g] for g in gws[i : i + horizon]]
        if len(horizon_pools[0]) < SQUAD_TOTAL:
            continue
        if current_squad_ids is None:
            res = optimize_squad(horizon_pools[0], pred_col=pred_col)
            res.update({"transfers_in": 0, "transfers_out": 0, "free_available": 1,
                        "paid_transfers": 0, "hit_cost": 0, "banked_next": 0,
                        "missing_from_pool": 0})
        else:
            res = optimize_squad_horizon(
                horizon_pools, current_squad_ids, banked, pred_col=pred_col,
                candidate_pool_size=candidate_pool_size,
            )

        realized = score_squad_realistic(res["squad"], "total_points")
        squad = res["squad"]

        for _, row in squad.iterrows():
            timeline_rows.append({
                "gw": int(gw),
                "element": int(row["element"]),
                "name": row.get("name", ""),
                "position": row["position"],
                "team_name": row.get("team_name", ""),
                "pred_points": float(row[pred_col]),
                "actual_points": float(row.get("total_points", 0) or 0),
                "minutes": float(row.get("minutes", 0) or 0),
                "in_xi": int(row["in_xi"]),
                "is_captain": int(row["is_captain"]),
                "is_vice": int(row["is_vice"]),
            })
        summary_rows.append({
            "gw": int(gw),
            "gw_score_net": realized["total"] - res["hit_cost"],
            "captain_picked": res["captain"]["name"],
            "transfers_in": res["transfers_in"],
            "hit_cost": res["hit_cost"],
        })

        current_squad_ids = set(squad["element"].astype(int))
        banked = res["banked_next"]

    return pd.DataFrame(summary_rows), pd.DataFrame(timeline_rows)
