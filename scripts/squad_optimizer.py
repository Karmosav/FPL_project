"""Phase 4 — single-GW squad optimizer.

Given predicted FPL points for one gameweek, pick the optimal 15-man squad
(2 GK / 5 DEF / 5 MID / 3 FWD), starting XI in a valid formation, and captain
subject to FPL constraints (squad cost <= GBP 100m, max 3 per club).

This is the static "fresh squad" problem with no transfers, horizon, or chips.
It's the upper bound on what single-GW optimization can do given a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import pulp


# --- FPL rules ---------------------------------------------------------------

POSITION_LIMITS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}  # per FPL API squad_min_play
XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
XI_TOTAL = 11
SQUAD_TOTAL = 15
BUDGET_TENTHS = 1000  # GBP 100m, prices stored as 0.1m units
MAX_PER_CLUB = 3
TRANSFERS_CAP_PER_GW = 20  # FPL hard limit
MAX_BANKED_TRANSFERS = 4   # max extras beyond this GW's auto-1 (2025/26 rule)
HIT_COST = 4               # points deducted per transfer beyond what's available

# vaastav and FPL API both drift between "GK"/"GKP" and the 2024-25 "AM"
# (attacking midfielder) bucket. Normalize to four canonical positions.
POSITION_NORMALIZE = {
    "GK": "GK", "GKP": "GK",
    "DEF": "DEF",
    "MID": "MID", "AM": "MID",
    "FWD": "FWD",
}


# --- Core ILP ----------------------------------------------------------------

def optimize_squad(
    pool: pd.DataFrame,
    pred_col: str = "pred_points",
    value_col: str = "value",
    position_col: str = "position",
    team_col: str = "team_name",
) -> dict:
    """Pick squad/XI/captain for a single GW from a pool DataFrame.

    Required columns in `pool`: pred_col, value_col, position_col, team_col.
    Other columns are passed through to the returned squad/xi frames.

    Returns dict with keys: squad, xi, captain, objective, cost, formation.
    """
    df = pool.copy()
    df[position_col] = df[position_col].map(POSITION_NORMALIZE).fillna(df[position_col])
    df = df.dropna(subset=[position_col, team_col, value_col, pred_col]).reset_index(drop=True)

    if df.empty:
        raise ValueError("Empty player pool after filtering.")

    idx = df.index.tolist()
    pred = df[pred_col].astype(float).to_dict()
    value = df[value_col].astype(float).to_dict()
    pos = df[position_col].to_dict()
    team = df[team_col].to_dict()

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("squad", idx, cat="Binary")  # in 15-man squad
    s = pulp.LpVariable.dicts("start", idx, cat="Binary")  # in starting XI
    c = pulp.LpVariable.dicts("capt", idx, cat="Binary")   # captain

    # Objective: XI points + captain bonus (captain's points counted twice).
    prob += (
        pulp.lpSum(pred[i] * s[i] for i in idx)
        + pulp.lpSum(pred[i] * c[i] for i in idx)
    )

    # Budget
    prob += pulp.lpSum(value[i] * x[i] for i in idx) <= BUDGET_TENTHS

    # Squad size and per-position counts
    prob += pulp.lpSum(x[i] for i in idx) == SQUAD_TOTAL
    for p, k in POSITION_LIMITS.items():
        prob += pulp.lpSum(x[i] for i in idx if pos[i] == p) == k

    # Max 3 per club
    for t in df[team_col].unique():
        prob += pulp.lpSum(x[i] for i in idx if team[i] == t) <= MAX_PER_CLUB

    # XI is a subset of the squad
    for i in idx:
        prob += s[i] <= x[i]
    prob += pulp.lpSum(s[i] for i in idx) == XI_TOTAL
    for p in POSITION_LIMITS:
        prob += pulp.lpSum(s[i] for i in idx if pos[i] == p) >= XI_MIN[p]
        prob += pulp.lpSum(s[i] for i in idx if pos[i] == p) <= XI_MAX[p]

    # Captain: exactly one, must be in XI
    prob += pulp.lpSum(c[i] for i in idx) == 1
    for i in idx:
        prob += c[i] <= s[i]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Solver status: {pulp.LpStatus[prob.status]}")

    df["in_squad"] = [int(round(x[i].value())) for i in idx]
    df["in_xi"] = [int(round(s[i].value())) for i in idx]
    df["is_captain"] = [int(round(c[i].value())) for i in idx]

    squad = df[df["in_squad"] == 1].copy()
    xi = squad[squad["in_xi"] == 1].copy()
    captain = squad[squad["is_captain"] == 1].iloc[0]

    formation = "-".join(
        str(int((xi[position_col] == p).sum())) for p in ["DEF", "MID", "FWD"]
    )

    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    squad = squad.assign(_pos_order=squad[position_col].map(pos_order)).sort_values(
        ["in_xi", "_pos_order", pred_col], ascending=[False, True, False]
    ).drop(columns="_pos_order")

    squad = assign_vice_and_bench_priority(squad, pred_col=pred_col, position_col=position_col)
    xi = squad[squad["in_xi"] == 1].copy()
    captain = squad[squad["is_captain"] == 1].iloc[0]

    return {
        "squad": squad,
        "xi": xi,
        "captain": captain,
        "vice_captain": squad[squad["is_vice"] == 1].iloc[0],
        "objective": float(pulp.value(prob.objective)),
        "cost": float(squad[value_col].sum() / 10.0),
        "formation": f"1-{formation}",
    }


def optimize_squad_with_transfers(
    pool: pd.DataFrame,
    current_squad_ids: set,
    banked_transfers: int,
    pred_col: str = "pred_points",
    value_col: str = "value",
    position_col: str = "position",
    team_col: str = "team_name",
    id_col: str = "element",
) -> dict:
    """Pick next-GW squad given the previous squad and banked transfers (2025/26 rules).

    Differences vs ``optimize_squad``:
      - You start from a previous 15-man squad (set of element IDs).
      - You get ``banked_transfers + 1`` "free" transfers this GW (cap 5, i.e.
        banked is 0..4). Extra transfers cost -4 points each.
      - Hard cap: ``TRANSFERS_CAP_PER_GW`` transfers max in one GW.

    Simplifying assumption (pass 1): prices in ``value_col`` are used uniformly
    for every player in the new squad. This ignores the 50% sell-on fee and any
    accumulated bank from prior price changes. The cumulative error over a
    season is small (~£1-3m) but worth fixing for a v2 simulator.
    """
    df = pool.copy()
    df[position_col] = df[position_col].map(POSITION_NORMALIZE).fillna(df[position_col])
    df = df.dropna(subset=[position_col, team_col, value_col, pred_col, id_col]).reset_index(drop=True)
    if df.empty:
        raise ValueError("Empty player pool after filtering.")

    banked = max(0, min(MAX_BANKED_TRANSFERS, int(banked_transfers)))
    free_available = banked + 1  # this GW's auto + carried-over

    idx = df.index.tolist()
    pred = df[pred_col].astype(float).to_dict()
    value = df[value_col].astype(float).to_dict()
    pos = df[position_col].to_dict()
    team = df[team_col].to_dict()
    element = df[id_col].astype(int).to_dict()

    prev_idx = [i for i in idx if element[i] in current_squad_ids]
    new_idx = [i for i in idx if element[i] not in current_squad_ids]

    # If some previous-squad players aren't in this GW's pool, they implicitly
    # leave (the optimizer can't keep what it can't see). Surface this so the
    # caller knows their realized state may differ from the FPL website.
    missing_from_pool = len(current_squad_ids) - len(prev_idx)

    prob = pulp.LpProblem("fpl_squad_xfer", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("squad", idx, cat="Binary")
    s = pulp.LpVariable.dicts("start", idx, cat="Binary")
    c = pulp.LpVariable.dicts("capt", idx, cat="Binary")
    # Slack for the -4 hit count: paid >= num_transfers - free_available, >= 0.
    paid = pulp.LpVariable("paid_transfers", lowBound=0, upBound=TRANSFERS_CAP_PER_GW, cat="Integer")

    # Transfers in = players we add who weren't in previous squad.
    # (Transfers out is mechanically equal because squad size stays at 15.)
    num_transfers = pulp.lpSum(x[i] for i in new_idx)

    prob += (
        pulp.lpSum(pred[i] * s[i] for i in idx)
        + pulp.lpSum(pred[i] * c[i] for i in idx)
        - HIT_COST * paid
    )

    # Budget (using current GW prices for the whole new squad)
    prob += pulp.lpSum(value[i] * x[i] for i in idx) <= BUDGET_TENTHS

    # Squad size and per-position counts (same as fresh-squad mode)
    prob += pulp.lpSum(x[i] for i in idx) == SQUAD_TOTAL
    for p, k in POSITION_LIMITS.items():
        prob += pulp.lpSum(x[i] for i in idx if pos[i] == p) == k

    # Max 3 per club
    for t in df[team_col].unique():
        prob += pulp.lpSum(x[i] for i in idx if team[i] == t) <= MAX_PER_CLUB

    # XI subset of squad + valid formation
    for i in idx:
        prob += s[i] <= x[i]
    prob += pulp.lpSum(s[i] for i in idx) == XI_TOTAL
    for p in POSITION_LIMITS:
        prob += pulp.lpSum(s[i] for i in idx if pos[i] == p) >= XI_MIN[p]
        prob += pulp.lpSum(s[i] for i in idx if pos[i] == p) <= XI_MAX[p]

    # Captain
    prob += pulp.lpSum(c[i] for i in idx) == 1
    for i in idx:
        prob += c[i] <= s[i]

    # Transfer accounting
    prob += num_transfers <= TRANSFERS_CAP_PER_GW
    prob += paid >= num_transfers - free_available

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Solver status: {pulp.LpStatus[prob.status]}")

    df["in_squad"] = [int(round(x[i].value())) for i in idx]
    df["in_xi"] = [int(round(s[i].value())) for i in idx]
    df["is_captain"] = [int(round(c[i].value())) for i in idx]

    squad = df[df["in_squad"] == 1].copy()
    xi = squad[squad["in_xi"] == 1].copy()
    captain = squad[squad["is_captain"] == 1].iloc[0]

    formation = "-".join(
        str(int((xi[position_col] == p).sum())) for p in ["DEF", "MID", "FWD"]
    )
    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    squad = squad.assign(_pos_order=squad[position_col].map(pos_order)).sort_values(
        ["in_xi", "_pos_order", pred_col], ascending=[False, True, False]
    ).drop(columns="_pos_order")
    squad = assign_vice_and_bench_priority(squad, pred_col=pred_col, position_col=position_col)

    new_ids = set(squad[id_col].astype(int))
    n_in = len(new_ids - current_squad_ids)
    n_out = len(current_squad_ids - new_ids)
    paid_int = max(0, n_in - free_available)
    used_free = n_in - paid_int
    banked_next = min(MAX_BANKED_TRANSFERS, banked + 1 - used_free)

    return {
        "squad": squad,
        "xi": xi,
        "captain": captain,
        "vice_captain": squad[squad["is_vice"] == 1].iloc[0],
        "objective": float(pulp.value(prob.objective)),
        "cost": float(squad[value_col].sum() / 10.0),
        "formation": f"1-{formation}",
        "transfers_in": int(n_in),
        "transfers_out": int(n_out),
        "free_available": int(free_available),
        "paid_transfers": int(paid_int),
        "hit_cost": int(paid_int * HIT_COST),
        "banked_next": int(banked_next),
        "missing_from_pool": int(missing_from_pool),
    }


def _filter_to_candidate_pool(
    horizon_pools: list[pd.DataFrame],
    current_squad_ids: set,
    pred_col: str,
    id_col: str,
    candidate_pool_size: int,
) -> list[pd.DataFrame]:
    """Trim each GW's pool to top-N by predicted points, then union with the
    current squad (which we always need available to "keep")."""
    top_ids: set = set(current_squad_ids)
    for p in horizon_pools:
        top_ids |= set(
            p.nlargest(candidate_pool_size, pred_col)[id_col].astype(int)
        )
    return [
        p[p[id_col].astype(int).isin(top_ids)].reset_index(drop=True)
        for p in horizon_pools
    ]


def optimize_squad_horizon(
    horizon_pools: list[pd.DataFrame],
    current_squad_ids: set,
    banked_transfers: int,
    pred_col: str = "pred_points",
    value_col: str = "value",
    position_col: str = "position",
    team_col: str = "team_name",
    id_col: str = "element",
    gw_weights: Optional[list[float]] = None,
    candidate_pool_size: int = 150,
) -> dict:
    """Plan transfers over a multi-GW horizon, return the FIRST GW's decision.

    Variables are indexed by (player_id, t) where t is 0..H-1 within the
    horizon. Transitions tie GW-to-GW state via ``in_var[i,t] >= x[i,t] - x[i,t-1]``
    (a player counted as transferred-in at GW t if they were absent at t-1 and
    present at t). The banked-transfer state machine carries forward:

        b_{t+1} <= min(MAX_BANKED, b_t + 1 - transfers_t + paid_t)

    Pool trimming: keep only the top ``candidate_pool_size`` players per GW by
    predicted points, unioned with ``current_squad_ids``. Without this the ILP
    has ~12k+ binaries and solves slowly; with it the solver finishes in seconds.
    """
    H = len(horizon_pools)
    if H == 0:
        raise ValueError("Empty horizon.")
    if gw_weights is None:
        gw_weights = [1.0] * H
    if len(gw_weights) != H:
        raise ValueError("gw_weights length must match horizon length.")

    # Normalize positions and drop incomplete rows in each pool.
    cleaned: list[pd.DataFrame] = []
    for p in horizon_pools:
        df = p.copy()
        df[position_col] = df[position_col].map(POSITION_NORMALIZE).fillna(df[position_col])
        df = df.dropna(subset=[position_col, team_col, value_col, pred_col, id_col])
        df[id_col] = df[id_col].astype(int)
        cleaned.append(df.reset_index(drop=True))

    cleaned = _filter_to_candidate_pool(
        cleaned, current_squad_ids, pred_col, id_col, candidate_pool_size,
    )

    # Per-GW per-player lookup
    pdata: list[dict] = [
        {int(r[id_col]): r for _, r in p.iterrows()} for p in cleaned
    ]
    all_pids = sorted({pid for d in pdata for pid in d.keys()})

    prob = pulp.LpProblem("fpl_squad_horizon", pulp.LpMaximize)

    x: dict[tuple[int, int], pulp.LpVariable] = {}
    s: dict[tuple[int, int], pulp.LpVariable] = {}
    c: dict[tuple[int, int], pulp.LpVariable] = {}
    in_v: dict[tuple[int, int], pulp.LpVariable] = {}
    for t in range(H):
        for pid in all_pids:
            if pid not in pdata[t]:
                continue
            x[pid, t] = pulp.LpVariable(f"x_{pid}_{t}", cat="Binary")
            s[pid, t] = pulp.LpVariable(f"s_{pid}_{t}", cat="Binary")
            c[pid, t] = pulp.LpVariable(f"c_{pid}_{t}", cat="Binary")
            in_v[pid, t] = pulp.LpVariable(f"in_{pid}_{t}", cat="Binary")

    paid = [
        pulp.LpVariable(f"paid_{t}", lowBound=0, upBound=TRANSFERS_CAP_PER_GW, cat="Integer")
        for t in range(H)
    ]
    # Banked transfers entering each GW (continuous is fine; integrality is implied
    # by the integer paid/transfer counts).
    b = [
        pulp.LpVariable(f"banked_{t}", lowBound=0, upBound=MAX_BANKED_TRANSFERS)
        for t in range(H + 1)
    ]
    prob += b[0] == banked_transfers

    # Objective
    obj_terms = []
    for t in range(H):
        for pid in all_pids:
            if (pid, t) not in s:
                continue
            pred = float(pdata[t][pid][pred_col])
            obj_terms.append(gw_weights[t] * pred * s[pid, t])
            obj_terms.append(gw_weights[t] * pred * c[pid, t])
        obj_terms.append(-gw_weights[t] * HIT_COST * paid[t])
    prob += pulp.lpSum(obj_terms)

    # Per-GW constraints
    for t in range(H):
        slot_x = [x[pid, t] for pid in all_pids if (pid, t) in x]
        prob += pulp.lpSum(slot_x) == SQUAD_TOTAL

        for pos_name, k in POSITION_LIMITS.items():
            prob += pulp.lpSum(
                x[pid, t] for pid in all_pids
                if (pid, t) in x and pdata[t][pid][position_col] == pos_name
            ) == k

        teams_t = {pdata[t][pid][team_col] for pid in all_pids if (pid, t) in x}
        for team_name in teams_t:
            prob += pulp.lpSum(
                x[pid, t] for pid in all_pids
                if (pid, t) in x and pdata[t][pid][team_col] == team_name
            ) <= MAX_PER_CLUB

        prob += pulp.lpSum(
            float(pdata[t][pid][value_col]) * x[pid, t]
            for pid in all_pids if (pid, t) in x
        ) <= BUDGET_TENTHS

        for pid in all_pids:
            if (pid, t) in s:
                prob += s[pid, t] <= x[pid, t]
        prob += pulp.lpSum(s[pid, t] for pid in all_pids if (pid, t) in s) == XI_TOTAL
        for pos_name in POSITION_LIMITS:
            xi_count = pulp.lpSum(
                s[pid, t] for pid in all_pids
                if (pid, t) in s and pdata[t][pid][position_col] == pos_name
            )
            prob += xi_count >= XI_MIN[pos_name]
            prob += xi_count <= XI_MAX[pos_name]

        prob += pulp.lpSum(c[pid, t] for pid in all_pids if (pid, t) in c) == 1
        for pid in all_pids:
            if (pid, t) in c:
                prob += c[pid, t] <= s[pid, t]

        # Transfer accounting: in_v[i,t] >= x[i,t] - x_prev[i]
        for pid in all_pids:
            if (pid, t) not in in_v:
                continue
            if t == 0:
                prev = 1 if pid in current_squad_ids else 0
            else:
                prev = x[pid, t - 1] if (pid, t - 1) in x else 0
            prob += in_v[pid, t] >= x[pid, t] - prev

        transfers_t = pulp.lpSum(in_v[pid, t] for pid in all_pids if (pid, t) in in_v)
        prob += transfers_t <= TRANSFERS_CAP_PER_GW

        # Hit slack: paid_t >= transfers_t - (b_t + 1)
        prob += paid[t] >= transfers_t - (b[t] + 1)
        # Banked state machine: b_{t+1} <= b_t + 1 - (transfers_t - paid_t)
        # = b_t + 1 - transfers_t + paid_t.   (Implicitly capped at MAX_BANKED by ub.)
        prob += b[t + 1] <= b[t] + 1 - transfers_t + paid[t]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Horizon solver status: {pulp.LpStatus[prob.status]}")

    # Extract first-GW decision and reshape into the same format
    # as ``optimize_squad_with_transfers``.
    t0 = 0
    first_pool = cleaned[t0].copy()
    first_pool["in_squad"] = [
        int(round(x[int(r[id_col]), t0].value())) if (int(r[id_col]), t0) in x else 0
        for _, r in first_pool.iterrows()
    ]
    first_pool["in_xi"] = [
        int(round(s[int(r[id_col]), t0].value())) if (int(r[id_col]), t0) in s else 0
        for _, r in first_pool.iterrows()
    ]
    first_pool["is_captain"] = [
        int(round(c[int(r[id_col]), t0].value())) if (int(r[id_col]), t0) in c else 0
        for _, r in first_pool.iterrows()
    ]

    squad = first_pool[first_pool["in_squad"] == 1].copy()
    xi = squad[squad["in_xi"] == 1].copy()
    captain = squad[squad["is_captain"] == 1].iloc[0]

    formation = "-".join(
        str(int((xi[position_col] == p).sum())) for p in ["DEF", "MID", "FWD"]
    )
    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    squad = squad.assign(_pos_order=squad[position_col].map(pos_order)).sort_values(
        ["in_xi", "_pos_order", pred_col], ascending=[False, True, False]
    ).drop(columns="_pos_order")
    squad = assign_vice_and_bench_priority(squad, pred_col=pred_col, position_col=position_col)

    new_ids = set(squad[id_col].astype(int))
    n_in = len(new_ids - current_squad_ids)
    n_out = len(current_squad_ids - new_ids)
    free_avail = banked_transfers + 1
    paid_int = max(0, n_in - free_avail)
    used_free = n_in - paid_int
    banked_next = min(MAX_BANKED_TRANSFERS, banked_transfers + 1 - used_free)
    missing_from_pool = len(current_squad_ids) - sum(
        1 for pid in current_squad_ids if pid in pdata[0]
    )

    # Surface the rest of the horizon for inspection (helpful for debugging the
    # "save now, spend later" behaviour).
    horizon_plan = []
    for t in range(H):
        chosen_ids = sorted([
            pid for pid in all_pids
            if (pid, t) in x and round(x[pid, t].value()) == 1
        ])
        horizon_plan.append({
            "gw_index": t,
            "squad_ids": chosen_ids,
            "transfers": int(round(sum(
                in_v[pid, t].value() for pid in all_pids if (pid, t) in in_v
            ))),
            "paid": int(round(paid[t].value())),
            "banked_in": float(b[t].value()),
            "banked_out": float(b[t + 1].value()),
        })

    return {
        "squad": squad,
        "xi": xi,
        "captain": captain,
        "vice_captain": squad[squad["is_vice"] == 1].iloc[0],
        "objective": float(pulp.value(prob.objective)),
        "cost": float(squad[value_col].sum() / 10.0),
        "formation": f"1-{formation}",
        "transfers_in": int(n_in),
        "transfers_out": int(n_out),
        "free_available": int(free_avail),
        "paid_transfers": int(paid_int),
        "hit_cost": int(paid_int * HIT_COST),
        "banked_next": int(banked_next),
        "missing_from_pool": int(missing_from_pool),
        "horizon_plan": horizon_plan,
        "horizon_length": H,
    }


def assign_vice_and_bench_priority(
    squad: pd.DataFrame,
    pred_col: str = "pred_points",
    position_col: str = "position",
) -> pd.DataFrame:
    """Post-solve helper: pick vice-captain and rank outfield bench by pred_points.

    Vice = highest-predicted XI player who isn't the captain (optimal because
    vice's points get doubled iff captain DNPs, so you want the best backup).

    Bench priority: 1, 2, 3 for outfield bench (1 = first to come on). Bench GK
    has priority 0 (only relevant for replacing the starting GK). Starters get 0.
    """
    squad = squad.copy()

    non_captain_xi = squad[(squad["in_xi"] == 1) & (squad["is_captain"] != 1)]
    vice_idx = non_captain_xi.sort_values(pred_col, ascending=False).index[0]
    squad["is_vice"] = 0
    squad.loc[vice_idx, "is_vice"] = 1

    bench = squad[squad["in_xi"] == 0]
    bench_outfield = bench[bench[position_col] != "GK"].sort_values(
        pred_col, ascending=False
    )
    squad["bench_priority"] = 0
    for rank, idx in enumerate(bench_outfield.index, start=1):
        squad.loc[idx, "bench_priority"] = rank
    return squad


# --- Realized scoring --------------------------------------------------------

def score_squad(squad: pd.DataFrame, actual_col: str = "total_points") -> dict:
    """Realized points for a chosen squad+XI+captain (simple, no auto-subs).

    Counts XI points + captain's points twice. No vice-captain fallback, no
    bench substitutions. Useful as a baseline to measure the auto-sub uplift.
    """
    xi = squad[squad["in_xi"] == 1]
    captain = squad[squad["is_captain"] == 1].iloc[0]
    xi_points = float(xi[actual_col].fillna(0).sum())
    captain_points = float(captain[actual_col] if pd.notna(captain[actual_col]) else 0.0)
    return {
        "xi_points": xi_points,
        "captain_bonus": captain_points,
        "total": xi_points + captain_points,
    }


def _is_legal_formation(positions: list[str]) -> bool:
    if len(positions) != XI_TOTAL:
        return False
    counts = {p: positions.count(p) for p in POSITION_LIMITS}
    return (
        counts.get("GK", 0) == 1
        and counts.get("DEF", 0) >= XI_MIN["DEF"]
        and counts.get("FWD", 0) >= XI_MIN["FWD"]
    )


def apply_auto_subs(
    squad: pd.DataFrame,
    minutes_col: str = "minutes",
    position_col: str = "position",
) -> pd.DataFrame:
    """Return the realized XI after applying FPL auto-sub rules.

    Rules:
      1. Bench GK only replaces the starting GK (and only if the starting GK
         didn't play and the bench GK did).
      2. Outfield bench iterates in priority order; each player looks for the
         first non-playing outfield starter they can replace while keeping
         a legal formation (1 GK / >=3 DEF / >=1 FWD / 11 total).
    """
    squad = squad.copy()
    played = (squad[minutes_col].fillna(0) > 0).to_dict()

    xi_idx = list(squad[squad["in_xi"] == 1].index)
    bench = squad[squad["in_xi"] == 0]
    bench_gk_idx = list(bench[bench[position_col] == "GK"].index)
    bench_outfield_idx = list(
        bench[bench[position_col] != "GK"]
        .sort_values("bench_priority")
        .index
    )

    # GK swap
    starting_gks = [i for i in xi_idx if squad.at[i, position_col] == "GK"]
    if starting_gks and not played.get(starting_gks[0], False):
        if bench_gk_idx and played.get(bench_gk_idx[0], False):
            xi_idx.remove(starting_gks[0])
            xi_idx.append(bench_gk_idx[0])

    # Outfield swaps in bench-priority order
    for sub_i in bench_outfield_idx:
        if not played.get(sub_i, False):
            continue
        non_playing_outfielders = [
            i for i in xi_idx
            if squad.at[i, position_col] != "GK" and not played.get(i, False)
        ]
        if not non_playing_outfielders:
            break
        for cand in non_playing_outfielders:
            test_positions = [
                squad.at[i, position_col]
                for i in xi_idx if i != cand
            ] + [squad.at[sub_i, position_col]]
            if _is_legal_formation(test_positions):
                xi_idx.remove(cand)
                xi_idx.append(sub_i)
                break

    return squad.loc[xi_idx]


def score_squad_realistic(
    squad: pd.DataFrame,
    actual_col: str = "total_points",
    minutes_col: str = "minutes",
    position_col: str = "position",
) -> dict:
    """Realized score with auto-subs and captain/vice fallback applied."""
    realized_xi = apply_auto_subs(squad, minutes_col=minutes_col, position_col=position_col)
    xi_points = float(realized_xi[actual_col].fillna(0).sum())

    captain = squad[squad["is_captain"] == 1].iloc[0]
    vice = squad[squad["is_vice"] == 1].iloc[0]
    cap_played = (captain[minutes_col] or 0) > 0
    vice_played = (vice[minutes_col] or 0) > 0
    if cap_played:
        captain_bonus = float(captain[actual_col]) if pd.notna(captain[actual_col]) else 0.0
        captain_used = captain["name"] if "name" in captain.index else None
    elif vice_played:
        captain_bonus = float(vice[actual_col]) if pd.notna(vice[actual_col]) else 0.0
        captain_used = vice["name"] if "name" in vice.index else None
    else:
        captain_bonus = 0.0
        captain_used = None

    # Track which bench players actually came on.
    started_idx = set(squad[squad["in_xi"] == 1].index)
    realized_idx = set(realized_xi.index)
    subs_in = list(realized_idx - started_idx)

    return {
        "xi_points": xi_points,
        "captain_bonus": captain_bonus,
        "total": xi_points + captain_bonus,
        "captain_used": captain_used,
        "subs_applied": len(subs_in),
    }


# --- Backtest harness --------------------------------------------------------

def _aggregate_dgw_rows(df: pd.DataFrame, pred_cols: list[str]) -> pd.DataFrame:
    """Collapse DGW duplicates so each (season, element, gw) is one row.

    Sums points-like columns (predictions, actual total_points, minutes) across
    fixtures and takes the first non-null for identity columns (position, team,
    name, value). In a DGW a player can play two games in one GW and their
    points stack — modelling them as separate rows would let the ILP "pick"
    both, which double-counts.
    """
    sum_cols = list({*pred_cols, "total_points", "minutes"})
    agg = {col: "sum" for col in sum_cols if col in df.columns}
    for col in df.columns:
        if col in agg or col in ["season", "element", "gw"]:
            continue
        agg[col] = "first"
    return df.groupby(["season", "element", "gw"], as_index=False).agg(agg)


def backtest(
    predictions_csv: Path,
    dataset_csv: Path,
    pred_col: str = "pred_mlp",
    include_oracle: bool = True,
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Iterate over GWs in the predictions CSV, optimize per-GW, score vs actual.

    Also runs an oracle (optimize using true points) as the upper bound — the
    gap between model and oracle is how much the model is leaving on the table.
    """
    preds = pd.read_csv(predictions_csv)
    hist = pd.read_csv(
        dataset_csv,
        low_memory=False,
        usecols=["season", "element", "gw", "value", "position", "team_name", "minutes"],
    )
    df = preds.merge(hist, on=["season", "element", "gw"], how="left")
    df = _aggregate_dgw_rows(df, pred_cols=[pred_col, "total_points"])

    # Drop rows without enough metadata to be optimizable.
    df = df.dropna(subset=["value", "position", "team_name", pred_col, "total_points"])

    rows = []
    for gw, chunk in df.groupby("gw", sort=True):
        if len(chunk) < SQUAD_TOTAL:
            continue
        try:
            model_res = optimize_squad(chunk, pred_col=pred_col)
        except Exception as exc:
            print(f"GW{int(gw):>2}: model optimizer failed: {exc}")
            continue

        simple = score_squad(model_res["squad"], "total_points")
        realistic = score_squad_realistic(model_res["squad"], "total_points")
        row = {
            "gw": int(gw),
            "model_score_simple": simple["total"],
            "model_score": realistic["total"],
            "model_xi_points": realistic["xi_points"],
            "model_captain_picked": model_res["captain"]["name"],
            "model_vice_picked": model_res["vice_captain"]["name"],
            "model_captain_used": realistic["captain_used"],
            "model_captain_points": realistic["captain_bonus"],
            "model_subs_applied": realistic["subs_applied"],
            "autosub_uplift": realistic["total"] - simple["total"],
            "model_formation": model_res["formation"],
            "model_cost_m": model_res["cost"],
        }

        if include_oracle:
            try:
                oracle_res = optimize_squad(chunk, pred_col="total_points")
                oracle_score = score_squad_realistic(oracle_res["squad"], "total_points")
                row["oracle_score"] = oracle_score["total"]
                row["oracle_captain"] = oracle_res["captain"]["name"]
                row["regret"] = oracle_score["total"] - realistic["total"]
            except Exception as exc:
                print(f"GW{int(gw):>2}: oracle optimizer failed: {exc}")

        rows.append(row)

    out = pd.DataFrame(rows)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
    return out


def backtest_with_transfers(
    predictions_csv: Path,
    dataset_csv: Path,
    pred_col: str = "pred_mlp",
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Walk the season GW1 -> GW38 carrying squad state and banked transfers.

    GW1: pick a fresh squad (no prior state). Every subsequent GW: pass the
    previous squad and banked count into ``optimize_squad_with_transfers``.

    Reports per-GW: chosen captain, transfers made, hit cost, realized score
    (auto-subs + captain/vice fallback), banked transfers carried forward.
    """
    preds = pd.read_csv(predictions_csv)
    hist = pd.read_csv(
        dataset_csv,
        low_memory=False,
        usecols=["season", "element", "gw", "value", "position", "team_name", "minutes"],
    )
    df = preds.merge(hist, on=["season", "element", "gw"], how="left")
    df = _aggregate_dgw_rows(df, pred_cols=[pred_col, "total_points"])
    df = df.dropna(subset=["value", "position", "team_name", pred_col, "total_points"])

    rows = []
    current_squad_ids: Optional[set] = None
    banked = 0

    for gw, chunk in df.groupby("gw", sort=True):
        if len(chunk) < SQUAD_TOTAL:
            continue

        if current_squad_ids is None:
            # GW1: fresh-squad mode.
            res = optimize_squad(chunk, pred_col=pred_col)
            res.update({
                "transfers_in": 0,
                "transfers_out": 0,
                "free_available": 1,
                "paid_transfers": 0,
                "hit_cost": 0,
                "banked_next": 0,
                "missing_from_pool": 0,
            })
        else:
            res = optimize_squad_with_transfers(
                chunk, current_squad_ids, banked, pred_col=pred_col,
            )

        realized = score_squad_realistic(res["squad"], "total_points")
        net_score = realized["total"] - res["hit_cost"]

        rows.append({
            "gw": int(gw),
            "gw_score_gross": realized["total"],
            "hit_cost": res["hit_cost"],
            "gw_score_net": net_score,
            "transfers_in": res["transfers_in"],
            "transfers_out": res["transfers_out"],
            "free_available": res["free_available"],
            "paid_transfers": res["paid_transfers"],
            "banked_before": banked,
            "banked_after": res["banked_next"],
            "captain_picked": res["captain"]["name"],
            "captain_used": realized["captain_used"],
            "subs_applied": realized["subs_applied"],
            "formation": res["formation"],
            "cost_m": res["cost"],
            "missing_from_pool": res["missing_from_pool"],
        })

        current_squad_ids = set(res["squad"]["element"].astype(int))
        banked = res["banked_next"]

    out = pd.DataFrame(rows)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
    return out


def backtest_with_horizon(
    predictions_csv: Path,
    dataset_csv: Path,
    pred_col: str = "pred_mlp",
    horizon: int = 4,
    gw_weights: Optional[list[float]] = None,
    candidate_pool_size: int = 150,
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Rolling-horizon backtest. At every GW, re-plan over the next ``horizon``
    GWs (clipped at season end), then commit only the first GW's decision."""
    preds = pd.read_csv(predictions_csv)
    hist = pd.read_csv(
        dataset_csv,
        low_memory=False,
        usecols=["season", "element", "gw", "value", "position", "team_name", "minutes"],
    )
    df = preds.merge(hist, on=["season", "element", "gw"], how="left")
    df = _aggregate_dgw_rows(df, pred_cols=[pred_col, "total_points"])
    df = df.dropna(subset=["value", "position", "team_name", pred_col, "total_points"])

    gws = sorted(df["gw"].unique())
    gw_to_pool = {gw: df[df["gw"] == gw].reset_index(drop=True) for gw in gws}

    rows = []
    current_squad_ids: Optional[set] = None
    banked = 0

    for i, gw in enumerate(gws):
        # Build horizon = [gw, gw+1, ..., up to `horizon` GWs] clipped to season end
        horizon_gws = gws[i : i + horizon]
        horizon_pools = [gw_to_pool[g] for g in horizon_gws]

        if current_squad_ids is None:
            # GW1: fall back to fresh-squad mode (no prior state, no transfer accounting needed).
            res = optimize_squad(horizon_pools[0], pred_col=pred_col)
            res.update({
                "transfers_in": 0, "transfers_out": 0, "free_available": 1,
                "paid_transfers": 0, "hit_cost": 0, "banked_next": 0,
                "missing_from_pool": 0, "horizon_length": 1,
            })
        else:
            res = optimize_squad_horizon(
                horizon_pools, current_squad_ids, banked,
                pred_col=pred_col, gw_weights=gw_weights,
                candidate_pool_size=candidate_pool_size,
            )

        realized = score_squad_realistic(res["squad"], "total_points")
        net = realized["total"] - res["hit_cost"]

        rows.append({
            "gw": int(gw),
            "horizon_used": res.get("horizon_length", 1),
            "gw_score_gross": realized["total"],
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

        current_squad_ids = set(res["squad"]["element"].astype(int))
        banked = res["banked_next"]

    out = pd.DataFrame(rows)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
    return out


def main():
    root = Path(__file__).resolve().parents[1]
    preds_path = root / "results" / "val_2024_25_predictions.csv"
    dataset_path = root / "data" / "processed" / "fpl_model_dataset.csv"
    fresh_out_path = root / "results" / "phase4_backtest_2024_25.csv"
    xfer_out_path = root / "results" / "phase4_backtest_transfers_2024_25.csv"

    # Transfer-constrained run (the realistic version).
    print("Transfer-constrained backtest (2025/26 rules, max 5 banked, -4 hits)...")
    xfer = backtest_with_transfers(
        preds_path, dataset_path, pred_col="pred_mlp", output_csv=xfer_out_path,
    )
    print()
    summary_cols = [
        "gw", "gw_score_gross", "hit_cost", "gw_score_net",
        "transfers_in", "banked_before", "banked_after",
        "captain_picked", "subs_applied", "formation",
    ]
    print("Per-GW summary (first 10):")
    print(xfer[summary_cols].head(10).to_string(index=False))
    print()
    print("Per-GW summary (last 5):")
    print(xfer[summary_cols].tail(5).to_string(index=False))
    print()
    total_gross = xfer["gw_score_gross"].sum()
    total_hits = xfer["hit_cost"].sum()
    total_net = xfer["gw_score_net"].sum()
    transfers_total = xfer["transfers_in"].sum()
    paid_total = xfer["paid_transfers"].sum()
    save_gws = (xfer["transfers_in"] == 0).sum()
    print("Season totals (transfer-constrained):")
    print(f"  Total gross points:   {total_gross:.0f} ({xfer['gw_score_gross'].mean():.1f}/GW)")
    print(f"  Total hits taken:     -{total_hits:.0f} pts ({paid_total} paid transfers)")
    print(f"  Total NET points:     {total_net:.0f} ({xfer['gw_score_net'].mean():.1f}/GW)")
    print(f"  Total transfers:      {transfers_total} ({transfers_total / 37:.1f}/GW after GW1)")
    print(f"  GWs with no transfer: {save_gws} (banking transfers)")
    print(f"  Avg banked after GW:  {xfer['banked_after'].mean():.2f} (cap 4)")
    print()
    print("Now the fresh-squad upper bound (no transfer constraint)...")
    results = backtest(preds_path, dataset_path, pred_col="pred_mlp", output_csv=fresh_out_path)

    print(f"Backtested {len(results)} gameweeks.")
    print()
    summary_cols = [
        "gw", "model_score_simple", "model_score", "autosub_uplift",
        "model_captain_picked", "model_captain_used", "model_subs_applied",
        "model_formation",
    ]
    if "oracle_score" in results.columns:
        summary_cols += ["oracle_score", "regret"]
    print("Per-GW summary (first 10):")
    print(results[summary_cols].head(10).to_string(index=False))
    print()
    print("Season totals:")
    print(f"  Model (simple scoring):    {results['model_score_simple'].sum():.0f} pts  ({results['model_score_simple'].mean():.1f}/GW)")
    print(f"  Model (auto-subs + vice):  {results['model_score'].sum():.0f} pts  ({results['model_score'].mean():.1f}/GW)")
    print(f"  Auto-sub uplift:           {results['autosub_uplift'].sum():.0f} pts  ({results['autosub_uplift'].mean():.2f}/GW)")
    if "oracle_score" in results.columns:
        print(f"  Oracle (perfect info):     {results['oracle_score'].sum():.0f} pts  ({results['oracle_score'].mean():.1f}/GW)")
        print(f"  Regret (oracle - model):   {results['regret'].sum():.0f} pts")
        print(f"  Model captures:            {100 * results['model_score'].sum() / results['oracle_score'].sum():.1f}% of oracle")
    print(f"  Avg formation:             {results['model_formation'].mode().iloc[0]}")
    print(f"  Most-picked captain:       {results['model_captain_picked'].value_counts().head(3).to_dict()}")
    vice_fallbacks = (results['model_captain_used'] != results['model_captain_picked']).sum()
    print(f"  Vice-captain triggered:    {vice_fallbacks} GW(s)")
    print()
    print(f"Transfer friction cost:    {results['model_score'].sum() - total_net:.0f} pts")
    print(f"  (fresh-squad upper bound: {results['model_score'].sum():.0f} pts, "
          f"transfer-constrained: {total_net:.0f} pts)")
    print()

    # Multi-GW horizon run (Task #9)
    print("=" * 60)
    print("Multi-GW horizon backtest (4-GW rolling planner)...")
    horizon_out_path = root / "results" / "phase4_backtest_horizon_2024_25.csv"
    hz = backtest_with_horizon(
        preds_path, dataset_path, pred_col="pred_mlp",
        horizon=4, output_csv=horizon_out_path,
    )
    hz_net = hz["gw_score_net"].sum()
    hz_gross = hz["gw_score_gross"].sum()
    hz_hits = hz["hit_cost"].sum()
    hz_transfers = hz["transfers_in"].sum()
    hz_save_gws = (hz["transfers_in"] == 0).sum()
    print(f"  Total NET points:      {hz_net:.0f} ({hz['gw_score_net'].mean():.1f}/GW)")
    print(f"  Gross / hits:          {hz_gross:.0f} / -{hz_hits:.0f} ({hz['paid_transfers'].sum()} paid)")
    print(f"  Total transfers:       {hz_transfers} ({hz_transfers / 37:.2f}/GW after GW1)")
    print(f"  GWs with no transfer:  {hz_save_gws} (banking opportunity used)")
    print(f"  Avg banked after GW:   {hz['banked_after'].mean():.2f} (cap {MAX_BANKED_TRANSFERS})")
    print(f"  Max banked seen:       {hz['banked_after'].max():.0f}")
    print()
    print("=" * 60)
    print("Comparison: greedy vs horizon (both transfer-constrained, 2025/26 rules)")
    print(f"  Greedy single-GW:  {total_net:.0f} pts ({total_net / 38:.1f}/GW), "
          f"never banks (avg {xfer['banked_after'].mean():.2f})")
    print(f"  Horizon (4 GW):    {hz_net:.0f} pts ({hz_net / 38:.1f}/GW), "
          f"avg banked {hz['banked_after'].mean():.2f}")
    print(f"  Delta:             {hz_net - total_net:+.0f} pts ({(hz_net - total_net) / 38:+.2f}/GW)")


if __name__ == "__main__":
    main()
