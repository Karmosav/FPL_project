"""Visualize the 2025/26 live simulation — squad evolution and score timeline.

Re-runs the chip-augmented backtest with per-(GW, player) detail capture so
the heatmap reflects the *actual* squads chosen each gameweek (including the
two wildcard turnovers).

Outputs:
    results/phase5_squad_tenure_2025_26.png   — Gantt heatmap of player tenure
    results/phase5_score_timeline_2025_26.png — per-GW + cumulative score chart
    results/phase5_live_timeline.csv          — per-(GW, player) detail (chip-aware)
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from squad_optimizer import (  # noqa: E402
    SQUAD_TOTAL,
    _aggregate_dgw_rows,
    optimize_squad,
    optimize_squad_horizon,
    optimize_squad_with_transfers,
    score_squad_realistic,
)
from chip_strategy import (  # noqa: E402
    collect_timeline_from_backtest,
    find_chip_schedule_from_baseline,
    _captain_or_vice_actual,
    _bench_actual,
)


PREDS = ROOT / "results" / "val_2025_26_predictions.csv"
DATASET = ROOT / "data" / "processed" / "fpl_model_dataset_2025_26.csv"
OUT_DIR = ROOT / "results"
PRED_COL = "pred_decomposed_tuned"
HORIZON = 4
POOL_SIZE = 150


# --- chip-aware backtest with rich logging ----------------------------------

def run_detailed_chip_backtest(chip_schedule: dict):
    preds = pd.read_csv(PREDS)
    hist = pd.read_csv(
        DATASET, low_memory=False,
        usecols=["season", "element", "gw", "value", "position", "team_name", "minutes"],
    )
    df = preds.merge(hist, on=["season", "element", "gw"], how="left")
    df = _aggregate_dgw_rows(df, pred_cols=[PRED_COL, "total_points"])
    df = df.dropna(subset=["value", "position", "team_name", PRED_COL, "total_points"])

    gws = sorted(df["gw"].unique())
    gw_to_pool = {gw: df[df["gw"] == gw].reset_index(drop=True) for gw in gws}

    summary_rows = []
    timeline_rows = []
    current_squad_ids: Optional[set] = None
    saved_squad_before_fh: Optional[set] = None
    saved_banked_before_fh: int = 0
    banked = 0

    def chip_active(name, gw):
        return gw in (chip_schedule.get(name) or [])

    for i, gw in enumerate(gws):
        is_tc = chip_active("triple_captain", gw)
        is_bb = chip_active("bench_boost", gw)
        is_wc = chip_active("wildcard", gw)
        is_fh = chip_active("free_hit", gw)

        horizon_pools = [gw_to_pool[g] for g in gws[i : i + HORIZON]]
        if len(horizon_pools[0]) < SQUAD_TOTAL:
            continue

        if current_squad_ids is None:
            res = optimize_squad(horizon_pools[0], pred_col=PRED_COL)
            res.update({"transfers_in": 0, "transfers_out": 0, "free_available": 1,
                        "paid_transfers": 0, "hit_cost": 0, "banked_next": 0})
        elif is_fh:
            saved_squad_before_fh = set(current_squad_ids)
            saved_banked_before_fh = banked
            res = optimize_squad(horizon_pools[0], pred_col=PRED_COL)
            res.update({"transfers_in": 0, "transfers_out": 0, "free_available": 0,
                        "paid_transfers": 0, "hit_cost": 0, "banked_next": banked})
        elif is_wc:
            res = optimize_squad_with_transfers(
                horizon_pools[0], current_squad_ids, banked_transfers=SQUAD_TOTAL,
                pred_col=PRED_COL,
            )
            res["banked_next"] = 0
            res["hit_cost"] = 0
            res["paid_transfers"] = 0
        else:
            res = optimize_squad_horizon(
                horizon_pools, current_squad_ids, banked, pred_col=PRED_COL,
                candidate_pool_size=POOL_SIZE,
            )

        squad = res["squad"]
        realized = score_squad_realistic(squad, "total_points")

        tc_uplift = _captain_or_vice_actual(squad) if is_tc else 0.0
        bb_uplift = _bench_actual(squad) if is_bb else 0.0
        gross = realized["total"] + tc_uplift + bb_uplift
        net = gross - res["hit_cost"]

        chip_label = ",".join(c for c, on in [
            ("TC", is_tc), ("BB", is_bb), ("WC", is_wc), ("FH", is_fh)
        ] if on) or ""

        for _, row in squad.iterrows():
            timeline_rows.append({
                "gw": int(gw),
                "element": int(row["element"]),
                "name": row.get("name", ""),
                "position": row["position"],
                "team_name": row.get("team_name", ""),
                "pred_points": float(row[PRED_COL]),
                "actual_points": float(row.get("total_points", 0) or 0),
                "minutes": float(row.get("minutes", 0) or 0),
                "in_xi": int(row["in_xi"]),
                "is_captain": int(row["is_captain"]),
                "is_vice": int(row["is_vice"]),
                "chip": chip_label,
            })

        summary_rows.append({
            "gw": int(gw),
            "gw_score_gross": gross,
            "hit_cost": res["hit_cost"],
            "gw_score_net": net,
            "tc_uplift": tc_uplift,
            "bb_uplift": bb_uplift,
            "chip": chip_label,
            "captain_picked": res["captain"]["name"],
            "captain_used": realized["captain_used"],
            "transfers_in": res["transfers_in"],
            "banked_before": banked,
            "banked_after": res["banked_next"],
        })

        if is_fh:
            current_squad_ids = saved_squad_before_fh
            banked = saved_banked_before_fh
        else:
            current_squad_ids = set(squad["element"].astype(int))
            banked = res["banked_next"]

    return pd.DataFrame(summary_rows), pd.DataFrame(timeline_rows)


# --- plots ------------------------------------------------------------------

def plot_squad_tenure(timeline: pd.DataFrame, out_path: Path) -> None:
    """Gantt-style heatmap. y=player (grouped by position), x=GW, cell shaded
    by role: bench / starting XI / captain."""
    POSITION_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    players = (
        timeline.groupby(["element", "name", "position"], as_index=False)["gw"]
        .count()
        .rename(columns={"gw": "weeks_owned"})
        .assign(_pos=lambda d: d["position"].map(POSITION_ORDER))
        .sort_values(["_pos", "weeks_owned"], ascending=[True, False])
        .reset_index(drop=True)
    )
    player_order = list(players["element"])
    player_labels = {row["element"]: f"{row['name'].replace('_', ' ')} ({row['position']})"
                     for _, row in players.iterrows()}

    gws = sorted(timeline["gw"].unique())
    grid = np.zeros((len(player_order), len(gws)))
    eid_to_row = {eid: i for i, eid in enumerate(player_order)}
    gw_to_col = {gw: i for i, gw in enumerate(gws)}
    for _, r in timeline.iterrows():
        i = eid_to_row[int(r["element"])]
        j = gw_to_col[int(r["gw"])]
        if r["is_captain"]:
            grid[i, j] = 3
        elif r["in_xi"]:
            grid[i, j] = 2
        else:
            grid[i, j] = 1

    cmap = ListedColormap(["#f4f4f4", "#c9d8e4", "#5eb1bf", "#e26d5c"])
    fig_h = max(6.0, 0.22 * len(player_order))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")

    ax.set_yticks(range(len(player_order)))
    ax.set_yticklabels([player_labels[e] for e in player_order], fontsize=8)
    ax.set_xticks(range(len(gws))[::2])
    ax.set_xticklabels([str(gws[i]) for i in range(0, len(gws), 2)], fontsize=9)
    ax.set_xlabel("Gameweek")
    ax.set_title(f"Squad tenure — 2025/26 live simulation "
                 f"({len(player_order)} unique players owned across the season)")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#c9d8e4"),
        plt.Rectangle((0, 0), 1, 1, color="#5eb1bf"),
        plt.Rectangle((0, 0), 1, 1, color="#e26d5c"),
    ]
    ax.legend(handles, ["Bench", "Starting XI", "Captain"],
              loc="upper right", framealpha=0.95, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_score_timeline(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={"hspace": 0.25})

    ax = axes[0]
    ax.bar(summary["gw"], summary["gw_score_gross"], color="#5eb1bf",
           label="Gross (XI + chip uplifts)", alpha=0.85)
    ax.bar(summary["gw"], -summary["hit_cost"], bottom=summary["gw_score_gross"],
           color="#e26d5c", label="Hit cost", alpha=0.85)
    ax.plot(summary["gw"], summary["gw_score_net"], color="#222",
            marker="o", markersize=3, linewidth=1.2, label="Net score")
    ax.axhline(summary["gw_score_net"].mean(), color="#222",
               linestyle="--", linewidth=0.8, alpha=0.5)

    # Annotate chip gameweeks
    chip_marks = summary[summary["chip"] != ""]
    for _, row in chip_marks.iterrows():
        ax.annotate(
            row["chip"],
            xy=(row["gw"], row["gw_score_gross"]),
            xytext=(0, 6), textcoords="offset points",
            ha="center", fontsize=8, color="#c8773a", fontweight="bold",
        )

    ax.set_ylabel("Points per GW")
    ax.set_title(
        f"2025/26 live simulation — net {summary['gw_score_net'].sum():.0f} pts "
        f"({summary['gw_score_net'].mean():.1f}/GW), captain {summary['captain_picked'].value_counts().idxmax().replace('_', ' ')}"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    cum_net = summary["gw_score_net"].cumsum()
    cum_gross = summary["gw_score_gross"].cumsum()
    ax.plot(summary["gw"], cum_gross, color="#5eb1bf", linewidth=2,
            label="Cumulative gross")
    ax.plot(summary["gw"], cum_net, color="#222", linewidth=2,
            label="Cumulative net")
    ax.fill_between(summary["gw"], cum_net, cum_gross,
                    color="#e26d5c", alpha=0.25, label="Lost to hits")
    ax.axhline(2582, color="#888", linestyle=":", linewidth=1,
               label="Top human (2,582)")
    ax.set_xlabel("Gameweek")
    ax.set_ylabel("Cumulative points")
    ax.set_xticks(summary["gw"][::2])
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Phase 5 — 2025/26 live backtest with chip strategy", fontsize=12, y=0.995)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Building baseline timeline (or loading cached)...")
    baseline_summary, baseline_timeline = collect_timeline_from_backtest(
        PREDS, DATASET, pred_col=PRED_COL,
    )

    print("Searching for heuristic chip schedule...")
    pool_df = pd.read_csv(PREDS).merge(
        pd.read_csv(
            DATASET, low_memory=False,
            usecols=["season", "element", "gw", "value", "position", "team_name", "minutes"],
        ),
        on=["season", "element", "gw"], how="left",
    )
    pool_df = _aggregate_dgw_rows(pool_df, pred_cols=[PRED_COL, "total_points"])
    pool_df = pool_df.dropna(subset=["value", "position", "team_name", PRED_COL, "total_points"])
    schedule = find_chip_schedule_from_baseline(
        baseline_summary, baseline_timeline, pool_df=pool_df, use_actuals=False,
    )
    print(f"  schedule: {schedule}")

    print("Running chip-aware backtest with detail capture...")
    summary, timeline = run_detailed_chip_backtest(schedule)
    timeline.to_csv(OUT_DIR / "phase5_live_timeline.csv", index=False)
    print(f"  net total: {summary['gw_score_net'].sum():.0f} pts "
          f"({summary['gw_score_net'].mean():.1f}/GW)")

    print("Generating charts...")
    plot_squad_tenure(timeline, OUT_DIR / "phase5_squad_tenure_2025_26.png")
    plot_score_timeline(summary, OUT_DIR / "phase5_score_timeline_2025_26.png")

    print()
    print("Outputs:")
    print(f"  {OUT_DIR / 'phase5_squad_tenure_2025_26.png'}")
    print(f"  {OUT_DIR / 'phase5_score_timeline_2025_26.png'}")
    print(f"  {OUT_DIR / 'phase5_live_timeline.csv'}")
    print()
    print(f"Unique players owned: {timeline['element'].nunique()}")
    print(f"Most-captained: {summary['captain_picked'].value_counts().head(3).to_dict()}")


if __name__ == "__main__":
    main()
