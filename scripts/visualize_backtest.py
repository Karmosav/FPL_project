"""Visualize the Phase 4 transfer-constrained backtest.

Re-runs the season GW1->GW38 (using ``optimize_squad`` for GW1 then
``optimize_squad_with_transfers`` thereafter) but logs per-(gw, player) detail.
Produces:

  results/phase4_squad_timeline.csv  — one row per (gw, player) the bot owned
  results/phase4_transfers_log.csv   — one row per transfer (gw, out -> in)
  results/phase4_squad_tenure.png    — Gantt-style chart of player tenure
  results/phase4_score_timeline.png  — gross/net/hit-cost over the season
  results/phase4_season_report.md    — readable per-GW writeup

Run from the project root:
    python scripts/visualize_backtest.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from squad_optimizer import (
    SQUAD_TOTAL,
    _aggregate_dgw_rows,
    optimize_squad,
    optimize_squad_with_transfers,
    score_squad_realistic,
)


def run_detailed_backtest(predictions_csv: Path, dataset_csv: Path, pred_col: str = "pred_mlp"):
    """Walk the season, capturing rich detail per-GW."""
    preds = pd.read_csv(predictions_csv)
    hist = pd.read_csv(
        dataset_csv,
        low_memory=False,
        usecols=["season", "element", "gw", "value", "position", "team_name", "minutes"],
    )
    df = preds.merge(hist, on=["season", "element", "gw"], how="left")
    df = _aggregate_dgw_rows(df, pred_cols=[pred_col, "total_points"])
    df = df.dropna(subset=["value", "position", "team_name", pred_col, "total_points"])

    gw_summary = []
    timeline_rows = []
    transfer_rows = []

    current_squad_ids: set | None = None
    last_squad_lookup: dict[int, dict] = {}
    banked = 0

    for gw, chunk in df.groupby("gw", sort=True):
        if len(chunk) < SQUAD_TOTAL:
            continue

        if current_squad_ids is None:
            res = optimize_squad(chunk, pred_col=pred_col)
            res.update({
                "transfers_in": 0, "transfers_out": 0, "free_available": 1,
                "paid_transfers": 0, "hit_cost": 0, "banked_next": 0,
                "missing_from_pool": 0,
            })
        else:
            res = optimize_squad_with_transfers(
                chunk, current_squad_ids, banked, pred_col=pred_col,
            )

        squad = res["squad"]
        realized = score_squad_realistic(squad, "total_points")

        # Captain / vice for this GW
        captain_id = int(squad.loc[squad["is_captain"] == 1, "element"].iloc[0])
        vice_id = int(squad.loc[squad["is_vice"] == 1, "element"].iloc[0])

        # Per-(gw, player) rows
        for _, row in squad.iterrows():
            timeline_rows.append({
                "gw": int(gw),
                "element": int(row["element"]),
                "name": row.get("name", ""),
                "position": row["position"],
                "team_name": row["team_name"],
                "value": float(row["value"]),
                "pred_points": float(row[pred_col]),
                "actual_points": float(row["total_points"]),
                "minutes": float(row.get("minutes", 0) or 0),
                "in_xi": int(row["in_xi"]),
                "is_captain": int(row["is_captain"]),
                "is_vice": int(row["is_vice"]),
                "bench_priority": int(row["bench_priority"]),
            })

        # Transfers in/out by matching against previous squad
        new_ids = set(squad["element"].astype(int))
        if current_squad_ids is not None:
            ids_in = new_ids - current_squad_ids
            ids_out = current_squad_ids - new_ids
            # Pair them in order of predicted-gain magnitude for readability.
            in_records = (
                squad[squad["element"].isin(ids_in)]
                [["element", "name", "position", "team_name", "value", pred_col, "total_points"]]
                .sort_values(pred_col, ascending=False)
                .reset_index(drop=True)
            )
            out_records = [
                last_squad_lookup[i] for i in ids_out if i in last_squad_lookup
            ]
            out_records = sorted(out_records, key=lambda r: -r.get(pred_col, 0))
            for k in range(max(len(in_records), len(out_records))):
                in_r = in_records.iloc[k] if k < len(in_records) else None
                out_r = out_records[k] if k < len(out_records) else None
                transfer_rows.append({
                    "gw": int(gw),
                    "out_name": out_r["name"] if out_r else "",
                    "out_position": out_r["position"] if out_r else "",
                    "out_team": out_r["team_name"] if out_r else "",
                    "out_pred": out_r.get(pred_col, np.nan) if out_r else np.nan,
                    "out_actual": out_r.get("total_points", np.nan) if out_r else np.nan,
                    "in_name": in_r["name"] if in_r is not None else "",
                    "in_position": in_r["position"] if in_r is not None else "",
                    "in_team": in_r["team_name"] if in_r is not None else "",
                    "in_pred": float(in_r[pred_col]) if in_r is not None else np.nan,
                    "in_actual": float(in_r["total_points"]) if in_r is not None else np.nan,
                })

        gw_summary.append({
            "gw": int(gw),
            "gw_score_gross": realized["total"],
            "hit_cost": res["hit_cost"],
            "gw_score_net": realized["total"] - res["hit_cost"],
            "xi_points": realized["xi_points"],
            "captain_bonus": realized["captain_bonus"],
            "transfers_in": res["transfers_in"],
            "free_available": res["free_available"],
            "paid_transfers": res["paid_transfers"],
            "banked_before": banked,
            "banked_after": res["banked_next"],
            "captain_picked": squad.loc[squad["element"] == captain_id, "name"].iloc[0],
            "vice_picked": squad.loc[squad["element"] == vice_id, "name"].iloc[0],
            "captain_used": realized["captain_used"],
            "subs_applied": realized["subs_applied"],
            "formation": res["formation"],
            "cost_m": res["cost"],
        })

        # Snapshot the current squad as a lookup for next GW's transfer-pairing.
        last_squad_lookup = {
            int(r["element"]): {
                "element": int(r["element"]),
                "name": r["name"],
                "position": r["position"],
                "team_name": r["team_name"],
                "value": float(r["value"]),
                pred_col: float(r[pred_col]),
                "total_points": float(r["total_points"]),
            }
            for _, r in squad.iterrows()
        }
        current_squad_ids = new_ids
        banked = res["banked_next"]

    return (
        pd.DataFrame(gw_summary),
        pd.DataFrame(timeline_rows),
        pd.DataFrame(transfer_rows),
    )


# --- Plots -------------------------------------------------------------------

def plot_score_timeline(gw_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"hspace": 0.25})

    # Top: per-GW gross/net + hit cost
    ax = axes[0]
    ax.bar(gw_df["gw"], gw_df["gw_score_gross"], color="#5eb1bf", label="Gross (XI + captain)", alpha=0.8)
    ax.bar(gw_df["gw"], -gw_df["hit_cost"], bottom=gw_df["gw_score_gross"], color="#e26d5c", label="Hit cost (-4 each)", alpha=0.85)
    ax.plot(gw_df["gw"], gw_df["gw_score_net"], color="#222", marker="o", markersize=3, linewidth=1.2, label="Net score")
    ax.axhline(gw_df["gw_score_net"].mean(), color="#222", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Points (per GW)")
    ax.set_title(f"Per-GW score — season net {gw_df['gw_score_net'].sum():.0f} pts ({gw_df['gw_score_net'].mean():.1f}/GW)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Bottom: cumulative
    ax = axes[1]
    ax.plot(gw_df["gw"], gw_df["gw_score_gross"].cumsum(), color="#5eb1bf", linewidth=2, label="Cumulative gross")
    ax.plot(gw_df["gw"], gw_df["gw_score_net"].cumsum(), color="#222", linewidth=2, label="Cumulative net")
    ax.fill_between(gw_df["gw"], gw_df["gw_score_net"].cumsum(), gw_df["gw_score_gross"].cumsum(), color="#e26d5c", alpha=0.25, label="Lost to hits")
    ax.set_xlabel("Gameweek")
    ax.set_ylabel("Cumulative points")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(gw_df["gw"][::2])

    fig.suptitle("Phase 4 — Transfer-constrained backtest (2024-25, pred_mlp)", fontsize=12, y=0.995)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_squad_tenure(timeline_df: pd.DataFrame, out_path: Path) -> None:
    """Gantt-style heatmap: y=player, x=GW, cell shaded if owned (XI/bench)."""
    POSITION_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    # Aggregate to one row per (player, gw) summarizing role.
    players = (
        timeline_df.groupby(["element", "name", "position"], as_index=False)["gw"]
        .count()
        .rename(columns={"gw": "weeks_owned"})
        .sort_values(["position", "weeks_owned"], key=lambda s: s.map(POSITION_ORDER) if s.name == "position" else -s)
        .reset_index(drop=True)
    )
    player_order = list(players["element"])
    player_labels = {row["element"]: f"{row['name']} ({row['position']})" for _, row in players.iterrows()}

    gws = sorted(timeline_df["gw"].unique())
    grid = np.zeros((len(player_order), len(gws)))  # 0=not owned, 1=bench, 2=XI, 3=captain

    eid_to_row = {eid: i for i, eid in enumerate(player_order)}
    gw_to_col = {gw: i for i, gw in enumerate(gws)}
    for _, r in timeline_df.iterrows():
        i = eid_to_row[int(r["element"])]
        j = gw_to_col[int(r["gw"])]
        if r["is_captain"]:
            grid[i, j] = 3
        elif r["in_xi"]:
            grid[i, j] = 2
        else:
            grid[i, j] = 1

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#f4f4f4", "#c9d8e4", "#5eb1bf", "#e26d5c"])

    fig_h = max(6.0, 0.22 * len(player_order))
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")

    ax.set_yticks(range(len(player_order)))
    ax.set_yticklabels([player_labels[e] for e in player_order], fontsize=8)
    ax.set_xticks(range(len(gws))[::2])
    ax.set_xticklabels([str(gws[i]) for i in range(0, len(gws), 2)], fontsize=8)
    ax.set_xlabel("Gameweek")
    ax.set_title(f"Squad tenure — {len(player_order)} unique players owned across 2024-25")

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#c9d8e4"),
        plt.Rectangle((0, 0), 1, 1, color="#5eb1bf"),
        plt.Rectangle((0, 0), 1, 1, color="#e26d5c"),
    ]
    ax.legend(handles, ["Bench", "Starting XI", "Captain"], loc="upper right", framealpha=0.9, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --- Markdown report ---------------------------------------------------------

def write_season_report(
    gw_df: pd.DataFrame,
    timeline_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = []
    lines.append("# Phase 4 transfer-constrained backtest — 2024-25 season report\n")
    lines.append("Model: `pred_mlp` (Karmo's direct MLP). 2025/26 transfer rules: max 5 banked, -4 per extra transfer.\n")

    total_net = gw_df["gw_score_net"].sum()
    total_gross = gw_df["gw_score_gross"].sum()
    total_hits = gw_df["hit_cost"].sum()
    lines.append("## Season totals")
    lines.append(f"- **Net points:** {total_net:.0f} ({gw_df['gw_score_net'].mean():.1f}/GW)")
    lines.append(f"- Gross points: {total_gross:.0f}")
    lines.append(f"- Hit cost: -{total_hits:.0f} ({gw_df['paid_transfers'].sum()} paid transfers, {(gw_df['hit_cost'] > 0).sum()} GWs with hits)")
    lines.append(f"- Total transfers: {gw_df['transfers_in'].sum()} ({gw_df['transfers_in'].sum() / max(1, len(gw_df) - 1):.1f}/GW after GW1)")
    lines.append(f"- Unique players owned: {timeline_df['element'].nunique()}")
    lines.append(f"- Most-picked captain: {gw_df['captain_picked'].value_counts().head(3).to_dict()}\n")

    lines.append("## Per-GW log\n")
    for _, row in gw_df.iterrows():
        gw = int(row["gw"])
        lines.append(f"### GW{gw} — net {row['gw_score_net']:.0f} pts (gross {row['gw_score_gross']:.0f}, hit -{row['hit_cost']:.0f})")
        lines.append(
            f"- Captain: **{row['captain_picked']}** (used: {row['captain_used']}), "
            f"vice: {row['vice_picked']}, formation: {row['formation']}, cost: £{row['cost_m']:.1f}m"
        )
        lines.append(
            f"- Transfers: {row['transfers_in']} in (free avail: {row['free_available']}, paid: {row['paid_transfers']}); "
            f"banked {row['banked_before']} -> {row['banked_after']}; auto-subs: {row['subs_applied']}"
        )
        # Transfer detail
        tf = transfer_df[transfer_df["gw"] == gw]
        if not tf.empty:
            lines.append("- Moves:")
            for _, t in tf.iterrows():
                out_desc = f"{t['out_name']} ({t['out_position']}, {t['out_team']})" if t["out_name"] else "—"
                in_desc = f"{t['in_name']} ({t['in_position']}, {t['in_team']})" if t["in_name"] else "—"
                pred_gain = (t.get("in_pred") or 0) - (t.get("out_pred") or 0)
                actual_gain = (t.get("in_actual") or 0) - (t.get("out_actual") or 0)
                lines.append(f"  - OUT {out_desc} -> IN {in_desc}  (pred Δ {pred_gain:+.1f}, actual Δ {actual_gain:+.1f})")
        # XI roster
        xi = timeline_df[(timeline_df["gw"] == gw) & (timeline_df["in_xi"] == 1)]
        bench = timeline_df[(timeline_df["gw"] == gw) & (timeline_df["in_xi"] == 0)]
        if not xi.empty:
            xi_str = ", ".join(
                f"{r['name']}{'*' if r['is_captain'] else ''}"
                f"{'^' if r['is_vice'] else ''} ({r['position']} {r['actual_points']:.0f}p)"
                for _, r in xi.iterrows()
            )
            lines.append(f"- XI: {xi_str}")
        if not bench.empty:
            bench_str = ", ".join(
                f"{r['name']} ({r['position']} {r['actual_points']:.0f}p)"
                for _, r in bench.iterrows()
            )
            lines.append(f"- Bench: {bench_str}")
        lines.append("")

    out_path.write_text("\n".join(lines))


def main():
    root = Path(__file__).resolve().parents[1]
    preds_path = root / "results" / "val_2024_25_predictions.csv"
    dataset_path = root / "data" / "processed" / "fpl_model_dataset.csv"
    out_dir = root / "results"

    print("Re-running transfer-constrained backtest with detail capture...")
    gw_df, timeline_df, transfer_df = run_detailed_backtest(preds_path, dataset_path)

    gw_df.to_csv(out_dir / "phase4_backtest_transfers_2024_25.csv", index=False)
    timeline_df.to_csv(out_dir / "phase4_squad_timeline.csv", index=False)
    transfer_df.to_csv(out_dir / "phase4_transfers_log.csv", index=False)

    print("Generating plots...")
    plot_score_timeline(gw_df, out_dir / "phase4_score_timeline.png")
    plot_squad_tenure(timeline_df, out_dir / "phase4_squad_tenure.png")

    print("Writing season report...")
    write_season_report(gw_df, timeline_df, transfer_df, out_dir / "phase4_season_report.md")

    print()
    print(f"Net season points: {gw_df['gw_score_net'].sum():.0f} ({gw_df['gw_score_net'].mean():.1f}/GW)")
    print(f"Unique players owned across the season: {timeline_df['element'].nunique()}")
    print(f"Most-captained: {gw_df['captain_picked'].value_counts().head(3).to_dict()}")
    print()
    print(f"Outputs written under {out_dir}:")
    print("  phase4_score_timeline.png       — per-GW + cumulative score chart")
    print("  phase4_squad_tenure.png         — player-tenure Gantt chart")
    print("  phase4_season_report.md         — readable per-GW writeup with rosters")
    print("  phase4_squad_timeline.csv       — one row per (gw, player owned)")
    print("  phase4_transfers_log.csv        — one row per transfer (out -> in)")


if __name__ == "__main__":
    main()
