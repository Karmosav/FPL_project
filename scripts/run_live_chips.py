"""End-to-end chip experiment on the 2025/26 live data.

Pipeline:
  1. Run the baseline 4-GW horizon backtest (no chips) and capture per-GW
     squad detail.
  2. Search for the best chip-deployment GWs two ways:
       - Heuristic: pick GWs by predicted captain / bench scores (realistic).
       - Oracle: pick GWs by actual captain / bench scores (upper bound).
  3. Re-run the backtest with each schedule applied.
  4. Report the uplift vs the no-chip baseline.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from chip_strategy import (  # noqa: E402
    backtest_with_chips,
    collect_timeline_from_backtest,
    find_chip_schedule_from_baseline,
)
from squad_optimizer import _aggregate_dgw_rows  # noqa: E402


PREDS = ROOT / "results" / "val_2025_26_predictions.csv"
DATASET = ROOT / "data" / "processed" / "fpl_model_dataset_2025_26.csv"
OUT_DIR = ROOT / "results"
LEADERBOARD_TOP = 2582


def summarise(label: str, results, baseline_net: float | None = None) -> float:
    net = results["gw_score_net"].sum()
    gross = results["gw_score_gross"].sum()
    hits = results["hit_cost"].sum()
    chip_gws = results[results["chip"] != ""][["gw", "chip", "tc_uplift", "bb_uplift"]]
    print(f"  {label}")
    print(f"    Net total:    {net:.0f} pts   ({net / 38:.1f}/GW)")
    print(f"    Gross / hits: {gross:.0f} / -{hits:.0f}")
    if baseline_net is not None:
        print(f"    vs baseline:  {net - baseline_net:+.0f} pts")
    if len(chip_gws):
        print(f"    Chips fired:")
        for _, row in chip_gws.iterrows():
            uplift = row["tc_uplift"] + row["bb_uplift"]
            print(f"      GW{int(row['gw']):>2}  {row['chip']:<6}  uplift {uplift:+.1f}")
    print()
    return net


def main():
    print("=" * 60)
    print("PHASE 5 — CHIP STRATEGY ON 2025/26 LIVE DATA")
    print("=" * 60)
    print()

    print("[1/4] Baseline backtest (no chips) + capturing squad detail...")
    summary, timeline = collect_timeline_from_backtest(
        PREDS, DATASET, pred_col="pred_decomposed_tuned",
    )
    timeline.to_csv(OUT_DIR / "phase5_baseline_timeline.csv", index=False)
    baseline_net = summary["gw_score_net"].sum()
    print(f"    Baseline (no chips):    {baseline_net:.0f} pts   ({baseline_net / 38:.1f}/GW)\n")

    print("[2/4] Searching for chip schedules...")
    # Build the merged pool DataFrame (preds + dataset) for wildcard search
    preds = pd.read_csv(PREDS)
    hist = pd.read_csv(
        DATASET, low_memory=False,
        usecols=["season", "element", "gw", "value", "position", "team_name", "minutes"],
    )
    pool_df = preds.merge(hist, on=["season", "element", "gw"], how="left")
    pool_df = _aggregate_dgw_rows(pool_df, pred_cols=["pred_decomposed_tuned", "total_points"])
    pool_df = pool_df.dropna(subset=["value", "position", "team_name", "pred_decomposed_tuned", "total_points"])

    heuristic = find_chip_schedule_from_baseline(summary, timeline, pool_df=pool_df, use_actuals=False)
    oracle = find_chip_schedule_from_baseline(summary, timeline, pool_df=pool_df, use_actuals=True)
    print(f"    Heuristic (from predictions): {heuristic}")
    print(f"    Oracle (perfect info):        {oracle}")
    print()

    print("[3/5] Heuristic schedule, TC + BB only (isolate WC effect)...")
    heur_tc_bb_only = {k: v for k, v in heuristic.items() if k != "wildcard"}
    heur_tc_bb_res = backtest_with_chips(
        PREDS, DATASET, pred_col="pred_decomposed_tuned",
        chip_schedule=heur_tc_bb_only,
    )
    print()

    print("[4/5] Heuristic schedule, all chips (TC + BB + WC)...")
    heuristic_res = backtest_with_chips(
        PREDS, DATASET, pred_col="pred_decomposed_tuned",
        chip_schedule=heuristic,
        output_csv=OUT_DIR / "phase5_heuristic_chips.csv",
    )
    print()

    print("[5/5] Oracle schedule, all chips...")
    oracle_res = backtest_with_chips(
        PREDS, DATASET, pred_col="pred_decomposed_tuned",
        chip_schedule=oracle,
        output_csv=OUT_DIR / "phase5_oracle_chips.csv",
    )
    print()

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    no_chip_net = summarise("No chips (baseline)", _wrap_no_chip(summary), baseline_net=None)
    tc_bb_net = summarise("Heuristic TC + BB only", heur_tc_bb_res, baseline_net=no_chip_net)
    heuristic_net = summarise("Heuristic TC + BB + WC", heuristic_res, baseline_net=no_chip_net)
    oracle_net = summarise("Oracle (upper bound)", oracle_res, baseline_net=no_chip_net)
    print(f"  WC marginal contribution (heuristic): {heuristic_net - tc_bb_net:+.0f} pts")
    print()

    print("=" * 60)
    print("vs LIVE LEADERBOARD")
    print("=" * 60)
    print(f"  Top live human:        {LEADERBOARD_TOP} pts")
    print(f"  Bot no chips:          {no_chip_net:.0f}  ({no_chip_net - LEADERBOARD_TOP:+.0f})")
    print(f"  Bot + heuristic TC+BB: {tc_bb_net:.0f}  ({tc_bb_net - LEADERBOARD_TOP:+.0f})")
    print(f"  Bot + heuristic all:   {heuristic_net:.0f}  ({heuristic_net - LEADERBOARD_TOP:+.0f})")
    print(f"  Bot + oracle:          {oracle_net:.0f}  ({oracle_net - LEADERBOARD_TOP:+.0f})")
    uplift = heuristic_net - no_chip_net
    oracle_uplift = oracle_net - no_chip_net
    if oracle_uplift > 0:
        print(f"  Heuristic captures {uplift / oracle_uplift * 100:.0f}% of the oracle chip uplift")


def _wrap_no_chip(summary):
    """Adapt the baseline summary to look like a backtest_with_chips output for
    consistent printing."""
    import pandas as pd
    return pd.DataFrame({
        "gw": summary["gw"],
        "gw_score_gross": summary["gw_score_net"] + summary["hit_cost"],
        "chip": "",
        "tc_uplift": 0.0,
        "bb_uplift": 0.0,
        "hit_cost": summary["hit_cost"],
        "gw_score_net": summary["gw_score_net"],
    })


if __name__ == "__main__":
    main()
