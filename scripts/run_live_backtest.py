"""Run the Phase 4 optimizer backtest against 2025/26 live data and predictions.

This is the headline experiment for the final report: how would our bot have
performed in the just-finished 2025/26 season, and how does it stack against
the global FPL leaderboard?

Inputs:
    results/val_2025_26_predictions.csv             — from run_live_inference.py
    data/processed/fpl_model_dataset_2025_26.csv    — from build_live_dataset.py

Output:
    results/phase4_backtest_2025_26_greedy.csv
    results/phase4_backtest_2025_26_horizon.csv
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from squad_optimizer import (  # noqa: E402
    MAX_BANKED_TRANSFERS,
    backtest,
    backtest_with_horizon,
    backtest_with_transfers,
)


# 2025/26 reference points (manually recorded — top of the live leaderboard).
LEADERBOARD_TOP = 2582
FPL_GLOBAL_AVG_PER_GW = (55, 58)  # historical band


def main():
    preds = ROOT / "results" / "val_2025_26_predictions.csv"
    dataset = ROOT / "data" / "processed" / "fpl_model_dataset_2025_26.csv"
    out_dir = ROOT / "results"

    print("=" * 60)
    print("2025/26 LIVE SEASON BACKTEST")
    print("=" * 60)
    print(f"Predictions: {preds.name}")
    print(f"Dataset:     {dataset.name}")
    print(f"Reference:   top live manager = {LEADERBOARD_TOP} pts")
    print()

    # Fresh-squad upper bound (unrealistic, for reference)
    print("[1/3] Fresh-squad upper bound (no transfer limits)...")
    fresh = backtest(
        preds, dataset,
        pred_col="pred_decomposed_tuned",
        include_oracle=False,
        output_csv=out_dir / "phase4_backtest_2025_26_fresh.csv",
    )
    fresh_total = fresh["model_score"].sum()
    print(f"  Total: {fresh_total:.0f} pts ({fresh['model_score'].mean():.1f}/GW)")
    print()

    # Greedy single-GW
    print("[2/3] Greedy single-GW (transfer-constrained, 2025/26 rules)...")
    greedy = backtest_with_transfers(
        preds, dataset,
        pred_col="pred_decomposed_tuned",
        output_csv=out_dir / "phase4_backtest_2025_26_greedy.csv",
    )
    greedy_net = greedy["gw_score_net"].sum()
    greedy_hits = greedy["hit_cost"].sum()
    print(f"  Net: {greedy_net:.0f} pts ({greedy['gw_score_net'].mean():.1f}/GW)")
    print(f"  Hits taken: -{greedy_hits:.0f} ({greedy['paid_transfers'].sum()} paid transfers)")
    print(f"  Total transfers: {greedy['transfers_in'].sum()}")
    print()

    # 4-GW horizon (best realistic)
    print("[3/3] 4-GW horizon (best realistic, 2025/26 rules)...")
    hz = backtest_with_horizon(
        preds, dataset,
        pred_col="pred_decomposed_tuned",
        horizon=4,
        output_csv=out_dir / "phase4_backtest_2025_26_horizon.csv",
    )
    hz_net = hz["gw_score_net"].sum()
    hz_hits = hz["hit_cost"].sum()
    print(f"  Net: {hz_net:.0f} pts ({hz['gw_score_net'].mean():.1f}/GW)")
    print(f"  Hits taken: -{hz_hits:.0f} ({hz['paid_transfers'].sum()} paid transfers)")
    print(f"  Total transfers: {hz['transfers_in'].sum()}")
    print(f"  GWs with no transfer (banking): {(hz['transfers_in'] == 0).sum()}")
    print(f"  Avg banked after GW: {hz['banked_after'].mean():.2f} / {MAX_BANKED_TRANSFERS}")
    print()

    print("=" * 60)
    print("2025/26 SUMMARY")
    print("=" * 60)
    print(f"  Top live human:     {LEADERBOARD_TOP} pts")
    print(f"  Bot — fresh squad:  {fresh_total:.0f} pts  (+{fresh_total - LEADERBOARD_TOP:+.0f} vs top, upper bound)")
    print(f"  Bot — horizon:      {hz_net:.0f} pts  ({hz_net - LEADERBOARD_TOP:+.0f} vs top)")
    print(f"  Bot — greedy:       {greedy_net:.0f} pts  ({greedy_net - LEADERBOARD_TOP:+.0f} vs top)")
    print()
    print(f"  Bot per-GW avg (horizon): {hz['gw_score_net'].mean():.1f} pts")
    print(f"  FPL global avg/GW:        {FPL_GLOBAL_AVG_PER_GW[0]}-{FPL_GLOBAL_AVG_PER_GW[1]} pts (historical band)")
    print(f"  Captain picks (horizon):  {hz['captain_picked'].value_counts().head(3).to_dict()}")
    print(f"  Avg formation (horizon):  {hz['formation'].mode().iloc[0]}")


if __name__ == "__main__":
    main()
