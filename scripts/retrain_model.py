"""Retrain the tuned decomposed model with 2025-26 folded into the data.

Split change vs the original model:
    old   train 2016-17..2023-24   val 2024-25
    new   train 2016-17..2024-25   val 2025-26

So the new model gains a full extra season of training data (2024-25) and is
validated on a season it has never seen (2025-26). The previous checkpoint is
re-scored on the same validation rows so the comparison is like-for-like.

Usage:
    python scripts/retrain_model.py                      # default split
    python scripts/retrain_model.py --val-season 2024-25 # reproduce old split
    python scripts/retrain_model.py --promote            # overwrite the live checkpoint
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_feature_ablations import (  # noqa: E402
    DecomposedFPLNet,
    build_head_targets,
    build_matrix,
    eval_metrics,
    expected_fpl_points_torch,
    load_modeling_df,
    tensor_to_numpy,
    train_decomposed,
)
from sklearn.preprocessing import StandardScaler  # noqa: E402

ALL_SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]
OLD_CKPT = ROOT / "results" / "phase3_decomposed_tuned.pt"
NEW_CKPT = ROOT / "results" / "phase3_decomposed_tuned_v2.pt"
PRED_OUT = ROOT / "results" / "retrain_val_predictions.csv"
CMP_OUT = ROOT / "results" / "phase3_retrain_comparison.csv"


def score_checkpoint(ckpt_path, X_val, y_val, played_mask):
    """Run a saved checkpoint over the validation matrix using ITS OWN scaler."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = DecomposedFPLNet(
        in_dim=ckpt["in_dim"], layers=tuple(ckpt["layers"]), dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
    Xs = ((X_val - mean) / np.where(scale == 0, 1.0, scale)).astype(np.float32)

    pos = torch.as_tensor(y_val["position_id"].tolist(), dtype=torch.long)
    with torch.no_grad():
        out = model(torch.tensor(Xs, dtype=torch.float32))
        pred = tensor_to_numpy(expected_fpl_points_torch(out, pos))
    return pred, eval_metrics(y_val["total_points"], pred, played_mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-season", default="2025-26")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--promote", action="store_true",
                    help="Copy the new model over the checkpoint used by live inference.")
    args = ap.parse_args()

    val_season = args.val_season
    train_seasons = [s for s in ALL_SEASONS if s < val_season]

    print(f"Train : {train_seasons[0]} .. {train_seasons[-1]}  ({len(train_seasons)} seasons)")
    print(f"Val   : {val_season}\n")

    df = load_modeling_df()
    train_df = df[df["season"].isin(train_seasons)].copy()
    val_df = df[df["season"] == val_season].copy()
    if val_df.empty:
        raise SystemExit(f"No rows for validation season {val_season}.")
    played = val_df["minutes"].fillna(0).to_numpy() > 0
    print(f"  train {len(train_df):,} rows | val {len(val_df):,} rows "
          f"({played.sum():,} played)\n")

    X_train, kept = build_matrix(train_df, None)
    X_val, _ = build_matrix(val_df, None)
    y_train = build_head_targets(train_df)
    y_val = build_head_targets(val_df)
    print(f"  features: {X_train.shape[1]}\n")

    # ---- baseline: existing checkpoint on the same val rows ----------------
    print(f"Scoring existing checkpoint ({OLD_CKPT.name}) on {val_season}...")
    old_pred, old_m = score_checkpoint(OLD_CKPT, X_val, y_val, played)
    print(f"  rho_played={old_m['spearman_played']:.4f}  rho_all={old_m['spearman_all']:.4f}  "
          f"MAE_played={old_m['mae_played']:.4f}\n")

    # ---- train the new model ----------------------------------------------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    print("Training new model (tuned config: [256,128,64], dropout 0.25, lr 1e-3)...")
    new_model, new_pred, epochs_ran = train_decomposed(
        X_train_s, y_train, X_val_s, y_val, epochs=args.epochs,
    )
    new_m = eval_metrics(y_val["total_points"], new_pred, played)
    print(f"  stopped after {epochs_ran} epochs")
    print(f"  rho_played={new_m['spearman_played']:.4f}  rho_all={new_m['spearman_all']:.4f}  "
          f"MAE_played={new_m['mae_played']:.4f}\n")

    # ---- report ------------------------------------------------------------
    rows = []
    for label, m in [("existing (train->2023-24)", old_m), ("retrained (train->2024-25)", new_m)]:
        r = dict(m)
        r["model"] = label
        rows.append(r)
    cmp = pd.DataFrame(rows)[
        ["model", "spearman_played", "spearman_all", "mae_played", "mae_all",
         "rmse_played", "r2_all"]
    ]
    cmp.to_csv(CMP_OUT, index=False)

    print("=" * 74)
    print(f"COMPARISON on held-out {val_season}")
    print("=" * 74)
    print(cmp.to_string(index=False))
    print()
    d_rho = new_m["spearman_played"] - old_m["spearman_played"]
    d_mae = new_m["mae_played"] - old_m["mae_played"]
    print(f"  delta rho_played : {d_rho:+.4f}")
    print(f"  delta MAE_played : {d_mae:+.4f}")
    print("  NOTE: single training run; variance on this architecture is ~+/-0.003-0.005 rho.")
    print()

    torch.save({
        "model_state": new_model.state_dict(),
        "in_dim": X_train_s.shape[1],
        "layers": [256, 128, 64],
        "dropout": 0.25,
        "lr": 1e-3,
        "aux_weight": 1.0,
        "name": "decomposed_tuned_v2",
        "feature_names": kept,
        "train_seasons": train_seasons,
        "val_season": val_season,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    }, NEW_CKPT)
    print(f"Saved model -> {NEW_CKPT.name}")

    if args.promote:
        shutil.copy2(OLD_CKPT, OLD_CKPT.with_suffix(".pt.bak"))
        shutil.copy2(NEW_CKPT, OLD_CKPT)
        print(f"PROMOTED: {NEW_CKPT.name} -> {OLD_CKPT.name} "
              f"(previous kept as {OLD_CKPT.name}.bak)")
    else:
        print("Not promoted. Re-run with --promote to make this the live model.")
    print()

    print(f"Writing validation predictions -> {PRED_OUT.name}")
    out = val_df[["season", "element", "gw", "player_id", "name"]].copy()
    out["total_points"] = y_val["total_points"]
    out["pred_existing"] = old_pred
    out["pred_retrained"] = new_pred
    out.to_csv(PRED_OUT, index=False)
    print(f"Comparison table -> {CMP_OUT.name}")


if __name__ == "__main__":
    main()
