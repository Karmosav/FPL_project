"""Run Karmo's tuned decomposed model on the 2025/26 live dataset and
produce per-(player, gw) point predictions.

Inputs:
    data/processed/fpl_model_dataset_2025_26.csv     — from build_live_dataset.py
    results/phase3_decomposed_tuned.pt               — Karmo's best model (notebook 03 sweep)

Output:
    results/val_2025_26_predictions.csv              — schema compatible with the
                                                       Phase 4 backtest harness
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "fpl_model_dataset_2025_26.csv"
CKPT_PATH = ROOT / "results" / "phase3_decomposed_tuned.pt"
OUT_PATH = ROOT / "results" / "val_2025_26_predictions.csv"


# ---- feature schema — must match notebook 03 exactly ----------------------
NUMERIC_FEATURES = [
    "total_points_roll3", "total_points_roll5",
    "minutes_roll3", "minutes_roll5",
    "goals_scored_roll3", "goals_scored_roll5",
    "assists_roll3", "assists_roll5",
    "expected_goals_roll3", "expected_goals_roll5",
    "expected_assists_roll3", "expected_assists_roll5",
    "team_goals_scored_gw_roll5", "team_goals_conceded_gw_roll5", "team_points_gw_roll5",
    "opponent_team_points_roll5", "opponent_team_gc_roll5",
    "last_season_ppg", "last_season_minutes_share",
    "was_home", "rest_days",
    "value", "selected", "transfers_in", "transfers_out", "transfers_balance",
]
POSITION_MAP = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "AM": 3, "FWD": 4}


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Match notebook 03's build_feature_matrix exactly."""
    x_num = (
        df[NUMERIC_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    pos_id = df["position"].map(POSITION_MAP).fillna(0).astype(int).to_numpy(dtype=np.float32)
    prom = df["is_promoted_team"].map({True: 1.0, False: 0.0}).fillna(0.0).astype(float).to_numpy(dtype=np.float32)
    return np.hstack([x_num, pos_id.reshape(-1, 1), prom.reshape(-1, 1)])


# ---- model — reconstruct DecomposedFPLNet from notebook 03 -----------------

class DecomposedFPLNet(nn.Module):
    """Shared backbone + 7 independent heads. Architecture is parameterised by
    `layers` so we can load Karmo's tuned variant ([256, 128, 64]) without
    hardcoding the original [128, 64, 32] one from notebook 03."""

    def __init__(self, in_dim: int, layers: list[int], dropout: float):
        super().__init__()
        blocks = []
        prev = in_dim
        for h in layers:
            blocks.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        self.backbone = nn.Sequential(*blocks)
        last = layers[-1]
        self.head_play = nn.Linear(last, 1)
        self.head_sixty = nn.Linear(last, 1)
        self.head_goal = nn.Linear(last, 1)
        self.head_assist = nn.Linear(last, 1)
        self.head_cs = nn.Linear(last, 1)
        self.head_bonus = nn.Linear(last, 1)
        self.head_gc = nn.Linear(last, 1)

    def forward(self, x):
        h = self.backbone(x)
        return {
            "play": torch.sigmoid(self.head_play(h)).squeeze(-1),
            "sixty": torch.sigmoid(self.head_sixty(h)).squeeze(-1),
            "goal": torch.sigmoid(self.head_goal(h)).squeeze(-1),
            "assist": torch.sigmoid(self.head_assist(h)).squeeze(-1),
            "cs": torch.sigmoid(self.head_cs(h)).squeeze(-1),
            "bonus": F.relu(self.head_bonus(h)).squeeze(-1),
            "gc": F.relu(self.head_gc(h)).squeeze(-1),
        }


# ---- recombination — verbatim from notebook 03 ----------------------------

def expected_fpl_points(
    p_play, p_sixty, p_goal, p_assist, p_cs, e_bonus, e_gc, position_id,
) -> np.ndarray:
    pos = position_id.astype(int)
    goal_pts = np.select(
        [pos == 1, pos == 2, pos == 3, pos == 4],
        [6.0, 6.0, 5.0, 4.0],
        default=5.0,
    )
    cs_pts = np.select([pos <= 2, pos == 3], [4.0, 1.0], default=0.0)
    appearance = p_play * (1.0 + p_sixty)
    goal_points = p_play * p_sixty * p_goal * goal_pts
    assist_points = p_play * p_sixty * p_assist * 3.0
    cs_points = p_play * p_cs * cs_pts
    gc_penalty = p_play * np.where(pos <= 2, -0.5 * np.clip(e_gc, 0.0, None), 0.0)
    bonus_points = p_play * e_bonus
    return (
        appearance + goal_points + assist_points + cs_points
        + gc_penalty + bonus_points
    ).astype(np.float32)


def main():
    print(f"Loading live dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["is_promoted_team"] = df["is_promoted_team"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    n_total = len(df)
    print(f"  rows: {n_total:,}, players: {df['element'].nunique()}, GWs: {df['gw'].nunique()}")

    print(f"Loading model: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    layers = ckpt["layers"]
    in_dim = ckpt["in_dim"]
    dropout = ckpt["dropout"]
    print(f"  layers: {layers}, in_dim: {in_dim}, dropout: {dropout}")

    model = DecomposedFPLNet(in_dim=in_dim, layers=layers, dropout=dropout)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Build features & apply the training-time StandardScaler
    X = build_feature_matrix(df)
    assert X.shape[1] == in_dim, f"feature count mismatch: got {X.shape[1]}, model wants {in_dim}"
    mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
    X_scaled = (X - mean) / np.where(scale == 0, 1.0, scale)

    print(f"Running inference on {len(X_scaled):,} rows...")
    with torch.no_grad():
        out = model(torch.tensor(X_scaled, dtype=torch.float32))
    heads = {k: v.cpu().numpy() for k, v in out.items()}

    pos_id_arr = df["position"].map(POSITION_MAP).fillna(0).astype(int).to_numpy()
    pred_decomposed_tuned = expected_fpl_points(
        heads["play"], heads["sixty"], heads["goal"], heads["assist"],
        heads["cs"], heads["bonus"], heads["gc"], pos_id_arr,
    )

    # Save in the same schema the Phase 4 backtest expects — note we do NOT
    # include `minutes` here because the backtest re-merges that column from
    # the dataset CSV and would collide on a duplicate key.
    out_df = df[["season", "element", "gw", "player_id", "name", "total_points"]].copy()
    out_df["pred_decomposed_tuned"] = pred_decomposed_tuned
    # Also expose under "pred_mlp" alias so the existing optimizer main()
    # can be pointed here with `pred_col="pred_mlp"` if desired.
    out_df["pred_mlp"] = pred_decomposed_tuned

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out_df):,} predictions → {OUT_PATH}")
    print()

    # Sanity check + correlation use the in-memory df (which has minutes).
    df["pred_decomposed_tuned"] = pred_decomposed_tuned
    played = df[df["minutes"].fillna(0) > 0]
    top = played.nlargest(10, "pred_decomposed_tuned")[
        ["gw", "name", "minutes", "total_points", "pred_decomposed_tuned"]
    ]
    print("Top 10 predicted player-GWs (where they played):")
    print(top.to_string(index=False))
    print()

    from scipy.stats import spearmanr
    rho_all = spearmanr(df["total_points"], df["pred_decomposed_tuned"]).statistic
    rho_played = spearmanr(played["total_points"], played["pred_decomposed_tuned"]).statistic
    print(f"Spearman ρ vs actual (all rows): {rho_all:.3f}")
    print(f"Spearman ρ vs actual (played):  {rho_played:.3f}")


if __name__ == "__main__":
    main()
