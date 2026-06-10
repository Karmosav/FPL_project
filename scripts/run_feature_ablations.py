"""Feature-group ablation study for the tuned decomposed model.

For each named feature group we retrain the model with those columns dropped
(reducing in_dim accordingly), then report val metrics. The full-feature
baseline is also re-trained here for a like-for-like comparison.

Training setup mirrors notebook 03's tuned configuration:
  - Architecture: backbone [256, 128, 64] with BN + ReLU + Dropout(0.25)
  - 7 sub-event heads recombined into FPL expected points
  - Adam lr=1e-3, ReduceLROnPlateau on val ρ_all (mode=max, factor=0.5, patience=2)
  - Multi-task loss: head_loss_weight=0.5 + aux_points_weight=1.0 on recombined pts
  - Early stopping on val ρ_all, patience=6

Output:
  results/phase6_feature_ablations.csv — one row per ablation, with both
  ρ_all/ρ_played/MAE/RMSE metrics and the delta vs the full-feature baseline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "fpl_model_dataset.csv"
OUT_PATH = ROOT / "results" / "phase6_feature_ablations.csv"

TRAIN_SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24",
]
VAL_SEASON = "2024-25"
TARGET = "total_points"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Karmo's exact column groups
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

# Group definitions — what gets dropped in each ablation
FEATURE_GROUPS = {
    "player_form": [
        "total_points_roll3", "total_points_roll5",
        "minutes_roll3", "minutes_roll5",
        "goals_scored_roll3", "goals_scored_roll5",
        "assists_roll3", "assists_roll5",
        "expected_goals_roll3", "expected_goals_roll5",
        "expected_assists_roll3", "expected_assists_roll5",
    ],
    "team_form": [
        "team_goals_scored_gw_roll5",
        "team_goals_conceded_gw_roll5",
        "team_points_gw_roll5",
    ],
    "opponent_strength": [
        "opponent_team_points_roll5",
        "opponent_team_gc_roll5",
    ],
    "anchors": [
        "last_season_ppg",
        "last_season_minutes_share",
    ],
    "fixture_context": [
        "was_home", "rest_days",
    ],
    "market_signals": [
        "value", "selected", "transfers_in", "transfers_out", "transfers_balance",
    ],
    "position": ["__position__"],   # special marker — drops the position_id column
    "promoted_flag": ["__promoted__"],  # special marker — drops is_promoted_team
}


# ---- model + loss (faithful to notebook 03 tuned config) -------------------

class DecomposedFPLNet(nn.Module):
    def __init__(self, in_dim: int, layers=(256, 128, 64), dropout: float = 0.25):
        super().__init__()
        blocks = []
        prev = in_dim
        for h in layers:
            blocks.extend([
                nn.Linear(prev, h), nn.BatchNorm1d(h),
                nn.ReLU(), nn.Dropout(dropout),
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


GOAL_PTS_LUT = torch.tensor([5.0, 6.0, 6.0, 5.0, 4.0])
CS_PTS_LUT = torch.tensor([0.0, 4.0, 4.0, 1.0, 0.0])


def expected_fpl_points_torch(out: dict, position_id: torch.Tensor) -> torch.Tensor:
    pos = position_id.long().clamp(0, 4)
    goal_pts = GOAL_PTS_LUT.to(position_id.device)[pos]
    cs_pts = CS_PTS_LUT.to(position_id.device)[pos]
    p_play, p_sixty = out["play"], out["sixty"]
    appearance = p_play * (1.0 + p_sixty)
    goal_points = p_play * p_sixty * out["goal"] * goal_pts
    assist_points = p_play * p_sixty * out["assist"] * 3.0
    cs_points = p_play * out["cs"] * cs_pts
    gc_penalty = p_play * torch.where(pos <= 2, -0.5 * out["gc"], torch.zeros_like(out["gc"]))
    bonus_points = p_play * out["bonus"]
    return appearance + goal_points + assist_points + cs_points + gc_penalty + bonus_points


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return np.asarray(t.detach().cpu().tolist(), dtype=np.float32)


def build_head_targets(df: pd.DataFrame) -> dict:
    minutes = df["minutes"].fillna(0).to_numpy()
    starts = pd.to_numeric(df.get("starts", 0), errors="coerce")
    y_play = (minutes > 0).astype(np.float32)
    y_sixty = np.where(
        starts == 1, 1.0,
        np.where(starts == 0, 0.0, (minutes >= 60).astype(float)),
    ).astype(np.float32)
    goals = pd.to_numeric(df["goals_scored"], errors="coerce").fillna(0).to_numpy()
    assists = pd.to_numeric(df["assists"], errors="coerce").fillna(0).to_numpy()
    cs = pd.to_numeric(df["clean_sheets"], errors="coerce").fillna(0).to_numpy()
    bonus = pd.to_numeric(df["bonus"], errors="coerce").fillna(0).to_numpy()
    gc = pd.to_numeric(df["goals_conceded"], errors="coerce").fillna(0).to_numpy()
    return {
        "play": y_play,
        "sixty": y_sixty,
        "goal": (goals > 0).astype(np.float32),
        "assist": (assists > 0).astype(np.float32),
        "cs": (cs > 0).astype(np.float32),
        "bonus": bonus.astype(np.float32),
        "gc": gc.astype(np.float32),
        "total_points": df[TARGET].to_numpy(dtype=np.float32),
        "position_id": df["position_id"].to_numpy(dtype=np.int64),
    }


def train_decomposed(
    X_train_s, y_train, X_val_s, y_val,
    epochs=80, batch_size=4096, lr=1e-3, patience=6,
    aux_points_weight=1.0, head_loss_weight=0.5,
    layers=(256, 128, 64), dropout=0.25,
):
    in_dim = X_train_s.shape[1]
    torch.manual_seed(SEED)
    model = DecomposedFPLNet(in_dim, layers=layers, dropout=dropout).to(DEVICE)

    train_ds = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(y_train["play"], dtype=torch.float32),
        torch.tensor(y_train["sixty"], dtype=torch.float32),
        torch.tensor(y_train["goal"], dtype=torch.float32),
        torch.tensor(y_train["assist"], dtype=torch.float32),
        torch.tensor(y_train["cs"], dtype=torch.float32),
        torch.tensor(y_train["bonus"], dtype=torch.float32),
        torch.tensor(y_train["gc"], dtype=torch.float32),
        torch.tensor(y_train["total_points"], dtype=torch.float32),
        torch.tensor(y_train["position_id"], dtype=torch.long),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32, device=DEVICE)
    y_pts_val = y_val["total_points"]
    pos_val_t = torch.as_tensor(y_val["position_id"].tolist(), dtype=torch.long, device=DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2,
    )

    best_state = None
    best_val_rho = -1.0
    stale = 0
    epochs_ran = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, y_play, y_sixty, y_goal, y_assist, y_cs, y_bonus, y_gc, y_pts, pos_b in loader:
            xb = xb.to(DEVICE)
            y_play, y_sixty = y_play.to(DEVICE), y_sixty.to(DEVICE)
            y_goal, y_assist = y_goal.to(DEVICE), y_assist.to(DEVICE)
            y_cs, y_bonus, y_gc, y_pts = y_cs.to(DEVICE), y_bonus.to(DEVICE), y_gc.to(DEVICE), y_pts.to(DEVICE)
            pos_b = pos_b.to(DEVICE)
            out = model(xb)

            loss_play = F.binary_cross_entropy(out["play"], y_play)
            loss_sixty = F.binary_cross_entropy(out["sixty"], y_sixty, reduction="none")
            loss_sixty = (loss_sixty * (0.15 + 0.85 * y_play)).mean()
            w_play = 0.15 + 0.85 * y_play
            goal_loss = F.binary_cross_entropy(out["goal"], y_goal, reduction="none")
            assist_loss = F.binary_cross_entropy(out["assist"], y_assist, reduction="none")
            loss_goal = (goal_loss * w_play).mean()
            loss_assist = (assist_loss * w_play).mean()

            cs_loss = F.binary_cross_entropy(out["cs"], y_cs, reduction="none")
            cs_mask = (pos_b <= 2) | (pos_b == 3)
            w_cs = y_play * (0.15 + 0.85 * y_sixty)
            if cs_mask.any():
                loss_cs = (cs_loss * w_cs)[cs_mask].mean()
            else:
                loss_cs = (cs_loss * w_cs).mean()

            loss_bonus = F.mse_loss(out["bonus"], y_bonus)
            gc_loss = F.mse_loss(out["gc"], y_gc, reduction="none")
            def_gk = pos_b <= 2
            loss_gc = gc_loss[def_gk].mean() if def_gk.any() else torch.tensor(0.0, device=DEVICE)

            pred_pts = expected_fpl_points_torch(out, pos_b)
            loss_aux = F.mse_loss(pred_pts, y_pts)

            head_loss = loss_play + loss_sixty + loss_goal + loss_assist + loss_cs + loss_bonus + 0.25 * loss_gc
            loss = head_loss_weight * head_loss + aux_points_weight * loss_aux
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Val eval
        model.eval()
        with torch.no_grad():
            out_val = model(X_val_t)
            pred_val = tensor_to_numpy(expected_fpl_points_torch(out_val, pos_val_t))
            val_rho = spearmanr(y_pts_val, pred_val).statistic

        scheduler.step(val_rho)
        epochs_ran = epoch
        if val_rho > best_val_rho:
            best_val_rho = val_rho
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_val = model(X_val_t)
        pred_val = tensor_to_numpy(expected_fpl_points_torch(out_val, pos_val_t))
    return pred_val, epochs_ran


# ---- feature matrix builder, parametrised by dropped group ----------------

def load_modeling_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["position_id"] = df["position"].map(POSITION_MAP).fillna(3).astype(int)
    df["is_promoted_team"] = df["is_promoted_team"].map({
        True: 1.0, False: 0.0, "True": 1.0, "False": 0.0,
    }).fillna(0.0)
    df["was_home"] = pd.to_numeric(df["was_home"], errors="coerce").fillna(0.0)
    return df


def build_matrix(df: pd.DataFrame, drop_group: str | None) -> tuple[np.ndarray, list[str]]:
    """Return (X, kept_feature_names). drop_group=None keeps every feature."""
    drop_cols = set()
    drop_position = False
    drop_promoted = False
    if drop_group is not None:
        for col in FEATURE_GROUPS[drop_group]:
            if col == "__position__":
                drop_position = True
            elif col == "__promoted__":
                drop_promoted = True
            else:
                drop_cols.add(col)

    numeric_kept = [c for c in NUMERIC_FEATURES if c not in drop_cols]
    x_num = df[numeric_kept].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    parts = [x_num]
    kept = list(numeric_kept)
    if not drop_position:
        parts.append(df[["position_id"]].to_numpy(dtype=np.float32))
        kept.append("position_id")
    if not drop_promoted:
        parts.append(df[["is_promoted_team"]].to_numpy(dtype=np.float32))
        kept.append("is_promoted_team")
    return np.hstack(parts), kept


# ---- main driver -----------------------------------------------------------

def eval_metrics(y_true, y_pred, played_mask):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "mae_all": mean_absolute_error(y_true, y_pred),
        "rmse_all": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2_all": r2_score(y_true, y_pred),
        "spearman_all": spearmanr(y_true, y_pred).statistic,
        "mae_played": mean_absolute_error(y_true[played_mask], y_pred[played_mask]),
        "rmse_played": mean_squared_error(y_true[played_mask], y_pred[played_mask]) ** 0.5,
        "r2_played": r2_score(y_true[played_mask], y_pred[played_mask]),
        "spearman_played": spearmanr(y_true[played_mask], y_pred[played_mask]).statistic,
    }


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Loading {DATA_PATH.name}...")
    df = load_modeling_df()
    train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()
    val_df = df[df["season"] == VAL_SEASON].copy()
    played_mask = val_df["minutes"].fillna(0).to_numpy() > 0
    print(f"  train {len(train_df):,} rows, val {len(val_df):,} rows ({played_mask.sum():,} played)\n")

    y_train = build_head_targets(train_df)
    y_val = build_head_targets(val_df)

    ablations = ["__full__"] + list(FEATURE_GROUPS.keys())
    rows = []
    baseline_metrics = None

    for name in ablations:
        drop = None if name == "__full__" else name
        X_train, kept = build_matrix(train_df, drop)
        X_val, _ = build_matrix(val_df, drop)
        in_dim = X_train.shape[1]
        n_dropped = len(NUMERIC_FEATURES) + 2 - in_dim

        print(f"[{name}] in_dim={in_dim} (dropped {n_dropped} cols), training...")
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train).astype(np.float32)
        X_val_s = scaler.transform(X_val).astype(np.float32)

        pred_val, epochs_ran = train_decomposed(X_train_s, y_train, X_val_s, y_val)
        m = eval_metrics(y_val["total_points"], pred_val, played_mask)
        m.update({
            "ablation": name,
            "in_dim": in_dim,
            "n_dropped": n_dropped,
            "epochs": epochs_ran,
        })
        rows.append(m)
        if name == "__full__":
            baseline_metrics = m
        else:
            d_rho = m["spearman_played"] - baseline_metrics["spearman_played"]
            d_mae = m["mae_played"] - baseline_metrics["mae_played"]
            print(f"  ρ_played={m['spearman_played']:.4f} (Δ {d_rho:+.4f}), "
                  f"MAE_played={m['mae_played']:.4f} (Δ {d_mae:+.4f}), "
                  f"ρ_all={m['spearman_all']:.4f}")
        if name == "__full__":
            print(f"  ρ_played={m['spearman_played']:.4f}, "
                  f"MAE_played={m['mae_played']:.4f}, "
                  f"ρ_all={m['spearman_all']:.4f}  (baseline)")
        print()

    out_df = pd.DataFrame(rows)
    # Add deltas vs baseline
    base = out_df[out_df["ablation"] == "__full__"].iloc[0]
    for metric in ["mae_played", "rmse_played", "spearman_played", "spearman_all", "mae_all"]:
        out_df[f"delta_{metric}"] = out_df[metric] - base[metric]
    out_df = out_df.sort_values("spearman_played", ascending=False).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH}")
    print()
    print("Ablation results sorted by ρ_played (higher = features dropped didn't hurt much):")
    print(out_df[["ablation", "in_dim", "spearman_played", "delta_spearman_played",
                  "mae_played", "delta_mae_played", "spearman_all"]].to_string(index=False))


if __name__ == "__main__":
    main()
