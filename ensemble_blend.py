"""
Ensemble: blend GRU sequence model with LightGBM residual predictions.
Loads trained GRU from outputs/sequence_model_best.pt, runs val inference,
sweeps blend weights, and reports RMSE at each alpha.
"""
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

ROOT      = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "train"
OUTPUT_DIR = ROOT / "outputs"

HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.1
BATCH_SIZE  = 128
MAX_SEQ_LEN = 50
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

GROUP_KEYS = ["game_id", "play_id", "nfl_id"]


def kaggle_rmse(dx, dy):
    return float(np.sqrt(np.concatenate([np.asarray(dx)**2, np.asarray(dy)**2]).mean()))


# ── 1. Load trajectory sequences (val weeks only) ─────────────────────────────
print("Loading trajectory sequences (weeks 13-18)...")
traj_seqs = {}

for week in range(13, 19):
    w   = str(week).zfill(2)
    inp = pd.read_csv(TRAIN_DIR / f"input_2023_w{w}.csv")
    inp = inp.sort_values(GROUP_KEYS + ["frame_id"])
    left = inp["play_direction"] == "left"
    inp  = inp.copy()
    inp.loc[left, "x"]   = 120 - inp.loc[left, "x"]
    inp.loc[left, "dir"] = (360 - inp.loc[left, "dir"]) % 360
    inp.loc[left, "o"]   = (360 - inp.loc[left, "o"])   % 360

    for key, grp in inp.groupby(GROUP_KEYS, sort=False):
        xs    = grp["x"].values.astype(np.float32)
        ys    = grp["y"].values.astype(np.float32)
        s     = grp["s"].values.astype(np.float32)
        a     = grp["a"].values.astype(np.float32)
        dir_r = np.deg2rad(grp["dir"].values).astype(np.float32)
        o_r   = np.deg2rad(grp["o"].values).astype(np.float32)
        seq   = np.stack([xs - xs[0], ys - ys[0], s, a,
                          np.sin(dir_r), np.cos(dir_r),
                          np.sin(o_r),   np.cos(o_r)], axis=1)
        traj_seqs[key] = seq
    print(f"  Loaded week {week}")

print(f"  Total trajectories: {len(traj_seqs)}")


# ── 2. Load ML dataset and compute normalization stats ────────────────────────
print("Loading ML dataset...")
ml = pd.read_csv(OUTPUT_DIR / "ml_dataset_w01_w18.csv")

FRAME_FEAT_COLS = ["progress", "t", "vx", "vy",
                   "distance_to_ball_x", "distance_to_ball_y"]

train_ml = ml[ml["week"] <= 12]
val_ml   = ml[ml["week"] >= 13].reset_index(drop=True)

frame_mean = train_ml[FRAME_FEAT_COLS].mean().values.astype(np.float32)
frame_std  = train_ml[FRAME_FEAT_COLS].std().values.astype(np.float32)
frame_std[frame_std < 1e-6] = 1.0

seq_feat_stats = {"rel_xy_std": 8.0, "s_std": 3.5, "a_std": 2.5}


def normalize_seq(seq):
    out = seq.copy()
    out[:, :2] /= seq_feat_stats["rel_xy_std"]
    out[:, 2]  /= seq_feat_stats["s_std"]
    out[:, 3]  /= seq_feat_stats["a_std"]
    return out


def normalize_frame(arr):
    return (arr - frame_mean) / frame_std


# ── 3. Val dataset with frame-level ID tracking ───────────────────────────────
class ValDataset(Dataset):
    def __init__(self, ml_df):
        self.samples   = []
        self.frame_ids = []   # list of (N_frames, 4) arrays: game_id, play_id, nfl_id, frame_id

        for key, grp in ml_df.groupby(GROUP_KEYS, sort=False):
            seq = traj_seqs.get(key)
            if seq is None:
                continue
            T      = min(len(seq), MAX_SEQ_LEN)
            padded = np.zeros((MAX_SEQ_LEN, 8), dtype=np.float32)
            padded[:T] = normalize_seq(seq[:T])

            frame_arr  = grp[FRAME_FEAT_COLS].values.astype(np.float32)
            frame_norm = normalize_frame(frame_arr)
            targets    = grp[["residual_x", "residual_y"]].values.astype(np.float32)

            self.samples.append({
                "seq":     padded,
                "seq_len": T,
                "frame":   frame_norm,
                "targets": targets,
            })
            self.frame_ids.append(
                grp[["game_id", "play_id", "nfl_id", "frame_id"]].values
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    seqs     = torch.FloatTensor(np.stack([b["seq"]    for b in batch]))
    seq_lens = [b["seq_len"] for b in batch]
    frames   = [torch.FloatTensor(b["frame"])   for b in batch]
    targets  = [torch.FloatTensor(b["targets"]) for b in batch]
    counts   = [len(b["targets"]) for b in batch]
    return seqs, seq_lens, torch.cat(frames), torch.cat(targets), counts


# ── 4. Model definition (must match train_sequence_model.py) ──────────────────
class SequenceResidualModel(nn.Module):
    def __init__(self, seq_input=8, frame_input=6,
                 hidden=HIDDEN_SIZE, layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.gru = nn.GRU(seq_input, hidden, layers,
                          batch_first=True, dropout=dropout if layers > 1 else 0.0)
        combined = hidden + frame_input
        self.head = nn.Sequential(
            nn.Linear(combined, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, seqs, seq_lens, frame_feats, counts):
        packed = nn.utils.rnn.pack_padded_sequence(
            seqs, seq_lens, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        emb = h_n[-1]
        emb_expanded = torch.cat(
            [emb[i].unsqueeze(0).expand(c, -1) for i, c in enumerate(counts)], dim=0
        )
        return self.head(torch.cat([emb_expanded, frame_feats], dim=1))


# ── 5. Load model ─────────────────────────────────────────────────────────────
model_path = OUTPUT_DIR / "sequence_model_best.pt"
if not model_path.exists():
    raise FileNotFoundError(f"No trained model at {model_path}. Run train_sequence_model.py first.")

print(f"\nLoading GRU model from {model_path}...")
model = SequenceResidualModel()
model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
model.eval()


# ── 6. Run inference ──────────────────────────────────────────────────────────
print("Building val dataset...")
val_ds = ValDataset(val_ml)
print(f"  Val trajectories: {len(val_ds)}")

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

print("Running GRU inference...")
all_preds = []
with torch.no_grad():
    for seqs, seq_lens, frames, targets, counts in val_loader:
        preds = model(seqs, seq_lens, frames, counts)
        all_preds.append(preds.numpy())

all_preds = np.concatenate(all_preds, axis=0)   # (N_rows, 2)

# Build DataFrame with join keys
id_rows = np.concatenate(val_ds.frame_ids, axis=0)   # (N_rows, 4)
gru_df  = pd.DataFrame(id_rows, columns=["game_id", "play_id", "nfl_id", "frame_id"])
gru_df["gru_pred_rx"] = all_preds[:, 0]
gru_df["gru_pred_ry"] = all_preds[:, 1]


# ── 7. Merge with LightGBM val predictions ───────────────────────────────────
print("Merging with LightGBM predictions...")
lgbm_val = pd.read_csv(OUTPUT_DIR / "val_predictions_w13_w18.csv")
merged   = lgbm_val.merge(gru_df, on=["game_id", "play_id", "nfl_id", "frame_id"], how="inner")

n_lgbm = len(lgbm_val)
n_merged = len(merged)
print(f"  LGBM rows: {n_lgbm:,}  |  Matched: {n_merged:,}  |  Unmatched: {n_lgbm - n_merged:,}")

true_rx = merged["residual_x"].values
true_ry = merged["residual_y"].values
lgbm_rx = merged["pred_residual_x"].values
lgbm_ry = merged["pred_residual_y"].values
gru_rx  = merged["gru_pred_rx"].values
gru_ry  = merged["gru_pred_ry"].values


# ── 8. Sweep blend weights ────────────────────────────────────────────────────
print("\n── Blend sweep (alpha = weight on GRU, 1-alpha on LightGBM) ──")
print(f"{'Alpha':>6}  {'RMSE':>8}")
print("-" * 18)

results = []
for alpha in np.arange(0.0, 1.01, 0.05):
    blend_rx   = alpha * gru_rx  + (1 - alpha) * lgbm_rx
    blend_ry   = alpha * gru_ry  + (1 - alpha) * lgbm_ry
    rmse       = kaggle_rmse(true_rx - blend_rx, true_ry - blend_ry)
    results.append((alpha, rmse))
    print(f"{alpha:>6.2f}  {rmse:>8.4f}")

best_alpha, best_rmse = min(results, key=lambda x: x[1])
lgbm_only_rmse = kaggle_rmse(true_rx - lgbm_rx, true_ry - lgbm_ry)
gru_only_rmse  = kaggle_rmse(true_rx - gru_rx,  true_ry - gru_ry)

print(f"\n── Summary ──────────────────────────────────────────")
print(f"LightGBM only RMSE  : {lgbm_only_rmse:.4f}")
print(f"GRU only RMSE       : {gru_only_rmse:.4f}")
print(f"Best ensemble RMSE  : {best_rmse:.4f}  (alpha={best_alpha:.2f})")
delta = best_rmse - lgbm_only_rmse
print(f"Ensemble vs LightGBM: {delta:+.4f}  ({'better' if delta < 0 else 'worse'})")

# Save blended predictions at best alpha
merged["ensemble_pred_rx"] = best_alpha * gru_rx + (1 - best_alpha) * lgbm_rx
merged["ensemble_pred_ry"] = best_alpha * gru_ry + (1 - best_alpha) * lgbm_ry
out_path = OUTPUT_DIR / "ensemble_val_predictions.csv"
merged.to_csv(out_path, index=False)
print(f"\nSaved ensemble predictions → {out_path}")
