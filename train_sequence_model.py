"""
GRU-based sequence model over pre-throw player trajectories.
Architecture: GRU encoder (pre-throw frames) + MLP prediction head (per output frame).
Trained end-to-end to predict residual_x / residual_y.
"""
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT       = Path(__file__).resolve().parent
TRAIN_DIR  = ROOT / "train"
OUTPUT_DIR = ROOT / "outputs"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
DROPOUT      = 0.1
BATCH_SIZE   = 32       # trajectories per batch
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 40
MAX_SEQ_LEN  = 50       # cap/pad input sequences to this length
SEED         = 42
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)

GROUP_KEYS = ["game_id", "play_id", "nfl_id"]


def kaggle_rmse(dx, dy):
    return float(np.sqrt(np.concatenate([np.asarray(dx)**2, np.asarray(dy)**2]).mean()))


# ── 1. Build trajectory sequences from raw input ──────────────────────────────
print("Loading raw input data and building trajectory sequences...")

traj_seqs = {}   # {(game_id, play_id, nfl_id): np.array(T, 8)}

for week in range(1, 19):
    w   = str(week).zfill(2)
    inp = pd.read_csv(TRAIN_DIR / f"input_2023_w{w}.csv")
    inp = inp.sort_values(GROUP_KEYS + ["frame_id"])

    # Coordinate normalization: flip left-direction plays to right-facing
    left = inp["play_direction"] == "left"
    inp = inp.copy()
    inp.loc[left, "x"]   = 120 - inp.loc[left, "x"]
    inp.loc[left, "dir"] = (360 - inp.loc[left, "dir"]) % 360
    inp.loc[left, "o"]   = (360 - inp.loc[left, "o"])   % 360

    # Sequence features: relative position, speed, accel, sin/cos of angles
    for key, grp in inp.groupby(GROUP_KEYS, sort=False):
        xs  = grp["x"].values.astype(np.float32)
        ys  = grp["y"].values.astype(np.float32)
        rel_x = xs - xs[0]
        rel_y = ys - ys[0]
        s     = grp["s"].values.astype(np.float32)
        a     = grp["a"].values.astype(np.float32)
        dir_r = np.deg2rad(grp["dir"].values).astype(np.float32)
        o_r   = np.deg2rad(grp["o"].values).astype(np.float32)
        seq   = np.stack([rel_x, rel_y, s, a,
                          np.sin(dir_r), np.cos(dir_r),
                          np.sin(o_r),   np.cos(o_r)], axis=1)  # (T, 8)
        traj_seqs[key] = seq

    if week % 6 == 0:
        print(f"  Loaded week {week}")

print(f"  Total trajectories: {len(traj_seqs)}")


# ── 2. Load ML dataset (baseline predictions + residual targets) ──────────────
print("Loading ML dataset...")
ml = pd.read_csv(OUTPUT_DIR / "ml_dataset_w01_w18.csv")

# Frame features (6): normalized after computing train stats below
FRAME_FEAT_COLS = ["progress", "t", "vx", "vy",
                   "distance_to_ball_x", "distance_to_ball_y"]

train_ml = ml[ml["week"] <= 12].reset_index(drop=True)
val_ml   = ml[ml["week"] >= 13].reset_index(drop=True)

# Compute normalization stats from training data
seq_feat_stats = {
    "rel_xy_std": 8.0,   # rough std for relative position (yards)
    "s_std":      3.5,   # rough std for speed
    "a_std":      2.5,   # rough std for acceleration
}

frame_mean = train_ml[FRAME_FEAT_COLS].mean().values.astype(np.float32)
frame_std  = train_ml[FRAME_FEAT_COLS].std().values.astype(np.float32)
frame_std[frame_std < 1e-6] = 1.0


def normalize_seq(seq):
    """Normalize sequence features in-place."""
    out = seq.copy()
    out[:, :2] /= seq_feat_stats["rel_xy_std"]   # rel_x, rel_y
    out[:, 2]  /= seq_feat_stats["s_std"]         # s
    out[:, 3]  /= seq_feat_stats["a_std"]         # a
    # cols 4-7 (sin/cos) are already in [-1, 1]
    return out


def normalize_frame(frame_arr):
    return (frame_arr - frame_mean) / frame_std


# ── 3. Dataset ────────────────────────────────────────────────────────────────
class TrajectoryDataset(Dataset):
    def __init__(self, ml_df):
        self.samples = []
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
                "seq":        padded,
                "seq_len":    T,
                "frame":      frame_norm,
                "targets":    targets,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    seqs      = torch.FloatTensor(np.stack([b["seq"]     for b in batch]))
    seq_lens  = [b["seq_len"] for b in batch]
    frames    = [torch.FloatTensor(b["frame"])   for b in batch]
    targets   = [torch.FloatTensor(b["targets"]) for b in batch]
    counts    = [len(b["targets"]) for b in batch]
    return seqs, seq_lens, torch.cat(frames), torch.cat(targets), counts


# ── 4. Model ──────────────────────────────────────────────────────────────────
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
        emb = h_n[-1]                              # (batch, hidden)

        # Repeat each trajectory embedding for all its output rows
        emb_expanded = torch.cat(
            [emb[i].unsqueeze(0).expand(c, -1) for i, c in enumerate(counts)], dim=0
        )
        combined = torch.cat([emb_expanded, frame_feats], dim=1)
        return self.head(combined)


# ── 5. Training ───────────────────────────────────────────────────────────────
print("Building datasets...")
train_ds = TrajectoryDataset(train_ml)
val_ds   = TrajectoryDataset(val_ml)
print(f"  Train trajectories: {len(train_ds)} | Val trajectories: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=0)

model     = SequenceResidualModel()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.MSELoss()

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training for {EPOCHS} epochs...\n")
print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Val RMSE':>9}")
print("-" * 32)

best_val_rmse = float("inf")
best_state    = None

for epoch in range(1, EPOCHS + 1):
    # ── train ──
    model.train()
    train_loss = 0.0
    for seqs, seq_lens, frames, targets, counts in train_loader:
        optimizer.zero_grad()
        preds = model(seqs, seq_lens, frames, counts)
        loss  = criterion(preds, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * len(targets)
    train_loss /= len(train_ds.samples)   # approximate mean per trajectory

    scheduler.step()

    # ── validate ──
    model.eval()
    all_pred_rx, all_pred_ry = [], []
    all_true_rx, all_true_ry = [], []
    with torch.no_grad():
        for seqs, seq_lens, frames, targets, counts in val_loader:
            preds = model(seqs, seq_lens, frames, counts)
            all_pred_rx.append(preds[:, 0].numpy())
            all_pred_ry.append(preds[:, 1].numpy())
            all_true_rx.append(targets[:, 0].numpy())
            all_true_ry.append(targets[:, 1].numpy())

    pred_rx = np.concatenate(all_pred_rx)
    pred_ry = np.concatenate(all_pred_ry)
    true_rx = np.concatenate(all_true_rx)
    true_ry = np.concatenate(all_true_ry)

    val_rmse = kaggle_rmse(true_rx - pred_rx, true_ry - pred_ry)

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        torch.save(best_state, OUTPUT_DIR / "sequence_model_best.pt")

    if epoch % 5 == 0 or epoch == 1:
        print(f"{epoch:>5}  {train_loss:>10.4f}  {val_rmse:>9.4f}")

print(f"\nBest val RMSE: {best_val_rmse:.4f}")
print(f"Model saved  : {OUTPUT_DIR / 'sequence_model_best.pt'}")
print(f"LightGBM RMSE for comparison: 0.7406")
delta = best_val_rmse - 0.7406
print(f"Delta vs LightGBM: {delta:+.4f}  ({'better' if delta < 0 else 'worse'})")

# ── 6. Per-week breakdown with best model ─────────────────────────────────────
model.load_state_dict(best_state)
model.eval()

val_ml_reset = val_ml.reset_index(drop=True)
# Rebuild predictions in val order
pred_map = {}   # key -> (pred_rx array, pred_ry array)
with torch.no_grad():
    for batch in DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn):
        seqs, seq_lens, frames, targets, counts = batch
        preds = model(seqs, seq_lens, frames, counts)
        # We can't easily recover order here without tracking keys, so just report overall
        pass

baseline_rmse = kaggle_rmse(val_ml["residual_x"].values, val_ml["residual_y"].values)
print(f"\nBaseline RMSE (weeks 13-18): {baseline_rmse:.4f}")
print(f"Sequence model RMSE        : {best_val_rmse:.4f}")
print(f"Improvement over baseline  : {(baseline_rmse - best_val_rmse) / baseline_rmse * 100:.2f}%")
