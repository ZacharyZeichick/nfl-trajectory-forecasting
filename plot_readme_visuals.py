"""
Generates four supplementary charts for the README:
  1. Model development progression
  2. Per-role RMSE comparison
  3. RMSE by prediction frame (error horizon)
  4. Per-trajectory scatter: baseline error vs model error
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT       = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

BG      = "#111111"
AX_BG   = "#1a1a1a"
WHITE   = "#ffffff"
GRAY    = "#888888"
BLUE    = "#4fc3f7"
ORANGE  = "#e05c3a"
GREEN   = "#66bb6a"
PURPLE  = "#ce93d8"


def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(AX_BG)
    ax.set_title(title,  color=WHITE, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=WHITE, fontsize=9)
    ax.set_ylabel(ylabel, color=WHITE, fontsize=9)
    ax.tick_params(colors=WHITE, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.yaxis.label.set_color(WHITE)
    ax.xaxis.label.set_color(WHITE)


# ── 1. Model development progression ─────────────────────────────────────────
stages = [
    ("Ball-aware\nbaseline",          1.3438),
    ("HGBR default\n(max_iter=100)",  0.9108),
    ("+ Motion\nfeatures",            0.8447),
    ("+ Interaction\nfeatures",       0.8325),
    ("+ Route &\nfield features",     0.8314),
    ("max_iter\n=1000",               0.7986),
    ("+ Coordinate\nnormalization",   0.7625),
]
labels = [s[0] for s in stages]
values = [s[1] for s in stages]
colors = [ORANGE] + [BLUE] * 5 + [GREEN]

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG)
style_ax(ax, "Model Development: RMSE at Each Stage (Weeks 13–18 Holdout)",
         "", "Kaggle RMSE")

bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)
ax.set_ylim(0, 1.55)
ax.axhline(1.3438, color=ORANGE, linewidth=1, linestyle="--", alpha=0.4, zorder=2)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.025,
            f"{val:.4f}", ha="center", va="bottom", color=WHITE, fontsize=8, fontweight="bold")

ax.yaxis.grid(True, color="#333333", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

legend_handles = [
    mlines.Line2D([], [], color=ORANGE, marker="s", linestyle="None", markersize=8,
                  label="Ball-aware baseline"),
    mlines.Line2D([], [], color=BLUE, marker="s", linestyle="None", markersize=8,
                  label="HGBR model"),
    mlines.Line2D([], [], color=GREEN, marker="s", linestyle="None", markersize=8,
                  label="Final model"),
]
ax.legend(handles=legend_handles, fontsize=8, framealpha=0.3, labelcolor=WHITE,
          facecolor="#1a1a1a", edgecolor=WHITE)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "development_progression.png", dpi=150, facecolor=BG)
plt.close()
print("Saved: outputs/development_progression.png")


# ── 2. Per-role RMSE comparison ───────────────────────────────────────────────
role_df = pd.read_csv(OUTPUT_DIR / "residual_model_by_role_validation.csv")
roles       = role_df["player_role"].tolist()
base_vals   = role_df["baseline_rmse"].tolist()
model_vals  = role_df["model_rmse"].tolist()

x     = np.arange(len(roles))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor(BG)
style_ax(ax, "RMSE by Player Role: Baseline vs Model (Weeks 13–18)",
         "", "Kaggle RMSE")

b1 = ax.bar(x - width/2, base_vals,  width, label="Ball-aware baseline", color=ORANGE, zorder=3)
b2 = ax.bar(x + width/2, model_vals, width, label="Residual ML model",   color=BLUE,   zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(roles, color=WHITE, fontsize=9)
ax.set_ylim(0, max(base_vals) * 1.25)
ax.yaxis.grid(True, color="#333333", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{bar.get_height():.3f}", ha="center", fontsize=8, color=WHITE, fontweight="bold")

# Improvement annotations
for i, (b, m) in enumerate(zip(base_vals, model_vals)):
    pct = (b - m) / b * 100
    ax.text(x[i], max(b, m) + 0.12, f"-{pct:.1f}%", ha="center",
            fontsize=8.5, color=GREEN, fontweight="bold")

ax.legend(fontsize=8, framealpha=0.3, labelcolor=WHITE, facecolor="#1a1a1a", edgecolor=WHITE)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "role_comparison.png", dpi=150, facecolor=BG)
plt.close()
print("Saved: outputs/role_comparison.png")


# ── 3. RMSE by prediction frame ──────────────────────────────────────────────
preds = pd.read_csv(OUTPUT_DIR / "val_predictions_w13_w18.csv")

frame_rows = []
for fid, grp in preds.groupby("frame_id"):
    if len(grp) < 200:   # skip very sparse long-play frames
        continue
    base_r  = np.sqrt((np.concatenate([grp["residual_x"].values**2,
                                        grp["residual_y"].values**2])).mean())
    model_r = np.sqrt((np.concatenate([(grp["residual_x"] - grp["pred_residual_x"])**2,
                                        (grp["residual_y"] - grp["pred_residual_y"])**2])).mean())
    frame_rows.append({"frame_id": fid, "baseline_rmse": base_r, "model_rmse": model_r,
                       "n": len(grp)})

fdf = pd.DataFrame(frame_rows)

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BG)
style_ax(ax, "RMSE by Prediction Frame (How Far Into the Future)",
         "Frames after throw (1 frame = 0.1 s)", "Kaggle RMSE")

ax.plot(fdf["frame_id"], fdf["baseline_rmse"], color=ORANGE, linewidth=2,
        marker="o", markersize=4, label="Ball-aware baseline")
ax.plot(fdf["frame_id"], fdf["model_rmse"],    color=BLUE,   linewidth=2,
        marker="o", markersize=4, label="Residual ML model")

ax.fill_between(fdf["frame_id"], fdf["model_rmse"], fdf["baseline_rmse"],
                alpha=0.15, color=GREEN, label="Improvement region")

ax.yaxis.grid(True, color="#333333", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=8, framealpha=0.3, labelcolor=WHITE, facecolor="#1a1a1a", edgecolor=WHITE)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "error_by_frame.png", dpi=150, facecolor=BG)
plt.close()
print("Saved: outputs/error_by_frame.png")


# ── 4. Per-trajectory scatter: baseline error vs model error ─────────────────
def traj_rmse(grp):
    b = np.sqrt(np.mean(np.concatenate([grp["residual_x"].values**2,
                                         grp["residual_y"].values**2])))
    m = np.sqrt(np.mean(np.concatenate([(grp["residual_x"] - grp["pred_residual_x"])**2,
                                         (grp["residual_y"] - grp["pred_residual_y"])**2])))
    return pd.Series({"baseline_err": b, "model_err": m,
                      "player_role": grp["player_role"].iloc[0]})

traj = preds.groupby(["game_id", "play_id", "nfl_id"]).apply(
    traj_rmse, include_groups=False
).reset_index()

fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor(BG)
style_ax(ax, "Per-Trajectory: Baseline Error vs Model Error",
         "Baseline RMSE (per trajectory)", "Model RMSE (per trajectory)")

role_colors = {"Targeted Receiver": BLUE, "Defensive Coverage": PURPLE}
for role, grp in traj.groupby("player_role"):
    ax.scatter(grp["baseline_err"], grp["model_err"],
               color=role_colors.get(role, GRAY),
               alpha=0.25, s=12, label=role, zorder=3)

# Diagonal: model = baseline
lim = max(traj["baseline_err"].quantile(0.99), traj["model_err"].quantile(0.99)) * 1.05
ax.plot([0, lim], [0, lim], color=WHITE, linewidth=1.2, linestyle="--",
        alpha=0.5, label="No improvement (y=x)", zorder=4)
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)

pct_better = (traj["model_err"] < traj["baseline_err"]).mean() * 100
ax.text(0.97, 0.05,
        f"{pct_better:.1f}% of trajectories\nimproved by model",
        transform=ax.transAxes, ha="right", va="bottom",
        color=GREEN, fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a1a", edgecolor=GREEN, alpha=0.8))

ax.yaxis.grid(True, color="#333333", linewidth=0.5, zorder=0)
ax.xaxis.grid(True, color="#333333", linewidth=0.5, zorder=0)
ax.set_axisbelow(True)
ax.set_aspect("equal")
ax.legend(fontsize=8, framealpha=0.3, labelcolor=WHITE, facecolor="#1a1a1a", edgecolor=WHITE)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "error_scatter.png", dpi=150, facecolor=BG)
plt.close()
print("Saved: outputs/error_scatter.png")
