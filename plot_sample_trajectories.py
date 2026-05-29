from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT       = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

# Selected trajectories (identified via percentile analysis of week 5 predictions)
# High-correction: 90th-95th percentile correction, strong improvement
HIGH = dict(game_id=2023100500, play_id=3697, nfl_id=54719)
# Typical: ~50th percentile correction, representative of median case
TYPICAL = dict(game_id=2023100804, play_id=2167, nfl_id=43454)

FIELD_GREEN  = "#2d6a35"
FIELD_LINE   = "#ffffff"
ENDZONE_GREEN = "#1f4d26"

YARD_SPACING = 10   # draw a line every 10 yards
FIELD_WIDTH  = 53.3 # yards


def draw_field_section(ax, x_min, x_max, y_min, y_max, pad=4):
    """Draw a zoomed football-field background within the given bounds."""
    xlo = max(0,   x_min - pad)
    xhi = min(120, x_max + pad)
    ylo = max(0,   y_min - pad)
    yhi = min(FIELD_WIDTH, y_max + pad)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_facecolor(FIELD_GREEN)
    ax.set_aspect("equal")

    # End-zone shading
    for ez_x in [0, 110]:
        ax.add_patch(mpatches.Rectangle(
            (ez_x, ylo), 10, yhi - ylo,
            color=ENDZONE_GREEN, zorder=0
        ))

    # Yard lines (every 10 yards)
    for yd in range(0, 121, YARD_SPACING):
        if xlo <= yd <= xhi:
            ax.axvline(yd, color=FIELD_LINE, linewidth=0.8, alpha=0.6, zorder=1)
            label_yd = yd if yd <= 60 else 120 - yd
            ax.text(yd, ylo + 0.6, str(label_yd),
                    color=FIELD_LINE, fontsize=6, ha="center", alpha=0.7, zorder=2)

    # Sidelines
    ax.axhline(0,            color=FIELD_LINE, linewidth=1.2, zorder=1)
    ax.axhline(FIELD_WIDTH,  color=FIELD_LINE, linewidth=1.2, zorder=1)

    # Hash marks (NFL: ~22.91 and ~30.39 yards from bottom)
    for hx in range(int(xlo), int(xhi) + 1):
        for hy in [22.91, 30.39]:
            if ylo <= hy <= yhi:
                ax.plot([hx - 0.3, hx + 0.3], [hy, hy],
                        color=FIELD_LINE, linewidth=0.6, alpha=0.5, zorder=1)

    return xlo, xhi, ylo, yhi


def plot_trajectory(ax, traj, title, subtitle):
    x_coords = np.concatenate([traj["x_true"], traj["baseline_x"], traj["model_x"]])
    y_coords = np.concatenate([traj["y_true"], traj["baseline_y"], traj["model_y"]])
    draw_field_section(ax, x_coords.min(), x_coords.max(),
                           y_coords.min(), y_coords.max(), pad=5)

    x0 = traj["x_true"].iloc[0]
    y0 = traj["y_true"].iloc[0]

    # Start marker
    ax.plot(x0, y0, "o", color="white", markersize=9, zorder=5, label="_nolegend_")
    ax.plot(x0, y0, "o", color="#333333", markersize=5, zorder=6, label="_nolegend_")

    # True path
    ax.plot(traj["x_true"],    traj["y_true"],
            color="white",     linewidth=2.2, zorder=4, label="True path")
    ax.plot(traj["x_true"].iloc[-1], traj["y_true"].iloc[-1],
            "o", color="white", markersize=7, zorder=5)

    # Baseline path
    ax.plot(traj["baseline_x"], traj["baseline_y"],
            color="#e05c3a", linewidth=2.0, linestyle="--", zorder=3,
            label="Ball-aware baseline")
    ax.plot(traj["baseline_x"].iloc[-1], traj["baseline_y"].iloc[-1],
            "s", color="#e05c3a", markersize=7, zorder=4)

    # Model path
    ax.plot(traj["model_x"], traj["model_y"],
            color="#4fc3f7", linewidth=2.2, zorder=4,
            label="Residual ML model")
    ax.plot(traj["model_x"].iloc[-1], traj["model_y"].iloc[-1],
            "^", color="#4fc3f7", markersize=8, zorder=5)

    ax.set_xlabel("Field position (x, yards)", color="white", fontsize=9)
    ax.set_ylabel("Field position (y, yards)", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
    ax.text(0.5, -0.14, subtitle, transform=ax.transAxes,
            ha="center", fontsize=8.5, color="#cccccc", style="italic")

    ax.legend(loc="upper right", fontsize=8, framealpha=0.35,
              labelcolor="white", facecolor="#1a1a1a", edgecolor="white")


def main():
    df = pd.read_csv(OUTPUT_DIR / "residual_model_week5_predictions.csv")

    for label, key, title, subtitle, out_name in [
        (
            "High-correction",
            HIGH,
            "High-Correction Example — Model vs Baseline",
            f"Game {HIGH['game_id']} · Play {HIGH['play_id']} · "
            f"Baseline RMSE 3.48 → Model RMSE 1.84 (improvement: 47%)",
            "sample_trajectory_high_correction.png",
        ),
        (
            "Typical",
            TYPICAL,
            "Typical Example — Model vs Baseline",
            f"Game {TYPICAL['game_id']} · Play {TYPICAL['play_id']} · "
            f"Baseline RMSE 0.95 → Model RMSE 0.51 (improvement: 46%)",
            "sample_trajectory_typical_correction.png",
        ),
    ]:
        traj = df[
            (df["game_id"] == key["game_id"]) &
            (df["play_id"] == key["play_id"]) &
            (df["nfl_id"]  == key["nfl_id"])
        ].sort_values("frame_id").reset_index(drop=True)

        if traj.empty:
            print(f"WARNING: No data found for {label} trajectory.")
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#111111")
        plot_trajectory(ax, traj, title, subtitle)
        plt.tight_layout()
        out_path = OUTPUT_DIR / out_name
        plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"Saved: outputs/{out_name}  ({len(traj)} frames)")


if __name__ == "__main__":
    main()
