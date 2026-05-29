from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

ROOT       = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

NUM_COLS = ["frame_id", "t", "last_x", "last_y", "last_s", "last_a", "last_dir", "last_o",
            "ball_land_x", "ball_land_y", "num_frames_output", "progress",
            "baseline_x", "baseline_y", "distance_to_ball",
            "velocity_pred_x", "velocity_pred_y", "vx", "vy",
            "distance_to_ball_x", "distance_to_ball_y",
            "angle_to_ball", "time_remaining",
            "is_targeted_receiver", "is_defensive_coverage", "is_offense", "is_defense",
            "delta_x_last_1", "delta_y_last_1",
            "delta_x_last_3", "delta_y_last_3",
            "speed_change_last_3", "direction_change_last_3",
            "acceleration_change_last_3", "orientation_change_last_3",
            "dist_to_targeted_receiver", "dx_to_targeted_receiver", "dy_to_targeted_receiver",
            "dist_to_nearest_opponent", "dx_to_nearest_opponent", "dy_to_nearest_opponent",
            "dist_to_nearest_teammate", "dx_to_nearest_teammate", "dy_to_nearest_teammate",
            "absolute_yardline_number"]
CAT_COLS = ["player_role", "player_side", "player_position", "play_direction"]
TARGETS  = ["residual_x", "residual_y"]
TOP_N    = 20

print("Loading data...")
df    = pd.read_csv(OUTPUT_DIR / "ml_dataset_w01_w18.csv")
train = df[df["week"] <= 12]
print(f"Train rows: {len(train)}")

pre = ColumnTransformer([
    ("num", "passthrough", NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
])
pipe = Pipeline([
    ("pre", pre),
    ("model", MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=300, random_state=42), n_jobs=-1
    )),
])

print("Training...")
pipe.fit(train[NUM_COLS + CAT_COLS], train[TARGETS])

# Permutation importance on a validation subsample (fast, model-agnostic)
val = df[df["week"] >= 13].sample(25000, random_state=42)
feat_names = NUM_COLS + CAT_COLS
print(f"Computing permutation importance on {len(val)} val rows...")
perm = permutation_importance(
    pipe,
    val[feat_names],
    val[TARGETS],
    n_repeats=8,
    random_state=42,
    n_jobs=-1,
)

importance = pd.Series(perm.importances_mean, index=feat_names).sort_values(ascending=False)
importance_std = pd.Series(perm.importances_std, index=feat_names)
importance.to_csv(OUTPUT_DIR / "feature_importance.csv")
print("Saved: outputs/feature_importance.csv")

# Plot top N
top = importance.head(TOP_N).iloc[::-1]  # reverse for horizontal bar (highest at top)

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor("#111111")
ax.set_facecolor("#1a1a1a")

colors = ["#4fc3f7" if i >= TOP_N - 5 else "#78909c" for i in range(TOP_N)]
top_std = importance_std[top.index].iloc[::-1]
bars = ax.barh(range(TOP_N), top.values, xerr=top_std.values,
               color=colors, edgecolor="none", height=0.7,
               error_kw={"ecolor": "#888888", "capsize": 3, "linewidth": 0.8})

ax.set_yticks(range(TOP_N))
ax.set_yticklabels(top.index, fontsize=9, color="white")
ax.set_xlabel("Mean Feature Importance (avg of residual_x + residual_y)", color="white", fontsize=9)
ax.set_title(f"Top {TOP_N} Feature Importances — HGBR Residual Model", color="white",
             fontsize=11, fontweight="bold", pad=10)
ax.tick_params(colors="white", labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor("#444444")
ax.xaxis.label.set_color("white")

plt.tight_layout()
out_path = OUTPUT_DIR / "feature_importance.png"
plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
plt.close()
print("Saved: outputs/feature_importance.png")
print(f"\nTop 10 features:")
print(importance.head(10).to_string())
