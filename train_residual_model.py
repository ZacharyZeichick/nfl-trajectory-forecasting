from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT       = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_COLS = ["frame_id", "t", "last_x", "last_y", "last_s", "last_a", "last_dir", "last_o",
            "ball_land_x", "ball_land_y", "num_frames_output", "progress",
            "baseline_x", "baseline_y", "distance_to_ball",
            "velocity_pred_x", "velocity_pred_y", "vx", "vy",
            "distance_to_ball_x", "distance_to_ball_y",
            "angle_to_ball", "time_remaining",
            "is_targeted_receiver", "is_defensive_coverage", "is_offense", "is_defense",
            "delta_x_last_1", "delta_y_last_1",
            "delta_x_last_3", "delta_y_last_3",
            "delta_x_last_5", "delta_y_last_5",
            "speed_change_last_3", "speed_change_last_5", "direction_change_last_3",
            "acceleration_change_last_3", "orientation_change_last_3",
            "first_x", "first_y", "route_dist_traveled", "mean_speed_input", "route_dir",
            "dist_to_targeted_receiver", "dx_to_targeted_receiver", "dy_to_targeted_receiver",
            "dist_to_nearest_opponent", "dx_to_nearest_opponent", "dy_to_nearest_opponent",
            "dist_to_nearest_teammate", "dx_to_nearest_teammate", "dy_to_nearest_teammate",
            "absolute_yardline_number"]
CAT_COLS = ["player_role", "player_side", "player_position", "play_direction"]
TARGETS  = ["residual_x", "residual_y"]

def kaggle_rmse(dx, dy):
    return np.sqrt(np.concatenate([np.asarray(dx)**2, np.asarray(dy)**2]).mean())

df = pd.read_csv(OUTPUT_DIR / "ml_dataset_w01_w18.csv")

train = df[df["week"] <= 12]
val   = df[df["week"] >= 13]

print(f"Train rows: {len(train)}")
print(f"Val rows  : {len(val)}")

pre = ColumnTransformer([
    ("num", "passthrough", NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
])
pipe = Pipeline([
    ("pre", pre),
    ("model", MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=1000, random_state=42), n_jobs=-1))
])

pipe.fit(train[NUM_COLS + CAT_COLS], train[TARGETS])

pred = pipe.predict(val[NUM_COLS + CAT_COLS])
model_x = val["baseline_x"].values + pred[:, 0]
model_y = val["baseline_y"].values + pred[:, 1]

overall_base  = kaggle_rmse(val["residual_x"].values, val["residual_y"].values)
overall_model = kaggle_rmse(val["residual_x"].values - pred[:, 0],
                             val["residual_y"].values - pred[:, 1])
overall_improv = (overall_base - overall_model) / overall_base * 100

print(f"\nBaseline RMSE (weeks 13-18) : {overall_base:.4f}")
print(f"Model RMSE    (weeks 13-18) : {overall_model:.4f}")
print(f"Improvement                 : {overall_improv:.2f}%")

print(f"\n{'week':>4}  {'rows':>6}  {'baseline':>8}  {'model':>7}  {'improv%':>7}")
print("-" * 42)

rows_out = []
val_pred_x = pred[:, 0]
val_pred_y = pred[:, 1]
val_reset  = val.reset_index(drop=True)

for week in range(13, 19):
    mask   = val_reset["week"] == week
    base_r  = kaggle_rmse(val_reset.loc[mask, "residual_x"].values,
                           val_reset.loc[mask, "residual_y"].values)
    model_r = kaggle_rmse(val_reset.loc[mask, "residual_x"].values - val_pred_x[mask],
                           val_reset.loc[mask, "residual_y"].values - val_pred_y[mask])
    improv  = (base_r - model_r) / base_r * 100
    print(f"{week:>4}  {mask.sum():>6}  {base_r:>8.4f}  {model_r:>7.4f}  {improv:>7.2f}%")
    rows_out.append({"val_week": week, "rows": int(mask.sum()),
                     "baseline_rmse": round(base_r, 4), "model_rmse": round(model_r, 4),
                     "improvement_pct": round(improv, 2)})

pd.DataFrame(rows_out).to_csv(OUTPUT_DIR / "residual_model_w13_w18_validation.csv", index=False)
print("\nSaved: outputs/residual_model_w13_w18_validation.csv")

# Per-role breakdown
print(f"\n{'role':<25}  {'rows':>6}  {'baseline':>8}  {'model':>7}  {'improv%':>7}")
print("-" * 58)
role_rows = []
for role in sorted(val_reset["player_role"].unique()):
    mask = val_reset["player_role"] == role
    base_r  = kaggle_rmse(val_reset.loc[mask, "residual_x"].values,
                           val_reset.loc[mask, "residual_y"].values)
    model_r = kaggle_rmse(val_reset.loc[mask, "residual_x"].values - val_pred_x[mask],
                           val_reset.loc[mask, "residual_y"].values - val_pred_y[mask])
    improv  = (base_r - model_r) / base_r * 100
    print(f"{role:<25}  {mask.sum():>6}  {base_r:>8.4f}  {model_r:>7.4f}  {improv:>7.2f}%")
    role_rows.append({"player_role": role, "rows": int(mask.sum()),
                      "baseline_rmse": round(base_r, 4), "model_rmse": round(model_r, 4),
                      "improvement_pct": round(improv, 2)})

pd.DataFrame(role_rows).to_csv(OUTPUT_DIR / "residual_model_by_role_validation.csv", index=False)
print("\nSaved: outputs/residual_model_by_role_validation.csv")
