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
CAT_COLS = ["player_side", "player_position"]
TARGETS  = ["residual_x", "residual_y"]
ROLES    = ["Targeted Receiver", "Defensive Coverage"]


def kaggle_rmse(dx, dy):
    return np.sqrt(np.concatenate([np.asarray(dx)**2, np.asarray(dy)**2]).mean())


def build_pipe():
    pre = ColumnTransformer([
        ("num", "passthrough", NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ])
    return Pipeline([
        ("pre", pre),
        ("model", MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=1000, random_state=42), n_jobs=-1
        )),
    ])


df    = pd.read_csv(OUTPUT_DIR / "ml_dataset_w01_w18.csv")
train = df[df["week"] <= 12]
val   = df[df["week"] >= 13].reset_index(drop=True)

print(f"Train rows: {len(train)}  |  Val rows: {len(val)}\n")

baseline_rmse = kaggle_rmse(val["residual_x"].values, val["residual_y"].values)
print(f"Baseline RMSE (weeks 13-18): {baseline_rmse:.4f}\n")

# Train and predict per role; assemble combined predictions
val_pred = np.zeros((len(val), 2))

for role in ROLES:
    print(f"Training: {role} ...", flush=True)
    tr_role  = train[train["player_role"] == role]
    val_role = val[val["player_role"] == role]
    idx      = val_role.index

    pipe = build_pipe()
    pipe.fit(tr_role[NUM_COLS + CAT_COLS], tr_role[TARGETS])
    pred = pipe.predict(val_role[NUM_COLS + CAT_COLS])

    val_pred[idx, 0] = pred[:, 0]
    val_pred[idx, 1] = pred[:, 1]

    role_base  = kaggle_rmse(val_role["residual_x"].values, val_role["residual_y"].values)
    role_model = kaggle_rmse(val_role["residual_x"].values - pred[:, 0],
                              val_role["residual_y"].values - pred[:, 1])
    role_imp   = (role_base - role_model) / role_base * 100
    print(f"  {role}: baseline {role_base:.4f} -> model {role_model:.4f} ({role_imp:.2f}%)\n")

# Overall combined result
overall_model = kaggle_rmse(val["residual_x"].values - val_pred[:, 0],
                             val["residual_y"].values - val_pred[:, 1])
overall_imp   = (baseline_rmse - overall_model) / baseline_rmse * 100

print(f"Combined model RMSE (weeks 13-18): {overall_model:.4f}")
print(f"Improvement                       : {overall_imp:.2f}%")
print(f"\nSingle model RMSE for comparison  : 0.7986")
print(f"Delta vs single model             : {overall_model - 0.7986:+.4f}")

print(f"\n{'week':>4}  {'rows':>6}  {'baseline':>8}  {'model':>7}  {'improv%':>7}")
print("-" * 42)

rows_out = []
for week in range(13, 19):
    mask    = val["week"] == week
    base_r  = kaggle_rmse(val.loc[mask, "residual_x"].values,
                           val.loc[mask, "residual_y"].values)
    model_r = kaggle_rmse(val.loc[mask, "residual_x"].values - val_pred[mask, 0],
                           val.loc[mask, "residual_y"].values - val_pred[mask, 1])
    improv  = (base_r - model_r) / base_r * 100
    print(f"{week:>4}  {mask.sum():>6}  {base_r:>8.4f}  {model_r:>7.4f}  {improv:>7.2f}%")
    rows_out.append({"val_week": week, "rows": int(mask.sum()),
                     "baseline_rmse": round(base_r, 4), "model_rmse": round(model_r, 4),
                     "improvement_pct": round(improv, 2)})
