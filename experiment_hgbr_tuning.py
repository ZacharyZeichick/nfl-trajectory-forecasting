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
            "acceleration_change_last_3", "orientation_change_last_3"]
CAT_COLS = ["player_role", "player_side", "player_position"]
TARGETS  = ["residual_x", "residual_y"]

CONFIGS = [
    ("current (default)",                  {}),
    ("max_iter=200",                        {"max_iter": 200}),
    ("max_iter=300",                        {"max_iter": 300}),
    ("lr=0.05, max_iter=300",               {"learning_rate": 0.05, "max_iter": 300}),
    ("max_leaf_nodes=31",                   {"max_leaf_nodes": 31}),
    ("max_leaf_nodes=63",                   {"max_leaf_nodes": 63}),
    ("l2_regularization=0.01",             {"l2_regularization": 0.01}),
]


def kaggle_rmse(dx, dy):
    return np.sqrt(np.concatenate([np.asarray(dx)**2, np.asarray(dy)**2]).mean())


def build_pipe(hgbr_kwargs):
    pre = ColumnTransformer([
        ("num", "passthrough", NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ])
    return Pipeline([
        ("pre", pre),
        ("model", MultiOutputRegressor(
            HistGradientBoostingRegressor(random_state=42, **hgbr_kwargs)
        )),
    ])


print("Loading data...")
df    = pd.read_csv(OUTPUT_DIR / "ml_dataset_w01_w18.csv")
train = df[df["week"] <= 12].reset_index(drop=True)
val   = df[df["week"] >= 13].reset_index(drop=True)
print(f"Train rows: {len(train)}  |  Val rows: {len(val)}\n")

baseline_rmse = kaggle_rmse(val["residual_x"].values, val["residual_y"].values)
print(f"Baseline RMSE (weeks 13-18): {baseline_rmse:.4f}\n")
print(f"{'#':>2}  {'Config':<30}  {'Model RMSE':>10}  {'Improvement':>11}")
print("-" * 60)

results = []

for idx, (name, kwargs) in enumerate(CONFIGS, start=1):
    print(f"[{idx}/{len(CONFIGS)}] Training: {name} ...", flush=True)
    pipe = build_pipe(kwargs)
    pipe.fit(train[NUM_COLS + CAT_COLS], train[TARGETS])

    pred   = pipe.predict(val[NUM_COLS + CAT_COLS])
    res_x  = val["residual_x"].values - pred[:, 0]
    res_y  = val["residual_y"].values - pred[:, 1]
    model_rmse = kaggle_rmse(res_x, res_y)
    improvement = (baseline_rmse - model_rmse) / baseline_rmse * 100

    print(f"{idx:>2}  {name:<30}  {model_rmse:>10.4f}  {improvement:>10.2f}%")
    results.append({
        "config": name,
        "model_rmse": round(model_rmse, 4),
        "baseline_rmse": round(baseline_rmse, 4),
        "improvement_pct": round(improvement, 2),
    })

print("\n--- Summary ---")
best = min(results, key=lambda r: r["model_rmse"])
print(f"Best config : {best['config']}")
print(f"Best RMSE   : {best['model_rmse']:.4f}  ({best['improvement_pct']:.2f}% improvement)")
