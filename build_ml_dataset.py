from pathlib import Path
import pandas as pd
import numpy as np

ROOT       = Path(__file__).resolve().parent
TRAIN_DIR  = ROOT / "train"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

KEEP_LAST = ["x", "y", "s", "a", "dir", "o", "ball_land_x", "ball_land_y", "num_frames_output",
             "player_role", "player_side", "player_position"]

frames = []

for week in range(1, 19):
    w = str(week).zfill(2)
    inp = pd.read_csv(TRAIN_DIR / f"input_2023_w{w}.csv")
    out = pd.read_csv(TRAIN_DIR / f"output_2023_w{w}.csv")

    last_obs = (
        inp.sort_values("frame_id")
        .groupby(["game_id", "play_id", "nfl_id"])[KEEP_LAST]
        .last()
        .reset_index()
        .rename(columns={"x": "last_x", "y": "last_y", "s": "last_s", "a": "last_a",
                         "dir": "last_dir", "o": "last_o"})
    )

    df = out.merge(last_obs, on=["game_id", "play_id", "nfl_id"], how="inner")
    df["week"] = week

    df["t"] = df["frame_id"] / 10.0
    dir_rad = np.deg2rad(df["last_dir"])
    df["vx"] = df["last_s"] * np.sin(dir_rad)
    df["vy"] = df["last_s"] * np.cos(dir_rad)
    df["velocity_pred_x"] = df["last_x"] + df["vx"] * df["t"]
    df["velocity_pred_y"] = df["last_y"] + df["vy"] * df["t"]
    df["progress"] = (df["frame_id"] / df["num_frames_output"]).clip(0, 1)
    df["ball_x"] = df["last_x"] + df["progress"] * (df["ball_land_x"] - df["last_x"])
    df["ball_y"] = df["last_y"] + df["progress"] * (df["ball_land_y"] - df["last_y"])
    df["baseline_x"] = 0.75 * df["velocity_pred_x"] + 0.25 * df["ball_x"]
    df["baseline_y"] = 0.75 * df["velocity_pred_y"] + 0.25 * df["ball_y"]
    df["residual_x"] = df["x"] - df["baseline_x"]
    df["residual_y"] = df["y"] - df["baseline_y"]
    df["distance_to_ball"] = np.sqrt((df["last_x"] - df["ball_land_x"])**2 +
                                     (df["last_y"] - df["ball_land_y"])**2)
    df["distance_to_ball_x"] = df["ball_land_x"] - df["last_x"]
    df["distance_to_ball_y"] = df["ball_land_y"] - df["last_y"]
    df["angle_to_ball"] = np.arctan2(df["distance_to_ball_y"], df["distance_to_ball_x"])
    df["time_remaining"] = df["num_frames_output"] - df["frame_id"]
    df["is_targeted_receiver"]  = (df["player_role"] == "Targeted Receiver").astype(int)
    df["is_defensive_coverage"] = (df["player_role"] == "Defensive Coverage").astype(int)
    df["is_offense"] = (df["player_side"] == "Offense").astype(int)
    df["is_defense"] = (df["player_side"] == "Defense").astype(int)

    frames.append(df)

out_cols = [
    "week", "game_id", "play_id", "nfl_id", "frame_id", "t",
    "last_x", "last_y", "last_s", "last_a", "last_dir", "last_o",
    "ball_land_x", "ball_land_y", "num_frames_output", "progress",
    "baseline_x", "baseline_y", "velocity_pred_x", "velocity_pred_y", "vx", "vy",
    "distance_to_ball", "distance_to_ball_x", "distance_to_ball_y",
    "angle_to_ball", "time_remaining",
    "is_targeted_receiver", "is_defensive_coverage", "is_offense", "is_defense",
    "player_role", "player_side", "player_position",
    "residual_x", "residual_y"
]

result = pd.concat(frames, ignore_index=True)[out_cols]
result.to_csv(OUTPUT_DIR / "ml_dataset_w01_w18.csv", index=False)

print(f"Saved: outputs/ml_dataset_w01_w18.csv")
print(f"Shape: {result.shape}")
print(f"Columns: {list(result.columns)}")
