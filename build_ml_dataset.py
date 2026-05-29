from pathlib import Path
import pandas as pd
import numpy as np

ROOT       = Path(__file__).resolve().parent
TRAIN_DIR  = ROOT / "train"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

KEEP_LAST = ["x", "y", "s", "a", "dir", "o", "ball_land_x", "ball_land_y", "num_frames_output",
             "player_role", "player_side", "player_position",
             "play_direction", "absolute_yardline_number"]

GROUP_KEYS = ["game_id", "play_id", "nfl_id"]

def circ_diff(a, b):
    """Signed circular difference in degrees, result in [-180, 180)."""
    return ((a - b + 180) % 360) - 180

frames = []

for week in range(1, 19):
    w = str(week).zfill(2)
    inp = pd.read_csv(TRAIN_DIR / f"input_2023_w{w}.csv")
    out = pd.read_csv(TRAIN_DIR / f"output_2023_w{w}.csv")

    # Normalize all plays to right-moving convention (offense attacks +x end zone).
    # For left-direction plays: flip x-coordinates and mirror directional angles.
    play_dirs = (inp[["game_id", "play_id", "play_direction"]]
                 .drop_duplicates(["game_id", "play_id"]))
    out = out.merge(play_dirs, on=["game_id", "play_id"], how="left")

    left_inp = inp["play_direction"] == "left"
    left_out = out["play_direction"] == "left"

    inp = inp.copy()
    inp.loc[left_inp, "x"]           = 120 - inp.loc[left_inp, "x"]
    inp.loc[left_inp, "ball_land_x"] = 120 - inp.loc[left_inp, "ball_land_x"]
    inp.loc[left_inp, "dir"]         = (360 - inp.loc[left_inp, "dir"]) % 360
    inp.loc[left_inp, "o"]           = (360 - inp.loc[left_inp, "o"]) % 360

    out = out.copy()
    out.loc[left_out, "x"] = 120 - out.loc[left_out, "x"]
    out = out.drop(columns=["play_direction"])

    inp_sorted = inp.sort_values(GROUP_KEYS + ["frame_id"])

    # Compute lag differences within each player/play group
    grp = inp_sorted.groupby(GROUP_KEYS, sort=False)
    inp_sorted = inp_sorted.copy()
    inp_sorted["delta_x_last_1"]    = grp["x"].diff(1)
    inp_sorted["delta_y_last_1"]    = grp["y"].diff(1)
    inp_sorted["delta_x_last_3"]    = grp["x"].diff(3)
    inp_sorted["delta_y_last_3"]    = grp["y"].diff(3)
    inp_sorted["delta_x_last_5"]    = grp["x"].diff(5)
    inp_sorted["delta_y_last_5"]    = grp["y"].diff(5)
    inp_sorted["speed_change_last_3"]       = grp["s"].diff(3)
    inp_sorted["speed_change_last_5"]       = grp["s"].diff(5)
    inp_sorted["acceleration_change_last_3"] = grp["a"].diff(3)
    # Circular diffs for angular columns
    inp_sorted["direction_change_last_3"]   = circ_diff(
        inp_sorted["dir"].values,
        grp["dir"].shift(3).values,
    )
    inp_sorted["orientation_change_last_3"] = circ_diff(
        inp_sorted["o"].values,
        grp["o"].shift(3).values,
    )

    MOTION_COLS = [
        "delta_x_last_1", "delta_y_last_1",
        "delta_x_last_3", "delta_y_last_3",
        "delta_x_last_5", "delta_y_last_5",
        "speed_change_last_3", "speed_change_last_5", "direction_change_last_3",
        "acceleration_change_last_3", "orientation_change_last_3",
    ]

    # --- Route-level features from full input sequence ---
    inp_sorted["_step_dist"] = np.sqrt(
        grp["x"].diff().fillna(0)**2 + grp["y"].diff().fillna(0)**2
    )
    route_stats = (
        inp_sorted.groupby(GROUP_KEYS, sort=False)
        .agg(route_dist_traveled=("_step_dist", "sum"),
             mean_speed_input=("s", "mean"))
        .reset_index()
    )
    inp_sorted = inp_sorted.drop(columns=["_step_dist"])

    first_obs = (
        inp_sorted.groupby(GROUP_KEYS, sort=False)[["x", "y"]]
        .first()
        .reset_index()
        .rename(columns={"x": "first_x", "y": "first_y"})
    )

    last_obs = (
        inp_sorted.sort_values("frame_id")
        .groupby(GROUP_KEYS)[KEEP_LAST + MOTION_COLS]
        .last()
        .reset_index()
        .rename(columns={"x": "last_x", "y": "last_y", "s": "last_s", "a": "last_a",
                         "dir": "last_dir", "o": "last_o"})
    )

    # --- Merge route-level features into last_obs ---
    last_obs = (
        last_obs
        .merge(first_obs,    on=GROUP_KEYS, how="left")
        .merge(route_stats,  on=GROUP_KEYS, how="left")
    )
    last_obs["route_dir"] = np.arctan2(
        last_obs["last_y"] - last_obs["first_y"],
        last_obs["last_x"] - last_obs["first_x"],
    )

    # --- Targeted receiver interaction features (no self-pair exclusion needed) ---
    tr_snap = (
        last_obs[last_obs["player_role"] == "Targeted Receiver"]
        [["game_id", "play_id", "last_x", "last_y"]]
        .drop_duplicates(subset=["game_id", "play_id"])
        .rename(columns={"last_x": "tr_x", "last_y": "tr_y"})
    )
    last_obs = last_obs.merge(tr_snap, on=["game_id", "play_id"], how="left")
    last_obs["dx_to_targeted_receiver"]   = last_obs["tr_x"] - last_obs["last_x"]
    last_obs["dy_to_targeted_receiver"]   = last_obs["tr_y"] - last_obs["last_y"]
    last_obs["dist_to_targeted_receiver"] = np.sqrt(
        last_obs["dx_to_targeted_receiver"]**2 + last_obs["dy_to_targeted_receiver"]**2
    )
    last_obs = last_obs.drop(columns=["tr_x", "tr_y"])

    # --- Nearest opponent / nearest teammate (self-pairs excluded) ---
    snap = last_obs[["game_id", "play_id", "nfl_id", "last_x", "last_y", "player_side"]].copy()
    pairs = snap.merge(snap, on=["game_id", "play_id"], suffixes=("", "_other"))
    pairs = pairs[pairs["nfl_id"] != pairs["nfl_id_other"]].copy()
    pairs["_dx"]   = pairs["last_x_other"] - pairs["last_x"]
    pairs["_dy"]   = pairs["last_y_other"] - pairs["last_y"]
    pairs["_dist"] = np.sqrt(pairs["_dx"]**2 + pairs["_dy"]**2)

    key = ["game_id", "play_id", "nfl_id"]
    opp_pairs  = pairs[pairs["player_side_other"] != pairs["player_side"]]
    team_pairs = pairs[pairs["player_side_other"] == pairs["player_side"]]

    nearest_opp = (
        opp_pairs.loc[opp_pairs.groupby(key)["_dist"].idxmin(), key + ["_dx", "_dy", "_dist"]]
        .rename(columns={"_dx": "dx_to_nearest_opponent",
                         "_dy": "dy_to_nearest_opponent",
                         "_dist": "dist_to_nearest_opponent"})
    )
    nearest_team = (
        team_pairs.loc[team_pairs.groupby(key)["_dist"].idxmin(), key + ["_dx", "_dy", "_dist"]]
        .rename(columns={"_dx": "dx_to_nearest_teammate",
                         "_dy": "dy_to_nearest_teammate",
                         "_dist": "dist_to_nearest_teammate"})
    )

    last_obs = (
        last_obs
        .merge(nearest_opp,  on=key, how="left")
        .merge(nearest_team, on=key, how="left")
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
    "delta_x_last_1", "delta_y_last_1",
    "delta_x_last_3", "delta_y_last_3",
    "delta_x_last_5", "delta_y_last_5",
    "speed_change_last_3", "speed_change_last_5", "direction_change_last_3",
    "acceleration_change_last_3", "orientation_change_last_3",
    "first_x", "first_y", "route_dist_traveled", "mean_speed_input", "route_dir",
    "dist_to_targeted_receiver", "dx_to_targeted_receiver", "dy_to_targeted_receiver",
    "dist_to_nearest_opponent", "dx_to_nearest_opponent", "dy_to_nearest_opponent",
    "dist_to_nearest_teammate", "dx_to_nearest_teammate", "dy_to_nearest_teammate",
    "play_direction", "absolute_yardline_number",
    "residual_x", "residual_y"
]

result = pd.concat(frames, ignore_index=True)[out_cols]
result.to_csv(OUTPUT_DIR / "ml_dataset_w01_w18.csv", index=False)

print(f"Saved: outputs/ml_dataset_w01_w18.csv")
print(f"Shape: {result.shape}")
print(f"Columns: {list(result.columns)}")
