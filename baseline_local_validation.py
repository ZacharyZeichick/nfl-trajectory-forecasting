from pathlib import Path
import pandas as pd
import numpy as np

ROOT       = Path(__file__).resolve().parent
TRAIN_DIR  = ROOT / "train"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def kaggle_rmse(dx, dy):
    return np.sqrt(np.concatenate([np.asarray(dx)**2, np.asarray(dy)**2]).mean())

def score_week(week):
    w = str(week).zfill(2)
    inp = pd.read_csv(TRAIN_DIR / f"input_2023_w{w}.csv")
    out = pd.read_csv(TRAIN_DIR / f"output_2023_w{w}.csv")

    last_obs = (
        inp.sort_values("frame_id")
        .groupby(["game_id", "play_id", "nfl_id"])[["x", "y", "s", "dir", "ball_land_x", "ball_land_y", "num_frames_output"]]
        .last()
        .reset_index()
        .rename(columns={"x": "last_x", "y": "last_y", "s": "last_s", "dir": "last_dir"})
    )

    s = out.merge(last_obs, on=["game_id", "play_id", "nfl_id"], how="inner")

    t = s["frame_id"] / 10.0
    rad = np.deg2rad(s["last_dir"])
    vel_x = s["last_x"] + s["last_s"] * np.sin(rad) * t
    vel_y = s["last_y"] + s["last_s"] * np.cos(rad) * t

    progress = (s["frame_id"] / s["num_frames_output"]).clip(0, 1)
    ball_x = s["last_x"] + progress * (s["ball_land_x"] - s["last_x"])
    ball_y = s["last_y"] + progress * (s["ball_land_y"] - s["last_y"])

    stat  = kaggle_rmse(s["x"] - s["last_x"], s["y"] - s["last_y"])
    vel   = kaggle_rmse(s["x"] - vel_x,        s["y"] - vel_y)
    b025  = kaggle_rmse(s["x"] - (0.75*vel_x + 0.25*ball_x),
                        s["y"] - (0.75*vel_y + 0.25*ball_y))

    return len(s), stat, vel, b025, s, vel_x, vel_y, ball_x, ball_y

print(f"{'week':>4}  {'rows':>6}  {'stationary':>10}  {'velocity':>8}  {'ball_0.25':>9}")
print("-" * 48)

results = []
rows_per_week = []
for week in range(1, 19):
    n, stat, vel, b025, s, vx, vy, bx, by = score_week(week)
    print(f"{week:>4}  {n:>6}  {stat:>10.4f}  {vel:>8.4f}  {b025:>9.4f}")
    results.append((s, vx, vy, bx, by))
    rows_per_week.append({"week": week, "rows": n, "stationary_rmse": round(stat, 4),
                          "velocity_rmse": round(vel, 4), "ball_025_rmse": round(b025, 4)})

print("-" * 48)

def overall_rmse(extractor):
    sq = np.concatenate([extractor(*r) for r in results])
    return np.sqrt(sq.mean())

stat_r = overall_rmse(lambda s,vx,vy,bx,by: np.concatenate([(s["x"]-s["last_x"])**2, (s["y"]-s["last_y"])**2]))
vel_r  = overall_rmse(lambda s,vx,vy,bx,by: np.concatenate([(s["x"]-vx)**2, (s["y"]-vy)**2]))
b025_r = overall_rmse(lambda s,vx,vy,bx,by: np.concatenate([(s["x"]-(0.75*vx+0.25*bx))**2, (s["y"]-(0.75*vy+0.25*by))**2]))
total  = sum(r[0].shape[0] for r in results)

print(f"{'ALL':>4}  {total:>6}  {stat_r:>10.4f}  {vel_r:>8.4f}  {b025_r:>9.4f}")

rows_per_week.append({"week": "ALL", "rows": total, "stationary_rmse": round(stat_r, 4),
                      "velocity_rmse": round(vel_r, 4), "ball_025_rmse": round(b025_r, 4)})
pd.DataFrame(rows_per_week).to_csv(OUTPUT_DIR / "baseline_full_season_results.csv", index=False)
print("\nSaved: outputs/baseline_full_season_results.csv")
