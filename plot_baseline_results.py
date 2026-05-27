from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT       = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(OUTPUT_DIR / "baseline_full_season_results.csv")
df = df[df["week"] != "ALL"].astype({"week": int})

plt.figure(figsize=(10, 5))
plt.plot(df["week"], df["stationary_rmse"], marker="o", label="Stationary")
plt.plot(df["week"], df["velocity_rmse"],   marker="o", label="Velocity")
plt.plot(df["week"], df["ball_025_rmse"],   marker="o", label="Ball-aware (w=0.25)")
plt.xlabel("Week")
plt.ylabel("Kaggle RMSE")
plt.title("NFL Big Data Bowl 2026 Baseline RMSE by Week")
plt.xticks(df["week"])
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "baseline_rmse_by_week.png", dpi=150)
print("Saved: outputs/baseline_rmse_by_week.png")
