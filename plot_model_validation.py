from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT       = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(OUTPUT_DIR / "residual_model_w13_w18_validation.csv")
df = df.astype({"val_week": int})

plt.figure(figsize=(9, 5))
plt.plot(df["val_week"], df["baseline_rmse"], marker="o", label="Ball-aware Baseline")
plt.plot(df["val_week"], df["model_rmse"],    marker="o", label="Residual ML Model")
plt.xlabel("Week")
plt.ylabel("Kaggle RMSE")
plt.title("Residual Model vs Ball-Aware Baseline, Weeks 13-18")
plt.xticks(df["val_week"])
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "residual_model_w13_w18_rmse.png", dpi=150)
print("Saved: outputs/residual_model_w13_w18_rmse.png")
