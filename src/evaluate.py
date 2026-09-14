"""
Compare model performance and produce the plots/breakdowns that make a
forecasting project reviewable at a glance:
    - predicted vs actual over time
    - error distribution
    - per-store error (where does the model fail?)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def plot_predicted_vs_actual(df: pd.DataFrame, store_id: int, out_path: str):
    store_df = df[df["store"] == store_id].sort_values("date")
    plt.figure(figsize=(14, 4))
    plt.plot(store_df["date"], store_df["sales"], label="Actual")
    plt.plot(store_df["date"], store_df["prediction"], label="Predicted", alpha=0.8)
    plt.title(f"Store {store_id}: predicted vs actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def per_store_error(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for store_id, group in df.groupby("store"):
        rows.append({
            "store": store_id,
            "rmse": rmse(group["sales"], group["prediction"]),
            "mape": mape(group["sales"], group["prediction"]),
            "n_days": len(group),
        })
    return pd.DataFrame(rows).sort_values("mape", ascending=False)


def main():
    pred_path = os.path.join(PROCESSED_DIR, "xgboost_predictions.csv")
    if not os.path.exists(pred_path):
        print(f"No predictions found at {pred_path}. Run: python src/models.py --model xgboost")
        return

    df = pd.read_csv(pred_path, parse_dates=["date"])

    overall_rmse = rmse(df["sales"], df["prediction"])
    overall_mape = mape(df["sales"], df["prediction"])
    print(f"Overall RMSE: {overall_rmse:.1f}   MAPE: {overall_mape:.1f}%")

    store_errors = per_store_error(df)
    store_errors.to_csv(os.path.join(PROCESSED_DIR, "per_store_error.csv"), index=False)
    print("\nWorst 5 stores by MAPE:")
    print(store_errors.head())
    print("\nBest 5 stores by MAPE:")
    print(store_errors.tail())

    # Plot predicted vs actual for the best, median, and worst store —
    # a quick way to show a reviewer where the model shines and where it struggles.
    sorted_stores = store_errors["store"].tolist()
    for label, store_id in [
        ("best", sorted_stores[-1]),
        ("median", sorted_stores[len(sorted_stores) // 2]),
        ("worst", sorted_stores[0]),
    ]:
        plot_predicted_vs_actual(df, store_id, f"eval_{label}_store_{store_id}.png")

    print("\nSaved plots: eval_best_*.png, eval_median_*.png, eval_worst_*.png")
    print("Saved: per_store_error.csv")


if __name__ == "__main__":
    main()
