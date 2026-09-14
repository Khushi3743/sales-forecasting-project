"""
Baseline forecasts to establish the error floor any 'real' model must beat.

Includes:
    - Naive: tomorrow = today
    - Seasonal naive: tomorrow = same weekday last week
    - Moving average: mean of last N days
"""

import os
import numpy as np
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def naive_forecast(series: pd.Series) -> pd.Series:
    return series.shift(1)


def seasonal_naive_forecast(series: pd.Series, season_length: int = 7) -> pd.Series:
    return series.shift(season_length)


def moving_average_forecast(series: pd.Series, window: int = 7) -> pd.Series:
    return series.shift(1).rolling(window=window).mean()


def evaluate_baselines(df: pd.DataFrame, store_id: int, test_days: int = 42):
    store_df = df[df["store"] == store_id].sort_values("date").set_index("date")
    series = store_df["sales"].asfreq("D").interpolate()

    train, test = series.iloc[:-test_days], series.iloc[-test_days:]

    results = {}
    for name, forecast_fn in [
        ("naive", naive_forecast),
        ("seasonal_naive", seasonal_naive_forecast),
        ("moving_average_7", moving_average_forecast),
    ]:
        full_forecast = forecast_fn(series)
        test_forecast = full_forecast.loc[test.index].dropna()
        aligned_actual = test.loc[test_forecast.index]
        results[name] = {
            "rmse": rmse(aligned_actual, test_forecast),
            "mape": mape(aligned_actual, test_forecast),
        }

    return results


def main():
    path = os.path.join(PROCESSED_DIR, "sales_clean.csv")
    df = pd.read_csv(path, parse_dates=["date"])

    # Run on a handful of stores and average — gives a more stable baseline read
    # than a single store.
    sample_stores = df["store"].unique()[:10]
    all_results = []
    for store_id in sample_stores:
        try:
            res = evaluate_baselines(df, store_id)
            all_results.append(res)
        except Exception as e:
            print(f"Skipping store {store_id}: {e}")

    summary = {}
    for method in ["naive", "seasonal_naive", "moving_average_7"]:
        rmses = [r[method]["rmse"] for r in all_results]
        mapes = [r[method]["mape"] for r in all_results]
        summary[method] = {"avg_rmse": np.mean(rmses), "avg_mape": np.mean(mapes)}

    print("Baseline results averaged across sample stores:")
    for method, scores in summary.items():
        print(f"  {method:20s} RMSE={scores['avg_rmse']:.1f}  MAPE={scores['avg_mape']:.1f}%")


if __name__ == "__main__":
    main()
