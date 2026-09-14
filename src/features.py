"""
Feature engineering for the ML (XGBoost/LightGBM) approach.

All features here are computed using only PAST information relative to
each row (shift before rolling), to avoid leakage from the future.
"""

import pandas as pd


def add_lag_features(df: pd.DataFrame, group_col="store", target_col="sales",
                      lags=(1, 7, 14, 28)) -> pd.DataFrame:
    df = df.sort_values([group_col, "date"]).copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df.groupby(group_col)[target_col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, group_col="store", target_col="sales",
                          windows=(7, 14, 28)) -> pd.DataFrame:
    df = df.sort_values([group_col, "date"]).copy()
    shifted = df.groupby(group_col)[target_col].shift(1)
    for window in windows:
        df[f"{target_col}_rollmean_{window}"] = (
            shifted.groupby(df[group_col]).rolling(window).mean().reset_index(level=0, drop=True)
        )
        df[f"{target_col}_rollstd_{window}"] = (
            shifted.groupby(df[group_col]).rolling(window).std().reset_index(level=0, drop=True)
        )
    return df


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    df = add_lag_features(df)
    df = add_rolling_features(df)
    # Drop rows where the longest lag/window isn't yet available
    df = df.dropna(subset=[c for c in df.columns if "lag" in c or "roll" in c])
    return df


FEATURE_COLUMNS_BASE = [
    "day_of_week", "month", "week_of_year", "is_weekend",
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_rollmean_7", "sales_rollmean_14", "sales_rollmean_28",
    "sales_rollstd_7", "sales_rollstd_14", "sales_rollstd_28",
]
# Extend with Promo/StateHoliday/CompetitionDistance etc. in models.py if present in your data.
