"""
Load and merge the raw Rossmann Store Sales CSVs into a single clean
DataFrame ready for EDA / feature engineering.

Expected raw files (from Kaggle) in data/raw/:
    train.csv   - Date, Store, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday
    store.csv   - Store, StoreType, Assortment, CompetitionDistance, Promo2, ...

If you're using the Store Item Demand Forecasting dataset instead, adjust
COLUMN_MAP below to rename your columns to: date, store, sales.
"""

import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# If using a different dataset, map its columns to this project's expected names here.
COLUMN_MAP = {
    "Date": "date",
    "Store": "store",
    "Sales": "sales",
}


def load_raw(train_file="train.csv", store_file="store.csv"):
    train_path = os.path.join(RAW_DIR, train_file)
    store_path = os.path.join(RAW_DIR, store_file)

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Couldn't find {train_path}. Download the dataset (see README) "
            f"and place train.csv / store.csv in data/raw/."
        )

    train = pd.read_csv(train_path, parse_dates=["Date"], low_memory=False)

    if os.path.exists(store_path):
        store = pd.read_csv(store_path)
        # Merge while both sides still use the raw "Store" column name,
        # then rename afterwards so we don't end up with mismatched keys.
        df = train.merge(store, on="Store", how="left")
    else:
        df = train

    df = df.rename(columns=COLUMN_MAP)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal cleaning: drop closed-store zero-sales rows, sort, fill gaps."""
    if "Open" in df.columns:
        df = df[df["Open"] == 1].copy()
    if "sales" in df.columns:
        df = df[df["sales"] > 0].copy()

    df = df.sort_values(["store", "date"]).reset_index(drop=True)

    # Fill common categorical NaNs that appear in store.csv
    for col in ["CompetitionDistance"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df = load_raw()
    df = basic_clean(df)
    df = add_calendar_features(df)

    out_path = os.path.join(PROCESSED_DIR, "sales_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved cleaned dataset: {out_path}  ({len(df):,} rows, {df['store'].nunique()} stores)")


if __name__ == "__main__":
    main()
