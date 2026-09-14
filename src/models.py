"""
Train and save forecasting models.

Usage:
    python src/models.py --model sarima --store 1
    python src/models.py --model xgboost
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "models")


def train_sarima(store_id: int, test_days: int = 42):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    df = pd.read_csv(os.path.join(PROCESSED_DIR, "sales_clean.csv"), parse_dates=["date"])
    series = (
        df[df["store"] == store_id]
        .sort_values("date")
        .set_index("date")["sales"]
        .asfreq("D")
        .interpolate()
    )
    train, test = series.iloc[:-test_days], series.iloc[-test_days:]

    # (order, seasonal_order) chosen as a reasonable weekly-seasonality starting point;
    # tune with pmdarima.auto_arima or a grid search for a real submission.
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
                     enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)

    forecast = fitted.forecast(steps=len(test))

    os.makedirs(MODELS_DIR, exist_ok=True)
    fitted.save(os.path.join(MODELS_DIR, f"sarima_store{store_id}.pkl"))

    return forecast, test


def train_prophet(store_id: int, test_days: int = 42):
    from prophet import Prophet

    df = pd.read_csv(os.path.join(PROCESSED_DIR, "sales_clean.csv"), parse_dates=["date"])
    store_df = df[df["store"] == store_id].sort_values("date")
    prophet_df = store_df[["date", "sales"]].rename(columns={"date": "ds", "sales": "y"})

    train = prophet_df.iloc[:-test_days]
    test = prophet_df.iloc[-test_days:]

    model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    model.fit(train)

    future = model.make_future_dataframe(periods=test_days)
    forecast = model.predict(future).set_index("ds")["yhat"].iloc[-test_days:]

    return forecast, test.set_index("ds")["y"]


def train_xgboost(test_days: int = 42):
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error

    from features import build_feature_set, FEATURE_COLUMNS_BASE

    df = pd.read_csv(os.path.join(PROCESSED_DIR, "sales_clean.csv"), parse_dates=["date"])
    df = build_feature_set(df)

    feature_cols = [c for c in FEATURE_COLUMNS_BASE if c in df.columns]
    cutoff_date = df["date"].max() - pd.Timedelta(days=test_days)

    train = df[df["date"] <= cutoff_date]
    test = df[df["date"] > cutoff_date]

    X_train, y_train = train[feature_cols], train["sales"]
    X_test, y_test = test[feature_cols], test["sales"]

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "xgboost_global.pkl"))

    test = test.copy()
    test["prediction"] = preds
    test.to_csv(os.path.join(PROCESSED_DIR, "xgboost_predictions.csv"), index=False)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"XGBoost test RMSE: {rmse:.1f}")
    return model, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["sarima", "prophet", "xgboost"], required=True)
    parser.add_argument("--store", type=int, default=1, help="Store ID (for sarima/prophet)")
    args = parser.parse_args()

    if args.model == "sarima":
        forecast, test = train_sarima(args.store)
        print(forecast.head())
    elif args.model == "prophet":
        forecast, test = train_prophet(args.store)
        print(forecast.head())
    elif args.model == "xgboost":
        train_xgboost()


if __name__ == "__main__":
    main()
