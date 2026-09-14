# Retail Sales Forecasting

A portfolio project forecasting store-level daily sales, built around the
[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) dataset
(Kaggle). The pipeline goes from raw data → EDA → baselines → classical
time series models → gradient-boosted ML model → evaluation & error analysis.

## Why this project

Sales forecasting is one of the most directly business-relevant time series
problems (inventory, staffing, cash flow planning). This project is
structured to show the full toolkit, not just one model:

- Proper time-respecting train/test splits (no shuffling)
- Baselines first, so improvements are provably real
- Classical statistical models (SARIMA / Prophet)
- ML with engineered lag/rolling features (XGBoost / LightGBM)
- Store-level error analysis, not just an aggregate metric

## Getting the data

1. Create a free Kaggle account if you don't have one.
2. Go to the [Rossmann Store Sales competition](https://www.kaggle.com/c/rossmann-store-sales/data)
   and download `train.csv`, `test.csv`, and `store.csv`.
3. Place them in `data/raw/`.

(If you'd rather start smaller/faster, the
[Store Item Demand Forecasting](https://www.kaggle.com/c/demand-forecasting-kernels-only)
dataset works with the same pipeline with minor column-name tweaks — see
notes in `src/data_loader.py`.)

## Project structure

```
sales_forecasting_project/
├── data/
│   ├── raw/              # put downloaded CSVs here (not committed)
│   └── processed/        # cleaned/merged data written by the pipeline
├── notebooks/
│   └── 01_eda.py         # exploratory analysis (run as script or paste into Jupyter)
├── src/
│   ├── data_loader.py    # load + merge raw files, basic cleaning
│   ├── features.py       # lag/rolling/calendar feature engineering
│   ├── baselines.py       # naive, moving average, seasonal naive
│   ├── models.py         # SARIMA/Prophet + XGBoost/LightGBM training
│   └── evaluate.py       # RMSE/MAPE, plots, per-store error breakdown
├── requirements.txt
└── README.md
```

## Suggested order of work

1. `python src/data_loader.py` — loads and merges raw CSVs, saves to `data/processed/`
2. Run `notebooks/01_eda.py` — trend/seasonality/holiday visual analysis
3. `python src/baselines.py` — establishes the RMSE/MAPE floor to beat
4. `python src/models.py --model sarima` and `--model xgboost` — trains and saves models
5. `python src/evaluate.py` — compares all models, plots predicted vs actual, breaks down error by store

## Metrics

- **RMSE** — penalizes large misses, standard for regression-style forecasting
- **MAPE** — easier to explain to a non-technical audience ("we're off by X% on average")
- Always report both, and always show a predicted-vs-actual plot — a single
  number is easy to game, a plot isn't.

## Ideas to extend further

- Wrap the best model in a small Streamlit app (upload a store ID, get a forecast)
- Add prediction intervals (quantile regression or Prophet's built-in uncertainty)
- Try a global model (one model across all stores) vs per-store models, compare
