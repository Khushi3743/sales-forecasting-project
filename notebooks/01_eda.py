"""
Exploratory analysis for the sales forecasting project.
Run as a plain script, or paste cells into a Jupyter notebook.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
df = pd.read_csv(os.path.join(PROCESSED_DIR, "sales_clean.csv"), parse_dates=["date"])

# ---------------------------------------------------------------
# 1. Aggregate daily sales across all stores
# ---------------------------------------------------------------
daily = df.groupby("date")["sales"].sum().reset_index()

plt.figure(figsize=(14, 4))
plt.plot(daily["date"], daily["sales"])
plt.title("Total daily sales across all stores")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("eda_daily_sales.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 2. Seasonal decomposition (trend / seasonal / residual)
# ---------------------------------------------------------------
daily_indexed = daily.set_index("date").asfreq("D").interpolate()
decomposition = seasonal_decompose(daily_indexed["sales"], model="additive", period=7)

fig = decomposition.plot()
fig.set_size_inches(14, 8)
plt.tight_layout()
plt.savefig("eda_decomposition.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 3. Day-of-week and monthly seasonality
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.boxplot(data=df, x="day_of_week", y="sales", ax=axes[0])
axes[0].set_title("Sales by day of week (0=Mon)")

sns.boxplot(data=df, x="month", y="sales", ax=axes[1])
axes[1].set_title("Sales by month")
plt.tight_layout()
plt.savefig("eda_seasonality.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 4. Holiday / promo effects (if columns present)
# ---------------------------------------------------------------
if "Promo" in df.columns:
    promo_effect = df.groupby("Promo")["sales"].mean()
    print("\nMean sales by promo status:")
    print(promo_effect)

if "StateHoliday" in df.columns:
    holiday_effect = df.groupby("StateHoliday")["sales"].mean()
    print("\nMean sales by state holiday flag:")
    print(holiday_effect)

# ---------------------------------------------------------------
# 5. Store-level variance — how heterogeneous are stores?
# ---------------------------------------------------------------
store_stats = df.groupby("store")["sales"].agg(["mean", "std"]).sort_values("mean", ascending=False)
print("\nTop 5 highest-average stores:")
print(store_stats.head())
print("\nBottom 5 lowest-average stores:")
print(store_stats.tail())

print("\nSaved: eda_daily_sales.png, eda_decomposition.png, eda_seasonality.png")
