"""
train.py — Household Power Consumption Forecasting
====================================================
Steps:
  1. Load & clean data (handle missing values)
  2. Resample minute-level → hourly
  3. Engineer time-based + lag + rolling features
  4. Standardise with StandardScaler
  5. Forward feature selection (sequential)
  6. Train Ridge, Lasso, PLS (PCR variant included)
  7. Evaluate with TimeSeriesSplit (MSE, RMSE)
  8. Save all artefacts for the Streamlit app
"""

import pandas as pd
import numpy as np
import joblib
import os
import json

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SequentialFeatureSelector

os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(
    "household_power_consumption.txt",
    sep=";", na_values="?", low_memory=False,
)

df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
)
df = df.set_index("Datetime")
df.drop(["Date", "Time"], axis=1, inplace=True)
df = df.astype(float)

missing_before = df.isnull().sum().sum()
df = df.ffill().bfill()          # forward-fill then back-fill
missing_after  = df.isnull().sum().sum()
print(f"  Missing values: {missing_before} → {missing_after} after fill")

# ─────────────────────────────────────────────
# 2. RESAMPLE TO HOURLY
# ─────────────────────────────────────────────
df_hourly = df.resample("h").mean()
print(f"  Hourly rows: {len(df_hourly):,}")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
target = "Global_active_power"

df_hourly["hour"]           = df_hourly.index.hour
df_hourly["day"]            = df_hourly.index.dayofweek
df_hourly["month"]          = df_hourly.index.month
df_hourly["is_weekend"]     = (df_hourly.index.dayofweek >= 5).astype(int)

df_hourly["lag_1"]          = df_hourly[target].shift(1)
df_hourly["lag_2"]          = df_hourly[target].shift(2)
df_hourly["lag_24"]         = df_hourly[target].shift(24)
df_hourly["lag_168"]        = df_hourly[target].shift(168)   # 1 week

df_hourly["rolling_mean_3"] = df_hourly[target].rolling(3).mean()
df_hourly["rolling_std_3"]  = df_hourly[target].rolling(3).std()
df_hourly["rolling_mean_6"] = df_hourly[target].rolling(6).mean()
df_hourly["rolling_mean_24"]= df_hourly[target].rolling(24).mean()

df_hourly = df_hourly.dropna()

ALL_FEATURES = [
    "hour", "day", "month", "is_weekend",
    "lag_1", "lag_2", "lag_24", "lag_168",
    "rolling_mean_3", "rolling_std_3",
    "rolling_mean_6", "rolling_mean_24",
]

X_all = df_hourly[ALL_FEATURES]
y     = df_hourly[target]

# ─────────────────────────────────────────────
# 4. STANDARDISE
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled_all = scaler.fit_transform(X_all)

# ─────────────────────────────────────────────
# 5. FORWARD FEATURE SELECTION
# ─────────────────────────────────────────────
print("\nRunning forward feature selection...")
tscv = TimeSeriesSplit(n_splits=5)

selector = SequentialFeatureSelector(
    Ridge(alpha=1.0),
    n_features_to_select=6,
    direction="forward",
    scoring="neg_mean_squared_error",
    cv=tscv,
    n_jobs=-1,
)
selector.fit(X_scaled_all, y)

FEATURES = [f for f, s in zip(ALL_FEATURES, selector.get_support()) if s]
print(f"  Selected features: {FEATURES}")

X      = df_hourly[FEATURES]
X_scaled = scaler.fit_transform(X)   # refit scaler on selected features only

# ─────────────────────────────────────────────
# 6 & 7. TRAIN + EVALUATE
# ─────────────────────────────────────────────
models = {
    "ridge": Ridge(alpha=1.0),
    "lasso": Lasso(alpha=0.01, max_iter=5000),
    "pls":   PLSRegression(n_components=3),
    "pcr":   Pipeline([
                 ("pca",   PCA(n_components=4)),
                 ("ridge", Ridge(alpha=1.0)),
             ]),
}

results = {}
print()
for name, model in models.items():
    mse_scores, rmse_scores = [], []

    for train_idx, test_idx in tscv.split(X_scaled):
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y.iloc[train_idx],   y.iloc[test_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        if hasattr(preds, "ravel"):
            preds = preds.ravel()

        mse  = mean_squared_error(y_te, preds)
        rmse = np.sqrt(mse)
        mse_scores.append(mse)
        rmse_scores.append(rmse)

    avg_mse  = float(np.mean(mse_scores))
    avg_rmse = float(np.mean(rmse_scores))
    results[name] = {"mse": avg_mse, "rmse": avg_rmse}
    print(f"  {name.upper():5s}  MSE: {avg_mse:.4f}  RMSE: {avg_rmse:.4f}")

    # Final fit on full data
    model.fit(X_scaled, y)
    joblib.dump(model, f"models/{name}.pkl")

# ─────────────────────────────────────────────
# 8. SAVE SHARED ARTEFACTS
# ─────────────────────────────────────────────
joblib.dump(scaler,   "scaler.pkl")
joblib.dump(FEATURES, "features.pkl")

with open("models/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ All models and artefacts saved.")
print(f"   Features used: {FEATURES}")
