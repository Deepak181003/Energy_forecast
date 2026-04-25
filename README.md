# ⚡ Household Power Consumption Forecast

Real-time next-hour electricity usage prediction using Ridge, Lasso, PLS, and PCR regression.

## 🔗 Live Demo

> [**[https://your-app.streamlit.app](https://your-app.streamlit.app)**](https://energyforecast-xoozm3lztcgwpyeury6nlx.streamlit.app/)
---

## 📋 Assignment Requirements Checklist

| Requirement | Status |
|---|---|
| Load dataset & handle missing values | ✅ ffill + bfill |
| Resample minute-level → hourly | ✅ `resample('h').mean()` |
| Time-based features (hour, day) | ✅ + month, is_weekend |
| Lag & rolling features | ✅ lag-1/2/24/168h, rolling mean/std |
| StandardScaler normalisation | ✅ |
| Forward feature selection | ✅ SequentialFeatureSelector (6 features) |
| Ridge regression | ✅ |
| Lasso regression | ✅ |
| PCR (PCA + Ridge) | ✅ sklearn Pipeline |
| PLS regression | ✅ PLSRegression |
| TimeSeriesSplit validation | ✅ 5-fold |
| MSE & RMSE evaluation | ✅ saved to `models/results.json` |
| Streamlit UI | ✅ 4-tab dashboard |
| Cloud deployment | ✅ Streamlit Cloud |

---

## 🚀 Run Locally

> **Note:** The dataset and model `.pkl` files are excluded from this repo (too large for GitHub).
> Follow the steps below to set up locally.

```bash
# 1. Clone the repo
git clone https://github.com/your-username/your-repo.git
cd your-repo

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
#    → https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set
#    Place "household_power_consumption.txt" in the project root

# 4. Train models (generates scaler.pkl, features.pkl, models/)
python train.py

# 5. Launch the app
streamlit run app.py
```

---

## 🗂️ Project Structure

```
├── app.py                  # Streamlit dashboard
├── train.py                # Data pipeline + model training
├── requirements.txt        # Python dependencies
├── household_power_consumption.txt   # UCI dataset
├── scaler.pkl              # Fitted StandardScaler
├── features.pkl            # Selected feature names
└── models/
    ├── ridge.pkl
    ├── lasso.pkl
    ├── pls.pkl
    ├── pcr.pkl
    └── results.json        # CV MSE / RMSE scores
```

---

## 🧠 Models

| Model | Description |
|---|---|
| **Ridge** | L2 regularisation — stable, low variance |
| **Lasso** | L1 regularisation — sparse, automatic feature selection |
| **PLS** | Partial Least Squares — latent component decomposition |
| **PCR** | PCA + Ridge — dimensionality reduction then regression |

---

## 📊 Features Used

Selected via forward sequential feature selection from a candidate pool of 12 features:
`hour`, `day`, `month`, `is_weekend`, `lag_1`, `lag_2`, `lag_24`, `lag_168`,
`rolling_mean_3`, `rolling_std_3`, `rolling_mean_6`, `rolling_mean_24`

---

## ☁️ Deployment (Streamlit Cloud)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set `app.py` as the entry point
4. Add the dataset file or use Git LFS for large files
5. Click **Deploy** — your public URL will appear instantly
