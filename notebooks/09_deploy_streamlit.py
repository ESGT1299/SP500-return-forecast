# 09_deploy_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib           import Path
from ta.momentum       import rsi
from ta.trend          import macd_diff
import joblib, xgboost as xgb
from tensorflow.keras.models import load_model
from sklearn.metrics          import mean_absolute_error, r2_score

# ─────────── Paths & Model Loading ───────────
BASE       = Path(__file__).resolve().parent.parent
models_dir = BASE/"models"

rf        = joblib.load(models_dir/"rf_tuned_model.joblib")
xgb_m     = xgb.XGBRegressor(); xgb_m.load_model(str(models_dir/"xgb_model.json"))
ensemble  = joblib.load(models_dir/"ensemble_model.joblib")
mlp       = load_model(models_dir/"mlp_improved_model.h5", compile=False)
lstm      = load_model(models_dir/"lstm_improved_model.h5", compile=False)
scaler    = joblib.load(models_dir/"scaler.pkl")

# ─────────── Precomputed Metrics ───────────
HOLD_OUT_MAE = 0.00751  # ≈0.75% log-return error
HOLD_OUT_R2  = 0.095    # explains ~9.5% of variance
WF_MAE_MEAN  = 0.00758
WF_MAE_STD   = 0.00226
WF_R2_MEAN   = 0.099
WF_R2_STD    = 0.045

# Sidebar
st.sidebar.header("📊 Model Performance")
st.sidebar.write(f"**Hold-out MAE**: {HOLD_OUT_MAE:.5f}  _(lower is better)_")
st.sidebar.write(f"**Hold-out R²**:  {HOLD_OUT_R2:.3f}   _(variance explained)_")
st.sidebar.write("---")
st.sidebar.write(f"**WF MAE**: {WF_MAE_MEAN:.5f} ± {WF_MAE_STD:.5f}")
st.sidebar.write(f"**WF R²**:  {WF_R2_MEAN:.3f} ± {WF_R2_STD:.3f}")

# ─────────── Page Title & Date Picker ───────────
st.title("📈 S&P 500 Next-Day Return Forecast")
as_of = st.date_input("Forecast date", pd.Timestamp.today().normalize())

# ─────────── Data Download with Fallback ───────────
start, end = "2010-01-01", "2025-01-01"
local_csv = BASE/"data"/"raw"/"sp500.csv"

# 1) Attempt yfinance download (single thread to minimize extra requests)
data = yf.download(
    "^GSPC",
    start=start,
    end=end,
    progress=False,
    threads=False
)

# 2) If it failed or is empty, fall back to the local CSV
if data is None or data.empty:
    st.warning("⚠️ Live download failed or returned empty—loading local CSV instead.")
    data = pd.read_csv(local_csv, index_col="Date", parse_dates=True)

# 3) If still empty, stop
if data.empty:
    st.error("No data available. Check your local CSV at data/raw/sp500.csv.")
    st.stop()


# ─────────── Price Chart ───────────
st.subheader("Historical S&P 500 Close Price")
st.line_chart(data["Close"], height=300)

# ─────────── Feature Engineering ───────────
df = data.copy()
df["return"] = np.log(df["Close"] / df["Close"].shift(1))

# ensure 1-D
close = df["Close"]
if getattr(close, "ndim", 1) > 1:
    close = close.squeeze()

df["rsi"]  = rsi(close, window=14)
df["macd"] = macd_diff(close)
for lag in range(1,6):
    df[f"ret_lag_{lag}"] = df["return"].shift(lag)
df["vol_10"]  = df["return"].rolling(10).std()
df["vol_pct"] = df["Volume"].pct_change()
df = df.dropna()

if df.empty:
    st.error("Not enough data after feature construction.")
    st.stop()

# ─────────── Prepare Latest Feature Vector ───────────
feature_cols = [
    "rsi","macd",
    *[f"ret_lag_{i}" for i in range(1,6)],
    "vol_10","vol_pct"
]

latest = df.iloc[-1]
X_new   = latest[feature_cols].values.reshape(1, -1)

# for LSTM, build the last 10-day window on scaled features
scaled = scaler.transform(df[feature_cols].values)
t      = 10
X_win  = np.array([scaled[-t:]])

# ─────────── Make Predictions ───────────
pred_rf   = rf.predict(X_new)[0]
pred_xgb  = xgb_m.predict(X_new)[0]
pred_mlp  = mlp.predict(scaler.transform(X_new)).flatten()[0]
pred_lstm = lstm.predict(X_win).flatten()[0]
pred_ens  = ensemble.predict([[pred_rf, pred_xgb, pred_mlp, pred_lstm]])[0]

# ─────────── Display Individual Forecasts ───────────
st.subheader("🔮 Individual Model Forecasts")
c1, c2, c3, c4 = st.columns(4)
c1.metric("RandomForest", f"{pred_rf:.4%}")
c2.metric("XGBoost",      f"{pred_xgb:.4%}")
c3.metric("MLP",          f"{pred_mlp:.4%}")
c4.metric("LSTM",         f"{pred_lstm:.4%}")

# ─────────── Display Ensemble Forecast ───────────
st.markdown("---")
st.subheader("🧮 Ensemble Forecast")
st.metric("Next-Day Return", f"{pred_ens:.4%}")

# ─────────── Explanation of Metrics ───────────
st.markdown(
    """
    - **MAE** (Mean Absolute Error): average absolute log-return error (0.0075 = 0.75%).  
    - **R²** (Coefficient of Determination): fraction of variance explained (0 = none, 1 = perfect).
    """
)
