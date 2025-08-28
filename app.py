import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Streamlit App Config
# ------------------------------
st.set_page_config(page_title="Uber NYC Demand Prediction", layout="wide")
st.title("🚖 Uber NYC Demand Analysis & Prediction")
st.markdown("This app explores Uber trip data in NYC and builds ML models to predict hourly demand.")

# ------------------------------
# Load Data (Direct Path)
# ------------------------------
DATA_PATH = "UberRaw-data-apr14.csv"  # adjust if you keep it elsewhere
df = pd.read_csv(DATA_PATH)

st.subheader("📊 Raw Data Preview")
st.write(df.head())

# ------------------------------
# Data Preprocessing
# ------------------------------
df['Date/Time'] = pd.to_datetime(df['Date/Time'], errors='coerce')
df = df.dropna(subset=['Date/Time']).sort_values('Date/Time').reset_index(drop=True)

# NYC bounds filter
df = df[df['Lat'].between(40.0, 41.2) & df['Lon'].between(-74.5, -72.8)].copy()

# Hourly aggregation
hourly = (
    df.set_index('Date/Time')
      .resample('H').size()
      .to_frame('pickups')
      .reset_index()
)

# Feature engineering
hourly['hour'] = hourly['Date/Time'].dt.hour
hourly['day'] = hourly['Date/Time'].dt.day
hourly['weekday'] = hourly['Date/Time'].dt.dayofweek
hourly['is_weekend'] = hourly['weekday'].isin([5, 6]).astype(int)

# Cyclical encodings
hourly['hour_sin'] = np.sin(2 * np.pi * hourly['hour'] / 24)
hourly['hour_cos'] = np.cos(2 * np.pi * hourly['hour'] / 24)
hourly['wday_sin'] = np.sin(2 * np.pi * hourly['weekday'] / 7)
hourly['wday_cos'] = np.cos(2 * np.pi * hourly['weekday'] / 7)

# ------------------------------
# Exploratory Data Analysis
# ------------------------------
st.subheader("🔎 Exploratory Data Analysis")

st.write("Hourly pickups over time:")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hourly['Date/Time'], hourly['pickups'], alpha=0.7)
ax.set_xlabel("Date/Time"); ax.set_ylabel("Pickups")
st.pyplot(fig)

pivot = hourly.pivot_table(index="weekday", columns="hour", values="pickups", aggfunc="mean")
fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(pivot, cmap="viridis", ax=ax)
st.write("Average pickups by weekday vs hour:")
st.pyplot(fig)

# ------------------------------
# Download Processed Data
# ------------------------------
st.subheader("💾 Download Processed Data")
csv = hourly.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Hourly Dataset as CSV",
    data=csv,
    file_name="uber_hourly_data.csv",
    mime="text/csv"
)

# ------------------------------
# Model Training & Comparison
# ------------------------------
st.subheader("🤖 Model Training & Comparison")

features = ['hour', 'weekday', 'is_weekend', 'hour_sin', 'hour_cos', 'wday_sin', 'wday_cos']
X = hourly[features]
y = hourly['pickups']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardize for MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train RandomForest ---
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# --- Train Neural Network ---
mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)
mlp_preds = mlp.predict(X_test_scaled)

# --- Evaluate Both Models ---
results = pd.DataFrame({
    "Model": ["RandomForest", "Neural Network"],
    "MSE": [
        mean_squared_error(y_test, rf_preds),
        mean_squared_error(y_test, mlp_preds)
    ],
    "MAE": [
        mean_absolute_error(y_test, rf_preds),
        mean_absolute_error(y_test, mlp_preds)
    ],
    "R²": [
        r2_score(y_test, rf_preds),
        r2_score(y_test, mlp_preds)
    ]
})

st.write("### 📊 Performance Comparison (Table)")
st.dataframe(results.style.format({"MSE": "{:.2f}", "MAE": "{:.2f}", "R²": "{:.2f}"}))

# --- Metric Bar Charts ---
st.write("### 🧮 Metric Bar Charts")

# MSE
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(results["Model"], results["MSE"])
ax.set_title("MSE by Model"); ax.set_ylabel("MSE")
st.pyplot(fig)

# MAE
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(results["Model"], results["MAE"])
ax.set_title("MAE by Model"); ax.set_ylabel("MAE")
st.pyplot(fig)

# R²
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(results["Model"], results["R²"])
ax.set_title("R² by Model"); ax.set_ylabel("R²")
st.pyplot(fig)

# --- Combined Curves ---
st.write("### 📈 Actual vs Predicted (Both Models)")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(y_test.values, label="Actual", color="black", alpha=0.7)
ax.plot(rf_preds, label="RandomForest Predicted", alpha=0.7)
ax.plot(mlp_preds, label="Neural Net Predicted", alpha=0.7)
ax.legend()
st.pyplot(fig)
