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
DATA_PATH = "../Datasets/UberRaw-data-apr14.csv"  # change if needed
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

# Time series plot
st.write("Hourly pickups over time:")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hourly['Date/Time'], hourly['pickups'], alpha=0.7)
ax.set_xlabel("Date/Time")
ax.set_ylabel("Pickups")
st.pyplot(fig)

# Heatmap (weekday vs hour)
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
# Model Training
# ------------------------------
st.subheader("🤖 Model Training & Evaluation")

features = ['hour', 'weekday', 'is_weekend', 'hour_sin', 'hour_cos', 'wday_sin', 'wday_cos']
X = hourly[features]
y = hourly['pickups']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardize for MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Choose model
model_choice = st.selectbox("Select Model", ["RandomForest", "Neural Network"])

if model_choice == "RandomForest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
else:
    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

# Evaluation
mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

st.metric("Mean Squared Error", f"{mse:.2f}")
st.metric("Mean Absolute Error", f"{mae:.2f}")
st.metric("R² Score", f"{r2:.2f}")

# Plot actual vs predicted
st.write("Actual vs Predicted:")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_test.values, label="Actual", alpha=0.7)
ax.plot(preds, label="Predicted", alpha=0.7)
ax.legend()
st.pyplot(fig)
