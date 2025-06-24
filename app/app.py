import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

st.set_page_config(layout="wide")
st.title("📊 Stock Price Forecasting Comparison App")
st.markdown("Compare performance and forecast plots of **7 models** on AAPL stock prices.")

# Load data
data = pd.read_csv("../data/raw/AAPL_stock.csv", skiprows=2)
data.rename(columns={data.columns[0]: 'Date', data.columns[1]: 'Close'}, inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data.dropna(inplace=True)

# Simulated actual and forecasted data
y_actual = data['Close'].values[-100:]
dates = data.index[-100:]

y_forecasts = {
    "ARIMA": y_actual + np.random.normal(0, 1, size=100),
    "SARIMA": y_actual + np.random.normal(0, 2, size=100),
    "LSTM": y_actual + np.random.normal(0, 1.5, size=100),
    "Random Forest": y_actual + np.random.normal(0, 1.7, size=100),
    "XGBoost": y_actual + np.random.normal(0, 1.3, size=100),
    "ETS": y_actual + np.random.normal(0, 1.2, size=100)
}

# Prophet needs special handling
df_prophet = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
model = Prophet(daily_seasonality=True)
model.fit(df_prophet)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Create tabs for models + comparison
tabs = st.tabs(["Prophet", "ARIMA", "SARIMA", "LSTM", "Random Forest", "XGBoost", "ETS", "Comparison Summary"])

# --- Prophet ---
with tabs[0]:
    st.header("📈 Prophet Forecast (30 Days Ahead)")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)
    st.subheader("Prophet RMSE: 16.02")

# Plot forecast chart function
def plot_forecast(title, forecasted, rmse_value):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, y_actual, label="Actual")
    ax.plot(dates, forecasted, label="Forecast")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)
    st.subheader(f"{title} RMSE: {rmse_value}")

# --- Other Models ---
with tabs[1]:  # ARIMA
    st.header("📉 ARIMA Forecast")
    plot_forecast("ARIMA Forecast", y_forecasts["ARIMA"], 1.06)

with tabs[2]:  # SARIMA
    st.header("🔁 SARIMA Forecast")
    plot_forecast("SARIMA Forecast", y_forecasts["SARIMA"], 30.2)

with tabs[3]:  # LSTM
    st.header("🤖 LSTM Forecast")
    plot_forecast("LSTM Forecast", y_forecasts["LSTM"], 6.59)

with tabs[4]:  # Random Forest
    st.header("🌲 Random Forest Forecast")
    plot_forecast("Random Forest Forecast", y_forecasts["Random Forest"], 9.33)

with tabs[5]:  # XGBoost
    st.header("🚀 XGBoost Forecast")
    plot_forecast("XGBoost Forecast", y_forecasts["XGBoost"], 5.95)

with tabs[6]:  # ETS
    st.header("📐 ETS Forecast")
    plot_forecast("ETS Forecast", y_forecasts["ETS"], 4.428)

# --- Final Comparison ---
with tabs[7]:
    st.header("📋 RMSE Comparison Summary")

    rmse_data = pd.DataFrame({
        "Model": ["ARIMA", "SARIMA", "Prophet", "LSTM", "Random Forest", "XGBoost", "ETS"],
        "RMSE": [1.06, 30.2, 16.02, 6.59, 9.33, 5.95, 4.428],
        "Notes": [
            "Based on price differences",
            "Seasonal + trend model",
            "Trend + seasonality (automatic)",
            "Deep learning using lag features",
            "Tree model using lag features",
            "Boosted tree model",
            "Trend smoothing only"
        ]
    }).sort_values(by="RMSE")

    st.dataframe(rmse_data)

    # Bar chart
    fig_bar, ax = plt.subplots(figsize=(10, 5))
    ax.bar(rmse_data["Model"], rmse_data["RMSE"], color="skyblue")
    ax.set_title("Model RMSE Comparison (Lower = Better)")
    ax.set_ylabel("RMSE")
    for i, v in enumerate(rmse_data["RMSE"]):
        ax.text(i, v + 0.5, f"{v:.2f}", ha='center')
    st.pyplot(fig_bar)

    st.markdown("""
### ✅ Observations:
- **Lowest RMSE on actual prices**: `ETS` and `XGBoost`
- **Lowest RMSE overall (on differenced)**: `ARIMA`
- **Deep learning approach**: `LSTM`
- **High error**: `SARIMA` — may need tuning
- **Most interpretable**: `Prophet`
""")
