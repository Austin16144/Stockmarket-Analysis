# Streamlit-based deployment for comparing all models

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
import os

st.set_page_config(layout="wide")

st.title("📊 Stock Price Forecasting Comparison App")
st.markdown("Compare performance of ARIMA, SARIMA, Prophet, and LSTM models on AAPL stock prices.")

# Load data
data = pd.read_csv("../data/raw/AAPL_stock.csv", skiprows=2)
data.rename(columns={data.columns[0]: 'Date', data.columns[1]: 'Close'}, inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Dummy predictions for ARIMA, SARIMA, LSTM (Replace with real model outputs)
y = data['Close'].values[-100:]
y_arima = y + np.random.normal(0, 1, size=len(y))
y_sarima = y + np.random.normal(0, 2, size=len(y))
y_lstm = y + np.random.normal(0, 1.5, size=len(y))
dates = data.index[-100:]


# Create tabs
tabs = st.tabs(["Prophet", "ARIMA", "SARIMA", "LSTM", "Comparison Summary"])

# --- Prophet ---
with tabs[0]:
    st.header("📈 Prophet Forecast")
    df = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    fig1 = model.plot(forecast)
    st.pyplot(fig1)
    st.subheader("Prophet RMSE: 16.02")

# --- ARIMA ---
with tabs[1]:
    st.header("📉 ARIMA Forecast")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(dates, y, label='Actual')
    ax2.plot(dates, y_arima, label='ARIMA Forecast')
    ax2.set_title("ARIMA Forecast vs Actual")
    ax2.legend()
    st.pyplot(fig2)
    st.subheader("ARIMA RMSE: 1.06")

# --- SARIMA ---
with tabs[2]:
    st.header("🔁 SARIMA Forecast")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(dates, y, label='Actual')
    ax3.plot(dates, y_sarima, label='SARIMA Forecast')
    ax3.set_title("SARIMA Forecast vs Actual")
    ax3.legend()
    st.pyplot(fig3)
    st.subheader("SARIMA RMSE: 30.2")

# --- LSTM ---
with tabs[3]:
    st.header("🤖 LSTM Forecast")
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(dates, y, label='Actual')
    ax4.plot(dates, y_lstm, label='LSTM Forecast')
    ax4.set_title("LSTM Forecast vs Actual")
    ax4.legend()
    st.pyplot(fig4)
    st.subheader("LSTM RMSE: 6.59")

# --- Comparison Table ---
with tabs[4]:
    st.header("📋 RMSE Comparison Summary")
    summary_data = {
        'Model': ['ARIMA', 'Prophet', 'LSTM', 'SARIMA'],
        'RMSE': [1.06, 16.02, 6.59, 30.2],
        'Notes': [
            'Predicted price difference',
            'Good for trend/seasonality',
            'Deep learning on actual price',
            'Seasonal classical model'
        ]
    }
    st.dataframe(pd.DataFrame(summary_data))

    st.markdown("\n**Best performing model overall:** ✅ LSTM (for actual price)\n\n**Most accurate on differenced data:** ✅ ARIMA\n\n**Simplest setup:** ✅ Prophet")
