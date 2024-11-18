import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.tsa.stattools as sts
import statsmodels.graphics.tsaplots as sgt
from pmdarima.arima import auto_arima

# Load dataset
@st.cache
def load_data():
    df = pd.read_csv('Short_order_dataset_final.csv')
    df['full_date'] = pd.to_datetime(df['full_date'])
    df_daily = df.groupby(df['full_date'].dt.date).agg({'Revenue': 'sum'}).reset_index()
    df_daily['full_date'] = pd.to_datetime(df_daily['full_date'])
    df_daily.set_index('full_date', inplace=True)
    df_daily = df_daily.asfreq('D')
    df_daily.fillna(0, inplace=True)
    return df_daily

df_daily = load_data()

st.title("Revenue Time Series Analysis")

# Plot the data
st.subheader("Revenue Over Time")
st.line_chart(df_daily['Revenue'])

# Stationarity Test
st.subheader("Dickey-Fuller Test for Stationarity")
adf_result = sts.adfuller(df_daily['Revenue'])
st.write(f"ADF Statistic: {adf_result[0]}")
st.write(f"p-value: {adf_result[1]}")
st.write("Critical Values:")
for key, value in adf_result[4].items():
    st.write(f'   {key}: {value}')

# ACF Plot
st.subheader("Auto-Correlation Function (ACF)")
fig, ax = plt.subplots()
sgt.plot_acf(df_daily['Revenue'], lags=40, zero=False, ax=ax)
st.pyplot(fig)

# Split data into train and test
size = int(len(df_daily) * 0.8)
df_train, df_test = df_daily.iloc[:size], df_daily.iloc[size:]

# ARIMA Model
st.subheader("ARIMA Model (1,1,1)")
model_arima = ARIMA(df_train['Revenue'], order=(1, 1, 1))
results_arima = model_arima.fit()
st.write(results_arima.summary())

# Prediction
st.subheader("ARIMA Prediction vs Actual")
start_index = "2015-12-30"
end_index = "2016-02-28"
pred_arima = results_arima.predict(start=start_index, end=end_index)

fig, ax = plt.subplots(figsize=(12, 6))
df_test['Revenue'][start_index:end_index].plot(ax=ax, label="Actual", color="blue")
pred_arima.plot(ax=ax, label="ARIMA Predictions", color="red")
plt.title("ARIMA Model: Prediction vs Actual", size=24)
plt.legend()
st.pyplot(fig)

# Auto-ARIMA Model
st.subheader("Auto-ARIMA Model")
model_auto_arima = auto_arima(df_train['Revenue'], trace=True, error_action='ignore', suppress_warnings=True)
st.write(model_auto_arima.summary())

# Auto-ARIMA Prediction
st.subheader("Auto-ARIMA Prediction vs Actual")
pred_auto_arima = model_auto_arima.predict(n_periods=len(df_test[start_index:end_index]))

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(df_test['Revenue'][start_index:end_index], label="Actual", color="blue")
plt.plot(df_test[start_index:end_index].index, pred_auto_arima, label="Auto-ARIMA Predictions", color="red")
plt.title("Auto-ARIMA Model: Prediction vs Actual", size=24)
plt.legend()
st.pyplot(fig)

# Show training data stats
st.subheader("Training Data Statistics")
st.write(df_train.describe())
