"""
Advanced Time Series Forecasting with Deep Learning and Attention Mechanisms

Includes:
- Synthetic multivariate-like electricity dataset
- SARIMA baseline
- Basic LSTM baseline
- LSTM with Attention mechanism
- Multi-step forecasting
- Attention weight interpretation
- RMSE & MAE evaluation

Author: Bashath H
"""

# =====================================================
# 1. IMPORT LIBRARIES
# =====================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Attention
)
from tensorflow.keras.callbacks import EarlyStopping

from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# 2. RANDOM SEED
# =====================================================
np.random.seed(42)
tf.random.set_seed(42)

# =====================================================
# 3. DATA GENERATION
# =====================================================
def generate_data(n_steps=3*365):
    t = np.arange(n_steps)
    trend = 0.004 * t
    seasonal_1 = 15 * np.sin(2 * np.pi * t / 24)
    seasonal_2 = 25 * np.sin(2 * np.pi * t / 365)
    noise = np.random.normal(0, 2, n_steps)

    data = 50 + trend + seasonal_1 + seasonal_2 + noise
    return pd.DataFrame({"consumption": data})

df = generate_data()

# =====================================================
# 4. VISUALIZATION
# =====================================================
plt.figure(figsize=(12,4))
plt.plot(df["consumption"])
plt.title("Synthetic Electricity Consumption Data")
plt.xlabel("Time")
plt.ylabel("Consumption")
plt.tight_layout()
plt.show()

# =====================================================
# 5. SCALING
# =====================================================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[["consumption"]])

# =====================================================
# 6. SEQUENCE CREATION
# =====================================================
def create_sequences(data, input_steps=30, output_steps=7):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps):
        X.append(data[i:i+input_steps])
        y.append(data[i+input_steps:i+input_steps+output_steps])
    return np.array(X), np.array(y)

INPUT_STEPS = 30
OUTPUT_STEPS = 7

X, y = create_sequences(scaled_data, INPUT_STEPS, OUTPUT_STEPS)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =====================================================
# 7. BASELINE 1: SARIMA
# =====================================================
train_series = df["consumption"][:split + INPUT_STEPS]
test_series = df["consumption"][split + INPUT_STEPS:]

sarima = SARIMAX(
    train_series,
    order=(1,1,1),
    seasonal_order=(1,1,1,12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_fit = sarima.fit(disp=False)
sarima_pred = sarima_fit.forecast(steps=len(test_series))

sarima_rmse = np.sqrt(mean_squared_error(test_series, sarima_pred))
sarima_mae = mean_absolute_error(test_series, sarima_pred)

# =====================================================
# 8. BASELINE 2: BASIC LSTM
# =====================================================
basic_lstm = tf.keras.Sequential([
    LSTM(64, input_shape=(INPUT_STEPS,1)),
    Dense(OUTPUT_STEPS)
])

basic_lstm.compile(optimizer="adam", loss="mse")

basic_lstm.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    verbose=0
)

basic_pred = basic_lstm.predict(X_test)

basic_pred_inv = scaler.inverse_transform(
    basic_pred.reshape(-1,1)
).reshape(basic_pred.shape)

y_test_inv = scaler.inverse_transform(
    y_test.reshape(-1,1)
).reshape(y_test.shape)

basic_rmse = np.sqrt(
    mean_squared_error(y_test_inv.flatten(), basic_pred_inv.flatten())
)
basic_mae = mean_absolute_error(
    y_test_inv.flatten(), basic_pred_inv.flatten()
)

# =====================================================
# 9. LSTM WITH ATTENTION MODEL
# =====================================================
inputs = Input(shape=(INPUT_STEPS, 1))

lstm_out = LSTM(64, return_sequences=True)(inputs)
lstm_out = Dropout(0.2)(lstm_out)

attention = Attention()([lstm_out, lstm_out])
context = tf.reduce_mean(attention, axis=1)

outputs = Dense(OUTPUT_STEPS)(context)

attention_model = Model(inputs, outputs)

attention_model.compile(
    optimizer="adam",
    loss="mse"
)

attention_model.summary()

# =====================================================
# 10. TRAINING
# =====================================================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

attention_model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# =====================================================
# 11. EVALUATION
# =====================================================
att_pred = attention_model.predict(X_test)

att_pred_inv = scaler.inverse_transform(
    att_pred.reshape(-1,1)
).reshape(att_pred.shape)

att_rmse = np.sqrt(
    mean_squared_error(y_test_inv.flatten(), att_pred_inv.flatten())
)
att_mae = mean_absolute_error(
    y_test_inv.flatten(), att_pred_inv.flatten()
)

# =====================================================
# 12. ATTENTION WEIGHT INTERPRETATION
# =====================================================
attention_extractor = Model(
    inputs=attention_model.input,
    outputs=attention
)

attention_weights = attention_extractor.predict(X_test[:50])
importance = np.mean(attention_weights, axis=(0,1))

plt.figure(figsize=(10,4))
plt.bar(range(INPUT_STEPS), importance)
plt.title("Attention Weight Importance Across Time Steps")
plt.xlabel("Past Time Steps")
plt.ylabel("Mean Attention Weight")
plt.tight_layout()
plt.show()

# =====================================================
# 13. FORECAST VISUALIZATION
# =====================================================
plt.figure(figsize=(10,4))
plt.plot(y_test_inv[0], label="Actual")
plt.plot(att_pred_inv[0], label="Attention Forecast")
plt.legend()
plt.title("Multi-Step Forecast Comparison")
plt.tight_layout()
plt.show()

# =====================================================
# 14. FINAL RESULTS
# =====================================================
print("\nMODEL PERFORMANCE COMPARISON")
print("="*45)
print(f"SARIMA        -> RMSE: {sarima_rmse:.3f}, MAE: {sarima_mae:.3f}")
print(f"Basic LSTM    -> RMSE: {basic_rmse:.3f}, MAE: {basic_mae:.3f}")
print(f"LSTM+Attention-> RMSE: {att_rmse:.3f}, MAE: {att_mae:.3f}")
print("="*45)

print("\nProject Completed Successfully ✅")
