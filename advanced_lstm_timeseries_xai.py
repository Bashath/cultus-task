"""
Advanced Time Series Forecasting with LSTM + Explainability (SHAP)

Features:
- Synthetic electricity consumption data
- Multi-step LSTM forecasting
- SARIMAX statistical baseline
- RMSE / MAE evaluation
- SHAP explainability for time-step importance

Author: Bashath H (Project Ready)
"""

# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from statsmodels.tsa.statespace.sarimax import SARIMAX

import shap
import warnings
warnings.filterwarnings("ignore")

# ===============================
# 2. RANDOM SEED
# ===============================
np.random.seed(42)
tf.random.set_seed(42)

# ===============================
# 3. DATA GENERATION
# ===============================
def generate_electricity_data(n_days=3*365):
    time = np.arange(n_days)
    trend = 0.005 * time
    seasonal_daily = 10 * np.sin(2 * np.pi * time / 24)
    seasonal_yearly = 20 * np.sin(2 * np.pi * time / 365)
    noise = np.random.normal(0, 2, n_days)

    consumption = 50 + trend + seasonal_daily + seasonal_yearly + noise
    return pd.DataFrame({"consumption": consumption})

df = generate_electricity_data()

# ===============================
# 4. VISUALIZE DATA
# ===============================
plt.figure(figsize=(12,4))
plt.plot(df["consumption"])
plt.title("Synthetic Electricity Consumption Data")
plt.xlabel("Time")
plt.ylabel("Consumption")
plt.tight_layout()
plt.show()

# ===============================
# 5. SCALING
# ===============================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[["consumption"]])

# ===============================
# 6. SEQUENCE CREATION
# ===============================
def create_sequences(data, input_steps=30, output_steps=7):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps):
        X.append(data[i:i+input_steps])
        y.append(data[i+input_steps:i+input_steps+output_steps])
    return np.array(X), np.array(y)

INPUT_STEPS = 30
OUTPUT_STEPS = 7

X, y = create_sequences(scaled_data, INPUT_STEPS, OUTPUT_STEPS)

split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# ===============================
# 7. LSTM MODEL
# ===============================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(INPUT_STEPS, 1)),
    Dropout(0.2),
    LSTM(32),
    Dense(OUTPUT_STEPS)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

# ===============================
# 8. TRAINING
# ===============================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ===============================
# 9. LSTM EVALUATION
# ===============================
y_pred = model.predict(X_test)

y_test_inv = scaler.inverse_transform(
    y_test.reshape(-1, 1)
).reshape(y_test.shape)

y_pred_inv = scaler.inverse_transform(
    y_pred.reshape(-1, 1)
).reshape(y_pred.shape)

lstm_rmse = np.sqrt(
    mean_squared_error(y_test_inv.flatten(), y_pred_inv.flatten())
)
lstm_mae = mean_absolute_error(
    y_test_inv.flatten(), y_pred_inv.flatten()
)

print("\nLSTM PERFORMANCE")
print("-------------------------")
print(f"RMSE: {lstm_rmse:.3f}")
print(f"MAE : {lstm_mae:.3f}")

# ===============================
# 10. SARIMAX BASELINE
# ===============================
train_series = df["consumption"][:split_idx + INPUT_STEPS]
test_series = df["consumption"][split_idx + INPUT_STEPS:]

sarimax_model = SARIMAX(
    train_series,
    order=(1,1,1),
    seasonal_order=(1,1,1,12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarimax_fit = sarimax_model.fit(disp=False)
sarimax_pred = sarimax_fit.forecast(steps=len(test_series))

sarimax_rmse = np.sqrt(
    mean_squared_error(test_series, sarimax_pred)
)
sarimax_mae = mean_absolute_error(
    test_series, sarimax_pred
)

print("\nSARIMAX PERFORMANCE")
print("-------------------------")
print(f"RMSE: {sarimax_rmse:.3f}")
print(f"MAE : {sarimax_mae:.3f}")

# ===============================
# 11. SHAP EXPLAINABILITY
# ===============================
background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(X_test[:50])

# Aggregate across output steps & features
time_importance = np.mean(np.abs(shap_values[0]), axis=(0,2))

plt.figure(figsize=(10,4))
plt.bar(range(INPUT_STEPS), time_importance)
plt.title("SHAP Time-Step Importance (LSTM)")
plt.xlabel("Past Time Steps")
plt.ylabel("Mean |SHAP Value|")
plt.tight_layout()
plt.show()

# ===============================
# 12. FORECAST VISUALIZATION
# ===============================
plt.figure(figsize=(10,4))
plt.plot(y_test_inv[0], label="Actual")
plt.plot(y_pred_inv[0], label="Predicted")
plt.title("7-Step Forecast vs Actual")
plt.legend()
plt.tight_layout()
plt.show()

# ===============================
# 13. FINAL COMPARISON
# ===============================
print("\nFINAL MODEL COMPARISON")
print("=" * 35)
print(f"LSTM     -> RMSE: {lstm_rmse:.3f}, MAE: {lstm_mae:.3f}")
print(f"SARIMAX  -> RMSE: {sarimax_rmse:.3f}, MAE: {sarimax_mae:.3f}")
print("=" * 35)

print("\nProject Execution Completed Successfully ✅")
