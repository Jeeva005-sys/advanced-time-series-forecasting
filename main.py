# ================================
# Advanced Time Series Forecasting
# Beginner Friendly Version
# ================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, MultiHeadAttention, LayerNormalization, Flatten

# ----------------
# 1. Generate Data
# ----------------
np.random.seed(42)

n_samples = 5000
time = np.arange(n_samples)

trend = time * 0.01
seasonal = 10 * np.sin(2 * np.pi * time / 50)
noise = np.random.normal(0, 2, n_samples)

f1 = trend + seasonal + noise
f2 = 5 * np.cos(2 * np.pi * time / 30) + noise
f3 = np.random.normal(0, 5, n_samples)

data = pd.DataFrame({
    "feature1": f1,
    "feature2": f2,
    "feature3": f3
})

data.to_csv("dataset.csv", index=False)

# Plot data
data.plot(figsize=(10,5))
plt.title("Generated Multivariate Time Series Data")
plt.savefig("data_plot.png")
plt.close()

print("Dataset generated and saved.")

# ----------------
# 2. Preprocessing
# ----------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_sequences(data, seq_length=30):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Data preprocessing completed.")

# ----------------
# 3. LSTM Baseline
# ----------------
lstm_model = Sequential([
    LSTM(64, input_shape=(30,3)),
    Dense(3)
])

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)

pred_lstm = lstm_model.predict(X_test)

mae_lstm = mean_absolute_error(y_test, pred_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test, pred_lstm))

print("LSTM MAE:", mae_lstm)
print("LSTM RMSE:", rmse_lstm)

# ----------------
# 4. Attention Model
# ----------------
inputs = Input(shape=(30,3))
attention = MultiHeadAttention(num_heads=2, key_dim=32)(inputs, inputs)
attention = LayerNormalization()(attention)
flat = Flatten()(attention)
outputs = Dense(3)(flat)

attention_model = Model(inputs, outputs)
attention_model.compile(optimizer='adam', loss='mse')
attention_model.fit(X_train, y_train, epochs=10, batch_size=32)

pred_attention = attention_model.predict(X_test)

mae_attention = mean_absolute_error(y_test, pred_attention)
rmse_attention = np.sqrt(mean_squared_error(y_test, pred_attention))

print("Attention MAE:", mae_attention)
print("Attention RMSE:", rmse_attention)

# ----------------
# 5. Results Table
# ----------------
results = pd.DataFrame({
    "Model": ["LSTM", "Attention"],
    "MAE": [mae_lstm, mae_attention],
    "RMSE": [rmse_lstm, rmse_attention]
})

results.to_csv("results.csv", index=False)

print("\nFinal Results:")
print(results)
