"""
Advanced Time Series Forecasting with LSTM and Attention Model.

This script generates synthetic multivariate time series data,
trains a baseline LSTM model and an attention-based encoder-decoder model,
performs hyperparameter tuning, and evaluates performance using MAE, RMSE, and MAPE.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Attention, Input
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# ---------------------------
# Utility Functions
# ---------------------------

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def create_sequences(data, seq_length):
    """
    Create input-output sequences for time series forecasting.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])
    return np.array(X), np.array(y)


# ---------------------------
# Model Builders
# ---------------------------

def build_lstm_model(input_shape, units=64, lr=0.001):
    """
    Build baseline LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr), loss='mse')
    return model


def build_attention_model(input_shape, units=64, lr=0.001):
    """
    Builds an encoder-decoder LSTM model with attention mechanism.
    """
    inputs = Input(shape=input_shape)

    encoder = LSTM(units, return_sequences=True)(inputs)
    attention = Attention()([encoder, encoder])

    decoder = LSTM(units)(attention)
    outputs = Dense(1)(decoder)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(lr), loss='mse')

    return model


# ---------------------------
# Data Generation
# ---------------------------

np.random.seed(42)

time_steps = 500
t = np.arange(time_steps)

signal1 = np.sin(0.02 * t)
signal2 = np.cos(0.02 * t)
noise = np.random.normal(0, 0.1, time_steps)

data = np.vstack([signal1 + noise, signal2 + noise, signal1 + signal2]).T

df = pd.DataFrame(data, columns=["feature1", "feature2", "target"])
df.to_csv("dataset.csv", index=False)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values)


# ---------------------------
# Experiment Settings
# ---------------------------

sequence_lengths = [10, 20]
lstm_units = [32, 64]
learning_rates = [0.001, 0.0005]
splits = [0.7, 0.8, 0.9]

results = []


# ---------------------------
# Training Loop
# ---------------------------

for split in splits:
    train_size = int(len(scaled_data) * split)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    for seq in sequence_lengths:
        X_train, y_train = create_sequences(train_data, seq)
        X_test, y_test = create_sequences(test_data, seq)

        for units in lstm_units:
            for lr in learning_rates:

                # Baseline LSTM
                lstm_model = build_lstm_model((seq, 3), units, lr)
                lstm_model.fit(X_train, y_train, epochs=10, verbose=0)

                lstm_pred = lstm_model.predict(X_test)
                lstm_mae = mean_absolute_error(y_test, lstm_pred)
                lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_pred))
                lstm_mape = calculate_mape(y_test, lstm_pred)

                results.append(["LSTM", split, seq, units, lr,
                                lstm_mae, lstm_rmse, lstm_mape])

                # Attention Model
                att_model = build_attention_model((seq, 3), units, lr)
                att_model.fit(X_train, y_train, epochs=10, verbose=0)

                att_pred = att_model.predict(X_test)
                att_mae = mean_absolute_error(y_test, att_pred)
                att_rmse = np.sqrt(mean_squared_error(y_test, att_pred))
                att_mape = calculate_mape(y_test, att_pred)

                results.append(["Attention", split, seq, units, lr,
                                att_mae, att_rmse, att_mape])


# ---------------------------
# Save Results
# ---------------------------

df_results = pd.DataFrame(results, columns=[
    "Model", "Train_Split", "Seq_Length", "Units", "Learning_Rate",
    "MAE", "RMSE", "MAPE"
])

df_results.to_csv("results.csv", index=False)
print("Results saved to results.csv")


# ---------------------------
# Plot Sample Prediction
# ---------------------------

plt.figure(figsize=(10, 5))
plt.plot(y_test[:100], label="True Values")
plt.plot(att_pred[:100], label="Predictions")
plt.title("Attention Model Forecast vs True Values")
plt.legend()
plt.savefig("data_plot.png")
plt.show()

print("Plot saved as data_plot.png")
print("Execution completed successfully.")
