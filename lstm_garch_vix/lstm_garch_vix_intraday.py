# lstm_garch_vix_intraday.py

import pandas as pd
import numpy as np
import os
import sys
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import custom functions
from lstm.lstm_functions import create_dataset, should_retrain

# ------------------ Load Datasets ------------------
# Load SPY LSTM data
spy_lstm_path = os.path.join("data", "SPY_15min_lstm.csv")
spy_lstm = pd.read_csv(spy_lstm_path)
spy_lstm["Date"] = pd.to_datetime(spy_lstm["Date"])

# Load GARCH results
garch_path = os.path.join("data", "results_garch_intraday.csv")
garch_results = pd.read_csv(garch_path)
garch_results["Date"] = pd.to_datetime(garch_results["Date"])

# Load VIX data
vix_path = os.path.join("data", "vix_15min.csv")
vix = pd.read_csv(vix_path)
vix["Date"] = pd.to_datetime(vix["timestamp"])

# Print VIX columns for debugging
print("VIX columns:", vix.columns.tolist())

# Rename Close column to Close_vix if available
if "Close" in vix.columns:
    vix = vix.rename(columns={"Close": "Close_vix"})
elif "close" in vix.columns:
    vix = vix.rename(columns={"close": "Close_vix"})
else:
    raise KeyError("VIX file is missing 'Close' or 'close' column needed for merging.")

# Drop unneeded columns
vix = vix.drop(
    columns=["timestamp", "Open", "High", "Low", "Adj Close", "Volume"], errors="ignore"
)


# ------------------ Merge Datasets ------------------
# Merge SPY LSTM data with GARCH predictions
merged = pd.merge(spy_lstm, garch_results, on="Date", how="left")
merged = merged.rename(
    columns={
        "volatility_x": "volatility",  # main volatility from SPY LSTM
        "prediction": "predicted_volatility_garch",
    }
)
# Drop 'volatility_y' if it exists
if "volatility_y" in merged.columns:
    merged = merged.drop(columns=["volatility_y"])

# Sort before merging VIX
merged = merged.sort_values("Date").reset_index(drop=True)
vix = vix.sort_values("Date").reset_index(drop=True)

# Merge VIX using asof merge
merged = pd.merge_asof(merged, vix, on="Date", direction="backward")

# Check columns
print(f"Columns in merged dataset after merging VIX: {merged.columns.tolist()}")
print(merged.head())

# Drop rows with missing essential columns
essential_cols = ["volatility", "predicted_volatility_garch", "Close_vix"]
merged = merged.dropna(subset=essential_cols).reset_index(drop=True)
print(f"Merged dataset shape after dropping NA rows: {merged.shape}")


# ------------------ Setup Model ------------------
# Select features and target
# Keep only numerical feature columns
# Select feature columns
feature_columns = [
    col
    for col in merged.columns
    if col not in ["volatility", "Date", "timestamp"]
    and pd.api.types.is_numeric_dtype(merged[col])
]
print(f"Final feature columns: {feature_columns}")


target_column = "volatility"

# Define time_steps and train/validation sizes
time_steps = 22  # ~5.5 hours
steps_per_day = 26  # approximate 6.5-hour trading day
initial_train_size = 21 * steps_per_day  # ~1 month
validation_size = 7 * steps_per_day  # ~1 week

# Create dataset
X, y = create_dataset(
    merged[feature_columns], merged[target_column].values.reshape(-1, 1), time_steps
)
print(f"X shape: {X.shape}, y shape: {y.shape}")


# Define model creation function
def create_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, activation="tanh", return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# Model settings
input_shape = (time_steps, X.shape[2])
model_save_path = os.path.join("lstm_garch", "lstm_garch_vix_intraday.weights.h5")
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

# ------------------ Walk-Forward Validation ------------------
results = []
counter = 0

for i in range(len(merged) - initial_train_size - validation_size - 1):
    if i + initial_train_size + validation_size + time_steps > len(X):
        print("Not enough data for a complete test sequence. Ending.")
        break

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Fit scalers
    scaler_X.fit(X[i : i + initial_train_size].reshape(-1, X.shape[2]))
    scaler_y.fit(y[i : i + initial_train_size].reshape(-1, 1))

    # Transform and reshape
    train_X = (
        scaler_X.transform(X[i : i + initial_train_size].reshape(-1, X.shape[2]))
        .reshape(-1, time_steps, X.shape[2])
        .astype(np.float32)
    )
    train_y = (
        scaler_y.transform(y[i : i + initial_train_size].reshape(-1, 1))
        .reshape(-1, 1)
        .astype(np.float32)
    )

    val_X = (
        scaler_X.transform(
            X[
                i + initial_train_size : i + initial_train_size + validation_size
            ].reshape(-1, X.shape[2])
        )
        .reshape(-1, time_steps, X.shape[2])
        .astype(np.float32)
    )
    val_y = (
        scaler_y.transform(
            y[
                i + initial_train_size : i + initial_train_size + validation_size
            ].reshape(-1, 1)
        )
        .reshape(-1, 1)
        .astype(np.float32)
    )

    test_X = (
        scaler_X.transform(
            X[
                i
                + initial_train_size
                + validation_size : i
                + initial_train_size
                + validation_size
                + 1
            ].reshape(-1, X.shape[2])
        )
        .reshape(-1, time_steps, X.shape[2])
        .astype(np.float32)
    )
    test_y = (
        scaler_y.transform(
            y[
                i
                + initial_train_size
                + validation_size : i
                + initial_train_size
                + validation_size
                + 1
            ].reshape(-1, 1)
        )
        .reshape(-1, 1)
        .astype(np.float32)
    )

    # Train or update the model
    if should_retrain(counter) or not os.path.exists(model_save_path):
        model = create_model(input_shape)
        model.fit(
            train_X,
            train_y,
            epochs=50,
            batch_size=64,
            validation_data=(val_X, val_y),
            verbose=0,
            callbacks=[early_stopping],
        )
        model.save_weights(model_save_path)
    else:
        model = create_model(input_shape)
        model.load_weights(model_save_path)
        model.fit(
            train_X[-1].reshape(1, *train_X[-1].shape),
            train_y[-1].reshape(1, 1),
            epochs=1,
            verbose=0,
        )

    # Predict
    predicted = model.predict(test_X)
    predicted = scaler_y.inverse_transform(predicted.reshape(-1, 1))
    actual = scaler_y.inverse_transform(test_y.reshape(-1, 1))
    mae = mean_absolute_error(actual, predicted)

    current_result = {
        "train_start": merged["Date"][i + time_steps],
        "train_end": merged["Date"][i + initial_train_size + time_steps - 1],
        "validation_start": merged["Date"][i + initial_train_size + time_steps],
        "validation_end": merged["Date"][
            i + initial_train_size + validation_size + time_steps - 1
        ],
        "test_date": merged["Date"][
            i + initial_train_size + validation_size + time_steps
        ],
        "prediction": float(predicted.flatten()[0]),
        "actual": float(actual.flatten()[0]),
        "mae": mae,
    }
    print(current_result)
    results.append(current_result)
    counter += 1

# ------------------ Save Results ------------------
results_path = os.path.join("data", "results_lstm_garch_vix_intraday.csv")
lstm_garch_vix_results = pd.DataFrame(results)
lstm_garch_vix_results.to_csv(results_path, index=False)
print(f"LSTM-GARCH-VIX Intraday modeling complete. Results saved to {results_path}.")
