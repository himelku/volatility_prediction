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


# Define a more robust create_model function
def create_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, activation="tanh", return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# Load intraday dataset
data_path = os.path.join("data", "SPY_15min_lstm.csv")
df = pd.read_csv(data_path)
print(f"Total data points: {len(df)}")

# Print columns to verify
print("Columns found in the csv file:")
print(df.columns)

# Parse DateTime with timestamps
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print(f"Data loaded: {df.shape[0]} rows, columns: {df.columns.tolist()}")
print(f"Sample data:\n{df.head()}")

# Select feature columns
feature_columns = [col for col in df.columns if col not in ["volatility", "Date"]]
target_column = "volatility"  # Adjust this as needed

# Set time_steps and dataset sizes
time_steps = 22  # ~5.5 hours
steps_per_day = 26  # approximate 6.5-hour trading day

# Adjust training and validation sizes
initial_train_size = 21 * steps_per_day  # 1 month
validation_size = 7 * steps_per_day  # 1 week

# Create dataset
X, y = create_dataset(
    df[feature_columns], df[target_column].values.reshape(-1, 1), time_steps
)
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Model settings
input_shape = (time_steps, X.shape[2])
model_save_path = os.path.join("lstm_garch", "lstm_garch_intraday.weights.h5")
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

results = []
counter = 0

for i in range(len(df) - initial_train_size - validation_size - 1):
    if i + initial_train_size + validation_size + time_steps > len(X):
        print("Not enough data for a complete test sequence. Ending.")
        break

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Fit scalers
    scaler_X.fit(X[i : i + initial_train_size].reshape(-1, X.shape[2]))
    scaler_y.fit(y[i : i + initial_train_size].reshape(-1, 1))

    # Transform and cast to float32
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

    # Check for NaNs/Infs
    for name, arr in zip(
        ["train_X", "train_y", "val_X", "val_y", "test_X", "test_y"],
        [train_X, train_y, val_X, val_y, test_X, test_y],
    ):
        if np.isnan(arr).any() or np.isinf(arr).any():
            print(f"⚠️ Warning: {name} contains NaNs or Infs!")

    # Train model
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

    # Predict and inverse transform
    predicted = model.predict(test_X)
    predicted = scaler_y.inverse_transform(predicted.reshape(-1, 1))
    actual = scaler_y.inverse_transform(test_y.reshape(-1, 1))
    mae = mean_absolute_error(actual, predicted)

    current_result = {
        "train_start": df["Date"][i + time_steps],
        "train_end": df["Date"][i + initial_train_size + time_steps - 1],
        "validation_start": df["Date"][i + initial_train_size + time_steps],
        "validation_end": df["Date"][
            i + initial_train_size + validation_size + time_steps - 1
        ],
        "test_date": df["Date"][i + initial_train_size + validation_size + time_steps],
        "prediction": float(predicted.flatten()[0]),
        "actual": float(actual.flatten()[0]),
        "mae": mae,
    }
    print(current_result)
    results.append(current_result)
    counter += 1

# Save results
results_path = os.path.join("data", "results_lstm_garch_intraday.csv")
lstm_garch_results = pd.DataFrame(results)
lstm_garch_results.to_csv(results_path, index=False)
print(f"Intraday LSTM-GARCH modeling complete. Results saved to {results_path}.")
