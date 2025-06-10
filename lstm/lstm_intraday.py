import pandas as pd
import numpy as np
import os
import sys
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lstm.lstm_functions import create_dataset, should_retrain, create_model

# Load intraday LSTM data
data_path = os.path.join("data", "SPY_15min_lstm.csv")
df = pd.read_csv(data_path)
print(f"Total data points: {len(df)}")

# Convert 'Date' to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Setup features and target
feature_columns = [col for col in df.columns if col not in ["volatility", "Date"]]
target_column = "volatility"

# Define time_steps (1 trading day in 15-min bars)
time_steps = 26

# Create dataset
X, y = create_dataset(
    df[feature_columns], df[target_column].values.reshape(-1, 1), time_steps
)

input_shape = (time_steps, X.shape[2])
model_save_path = os.path.join("lstm_intraday.weights.h5")

# Early stopping callback
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

# Approximate training size
initial_train_size = 1000
validation_size = 250

print(f"Initial Train Size: {initial_train_size}")
print(f"Validation Size: {validation_size}")

# Check if enough data is available
if len(df) < initial_train_size + validation_size + time_steps + 1:
    print(
        "Not enough data for training + validation + test. Please reduce split sizes or collect more data."
    )
    exit()

results = []
counter = 0

# Walk-forward training loop
for i in range(len(df) - initial_train_size - validation_size - 1):
    if i + initial_train_size + validation_size + time_steps > len(X):
        break

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Fit scalers on training data
    scaler_X.fit(X[i : i + initial_train_size].reshape(-1, X.shape[2]))
    scaler_y.fit(y[i : i + initial_train_size].reshape(-1, 1))

    # Transform train/validation/test sets
    train_X = scaler_X.transform(
        X[i : i + initial_train_size].reshape(-1, X.shape[2])
    ).reshape(-1, time_steps, X.shape[2])
    train_y = scaler_y.transform(y[i : i + initial_train_size].reshape(-1, 1)).reshape(
        -1, 1
    )

    val_X = scaler_X.transform(
        X[i + initial_train_size : i + initial_train_size + validation_size].reshape(
            -1, X.shape[2]
        )
    ).reshape(-1, time_steps, X.shape[2])
    val_y = scaler_y.transform(
        y[i + initial_train_size : i + initial_train_size + validation_size].reshape(
            -1, 1
        )
    ).reshape(-1, 1)

    test_X = scaler_X.transform(
        X[
            i
            + initial_train_size
            + validation_size : i
            + initial_train_size
            + validation_size
            + 1
        ].reshape(-1, X.shape[2])
    ).reshape(-1, time_steps, X.shape[2])
    test_y = scaler_y.transform(
        y[
            i
            + initial_train_size
            + validation_size : i
            + initial_train_size
            + validation_size
            + 1
        ].reshape(-1, 1)
    ).reshape(-1, 1)

    # Train model
    if should_retrain(counter, interval=252 * 26) or not os.path.exists(
        model_save_path
    ):
        model = create_model(input_shape)
        model.fit(
            train_X,
            train_y,
            epochs=100,
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
        "prediction": predicted.flatten()[0],
        "actual": actual.flatten()[0],
        "mae": mae,
    }
    print(current_result)
    results.append(current_result)
    counter += 1

# Save results
results_dir = os.path.join("data")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

results_path = os.path.join(results_dir, "results_lstm_intraday.csv")
lstm_results = pd.DataFrame(results)
lstm_results.to_csv(results_path, index=False)
print(f"Results saved to {results_path}")
