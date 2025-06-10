import pandas as pd
import numpy as np
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lstm.lstm_functions import create_dataset

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch

# Load intraday data
data_path = os.path.join("data", "SPY_15min_lstm.csv")
df = pd.read_excel(data_path)
df["Date"] = pd.to_datetime(df["Date"])

# Prepare features and target
feature_columns = [col for col in df.columns if col not in ["volatility", "Date"]]
target_column = "volatility"

# Set time_steps and dataset sizes
time_steps = 26
X, y = create_dataset(
    df[feature_columns], df[target_column].values.reshape(-1, 1), time_steps
)
input_shape = (time_steps, X.shape[2])

# Define index for one-year worth of intraday data
index_1yr = 252 * 26  # 252 trading days, 26 intervals per day

# Subset the data
tune_X, tune_y = X[:index_1yr], y[:index_1yr]


# Define the model building function
def build_model(hp):
    model = Sequential()
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            LSTM(
                units=hp.Choice(f"units_lstm_{i}", [32, 64, 128]),
                activation=hp.Choice(f"activation_{i}", ["tanh", "relu"]),
                return_sequences=True if i < hp.get("num_layers") - 1 else False,
                input_shape=input_shape if i == 0 else None,
            )
        )
        model.add(
            Dropout(
                rate=hp.Float(f"dropout_{i}", min_value=0.0, max_value=0.3, step=0.1)
            )
        )
    model.add(Dense(1, activation="relu"))
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice("learning_rate", [0.01, 0.001, 0.0001])),
        loss=hp.Choice("loss", ["mean_squared_error", "mean_absolute_error"]),
    )
    return model


# Setup Hyperparameter Tuner
tuner = RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=20,
    executions_per_trial=3,
    directory="model_tuning_intraday",
    project_name="LSTM_Tuning_Intraday",
)

# Early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

# Run the tuner
tuner.search(
    tune_X,
    tune_y,
    epochs=50,
    validation_split=0.6,
    callbacks=[early_stopping],
    verbose=1,
)

# Print the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters: {best_hps.values}")
