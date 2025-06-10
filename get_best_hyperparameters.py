import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras_tuner.tuners import RandomSearch

from lstm.lstm_functions import create_dataset  # reuse any needed utils


# Dummy build_model function to match the tuner structure
def build_model(hp):
    model = Sequential()
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            LSTM(
                units=hp.Choice("units_lstm_" + str(i), [32, 64, 128]),
                activation=hp.Choice("activation_" + str(i), ["tanh", "relu"]),
                return_sequences=True if i < hp.get("num_layers") - 1 else False,
                input_shape=(26, 1),
            )
        )  # Placeholder shape
        model.add(
            Dropout(
                rate=hp.Float(
                    "dropout_" + str(i), min_value=0.0, max_value=0.3, step=0.1
                )
            )
        )
    model.add(Dense(1, activation="relu"))
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice("learning_rate", [0.01, 0.001, 0.0001])),
        loss=hp.Choice("loss", ["mean_squared_error", "mean_absolute_error"]),
    )
    return model


# Load the tuner
tuner = RandomSearch(
    build_model,
    objective="val_loss",
    directory="model_tuning_intraday",
    project_name="LSTM_Tuning_Intraday",
)

# Fetch the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best hyperparameters found:")
for param, value in best_hps.values.items():
    print(f"{param}: {value}")
