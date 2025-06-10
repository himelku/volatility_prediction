import pandas as pd
import numpy as np
from arch import arch_model

# Download the dataset
sp = pd.read_csv("SPY_15min_intraday.csv")
sp["timestamp"] = pd.to_datetime(sp["timestamp"])  # Ensure 'Date' is in datetime format

# Calculate daily log returns
sp["log_returns"] = np.log(sp["close"] / sp["close"].shift(1))

# Calculate rolling volatility using log returns
rolling_window_size = 10
sp["volatility"] = sp["log_returns"].rolling(window=rolling_window_size).std()

# Scale log returns for model input
sp["scaled_log_returns"] = sp["log_returns"] * 1000
sp.dropna(inplace=True)

# Filter data starting from 1985 to use the first 15 years to look for best parameteres
filtered_sp = sp.copy()

print(f"Number of data points: {len(filtered_sp)}")
print(filtered_sp.head())
print(filtered_sp.tail())
print(filtered_sp["scaled_log_returns"].describe())
print(f"Filtered data shape: {filtered_sp.shape}")
print(
    f"Start Date: {filtered_sp['timestamp'].min()} - End Date: {filtered_sp['timestamp'].max()}"
)


# Parameters choice
p_values = range(1, 4)
q_values = range(1, 4)

# Initialize variables to store the best model's parameters and score
best_aic = np.inf
best_p = None
best_q = None
best_model = None

# Iterate over possible values of p and q
for p in p_values:
    for q in q_values:
        try:
            # Fit the GARCH model
            model = arch_model(filtered_sp["scaled_log_returns"], vol="Garch", p=p, q=q)
            model_fit = model.fit(disp="off")
            print(f"Tried p={p}, q={q}, AIC={model_fit.aic}")
            # Check if this model has a better AIC score
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_p = p
                best_q = q
                best_model = model_fit
        except:  # Catch any errors during model fitting
            continue


# Print the best model's parameters and AIC
print(f"Best GARCH Model: p={best_p}, q={best_q}, AIC={best_aic}")
