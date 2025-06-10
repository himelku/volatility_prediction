import pandas as pd
import numpy as np
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Define paths
data_dir = os.path.join("data")
input_file = os.path.join(data_dir, "SPY_15min_intraday.csv")
output_file = os.path.join(data_dir, "SPY_15min_lstm.csv")

# Load intraday data
sp = pd.read_csv(input_file)

# Rolling window size (approx. one trading day)
rolling_window_size = 26

# Calculate 15-min log returns
sp["log_returns"] = np.log(sp["close"] / sp["close"].shift(1))

# Calculate volatility as the standard deviation of log returns
sp["volatility"] = sp["log_returns"].rolling(window=rolling_window_size).std()

# Add lagged volatility
lag_days = 1
for i in range(1, lag_days + 1):
    sp[f"lagged_volatility_{i}"] = sp["volatility"].shift(i)

# Convert 'timestamp' to datetime
sp["Date"] = pd.to_datetime(sp["timestamp"])

# Drop unnecessary columns
sp.dropna(inplace=True)
sp.drop(columns=["open", "high", "low", "close", "volume", "timestamp"], inplace=True)

# Save processed data
sp.to_csv(output_file, index=False)
print(f"Processed data saved to: {output_file}")
