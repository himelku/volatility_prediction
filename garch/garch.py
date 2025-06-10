# garch.py
import pandas as pd
import numpy as np
import os
import sys
from arch import arch_model
import matplotlib.pyplot as plt

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from garch.garch_functions import garch_data_prep, train_garch_model

# Load the intraday data
data_path = os.path.join("data", "SPY_15min_intraday.csv")
df = pd.read_csv(data_path)

# Add 'Date' column
df["Date"] = pd.to_datetime(df["timestamp"])

# Prepare the data
sp = garch_data_prep(df)

# Choose a start date that makes sense
start_date = "2025-05-07 06:30:00"  # Adjust if needed

print(f"Data range: {sp['Date'].min()} to {sp['Date'].max()}")
print(f"Using start_date: {start_date}")

# Train the GARCH model
predicted_volatility = train_garch_model(sp, start_date)

# Prepare actual and predicted volatility for comparison
actual_volatility = sp[["Date", "volatility"]].set_index("Date")
actual_volatility = actual_volatility.rename(columns={"volatility": "actual"})

# Merge
merged_df = sp.merge(actual_volatility, on="Date", how="outer").merge(
    predicted_volatility, on="Date", how="outer"
)
merged_df.dropna(inplace=True)

# Drop unneeded columns
garch_results = merged_df.drop(
    columns=["open", "high", "low", "volume", "scaled_log_returns", "log_returns"],
    errors="ignore",
)

print(garch_results)

# Save results
results_path = os.path.join("data", "results_garch_intraday.csv")
garch_results.to_csv(results_path, index=False)
print(f"GARCH results saved to {results_path}")

# Plot
plt.figure(figsize=(14, 6))
plt.plot(garch_results["Date"], garch_results["actual"], label="Actual Volatility")
plt.plot(
    garch_results["Date"],
    garch_results["prediction"],
    label="Predicted Volatility",
    alpha=0.7,
)
plt.legend()
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.title("Intraday Volatility: Actual vs. GARCH Predicted")
plt.grid()
plt.show()
