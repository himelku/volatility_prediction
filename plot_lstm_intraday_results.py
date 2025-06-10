import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the results
results = pd.read_excel("results/results_lstm_intraday.xlsx")

# Plot Actual vs. Predicted Volatility
plt.figure(figsize=(14, 6))
plt.plot(results["test_date"], results["actual"], label="Actual Volatility", alpha=0.9)
plt.plot(
    results["test_date"], results["prediction"], label="Predicted Volatility", alpha=0.7
)
plt.xlabel("Test Date")
plt.ylabel("Volatility")
plt.title("Intraday LSTM - Actual vs. Predicted Volatility")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate Rolling Averages for Smoothing
window_size = 10
results["actual_smooth"] = results["actual"].rolling(window=window_size).mean()
results["prediction_smooth"] = results["prediction"].rolling(window=window_size).mean()

# Plot Smoothed Results
plt.figure(figsize=(14, 6))
plt.plot(
    results["test_date"],
    results["actual_smooth"],
    label="Actual Volatility (Smoothed)",
    alpha=0.9,
)
plt.plot(
    results["test_date"],
    results["prediction_smooth"],
    label="Predicted Volatility (Smoothed)",
    alpha=0.7,
)
plt.xlabel("Test Date")
plt.ylabel("Volatility")
plt.title("Intraday LSTM - Actual vs. Predicted Volatility (Smoothed)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate Error Metrics
rmse = mean_squared_error(results["actual"], results["prediction"], squared=False)
mae = mean_absolute_error(results["actual"], results["prediction"])

print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
