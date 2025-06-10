# garch_functions.py
import numpy as np
import pandas as pd
from arch import arch_model


def garch_data_prep(df):
    df = df.copy()
    df = df.sort_values("Date")
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = df["log_returns"].rolling(window=10).std()
    df.dropna(subset=["log_returns", "volatility"], inplace=True)
    return df


def train_garch_model(df, start_date):
    """
    Fits a GARCH(1,1) model on the log returns from start_date onward.
    """
    train_data = df[df["Date"] >= start_date]
    if train_data.empty:
        raise ValueError(
            f"Not enough training data (only {len(train_data)} rows). "
            f"Available range: {df['Date'].min()} to {df['Date'].max()}"
        )

    model = arch_model(train_data["log_returns"], vol="Garch", p=1, q=1)
    fitted_model = model.fit(disp="off")

    # Get predicted volatility
    predicted_volatility = fitted_model.conditional_volatility
    results = pd.DataFrame(
        {"Date": train_data["Date"], "prediction": predicted_volatility}
    )

    return results
