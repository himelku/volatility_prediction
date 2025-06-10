# fetch_intraday_data.py
import os
import time
import pandas as pd
import requests
from io import StringIO

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

# Alpha Vantage API key
api_key = "API_KEY"
symbol = "SPY"
interval = "15min"
output_file = f"{symbol}_{interval}_intraday.csv"

# Fetch recent intraday data (outputsize=full)
url = (
    f"https://www.alphavantage.co/query"
    f"?function=TIME_SERIES_INTRADAY"
    f"&symbol={symbol}"
    f"&interval={interval}"
    f"&outputsize=full"
    f"&apikey={api_key}"
    f"&datatype=csv"
)

print(f"Fetching {symbol} {interval} intraday data (latest)...")
response = requests.get(url)
if response.status_code != 200:
    print(f"Request failed with status {response.status_code}.")
else:
    content = response.text
    if "timestamp" not in content:
        print(f"No data returned. Response: {content[:500]}")
    else:
        df = pd.read_csv(StringIO(content))
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.sort_values(by="timestamp", inplace=True)
        df.to_csv(output_file, index=False)
        print(f"Data saved to '{output_file}'.")
