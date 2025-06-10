import requests
import pandas as pd
import time

# Alpha Vantage API Key
api_key = "API_KEY"  # Replace with your actual API key

# VIX Proxy Ticker
symbol = "VXX"  # Or use 'UVXY' or other VIX proxies

# API parameters
interval = "15min"
outputsize = "full"  # to get full intraday data history
datatype = "csv"

url = (
    f"https://www.alphavantage.co/query"
    f"?function=TIME_SERIES_INTRADAY"
    f"&symbol={symbol}"
    f"&interval={interval}"
    f"&outputsize={outputsize}"
    f"&apikey={api_key}"
    f"&datatype={datatype}"
)

# Fetching data
print(f"Fetching {symbol} {interval} intraday data from Alpha Vantage...")

response = requests.get(url)

if response.status_code == 200:
    content = response.text
    with open("data/vix_15min.csv", "w") as f:
        f.write(content)
    print(f"Data saved to 'data/vix_15min.csv'.")
else:
    print(f"Error: {response.status_code}")

# Optional: Sleep between calls to respect API rate limits
time.sleep(12)  # Alpha Vantage allows 5 calls/minute, so 12 seconds ensures compliance
