
# Volatility Forecasting Using LSTM on Intraday Data

This project implements a volatility forecasting pipeline using LSTM (Long Short-Term Memory) neural networks on 15-minute intraday SPY data. The project is structured with modular, reusable scripts for data preparation, training, hyperparameter tuning, and evaluation.

---

## 📂 Project Structure

```
volatility_prediction-main/
│
├── data/
│   └── SPY_15min_intraday.csv          # Raw intraday data
│
├── results/
│   └── results_lstm_intraday.xlsx      # Predictions and actual values
│
├── lstm/
│   ├── __init__.py                     # Marks lstm/ as a package
│   ├── lstm_functions.py               # Utility functions for dataset creation, etc.
│   ├── prepare_data_lstm_intraday.py   # Data preparation script
│   ├── lstm_intraday.py                # Walk-forward LSTM training and prediction
│   └── lstm_hyperparameters_tuning_intraday.py  # Hyperparameter tuning script
│
├── get_best_hyperparameters.py         # Script to print best hyperparameters
└── plot_lstm_intraday_results.py       # Script to plot actual vs. predicted volatility
```

---

## 🚀 Features

✅ Rolling volatility calculation on 15-minute data  
✅ Walk-forward LSTM training and prediction  
✅ Hyperparameter tuning with Keras Tuner  
✅ Smoothed visualization of predicted vs. actual volatility  
✅ Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) metrics  
✅ Modular design with separate scripts for each step  
✅ Package structure for easy imports  

---

## 🔧 Setup Instructions

1️⃣ Clone the repository or organize files according to the structure above.  
2️⃣ Create a virtual environment:
```bash
python -m venv venv
```

3️⃣ Activate the virtual environment:
```bash
# Windows:
.env\Scriptsctivate
```

4️⃣ Install dependencies:
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow keras keras-tuner openpyxl
```

5️⃣ Ensure the `data/` folder contains:
- `SPY_15min_intraday.csv`: the raw intraday data file.

---

## 📝 Step-by-Step Usage

### 1️⃣ Prepare the Data

Generates a cleaned dataset with log returns and rolling volatility:

```bash
python -m lstm.prepare_data_lstm_intraday
```

Output: `data/SPY_15min_lstm.xlsx`

---

### 2️⃣ (Optional) Hyperparameter Tuning

Uses Keras Tuner to search for optimal LSTM configurations:

```bash
python -m lstm.lstm_hyperparameters_tuning_intraday
```

Monitor the trials printed in the console. You can interrupt with `Ctrl+C` at any time to stop tuning.

---

### 3️⃣ Retrieve Best Hyperparameters

After tuning completes (or after you stop it), extract the best hyperparameters:

```bash
python get_best_hyperparameters.py
```

Use the printed values to update your model architecture in `lstm/lstm_intraday.py`.

---

### 4️⃣ Train the LSTM Model

Run the main training script with walk-forward validation:

```bash
python -m lstm.lstm_intraday
```

Outputs:
- `results/results_lstm_intraday.xlsx`: containing actual vs. predicted volatility.

---

### 5️⃣ Plot the Results

Visualize model performance:

```bash
python plot_lstm_intraday_results.py
```

Shows:
- Raw actual vs. predicted volatility  
- Smoothed actual vs. predicted volatility  
- RMSE and MAE metrics

---

## 🛠️ Adjustments You Can Make

✅ **Time Steps**  
- Default: `time_steps = 26` (~1 trading day of 15-min data)  
- Increase or decrease in `lstm_intraday.py` depending on your model’s memory.

✅ **Training and Validation Sizes**  
- Adjust `initial_train_size` and `validation_size` in `lstm_intraday.py`.  
- Default: `1000` and `250` respectively.

✅ **Hyperparameters**  
- Integrate the best hyperparameters from `get_best_hyperparameters.py` into `create_model()` in `lstm_intraday.py`.

✅ **Data Scaling**  
- If DataScaleWarnings appear in GARCH or LSTM, scale your features by multiplying by 1000 or 10000 in the data preparation step.

✅ **Model Saving**  
- Keras requires that weights files end in `.weights.h5`.  
- Update all saves to use this convention:
  ```python
  model.save_weights('lstm_intraday.weights.h5')
  ```

---

## 🔍 What Else Can Be Done?

✨ **Expand Feature Engineering**  
- Add technical indicators like RSI, moving averages, or volume-based features.

✨ **Include Exogenous Variables**  
- Incorporate market news or economic indicators.

✨ **Test Different Architectures**  
- Try GRU, Bi-LSTM, or Transformer models.

✨ **Implement Early Stopping Globally**  
- Stop the entire hyperparameter search if overall improvement stagnates.

✨ **Save and Load Models Automatically**  
- Save the best model’s architecture using `model.save()` instead of just weights.

✨ **Use Cross-Validation**  
- Instead of rolling windows, implement time series cross-validation frameworks.

✨ **Deploy a Dashboard**  
- Use Streamlit or Dash to visualize live volatility forecasts.

---

## ⚡️ Notes

✅ Make sure to create `results/` and `data/` folders before running scripts.  
✅ Use consistent file paths to avoid `FileNotFoundError`.  
✅ Adjust hyperparameters based on your actual dataset size and business goals.

---

## 🙌 Acknowledgements

Thanks to the iterative conversations and your attention to detail that made this pipeline robust and adaptable! If you’d like to extend it further or integrate new features, just let me know! 🚀✨

---
