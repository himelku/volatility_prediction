
# Intraday LSTM-GARCH Modeling Pipeline 🚀

This guide documents all the steps, code, and reasoning we followed to implement the LSTM-GARCH volatility forecasting model on **intraday data** (15-minute frequency) in Python using Keras.

---

## 📦 Project Structure

```
volatility_prediction-main/
│
├── data/
│   └── SPY_15min_intraday.csv
│
├── lstm/
│   └── lstm_functions.py
│
├── lstm_garch/
│   └── lstm_garch_intraday.py
│
└── results/
```

---

## 📋 Step-by-Step Process

### 1️⃣ Import Packages

We use:
- `pandas`, `numpy`: Data handling.
- `keras`: LSTM modeling.
- `sklearn`: Scaling and metrics.

### 2️⃣ Import Custom Functions

```python
from lstm.lstm_functions import create_dataset, create_model, should_retrain
```

We fixed path issues by adding:
```python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

This ensures the script can find sibling folders like `lstm/`.

---

### 3️⃣ Load Dataset

We loaded the intraday dataset from:
```
data/SPY_15min_intraday.csv
```

Initially, encoding issues (`UnicodeDecodeError`) were resolved by specifying:
```python
df = pd.read_csv(data_path, sep=None, engine='python', encoding='latin1')
```

**Reasoning:**  
- `sep=None` allows pandas to auto-detect the delimiter.
- `encoding='latin1'` handles non-UTF-8 characters.

We then inspected the columns to locate the time column:
```python
print(df.columns)
```

---

### 4️⃣ Parse Date Column

We set:
```python
time_column = 'Date'
df['Date'] = pd.to_datetime(df[time_column])
df = df.sort_values('Date').reset_index(drop=True)
```

---

### 5️⃣ Define Features and Target

```python
feature_columns = [col for col in df.columns if col not in ['volatility', 'Date']]
target_column = 'volatility'
```

---

### 6️⃣ Define Model Parameters

Original settings were too large for our 1338-row dataset. We reduced:

```python
steps_per_day = 26  # ~6.5 hours/day
time_steps = 22     # ~5.5 hours

initial_train_size = 21 * steps_per_day  # 1 month
validation_size = 7 * steps_per_day      # 1 week
```

**Reasoning:**  
This ensures enough data for training, validation, and testing.

---

### 7️⃣ Create the Dataset

```python
X, y = create_dataset(df[feature_columns], df[target_column].values.reshape(-1, 1), time_steps)
```

**Explanation:**  
- `X.shape`: (samples, time_steps, features)
- `y.shape`: (samples, 1)

---

### 8️⃣ Define Model Save Path

We updated:
```python
model_save_path = os.path.join('lstm_garch', 'lstm_garch_intraday.weights.h5')
```

**Reasoning:**  
Keras requires `.weights.h5` extension to store model weights.

---

### 9️⃣ Train and Predict

The main loop:
```python
for i in range(len(df) - initial_train_size - validation_size - 1):
    ...
```

Inside:
- Scale features with `MinMaxScaler`.
- Train model with EarlyStopping on train+validation sets.
- Save model weights after training.
- Predict next step using the test set.
- Store predictions in results.

**Conditional Retraining:**  
- If the model is new or based on `should_retrain(counter)`, it retrains on the training set.
- Otherwise, it does one-step incremental training.

---

### 🔟 Save Results

```python
results_path = os.path.join('results', 'results_lstm_garch_intraday.csv')
lstm_garch_results.to_csv(results_path, index=False)
```

---

## 🚀 How to Run

1️⃣ Navigate to the project root:

```bash
cd volatility_prediction-main
```

2️⃣ Activate your environment (if applicable):

```bash
conda activate myenv
```

3️⃣ Run:

```bash
python -m lstm_garch.lstm_garch_intraday
```

**OR:**

```bash
python lstm_garch/lstm_garch_intraday.py
```

---

## 📊 Notes

- Make sure the target column **`volatility`** exists in your dataset.
- Adjust **`time_steps`**, **`initial_train_size`**, and **`validation_size`** as needed.
- Ensure that `lstm_functions.py` contains:
  - `create_dataset`
  - `create_model`
  - `should_retrain`
- Check that `results/` directory exists.

---

## 🎯 Summary

This guide walked you through:
✅ Loading intraday data  
✅ Handling encoding issues  
✅ Fixing file path issues  
✅ Adjusting dataset size  
✅ Saving model weights with correct file extension  
✅ Ensuring the loop runs properly with available data  
✅ Exporting results

---

Happy modeling! 📈✨
