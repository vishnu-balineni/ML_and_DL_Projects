# Weather Prediction Project

This project predicts weather temperature using **Random Forest (RF)**, **XGBoost (XGB)**, and **LSTM** models. It demonstrates traditional machine learning as well as deep learning approaches for time series prediction.

## Project Overview

- **Objective:** Predict hourly/daily temperature using historical weather data.  
- **Techniques Used:** Random Forest, XGBoost, LSTM (Deep Learning), Feature Scaling, Data Preprocessing.  
- **Visualization:** Heatmaps, Feature Importance, and Prediction vs Actual charts.  
- **Manual Testing:** Models can be tested by giving manual input.  

## Files in this Repository

| File | Description |
|------|-------------|
| `rf_model.pkl` | Trained Random Forest model |
| `xgb_model.pkl` | Trained XGBoost model |
| `lstm_model.h5` | Trained LSTM model |
| `scaler_ml.pkl` | Scaler for ML models (RF & XGB) |
| `scaler_lstm.pkl` | Scaler for LSTM input/output |
| `streamlit_app.py` | (Optional) Streamlit app to interact with models |
| `README.md` | This file |

## Dataset

- The dataset contains hourly weather observations with columns like:
  - `Temperature`, `Apparent Temperature`, `Humidity`
  - `Wind Speed`, `Wind Bearing`, `Visibility`, `Pressure`
  - `Precipitation Type`, `Date/Time`
- Used for both training ML models and LSTM for time series prediction.

## How to Run

1.
 **Install Dependencies:**
```bash
pip install -r requirements.txt

2.
 **Load Models and Make Predictions:**
```python
import joblib
import tensorflow as tf
import numpy as np

# Load ML models
rf_model = joblib.load('rf_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')
scaler_ml = joblib.load('scaler_ml.pkl')

# Load LSTM model
lstm_model = tf.keras.models.load_model('lstm_model.h5')
scaler_lstm = joblib.load('scaler_lstm.pkl')

3. 
**Give Manual Input for Prediction:**
- Provide feature values (Temperature, Humidity, Wind, etc.) for RF/XGB.  
- Provide last N days of temperature for LSTM.  
- Models will output predicted temperature.

```python
# Example input for ML models
manual_input = [10.0, 0.5, 12.0, 180.0, 10.0, 5.0, 1013.0, 0, 14, 23, 10, 2025, 3]
X_input_scaled = scaler_ml.transform([manual_input])
rf_pred = rf_model.predict(X_input_scaled)[0]
xgb_pred = xgb_model.predict(X_input_scaled)[0]
print(f"RF: {rf_pred:.2f} °C, XGB: {xgb_pred:.2f} °C")

# Example for LSTM
last_30_days = [10,12,11,13,12,14,13,12,11,10,12,11,13,12,14,
                13,12,11,10,12,11,13,12,14,13,12,11,10,12,11]
lstm_input_scaled = scaler_lstm.transform(np.array(last_30_days).reshape(-1,1))
lstm_input_scaled = lstm_input_scaled.reshape(1,30,1)
lstm_pred_scaled = lstm_model.predict(lstm_input_scaled)
lstm_pred = scaler_lstm.inverse_transform(lstm_pred_scaled)
print(f"LSTM: {lstm_pred[0][0]:.2f} °C")

## Visualization

Include screenshots or plots of:
- Heatmaps  
- Feature Importance  
- Prediction vs Actual charts  

Example:
![Heatmap](heatmap.png)
![Prediction vs Actual](prediction_chart.png)


## Future Improvements

- Deploy a fully interactive Streamlit web app.  
- Add additional weather features like rainfall probability.  
- Experiment with advanced deep learning models like GRU or Transformer-based forecasting.  
- Automate data preprocessing for new datasets.


