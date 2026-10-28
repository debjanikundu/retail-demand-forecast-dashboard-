import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. ARIMA Wrapper
# ---------------------------------------------------------
def fit_predict_arima(train_series: pd.Series, horizon: int, order=(1, 1, 1)):
    """
    Fits an ARIMA model and outputs point forecast and 95% confidence intervals.
    """
    model = ARIMA(train_series.values, order=order)
    fitted = model.fit()
    forecast_res = fitted.get_forecast(steps=horizon)
    point_pred = forecast_res.predicted_mean
    ci = forecast_res.conf_int(alpha=0.05)
    
    return {
        "pred": np.maximum(0, point_pred),
        "lower": np.maximum(0, ci[:, 0]),
        "upper": np.maximum(0, ci[:, 1])
    }

# ---------------------------------------------------------
# 2. Prophet Wrapper
# ---------------------------------------------------------
def fit_predict_prophet(train_df: pd.DataFrame, horizon: int):
    """
    Fits Facebook Prophet with weekly/yearly seasonality and 95% uncertainty intervals.
    train_df must contain 'ds' (datetime) and 'y' (target).
    """
    m = Prophet(
        interval_width=0.95,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    m.fit(train_df[['ds', 'y']])
    future = m.make_future_dataframe(periods=horizon, freq='D')
    forecast = m.predict(future).iloc[-horizon:]
    
    return {
        "pred": np.maximum(0, forecast['yhat'].values),
        "lower": np.maximum(0, forecast['yhat_lower'].values),
        "upper": np.maximum(0, forecast['yhat_upper'].values)
    }

# ---------------------------------------------------------
# 3. PyTorch LSTM with Monte Carlo Dropout for CIs
# ---------------------------------------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i + seq_len])
        ys.append(data[i + seq_len])
    return np.array(xs), np.array(ys)

def fit_predict_lstm(train_series: pd.Series, horizon: int, seq_len: int = 14, epochs: int = 35, n_mc_samples: int = 50):
    """
    Fits a PyTorch LSTM model. Computes 95% CI via Monte Carlo Dropout during inference.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(train_series.values.reshape(-1, 1))
    
    if len(scaled_data) <= seq_len + 5:
        # Fallback if series is too short
        pred_val = train_series.mean()
        return {
            "pred": np.full(horizon, pred_val),
            "lower": np.full(horizon, max(0, pred_val * 0.7)),
            "upper": np.full(horizon, pred_val * 1.3)
        }
        
    X, y = create_sequences(scaled_data, seq_len)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    model = LSTMForecaster()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_tensor)
        loss = criterion(out, y_tensor)
        loss.backward()
        optimizer.step()

    # Autoregressive roll-out forecast using Monte Carlo Dropout
    model.train()  # Keep dropout active for MC inference
    mc_forecasts = []
    
    for _ in range(n_mc_samples):
        current_seq = scaled_data[-seq_len:].copy()
        seq_preds = []
        for _ in range(horizon):
            inp = torch.tensor(current_seq.reshape(1, seq_len, 1), dtype=torch.float32)
            with torch.no_grad():
                pred = model(inp).item()
            seq_preds.append(pred)
            current_seq = np.append(current_seq[1:], [[pred]], axis=0)
        mc_forecasts.append(scaler.inverse_transform(np.array(seq_preds).reshape(-1, 1)).flatten())

    mc_forecasts = np.array(mc_forecasts)
    mean_pred = np.mean(mc_forecasts, axis=0)
    lower_ci = np.percentile(mc_forecasts, 2.5, axis=0)
    upper_ci = np.percentile(mc_forecasts, 97.5, axis=0)

    return {
        "pred": np.maximum(0, mean_pred),
        "lower": np.maximum(0, lower_ci),
        "upper": np.maximum(0, upper_ci)
    }
