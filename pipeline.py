import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models import fit_predict_arima, fit_predict_prophet, fit_predict_lstm

def get_or_generate_m5_data(filepath="m5_sample.csv") -> pd.DataFrame:
    """
    Loads M5 / Rossmann formatted dataset or synthesizes a representative
    hierarchical daily sales dataset across multiple Stores and SKUs.
    """
    try:
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception:
        # Create a synthetic M5-like dataset: 2 stores, 3 products, 365 days
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=365, freq="D")
        stores = ["CA_1", "TX_1"]
        items = ["FOODS_3_001", "HOBBIES_1_004", "HOUSEHOLD_2_105"]
        
        records = []
        for store in stores:
            for item in items:
                base_level = np.random.randint(15, 60)
                trend = np.linspace(0, 15, len(dates))
                weekly_seasonality = 8 * np.sin(2 * np.pi * dates.dayofweek / 7)
                noise = np.random.normal(0, 4, len(dates))
                demand = np.maximum(0, (base_level + trend + weekly_seasonality + noise).round())
                
                for d, val in zip(dates, demand):
                    records.append({
                        "date": d,
                        "store_id": store,
                        "item_id": item,
                        "sales": val
                    })
        df = pd.DataFrame(records)
        df.to_csv("m5_sample.csv", index=False)
        return df

def evaluate_predictions(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # WAPE (Weighted Absolute Percentage Error) avoids division-by-zero on zero demand days
    wape = np.sum(np.abs(y_true - y_pred)) / (np.sum(y_true) + 1e-6) * 100
    return {"RMSE": rmse, "MAE": mae, "WAPE": wape}

def run_forecast_benchmark(df: pd.DataFrame, store_id: str, item_id: str, horizon: int = 28):
    """
    Runs ARIMA vs Prophet vs LSTM on a given store/SKU series and returns
    predictions, confidence bands, and the winning model by RMSE.
    """
    sub_df = df[(df['store_id'] == store_id) & (df['item_id'] == item_id)].sort_values('date').copy()
    
    train_df = sub_df.iloc[:-horizon]
    test_df = sub_df.iloc[-horizon:]
    
    # 1. ARIMA
    arima_res = fit_predict_arima(train_df['sales'], horizon=horizon)
    arima_metrics = evaluate_predictions(test_df['sales'].values, arima_res['pred'])
    
    # 2. Prophet
    prophet_train = train_df.rename(columns={'date': 'ds', 'sales': 'y'})
    prophet_res = fit_predict_prophet(prophet_train, horizon=horizon)
    prophet_metrics = evaluate_predictions(test_df['sales'].values, prophet_res['pred'])
    
    # 3. LSTM
    lstm_res = fit_predict_lstm(train_df['sales'], horizon=horizon)
    lstm_metrics = evaluate_predictions(test_df['sales'].values, lstm_res['pred'])
    
    # Compile comparison
    comparison_df = pd.DataFrame({
        "Model": ["ARIMA", "Prophet", "LSTM"],
        "RMSE": [arima_metrics["RMSE"], prophet_metrics["RMSE"], lstm_metrics["RMSE"]],
        "MAE": [arima_metrics["MAE"], prophet_metrics["MAE"], lstm_metrics["MAE"]],
        "WAPE (%)": [arima_metrics["WAPE"], prophet_metrics["WAPE"], lstm_metrics["WAPE"]]
    }).sort_values(by="RMSE").reset_index(drop=True)
    
    winner = comparison_df.iloc[0]["Model"]
    
    return {
        "train_df": train_df,
        "test_df": test_df,
        "models": {
            "ARIMA": arima_res,
            "Prophet": prophet_res,
            "LSTM": lstm_res
        },
        "leaderboard": comparison_df,
        "winner": winner
    }
