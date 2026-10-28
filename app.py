import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pipeline import get_or_generate_m5_data, run_forecast_benchmark

# Dashboard Configuration
st.set_page_config(page_title="Retail Demand Forecasting Hub", layout="wide")
st.title("📦 Retail Demand Forecasting Dashboard")
st.caption("Benchmark ARIMA vs. Prophet vs. PyTorch LSTM with Confidence Bounds (M5 Dataset)")

# Load Data
@st.cache_data
def load_data():
    return get_or_generate_m5_data()

df = load_data()

# Sidebar: Selection Controls
st.sidebar.header("🎯 Target Selection")
all_stores = sorted(df['store_id'].unique())
selected_store = st.sidebar.selectbox("Select Store", all_stores)

filtered_items = sorted(df[df['store_id'] == selected_store]['item_id'].unique())
selected_item = st.sidebar.selectbox("Select Product / SKU", filtered_items)

horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=7, max_value=60, value=28, step=7)
confidence_model = st.sidebar.radio("Display Model & Confidence Interval", ["Winner Only", "Prophet", "ARIMA", "LSTM", "All Models"])

# Trigger Pipeline
with st.spinner("Training models and calculating confidence intervals..."):
    results = run_forecast_benchmark(df, selected_store, selected_item, horizon=horizon)

train_df = results["train_df"]
test_df = results["test_df"]
leaderboard = results["leaderboard"]
winner = results["winner"]
model_outputs = results["models"]

# Metrics Highlights
col1, col2, col3, col4 = st.columns(4)
col1.metric("Selected Store", selected_store)
col2.metric("Selected SKU", selected_item)
col3.metric("Winning Model", winner)
best_rmse = leaderboard.loc[leaderboard['Model'] == winner, 'RMSE'].values[0]
col4.metric("Best RMSE", f"{best_rmse:.2f}")

# Main Chart Visualization (Plotly)
st.subheader("📈 Forecast vs Actual Demand")

fig = go.Figure()

# Plot historical actuals
recent_train = train_df.tail(60) # Show last 60 days of history for clarity
fig.add_trace(go.Scatter(
    x=recent_train['date'], y=recent_train['sales'],
    name="Historical (Train)", mode="lines", line=dict(color="#6c757d")
))

# Plot test actuals
fig.add_trace(go.Scatter(
    x=test_df['date'], y=test_df['sales'],
    name="Actual Demand (Test)", mode="lines+markers", line=dict(color="#000000", width=2.5)
))

palette = {
    "ARIMA": "#ff7f0e",
    "Prophet": "#1f77b4",
    "LSTM": "#2ca02c"
}

def add_model_traces(fig, name, data, dates, color, show_ci=False):
    fig.add_trace(go.Scatter(
        x=dates, y=data['pred'],
        name=f"{name} Forecast", mode="lines", line=dict(color=color, width=2)
    ))
    if show_ci:
        # Shaded Confidence Interval
        fig.add_trace(go.Scatter(
            x=list(dates) + list(dates)[::-1],
            y=list(data['upper']) + list(data['lower'])[::-1],
            fill='toself',
            fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.18])}",
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name=f"{name} 95% CI"
        ))

# Determine which models to plot
dates = test_df['date'].values
if confidence_model == "Winner Only":
    add_model_traces(fig, winner, model_outputs[winner], dates, palette[winner], show_ci=True)
elif confidence_model == "All Models":
    for m_name in ["ARIMA", "Prophet", "LSTM"]:
        add_model_traces(fig, m_name, model_outputs[m_name], dates, palette[m_name], show_ci=False)
else:
    add_model_traces(fig, confidence_model, model_outputs[confidence_model], dates, palette[confidence_model], show_ci=True)

fig.update_layout(
    height=500,
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title="Units Sold",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# Comparison Leaderboard
st.subheader("🏆 Model Evaluation Leaderboard")
st.dataframe(
    leaderboard.style.format({
        "RMSE": "{:.2f}",
        "MAE": "{:.2f}",
        "WAPE (%)": "{:.2f}%"
    }).highlight_min(subset=["RMSE", "MAE", "WAPE (%)"], color="#d4edda", axis=0),
    use_container_width=True
)

# Granular Forecast Data Table
with st.expander("🔍 View Raw Forecast Numbers & Confidence Intervals"):
    forecast_table = pd.DataFrame({"Date": test_df['date'].dt.strftime('%Y-%m-%d'), "Actual": test_df['sales'].values})
    for m in ["ARIMA", "Prophet", "LSTM"]:
        forecast_table[f"{m}_Pred"] = model_outputs[m]['pred'].round(1)
        forecast_table[f"{m}_Lower_95"] = model_outputs[m]['lower'].round(1)
        forecast_table[f"{m}_Upper_95"] = model_outputs[m]['upper'].round(1)
    st.dataframe(forecast_table.set_index("Date"), use_container_width=True)
