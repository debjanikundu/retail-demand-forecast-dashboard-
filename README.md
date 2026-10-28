# retail-demand-forecast-dashboard-

# 📦 Retail Demand Forecasting & Model Selection Dashboard

An interactive time-series forecasting and decision-support dashboard built using the **Walmart M5 Competition dataset**. The application automatically benchmarks classical statistical, generalized additive, and deep learning architectures per store/SKU to determine the best-performing model.

---

## 🚀 Key Features

- **Automated Model Tournament:** Compares **ARIMA**, **Facebook Prophet**, and **PyTorch LSTM** per product/store combination.
- **Uncertainty Bounds:** Generates 95% Confidence Intervals across all architectures (including Monte Carlo Dropout inference for LSTM).
- **Out-of-Sample Evaluation:** Computes **RMSE**, **MAE**, and **WAPE** (Weighted Absolute Percentage Error) over a 28-day forecast horizon.
- **Interactive UI:** Built with **Streamlit** and **Plotly** for granular series inspection, KPI monitoring, and error leaderboards.

---

## 🛠 Tech Stack

- **Frameworks:** Python, Streamlit, Plotly
- **Time Series & ML:** PyTorch, Facebook Prophet, Statsmodels, Scikit-learn
- **Data Engineering:** Pandas, NumPy

---

## ⚙️ Quickstart

### 1. Clone the repository
```bash
git clone [https://github.com/](https://github.com/)<your-username>/retail-demand-forecast-dashboard.git
cd retail-demand-forecast-dashboard
```

### 2. Set up virtual environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare data & launch app
```bash
# Generate processed sample from Kaggle M5 raw data (optional if sample included)
python prepare_m5.py

# Launch Streamlit dashboard
streamlit run app.py
```
