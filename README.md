#  End-to-End Inventory Optimization: Forecasting + Prescription

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Optimization](https://img.shields.io/badge/Domain-Operations_Research-orange)

##  Project Overview
Most data science projects stop at **Predictive Analytics** (e.g., "What will the demand be tomorrow?"). However, businesses ultimately need **Prescriptive Analytics** (e.g., "How many units should we actually stock today to maximize profit?"). 

This project bridges the gap between **Machine Learning (Forecasting)** and **Operations Research (Optimization)**. It builds an end-to-end intelligent replenishment system designed for high-risk, perishable goods (e.g., fresh groceries), where understocking leads to customer goodwill loss and overstocking leads to spoilage.

##  Business Scenario & Value
* **The Challenge:** Stocking the "average" forecasted demand (P50) naturally results in stockouts 50% of the time, alienating customers. 
* **The Solution:** By integrating the **Dynamic Newsvendor Model** with **Quantile Regression**, the system calculates the financially optimal service level and dynamically adjusts safety stock.
* **The Impact:** Through historical backtesting, the AI-driven P90 prescriptive strategy significantly outperforms traditional naive P50 replenishment, achieving a substantial **% Increase in Net Profit** while maintaining a >95% in-stock rate.

---

##  Core Methodology

### 1. Demand Forecasting (Data Science)
Instead of using Mean Squared Error (MSE) to predict a single average value, this project utilizes **LightGBM** with a **Pinball Loss Function** to predict the *probability distribution* of future demand.
* **Features Engineered:** Lagged sales (1, 7, 28 days), rolling window statistics (mean, std, max), calendar events, and price elasticity.
* **Output:** `P50` (Median baseline) and `P90` (Upper-bound safety stock).

### 2. Inventory Optimization (Operations Research)
Applies the **Newsvendor Model** to find the optimal balance between the Cost of Underage ($C_u$) and Cost of Overage ($C_o$).
* Computes the **Critical Ratio (CR)**: $CR = C_u / (C_u + C_o)$ based on user-defined financial inputs (Gross Margin, Holding Cost, Spoilage Rate).
* Maps the mathematical CR to the corresponding machine learning quantile prediction to prescribe the exact optimal order quantity ($Q^*$).

### 3. Backtesting Engine & Interactive UI
* **Simulation:** A daily ledger engine that mimics real-world operations (morning replenishment -> daytime sales -> nighttime settlement & spoilage calculation).
* **Dashboard:** A `Streamlit` web application allowing stakeholders to tweak financial parameters and visualize cumulative profit uplifts in real-time.

---

## 📂 Project Structure

```text
ETE_InventoryOptimization/
│
├── data/
│   ├── raw/                  # Original M5 Forecasting dataset (calendar, sales, prices)
│   └── processed/            # Generated features, model predictions, and backtest results
│
├── src/
│   ├── build_feature.py      # Time-series feature engineering (Lags & Rolling windows)
│   ├── train_model.py        # LightGBM Quantile Regression training (Pinball loss)
│   ├── backtest.py           # Financial simulation engine comparing Strategy A vs B
│   ├── features.py           # Create original dataframe
│   └── dashboard.py          # Interactive Streamlit frontend
