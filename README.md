# US-Dividend Quality Screener (Bio-Stats x Finance Edition) 🧬📈
### AI-Powered Portfolio Construction with Survival Analysis & Machine Learning

## Overview
This is a quantitative analysis tool designed to construct a **"Low-Volatility, High-Quality Dividend Portfolio"**.
By applying **Survival Analysis (Kaplan-Meier & Cox Proportional Hazards)**—methods traditionally used in medical research—to financial data, this tool quantitatively predicts the "lifespan" of a company's dividend sustainability.

🚀 **New Feature:** The project now includes an **Interactive Web Dashboard (Streamlit)**, allowing users to visualize strategies, perform A/B testing, and generate future portfolio predictions (2026) with a single click.

## 💡 Core Concept: "Bio-Statistics meets Finance"
In this project, I reinterpreted financial events through the lens of biological survival:
* **Patient Death** $\rightarrow$ **Dividend Cut / Stagnation**
* **Survival Duration** $\rightarrow$ **Financial Runway (Inverse of Payout Ratio)**
* **Risk Factors (Covariates)** $\rightarrow$ **Financial Metrics (Margin, ROE, Yield)**

Just as doctors predict patient survival probabilities, this tool calculates a **"Hazard Score"** for every S&P 500 company to filter out future "value traps."

## Features
1. **Bio-Statistical Risk Assessment (Survival Analysis):**
    * **Kaplan-Meier Estimator:** Visualizes the "survival probability" (dividend maintenance rate) of the market over time.
    * **Cox Proportional Hazards Model:** Calculates a proprietary **Hazard Score** for each stock. High-risk companies (e.g., high yield traps, declining margins) are penalized mathematically before they cut dividends.
    * **Library:** Implemented using `lifelines` (Python).

2. **📊 Interactive Web Dashboard (Streamlit):**
    * **No Coding Required:** Run simulations via a GUI.
    * **Real-time Visualization:** View cumulative returns, drawdowns, and survival curves instantly.
    * **A/B Testing Mode:** Compare the "AI Strategy" directly against a "Simple High-Yield Strategy" to prove model superiority.

3. **🔮 Future Prediction Module (2026 Portfolio):**
    * Instead of just backtesting, the model now trains on full historical data (up to Dec 31, 2025) to generate a **predictive portfolio for 2026**.
    * Outputs a list of 10 "Iron-Clad" dividend stocks with the lowest Hazard Scores.

4. **✅ Walk-Forward Analysis (WFA) & Dynamic Rebalancing:**
    * **No Look-Ahead Bias:** Uses an **Expanding Window** approach.
    * **Annual Rebalancing:** Simulates a realistic trading environment, proving the model's ability to adapt to unseen market regimes (e.g., COVID-19 crash, 2022 inflation).

5. **AI-Driven Diversification (K-Means Clustering):**
    * Uses `scikit-learn` to cluster stocks based on **historical price movement correlations**.
    * Ensures the portfolio is **mathematically diversified**, selecting stocks from distinct clusters rather than relying on arbitrary sector labels.

## Tech Stack
* **Language:** Python 3.10+
* **Web App:** `streamlit` (Dashboard), `pyngrok` (Tunneling for Colab)
* **Bio-Statistics:** `lifelines` (Survival Analysis)
* **Machine Learning:** `scikit-learn` (Clustering)
* **Data Science:** `pandas`, `numpy`, `scipy`
* **Financial Data:** `yfinance`, `requests` (Scraping)
* **Visualization:** `matplotlib`, `seaborn`

## 📂 Directory Structure
```text
US-Dividend-Screener/
├── 📁 data/                  # Raw CSV storage
├── 📁 output/                #
│   └── 📁 figures/           # Generated Charts (Survival curves, Heatmaps, etc.)
├── 📁 src/                   # Source Code
│   ├── 📁 data/              # Data Fetching & Cleaning (loader.py)
│   ├── 📁 analysis/          # Survival Analysis (survival.py) & Scoring
│   ├── 📁 models/            # K-Means Clustering
│   ├── 📁 backtesting/       # Performance Engine
│   └── 📁 visualization/     # Plotting Logic
├── 📄 app.py                 # Streamlit Web Application (GUI)
├── 📄 main.py                # Main Execution Script (CLI)
└── 📄 requirements.txt       # Dependencies
```

## Usage

### Option A: Run the Web App (Recommended) 🚀
Experience the analysis through an interactive dashboard.

```bash
streamlit run app.py
```
- Features: Adjust parameters (Start Year, Portfolio Size), Run A/B Tests, View 2026 Predictions.
- Colab Users: Use pyngrok to tunnel the app if running on Google Colab.

### Option B: Run Command Line Tool
Execute the full pipeline and generate reports in the console.

1. Data Acquisition Fetches the latest S&P 500 list and 10 years of historical data.

```bash
python src/data/fetch_data.py
```
2. Analysis & Execution Runs Screening -> Survival Analysis -> AI Clustering -> Backtest -> 2026 Prediction.

```bash
python main.py
```

## Performance Results (Walk-Forward 2017-2025)
The AI Strategy was tested against a "Simple High-Yield Strategy" (buying the top 10 highest yielders)

**Key Metrics:**
* **CAGR (Annual Return):** **16.26%** (vs S&P 500 benchmark)
* **Sharpe Ratio:** **0.84** (High risk-adjusted return)
* **Max Drawdown:** -43.13% (Limited due to COVID-19, but recovered quickly)

**Conclusion**: The AI model successfully avoided "Yield Traps" (stocks with high yields but poor financials). By prioritizing "Hazard Scores" derived from survival analysis, the portfolio achieved higher returns with significantly less risk during market downturns.

**Key Visualizations:**
* `1_cumulative_return.png`: Evidence of outperformance against S&P 500.
* `3_annual_returns.png`: Shows resilience during bear markets (e.g., 2022).
* `7_rolling_beta.png`: Demonstrates dynamic risk reduction (Beta < 0.7 in recent years).
* `9_survival_curve.png`: Validates the binary separation of safe vs. risky firms.
* `13_cox_coefficients.png`: Visual proof that **Free Cash Flow** drives survival.

## Author
* **Background:** Master's Student in Organic Chemistry / Biochemistry
* **Focus:** Quantitative Finance, Data Science, Bio-Statistics Application
* **Project Goal:**
    This project demonstrates the cross-disciplinary application of **survival analysis algorithms** (used in clinical trials) to **financial risk management**. It serves as a proof-of-concept for using alternative datasets and methodologies in quantitative research.