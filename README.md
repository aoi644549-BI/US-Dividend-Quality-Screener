# US-Dividend Quality Screener (Bio-Stats x Finance Edition) 🧬📈
### AI-Powered Portfolio Construction with Survival Analysis & Machine Learning

## Overview
This is a quantitative analysis tool designed to construct a **"Low-Volatility, High-Quality Dividend Portfolio"**.
By applying **Survival Analysis (Kaplan-Meier & Cox Proportional Hazards)**—methods traditionally used in medical research—to financial data, this tool quantitatively predicts the "lifespan" of a company's dividend sustainability.

Combined with **Unsupervised Machine Learning (K-Means Clustering)** for correlation-based diversification, it builds a portfolio resilient to market crashes and inflation.

## 💡 Core Concept: "Bio-Statistics meets Finance"
In this project, I reinterpreted financial events through the lens of biological survival:
* **Patient Death** $\rightarrow$ **Dividend Cut / Stagnation**
* **Survival Duration** $\rightarrow$ **Financial Runway (Inverse of Payout Ratio)**
* **Risk Factors (Covariates)** $\rightarrow$ **Financial Metrics (Margin, ROE, Beta)**

Just as doctors predict patient survival probabilities, this tool calculates a **"Hazard Score"** for every S&P 500 company to filter out future "value traps."

## Features
1.  **Bio-Statistical Risk Assessment (Survival Analysis):**
    * **Kaplan-Meier Estimator:** Visualizes the "survival probability" (dividend maintenance rate) of the market over time.
    * **Cox Proportional Hazards Model:** Calculates a proprietary **Hazard Score** for each stock. High-risk companies (e.g., high payout ratio, declining growth) are penalized mathematically before they cut dividends.
    * **Library:** Implemented using `lifelines` (Python).

2.  **AI-Driven Diversification (K-Means Clustering):**
    * Uses `scikit-learn` to cluster stocks based on **10-year price movement correlations**.
    * Ensures the portfolio is **mathematically diversified**, selecting stocks from distinct clusters (e.g., mixing low-beta Utilities with high-growth Tech) rather than relying on arbitrary sector labels.

3.  **Low-Volatility Scoring Model:**
    * Scores candidates based on a defensive weighting logic:
        * **Stability (Beta):** 50% (Prioritizing downside protection)
        * **Survival (Hazard Score):** 30% (Dividend sustainability)
        * **Profitability (Margins):** 20% (Earning power)
    * Selects the top **10 stocks** to balance risk and return.

4.  **Robust Data Pipeline:**
    * Automatically fetches S&P 500 data via Wikipedia and Yahoo Finance API (`yfinance`).
    * Includes a **Data Cleaning Layer** that automatically detects and standardizes mixed data units (e.g., converting percentage `5.0` to decimal `0.05`).

5.  **Comprehensive Backtesting & Visualization:**
    * Simulates performance from 2016–2026.
    * Generates **9 types of analytical charts**, including:
        * Kaplan-Meier Survival Curve
        * Monte Carlo Simulation (Future Forecasting)
        * Return Distribution (Fat Tail Analysis)
        * Rolling Beta & Sharpe Ratio

## Tech Stack
* **Language:** Python 3.10+
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
├── 📄 main.py                # Main Execution Script
└── 📄 requirements.txt       # Dependencies
```

##  Usage
The workflow is divided into **Data Acquisition** and **Analysis** phases to optimize efficiency.

### 1. Data Acquisition
Fetches the latest S&P 500 list and 10 years of historical data. Robust against anti-scraping measures.
*Note: This process is separated because downloading data takes time. Run this periodically (e.g., weekly).*

```bash
python src/data/fetch_data.py
```
Output:

Populates data/ directory with sp500_stock_prices.csv and sp500_fundamentals.csv.


### 2. Analysis & Execution
Runs the full pipeline: Screening -> Survival Analysis -> AI Clustering -> Scoring -> Backtest.

```bash
python main.py
```
Output: 

Console: Displays AI cluster assignments and the final selected portfolio metrics (Sharpe Ratio, CAGR).

File: Saves the performance chart to output/figures/ai_strategy_result.png.


## Sample Output
The tool generates a portfolio of **10 stocks** optimized for risk-adjusted returns.

**Performance Metrics (2016-2026):**
* **CAGR (Annual Return):** ~13.0% (Outperforming typical High-Dividend ETFs)
* **Max Drawdown:** Significantly reduced compared to pure market indices due to the "Low Beta" focus.
* **Sharpe Ratio:** Optimized for stability.

**Key Visualizations:**
* `9_survival_curve.png`: Shows the probability of dividend sustainability.
* `8_monte_carlo.png`: Probabilistic future portfolio value projection.

## Author
* **Background:** Master's Student in Organic Chemistry / Biochemistry
* **Focus:** Quantitative Finance, Data Science, Bio-Statistics Application
* **Project Goal:**
    This project demonstrates the cross-disciplinary application of **survival analysis algorithms** (used in clinical trials) to **financial risk management**. It serves as a proof-of-concept for using alternative datasets and methodologies in quantitative research.