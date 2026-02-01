# US-Dividend Quality Screener (Bio-Stats x Finance Edition) 🧬📈
### AI-Powered Portfolio Construction with Survival Analysis & Machine Learning

## Overview
This is a quantitative analysis tool designed to construct a **"Low-Volatility, High-Quality Dividend Portfolio"**.
By applying **Survival Analysis (Kaplan-Meier & Cox Proportional Hazards)**—methods traditionally used in medical research—to financial data, this tool quantitatively predicts the "lifespan" of a company's dividend sustainability.

Combined with **Unsupervised Machine Learning (K-Means Clustering)** and **Walk-Forward Analysis**, it builds a dynamic portfolio that adapts to market regime changes without look-ahead bias.

## 💡 Core Concept: "Bio-Statistics meets Finance"
In this project, I reinterpreted financial events through the lens of biological survival:
* **Patient Death** $\rightarrow$ **Dividend Cut / Stagnation**
* **Survival Duration** $\rightarrow$ **Financial Runway (Inverse of Payout Ratio)**
* **Risk Factors (Covariates)** $\rightarrow$ **Financial Metrics (Margin, ROE, Beta)**

Just as doctors predict patient survival probabilities, this tool calculates a **"Hazard Score"** for every S&P 500 company to filter out future "value traps."

Just as doctors predict patient survival probabilities, this tool calculates a **"Hazard Score"** for every S&P 500 company to filter out future "value traps."

## Features
1.  **Bio-Statistical Risk Assessment (Survival Analysis):**
    * **Kaplan-Meier Estimator:** Visualizes the "survival probability" (dividend maintenance rate) of the market over time.
    * **Cox Proportional Hazards Model:** Calculates a proprietary **Hazard Score** for each stock. High-risk companies (e.g., high payout ratio, declining growth) are penalized mathematically before they cut dividends.
    * **Library:** Implemented using `lifelines` (Python).

2.  **✅ Walk-Forward Analysis (WFA) & Dynamic Rebalancing:**
    * **No Look-Ahead Bias:** Unlike traditional static backtests, this tool uses an **Expanding Window** approach.
    * **Annual Rebalancing:**
        * *Year $N$:* Train AI on data from $2010$ to $N-1$.
        * *Year $N$:* Select portfolio based *only* on past knowledge.
        * *Year $N$:* Hold for 1 year, then repeat.
    * This simulates a **realistic trading environment**, proving the model's ability to adapt to unseen market regimes (e.g., COVID-19 crash, 2022 inflation).

3.  **AI-Driven Diversification (K-Means Clustering):**
    * Uses `scikit-learn` to cluster stocks based on **historical price movement correlations** (updated annually).
    * Ensures the portfolio is **mathematically diversified**, selecting stocks from distinct clusters (e.g., mixing low-beta Utilities with high-growth Tech) rather than relying on arbitrary sector labels.

4.  **Low-Volatility Scoring Model:**
    * Scores candidates based on a defensive weighting logic:
        * **Stability (Beta):** 50% (Prioritizing downside protection)
        * **Survival (Hazard Score):** 30% (Dividend sustainability)
        * **Profitability (Margins):** 20% (Earning power)
    * Selects the top **10 stocks** to balance risk and return.

5.  **Robust Data Pipeline:**
    * Automatically fetches S&P 500 data via Wikipedia and Yahoo Finance API (`yfinance`).
    * Includes a **Data Cleaning Layer** that automatically detects and standardizes mixed data units (e.g., converting percentage `5.0` to decimal `0.05`).

6.  **Comprehensive Backtesting & Visualization:**
    * Simulates performance from 2017–2025 using Walk-Forward methodology.
    * Generates **13 types of analytical charts** to explain *why* the model worked, including:
        * Kaplan-Meier Survival Curve
        * Rolling Beta (Dynamic Risk Exposure)
        * Risk-Return Scatter Plot
        * Feature Importance (Cox Coefficients)

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


## Performance Results (Walk-Forward 2017-2025)
The tool generates a dynamic portfolio of **10 stocks**, rebalanced annually.

**Key Metrics:**
* **CAGR (Annual Return):** **16.26%** (vs S&P 500 benchmark)
* **Sharpe Ratio:** **0.84** (High risk-adjusted return)
* **Max Drawdown:** -43.13% (Limited due to COVID-19, but recovered quickly)

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