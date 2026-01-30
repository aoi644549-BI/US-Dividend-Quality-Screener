# US-Dividend Quality Screener 🇺🇸
### AI-Powered Analysis Tool for Financial Health & Downside Resilience

## Overview
This is a Python-based screening tool designed to identify **"True Quality High-Dividend Stocks"**—companies with solid financial foundations that can withstand market crashes—while avoiding "Value Traps."

Targeting all S&P 500 constituents, the tool combines **"Fundamental Analysis with Unsupervised Machine Learning (K-Means Clustering) "**to construct a mathematically diversified portfolio that minimizes correlation risk.

## Background
In dividend investing, simply chasing high yields often leads to **"dividend cuts"** or concentration risk in specific sectors (e.g., holding only Energy stocks). To solve this, I developed a tool that filters stocks based on three core axes:

1.  **Financial Backbone** (Profitability relative to sector peers)
2.  **Crash Durability** (Historical Resilience)
3.  **AI-Based Diversification** (Mathematical Correlation Clustering)


## Features
1.  **Automated Data Pipeline:**
    * Automatically fetches the full S&P 500 list and 10 years of historical price/financial data via Wikipedia and the Yahoo Finance API.
    * Includes logic to handle **survivorship bias** and clean incomplete data for the current fiscal year.
2.  **Relative Value Screening (Sector Neutral):**
    * **Dividend Yield:** ≥ 3.0%
    * **Payout Ratio:** ≤ 90% (Ensuring sustainability)
    * **Dividend Growth (10Y):** ≥ 10% (Inflation protection)
    * **Relative ROE:** Must exceed **Sector Average** (Selecting only the most efficient compounders in each industry).
3.  **AI-Driven Diversification (Machine Learning):**
    * Uses **K-Means Clustering** (`scikit-learn`) to group stocks based on their actual price movement correlations over 10 years.
    * Identifies "mathematically distinct" asset groups rather than relying on traditional sector labels.
4.  **Proprietary Scoring & Selection:**
    * Calculates a comprehensive score based on **Operating Margin** (Earning Power) and **Beta** (Stability).
    * Selects the highest-scoring stock from **each AI-generated cluster** to construct a truly diversified "All-Weather" portfolio.
5.  **Historical Stress Test:**
    * Backtests performance against the S&P 500 over the past decade (2016–2026), verifying resilience during crises like the **COVID-19 Crash (2020)**.

## Tech Stack
* **Language:** Python 3.10+
* **Infrastructure:** Docker / Docker Compose
* **Machine Learning:** scikit-learn (K-Means Clustering)
* **Web Framework:** Streamlit
* **Libraries:**
    * `pandas` (Data Manipulation)
    * `yfinance` (Financial Data Extraction)
    * `requests` (Web Scraping)
    * `matplotlib` (Data Visualization)
    * `tqdm` (Progress Tracking)

## 📂 Directory Structure
This project adopts a modular design based on **Separation of Concerns (SoC)** to ensure scalability and maintainability.

```text
US-Dividend-Screener/
├── 📁 data/                  # Raw data storage (csv)
├── 📁 output/                # Analysis results & Charts
├── 📁 src/                   # Source Code
│   ├── 📁 data/              # Data Loading & Acquisition
│   ├── 📁 analysis/          # Screening & Scoring Logic
│   ├── 📁 models/            # Machine Learning Models (K-Means)
│   ├── 📁 backtesting/       # Historical Simulation
│   └── 📁 visualization/     # Plotting
├── 📄 main.py                # Main Entry Point
└── 📄 requirements.txt       # Dependencies
```

##  Usage
The workflow is divided into **Data Acquisition** and **Analysis** phases to optimize efficiency.

### 1. Data Acquisition
Fetches S&P 500 data (approx. 5-10 mins). Run this periodically to update the dataset.
*Note: This process is separated because downloading data takes time. Run this periodically (e.g., weekly).*

```bash
python src/data/fetch_data.py
```
Output:

Populates data/ directory with sp500_stock_prices.csv and sp500_fundamentals.csv.


### 2. Analysis & Execution
Executes the full pipeline: Relative Screening -> AI Clustering -> Scoring -> Backtesting.

```bash
python main.py
```
Output: 

Console: Displays AI cluster assignments and the final selected portfolio metrics (Sharpe Ratio, CAGR).

File: Saves the performance chart to output/figures/ai_strategy_result.png.


## Sample Result
Below is a performance comparison between the AI-Selected Portfolio and the S&P 500 over the past 10 years (2016-2026).



* **Red Zone:** COVID-19 Crash period.
* **CAGR (Annual Return):** 13.79% (Outperforming Market Avg)
* **Strategy:** The AI successfully identified low-correlation assets (e.g., mixing Merck [Healthcare] with EOG Resources [Energy] and Essex Property [Real Estate]), balancing growth and defensive capabilities.

## Author
* **Major:**  1st year Master's Student (Organic Chemistry、bio chemistry)
* **Interests:** Data Science, Asset Management, Quantitative Analysis
* **Project Goal:**  
    As an aspiring quantitative researcher, I initiated this project to deepen my understanding of the financial industry and gain data-driven insights for my personal investing. It also serves as a practical study in leveraging Generative AI to accelerate the implementation of advanced financial algorithms.