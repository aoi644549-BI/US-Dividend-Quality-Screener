# US-Dividend Quality Screener 🇺🇸
### Analysis Tool for Financial Health & Downside Resilience

## Overview
This is a Python-based screening tool designed to identify **"True Quality High-Dividend Stocks"**—companies with solid financial foundations that can withstand market crashes—while avoiding "Value Traps" (stocks that appear cheap but have underlying issues).

Targeting S&P 500 constituents, the tool automatically performs fundamental analysis and historical stress tests (evaluating resilience during past market crashes).

## Background
In my personal asset management, I focus on dividend investing. However, I identified a problem: simply chasing high yields often leads to **"dividend cut risks"** or **"structural downtrends."**

To solve this, I developed a tool that filters stocks based on two core axes:
1.  **Financial Backbone** (Profitability & Sustainability)
2.  **Crash Durability** (Historical Resilience)

## Features
1.  **Automated Data Collection:**
    * Automatically fetches S&P 500 lists and financial/price data via Wikipedia and the Yahoo Finance API.
2.  **Quality Screening:**
    * **Dividend Yield:** ≥ 3.0%
    * **Payout Ratio:** ≤ 80% (Ensuring dividend sustainability)
    * **Operating Margin:** > 0% (Must be profitable)
3.  **Proprietary Scoring (My Logic):**
    * Calculates a comprehensive score by normalizing **Operating Margin** (Earning Power) and **Beta** (Stability), applying a custom weighting algorithm.
4.  **Historical Stress Test:**
    * Backtests performance against the S&P 500 during specific crisis periods, such as the **COVID-19 Crash (Feb-Mar 2020)**.

## Tech Stack
* **Language:** Python 3.10+
* **Infrastructure:** Docker / Docker Compose
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
│   ├── 📁 data/              # Data Loading & Acquisition (Loader)
│   ├── 📁 analysis/          # Analysis Logic (Screener, Scorer)
│   ├── 📁 backtesting/       # Historical Simulation (BacktestEngine)
│   └── 📁 visualization/     # Plotting & Visualization (Plotter)
├── 📄 main.py                # Main Entry Point
└── 📄 Dockerfile             # Container definition
```

##  Usage
The workflow is divided into **Data Acquisition** and **Analysis** phases to optimize efficiency.

### 1. Data Acquisition
Fetches the latest S&P 500 list and historical financial data.
*Note: This process is separated because downloading data takes time. Run this periodically (e.g., weekly).*

```bash
python main.py
```
Output:

Console: Displays the selected portfolio and performance metrics (Sharpe Ratio, Max Drawdown).

File: Saves the backtest chart to output/figures/backtest_result.png.

### 2. Analysis & Execution
Executes the full pipeline: screening, scoring, backtesting, and visualization. Note: This step is fast as it uses the locally cached data.

```bash
python src/data/fetch_data.py
```
Output: data/ directory will be populated with sp500_stock_prices.csv etc.


## Sample Result
Below is a performance comparison between the Top-selected stocks and the market average (S&P 500) over the past 5 years.

*(Please insert your image here: `output/figures/performance_chart.png`)*

* **Red Zone:** COVID-19 Crash period.
* **Insight:** The Utilities sector stocks (e.g., LNT, AEP) demonstrated significant downside resistance compared to the market average during the crash, validating the screening logic.

## Author
* **Major:**  1st year Master's Student
* **Interests:** Data Science, Asset Management, Quantitative Analysis