import pandas as pd
import numpy as np
from src.data.loader import DataLoader
from src.analysis.screener import QualityScreener
from src.analysis.scoring import Scorer
from src.backtesting.engine import BacktestEngine
from src.visualization.plotter import Plotter
from src.models.clustering import StockClusterer
from src.analysis.survival import DividendSurvivalAnalysis


def train_and_select(df_prices_train, df_fund, target_year_str):
    # 1. Screening
    screener = QualityScreener(df_fund)
    df_filtered = screener.apply_filters(
        min_yield=0.015, max_payout=1.0, min_growth_10y=0.0
    )
    
    survival_data_result = None

    if len(df_filtered) < 5:
        return [], None

    # 2. Survival Analysis
    survival_analyzer = DividendSurvivalAnalysis()
    try:
        risk_scores, survival_data = survival_analyzer.fit_and_predict(df_fund, df_prices_train)
        df_filtered = df_filtered.merge(risk_scores, on='Symbol', how='left')
        df_filtered['Hazard_Score'] = df_filtered['Hazard_Score'].fillna(df_filtered['Hazard_Score'].mean())
        survival_data_result = survival_data # データを保存
    except:
        df_filtered['Hazard_Score'] = 0

    # 3. Clustering
    valid_tickers = [t for t in df_filtered['Symbol'] if t in df_prices_train.columns]
    
    if len(valid_tickers) < 2 or len(df_prices_train) < 252:
        df_filtered['Cluster'] = 0
    else:
        subset_prices = df_prices_train[valid_tickers]
        subset_prices = subset_prices.fillna(method='ffill').fillna(method='bfill')
        
        n_samples = len(valid_tickers)
        optimal_clusters = max(2, min(10, int(n_samples / 2)))
        
        clusterer = StockClusterer(n_clusters=optimal_clusters)
        try:
            clusters = clusterer.fit_predict(subset_prices)
            df_filtered = df_filtered.merge(clusters, left_on='Symbol', right_index=True, how='left')
            if 'Cluster_y' in df_filtered.columns:
                df_filtered['Cluster'] = df_filtered['Cluster_y'].fillna(-1).astype(int)
                df_filtered = df_filtered.drop(columns=['Cluster_x', 'Cluster_y'], errors='ignore')
        except:
            df_filtered['Cluster'] = 0

    # 4. Scoring & Selection
    scorer = Scorer(df_filtered)
    df_for_selection = df_filtered.copy()
    
    if 'Cluster' in df_for_selection.columns:
        df_for_selection['Original_Sector'] = df_for_selection['Sector']
        df_for_selection['Sector'] = df_for_selection['Cluster']
    
    target_n = 10
    df_selected = scorer.select_diversified_portfolio(df_for_selection, n=target_n, max_per_sector=2)
    
    if len(df_selected) < target_n and len(df_filtered) >= target_n:
        existing = df_selected['Symbol'].tolist()
        remaining = scorer.calculate_score(df_for_selection)
        remaining = remaining[~remaining['Symbol'].isin(existing)]
        needed = target_n - len(df_selected)
        df_selected = pd.concat([df_selected, remaining.head(needed)])
        
    return df_selected['Symbol'].tolist(), survival_data_result

def main():
    print("=== US Dividend Screener (Walk-Forward Analysis Edition) Started ===")

    # 1. Load Data
    loader = DataLoader()
    try:
        df_fund = loader.load_fundamentals()
        df_prices = loader.load_prices()
    except Exception as e:
        print(f"Error: {e}")
        return

    # --- Walk-Forward Setting ---
    start_year = 2017
    end_year = 2025
    
    portfolio_history = {}
    last_survival_data = None # 最後の年の生存解析データを保存する変数
    
    print(f"\n[Walk-Forward] Simulating from {start_year} to {end_year}...")
    print("---------------------------------------------------------------")
    
    for year in range(start_year, end_year + 1):
        train_end_date = f"{year - 1}-12-31"
        print(f" > Training Model for Year {year} (Data up to {train_end_date})")
        
        df_prices_train = df_prices.loc[:train_end_date]
        if df_prices_train.empty:
            continue
            
        # 戻り値を受け取る
        selected_tickers, survival_data = train_and_select(df_prices_train, df_fund, str(year))
        
        portfolio_history[year] = selected_tickers
        
        # 最後の年のデータを保存しておく
        if survival_data is not None:
            last_survival_data = survival_data
            
        print(f"   -> Selected {len(selected_tickers)} stocks")

    # 4. Backtesting (Dynamic)
    engine = BacktestEngine()
    result = engine.run_dynamic(portfolio_history, df_prices)
    
    if result:
        metrics = result['metrics']
        print(f"\n【Walk-Forward パフォーマンス結果】")
        print(f"年平均リターン: {metrics['CAGR']:.2%}")
        print(f"シャープレシオ: {metrics['Sharpe']:.2f}")
        print(f"最大ドローダウン: {metrics['MaxDrawdown']:.2%}")

        # Visualization
        plotter = Plotter()
        #最後に保存した survival_data を渡す
        plotter.plot_all(result, pd.DataFrame(), survival_data=last_survival_data)

if __name__ == "__main__":
    main()