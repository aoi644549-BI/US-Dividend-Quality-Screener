from src.data.loader import DataLoader
from src.analysis.screener import QualityScreener
from src.analysis.scoring import Scorer
from src.backtesting.engine import BacktestEngine
from src.visualization.plotter import Plotter
from src.models.clustering import StockClusterer
from src.analysis.survival import DividendSurvivalAnalysis
import pandas as pd 


def main():
    print("=== US Dividend Screener (Low Volatility Edition) Started ===")

    # 1. Load Data
    loader = DataLoader()
    try:
        df_fund = loader.load_fundamentals()
        df_prices = loader.load_prices()
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Screening
    screener = QualityScreener(df_fund)
    df_filtered = screener.apply_filters(
        min_yield=0.015,    
        max_payout=1.0,     
        min_growth_10y=0.00  
    )
    print(f"スクリーニング通過: {len(df_filtered)} 銘柄")
    
    if len(df_filtered) < 5:
        print("候補が少なすぎます。")
        return

    # --- Survival Analysis ---
    print("\n[Survival Analysis] コックス比例ハザードモデルによる減配リスク算出中...")
    survival_analyzer = DividendSurvivalAnalysis()
    try:
        risk_scores, survival_data = survival_analyzer.fit_and_predict(df_fund, df_prices)
        df_filtered = df_filtered.merge(risk_scores, on='Symbol', how='left')
        df_filtered['Hazard_Score'] = df_filtered['Hazard_Score'].fillna(df_filtered['Hazard_Score'].mean())
    except Exception as e:
        print(f"Skipping Survival Analysis: {e}")
        df_filtered['Hazard_Score'] = 0
        survival_data = None

    # --- Machine Learning ---
    print("\n[Machine Learning] 株価の動きに基づきクラスタリングを実行中...")
    valid_tickers = [t for t in df_filtered['Symbol'] if t in df_prices.columns]
    
    # クラスタリング用の列を初期化
    df_filtered['Cluster'] = -1 
    
    if len(valid_tickers) < 2:
        print("  -> クラスタリング対象が少なすぎるためスキップ")
    else:
        subset_prices = df_prices[valid_tickers]
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
            elif 'Cluster' in clusters.columns:
                 pass 
        except Exception as e:
             print(f"  [Warning] Clustering failed: {e}")
             df_filtered['Cluster'] = -1

    # 3. Scoring & Selection
    scorer = Scorer(df_filtered)
    
    df_for_selection = df_filtered.copy()
    
    # クラスタリングが成功していればそれを使う、失敗(-1)なら元のセクターを使う
    if 'Cluster' in df_for_selection.columns and df_for_selection['Cluster'].nunique() > 1:
        df_for_selection['Original_Sector'] = df_for_selection['Sector']
        df_for_selection['Sector'] = df_for_selection['Cluster']
    else:
        print("  -> クラスタリング結果が不十分なため、元のセクター情報を使用します。")
    
    # 銘柄選定 (n=10)
    target_n = 10
    df_selected = scorer.select_diversified_portfolio(df_for_selection, n=target_n, max_per_sector=2)
    
    # もし銘柄数が足りなかったら、セクター制限を無視して埋める
    if len(df_selected) < target_n and len(df_filtered) >= target_n:
        print(f"  [Info] 分散投資枠で埋まりきらなかったため、スコア上位から補充します ({len(df_selected)} -> {target_n})")
        existing_symbols = df_selected['Symbol'].tolist()
        # 選ばれていない銘柄をスコア順に取得
        remaining = scorer.calculate_score(df_for_selection)
        remaining = remaining[~remaining['Symbol'].isin(existing_symbols)]
        
        # 不足分を追加
        needed = target_n - len(df_selected)
        df_selected = pd.concat([df_selected, remaining.head(needed)])

    top_tickers = df_selected['Symbol'].tolist()
    print(f"\nAI & 生存解析選定ポートフォリオ ({len(top_tickers)}銘柄): {top_tickers}")
    
    cols = ['Symbol', 'Original_Sector', 'Cluster', 'Total_Score', 'Hazard_Score', 'Beta']
    disp_cols = [c for c in cols if c in df_selected.columns]
    print(df_selected[disp_cols])

    # 4. Backtesting
    if len(top_tickers) > 0:
        engine = BacktestEngine(start_date="2016-01-01", end_date="2026-01-01")
        result = engine.run(top_tickers)
        
        if result:
            metrics = result['metrics']
            print(f"\n【10年間のパフォーマンス指標】")
            print(f"年平均リターン: {metrics['CAGR']:.2%}")
            print(f"シャープレシオ: {metrics['Sharpe']:.2f}")
            print(f"最大ドローダウン: {metrics['MaxDrawdown']:.2%}")

            plotter = Plotter()
            plotter.plot_all(result, df_selected, survival_data=survival_data)

if __name__ == "__main__":
    main()