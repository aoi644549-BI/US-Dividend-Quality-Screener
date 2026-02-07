import pandas as pd
import numpy as np
import yfinance as yf
from src.data.loader import DataLoader
from src.analysis.screener import QualityScreener
from src.analysis.scoring import Scorer
from src.backtesting.engine import BacktestEngine
from src.visualization.plotter import Plotter
from src.models.clustering import StockClusterer
from src.analysis.survival import DividendSurvivalAnalysis

# --- 設定 ---
START_YEAR = 2017
END_YEAR = 2025  # バックテストの終了年
TARGET_N = 10    # ポートフォリオの銘柄数

def run_ai_strategy(df_fund, df_prices):
    """AI戦略を実行し、毎年のポートフォリオ履歴を返す"""
    history = {}
    last_survival_data = None
    last_model_summary = None

    print(f"\n[Simulation] Comparing Strategies from {START_YEAR} to {END_YEAR}...")

    # ウォークフォワード分析 (1年ずつスライドして学習・予測)
    for year in range(START_YEAR, END_YEAR + 1):
        print(f" > Training Year {year}...")
        
        # 1. データの分割 (前年末までのデータを使う)
        train_end_date = f"{year - 1}-12-31"
        df_prices_train = df_prices.loc[:train_end_date]
        
        if df_prices_train.empty:
            print(f"   [Skip] No data for year {year}")
            continue

        # 2. スクリーニング (足切り)
        screener = QualityScreener(df_fund)
        df_filtered = screener.apply_filters(
            min_yield=0.015,
            max_payout=1.0,
            min_growth_10y=0.0
        )
        
        # 3. 生存解析 (Survival Analysis)
        survival_analyzer = DividendSurvivalAnalysis()
        try:
            result = survival_analyzer.fit_and_predict(df_fund, df_prices_train)
            if len(result) == 3:
                risk_scores, survival_data, summary = result
            else:
                risk_scores, survival_data = result
                summary = None

            if risk_scores is not None:
                df_filtered = df_filtered.merge(risk_scores, on='Symbol', how='left')
                # 欠損値は平均で埋める
                df_filtered['Hazard_Score'] = df_filtered['Hazard_Score'].fillna(df_filtered['Hazard_Score'].mean())
                
                # 最終年のデータだけ保存（グラフ描画用）
                if year == END_YEAR:
                    last_survival_data = survival_data
                    last_model_summary = summary
            else:
                df_filtered['Hazard_Score'] = 0
        except Exception as e:
            print(f"   [Warning] Survival Analysis failed: {e}")
            df_filtered['Hazard_Score'] = 0

        # 4. クラスタリング (株価の動きでグループ化)
        valid_tickers = [t for t in df_filtered['Symbol'] if t in df_prices_train.columns]
        
        if len(valid_tickers) > 10:
            subset_prices = df_prices_train[valid_tickers].ffill().bfill()
            
            try:
                clusterer = StockClusterer(n_clusters=max(2, min(10, int(len(valid_tickers)/2))), random_state=1)
            except:
                clusterer = StockClusterer(n_clusters=max(2, min(10, int(len(valid_tickers)/2))))
                
            clusters = clusterer.fit_predict(subset_prices)
            df_filtered = df_filtered.merge(clusters, left_on='Symbol', right_index=True, how='left')
            
            if 'Cluster_y' in df_filtered.columns:
                df_filtered['Cluster'] = df_filtered['Cluster_y'].fillna(-1).astype(int)

        # 5. スコアリングと選抜
        scorer = Scorer(df_filtered)
        if 'Cluster' in df_filtered.columns:
            df_filtered['Sector'] = df_filtered['Cluster'] # Use clusters as sectors
        
        df_selected = scorer.select_diversified_portfolio(df_filtered, n=TARGET_N, max_per_sector=2)
        
        # 銘柄数が足りない場合の補充
        if len(df_selected) < TARGET_N:
             remaining = scorer.calculate_score(df_filtered)
             remaining = remaining[~remaining['Symbol'].isin(df_selected['Symbol'])]
             needed = TARGET_N - len(df_selected)
             df_selected = pd.concat([df_selected, remaining.head(needed)])

        tickers = df_selected['Symbol'].tolist()
        history[year] = tickers
        print(f"   [AI Model] Selected ({len(tickers)}): {', '.join(tickers)}")
        
    # バックテスト実行
    print("\n--- Calculating AI Strategy Performance ---")
    engine = BacktestEngine()
    result = engine.run_dynamic(history, df_prices)
    
    return result, history, last_survival_data, last_model_summary

def run_simple_strategy(df_fund, df_prices):
    """比較用の単純戦略 (配当利回り順)"""
    history = {}
    for year in range(START_YEAR, END_YEAR + 1):
        screener = QualityScreener(df_fund)
        df_filtered = screener.apply_filters(min_yield=0.015, max_payout=1.0, min_growth_10y=0.0)
        df_sorted = df_filtered.sort_values('DividendYield', ascending=False)
        history[year] = df_sorted.head(TARGET_N)['Symbol'].tolist()
        
    print("\n--- Calculating Simple Strategy Performance ---")
    engine = BacktestEngine()
    return engine.run_dynamic(history, df_prices)

def predict_future_portfolio(df_fund, df_prices, target_year=2026):
    """未来(2026年)のポートフォリオを予測・提示する"""
    print(f"\n🔮 Generating AI Portfolio for {target_year} (Future Prediction)...")
    print("--------------------------------------------------")
    
    # 学習データは「データの最後（2025-12-31）」までフルに使う
    train_end_date = "2025-12-31"
    df_prices_train = df_prices.loc[:train_end_date]
    
    # 1. スクリーニング
    screener = QualityScreener(df_fund)
    df_filtered = screener.apply_filters(min_yield=0.015, max_payout=1.0, min_growth_10y=0.0)
    
    # 2. 生存解析
    survival_analyzer = DividendSurvivalAnalysis()
    try:
        result = survival_analyzer.fit_and_predict(df_fund, df_prices_train)
        if len(result) == 3:
            risk_scores, _, _ = result
        else:
            risk_scores, _ = result
            
        if risk_scores is not None:
            df_filtered = df_filtered.merge(risk_scores, on='Symbol', how='left')
            df_filtered['Hazard_Score'] = df_filtered['Hazard_Score'].fillna(df_filtered['Hazard_Score'].mean())
        else:
            df_filtered['Hazard_Score'] = 0
    except Exception as e:
        print(f"Warning: {e}")
        df_filtered['Hazard_Score'] = 0
        
    # 3. クラスタリング
    valid_tickers = [t for t in df_filtered['Symbol'] if t in df_prices_train.columns]
    if len(valid_tickers) > 10:
        subset_prices = df_prices_train[valid_tickers].ffill().bfill()
        try:
            clusterer = StockClusterer(n_clusters=10, random_state=1)
        except:
            clusterer = StockClusterer(n_clusters=10)
        clusters = clusterer.fit_predict(subset_prices)
        df_filtered = df_filtered.merge(clusters, left_on='Symbol', right_index=True, how='left')
        if 'Cluster_y' in df_filtered.columns:
            df_filtered['Cluster'] = df_filtered['Cluster_y'].fillna(-1).astype(int)

    # 4. 選定
    scorer = Scorer(df_filtered)
    if 'Cluster' in df_filtered.columns:
        df_filtered['Sector'] = df_filtered['Cluster']
    
    df_selected = scorer.select_diversified_portfolio(df_filtered, n=TARGET_N, max_per_sector=2)
    
    # 補充
    if len(df_selected) < TARGET_N:
        remaining = scorer.calculate_score(df_filtered)
        remaining = remaining[~remaining['Symbol'].isin(df_selected['Symbol'])]
        needed = TARGET_N - len(df_selected)
        df_selected = pd.concat([df_selected, remaining.head(needed)])
        
    # 結果表示
    print(f"✅ AI Selected Portfolio for {target_year} Start:")
    print("==================================================")
    
    cols = ['Symbol', 'Security', 'GICS Sector', 'DividendYield', 'PayoutRatio', 'Hazard_Score']
    display_cols = [c for c in cols if c in df_selected.columns]
    
    df_display = df_selected[display_cols].copy()
    if 'Hazard_Score' in df_display.columns:
        df_display = df_display.sort_values('Hazard_Score')
        
    print(df_display.to_markdown(index=False))
    print("--------------------------------------------------")
    print(f"💰 Average Portfolio Yield: {df_display['DividendYield'].mean():.2%}")
    print("==================================================")

def main():
    print("=== A/B Testing: AI Strategy vs Simple Strategy ===")
    
    # 1. データ読み込み
    loader = DataLoader()
    try:
        df_fund = loader.load_fundamentals()
        df_prices = loader.load_prices()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. AI戦略の実行
    result_ai, history_ai, survival_data, model_summary = run_ai_strategy(df_fund, df_prices)

    # 3. 単純戦略の実行
    result_simple = run_simple_strategy(df_fund, df_prices)

    # 4. 結果の比較と表示
    metrics_ai = result_ai['metrics']
    metrics_sim = result_simple['metrics']

    print("\n==================================================")
    print(f" 🏆 A/B TEST RESULTS ({START_YEAR}-{END_YEAR})")
    print("==================================================")
    print(f"{'Metric':<20} | {'AI Strategy (Proposal)':<22} | {'Simple Yield (Control)':<22}")
    print("-" * 70)
    print(f"{'CAGR (Return)':<20} | {metrics_ai['CAGR']:>21.2%} | {metrics_sim['CAGR']:>21.2%}")
    print(f"{'Sharpe Ratio':<20} | {metrics_ai['Sharpe']:>21.2f} | {metrics_sim['Sharpe']:>21.2f}")
    print(f"{'Max Drawdown':<20} | {metrics_ai['MaxDrawdown']:>21.2%} | {metrics_sim['MaxDrawdown']:>21.2%}")
    print("-" * 70)
    
    if metrics_ai['CAGR'] > metrics_sim['CAGR'] and metrics_ai['MaxDrawdown'] > metrics_sim['MaxDrawdown']:
        print("✅ Conclusion: AI Strategy is MORE EFFICIENT.")
    else:
        print("⚠️ Conclusion: Mixed results.")
    print("==================================================")

    # 5. グラフ描画
    plotter = Plotter()
    plotter.plot_all(result_ai, pd.DataFrame(), survival_data=survival_data, model_summary=model_summary)
    plotter.plot_ab_test_results(result_ai, result_simple)
    
    # 6. ポートフォリオ利回りの推移グラフ
    plotter.plot_portfolio_yield(history_ai, df_fund)

    # 7. 未来(2026年)のポートフォリオ予測
    predict_future_portfolio(df_fund, df_prices, target_year=2026)

if __name__ == "__main__":
    main()