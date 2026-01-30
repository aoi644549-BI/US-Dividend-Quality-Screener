from src.data.loader import DataLoader
from src.analysis.screener import QualityScreener
from src.analysis.scoring import Scorer
from src.backtesting.engine import BacktestEngine
from src.visualization.plotter import Plotter


def main():
    print("=== US Dividend Screener (10-Year Quality) Started ===")

    # 1. Load Data
    loader = DataLoader()
    try:
        df = loader.load_fundamentals()
    except Exception as e:
        print(e)
        return

    # 2. Screening
    screener = QualityScreener(df)
    
    # ★設定確認（画像データに基づく）
    df_filtered = screener.apply_filters(
        min_yield=3.0, 
        max_payout=0.8, 
        min_growth_10y=0.2, 
        min_roe=0.1,        
        min_mkt_cap=1_000_000
    )
    
    print(f"スクリーニング通過: {len(df_filtered)} 銘柄")
    
    if len(df_filtered) > 0:
        display_cols = ['Symbol', 'Name', 'ReturnOnEquity', 'DividendGrowth10Y']
        
        print(df_filtered[display_cols].rename(
            columns={'ReturnOnEquity': 'ROE', 'DividendGrowth10Y': 'DivGrowth10Y'}
        ).head())

    # 3. Scoring & Selection
    scorer = Scorer(df_filtered)
    # セクター分散
    df_selected = scorer.select_diversified_portfolio(n=5, max_per_sector=2)
    
    if len(df_selected) == 0:
        print("条件に合う銘柄がありません。")
        return

    top_tickers = df_selected['Symbol'].tolist()
    print(f"\n分散ポートフォリオ: {top_tickers}")

    # 4. Backtesting
    # 2016年からの10年チャートを描く
    engine = BacktestEngine(start_date="2016-01-01", end_date="2026-01-01")
    result = engine.run(top_tickers)
    
    if result:
        metrics = result['metrics']
        print(f"\n【10年間のパフォーマンス指標】")
        print(f"年平均リターン: {metrics['CAGR']:.2%}")
        print(f"シャープレシオ: {metrics['Sharpe']:.2f}")
        print(f"最大ドローダウン: {metrics['MaxDrawdown']:.2%}")

        # 5. Visualization
        plotter = Plotter()
        plotter.plot_backtest(result, filename="10year_strategy.png")

if __name__ == "__main__":
    main()