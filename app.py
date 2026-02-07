import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image

from src.data.loader import DataLoader
from src.analysis.screener import QualityScreener
from src.analysis.scoring import Scorer
from src.backtesting.engine import BacktestEngine
from src.visualization.plotter import Plotter
from src.models.clustering import StockClusterer
from src.analysis.survival import DividendSurvivalAnalysis

# --- ページ設定 ---
st.set_page_config(
    page_title="Bio-Finance Dividend Screener",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- タイトルと説明 ---
st.title("🧬 Bio-Finance Dividend Screener")
st.markdown("""
**「企業の減配」を「患者の死亡」と見立てた生存解析モデル (Survival Analysis)** を用いて、
S&P 500 構成銘柄から財務的に堅牢なポートフォリオを構築します。
""")

# --- サイドバー設定 ---
st.sidebar.header("⚙️ Settings")

start_year = st.sidebar.slider("Start Year", 2015, 2023, 2017)
end_year = st.sidebar.slider("End Year", 2018, 2026, 2025)
target_n = st.sidebar.slider("Portfolio Size (Stocks)", 5, 20, 10)
run_ab_test = st.sidebar.checkbox("Run A/B Test (vs Simple Yield)", value=True)

# --- データ読み込み (キャッシュ化で高速化) ---
@st.cache_data
def load_data():
    loader = DataLoader()
    try:
        df_fund = loader.load_fundamentals()
        df_prices = loader.load_prices()
        return df_fund, df_prices
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

df_fund, df_prices = load_data()

if df_fund is None:
    st.stop()

st.sidebar.success("✅ Data Loaded Successfully")


# --- AI戦略ロジック (関数化) ---
def run_ai_strategy(df_fund, df_prices, start_year, end_year, n_stocks):
    history = {}
    last_survival_data = None
    
    # プログレスバー
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_years = end_year - start_year + 1
    
    for i, year in enumerate(range(start_year, end_year + 1)):
        status_text.text(f"Training Model for Year {year}...")
        train_end_date = f"{year - 1}-12-31"
        df_prices_train = df_prices.loc[:train_end_date]
        
        if df_prices_train.empty: continue
            
        # 1. Screening
        screener = QualityScreener(df_fund)
        df_filtered = screener.apply_filters(min_yield=0.015, max_payout=1.0, min_growth_10y=0.0)
        
        # 2. Survival Analysis
        survival_analyzer = DividendSurvivalAnalysis()
        try:
            # ★修正: 3つの値を返すように変更
            result = survival_analyzer.fit_and_predict(df_fund, df_prices_train)
            
            # 結果が2つの場合（古いコード対応）と3つの場合で分岐
            if len(result) == 3:
                risk_scores, survival_data, _ = result
            else:
                risk_scores, survival_data = result

            if risk_scores is None:
                df_filtered['Hazard_Score'] = 0
            else:
                df_filtered = df_filtered.merge(risk_scores, on='Symbol', how='left')
                df_filtered['Hazard_Score'] = df_filtered['Hazard_Score'].fillna(df_filtered['Hazard_Score'].mean())
                last_survival_data = survival_data
        except Exception as e:
            print(f"Error in Survival Analysis: {e}")
            df_filtered['Hazard_Score'] = 0

        # 3. Clustering
        valid_tickers = [t for t in df_filtered['Symbol'] if t in df_prices_train.columns]
        if len(valid_tickers) > 2:
            subset = df_prices_train[valid_tickers].fillna(method='ffill').fillna(method='bfill')
            
            # random_state=1 を指定して固定
            # clustering.pyが未対応でも動くよう、try-exceptは使わず引数指定
            clusterer = StockClusterer(n_clusters=max(2, min(10, int(len(valid_tickers)/2))), random_state=1)
            
            clusters = clusterer.fit_predict(subset)
            df_filtered = df_filtered.merge(clusters, left_on='Symbol', right_index=True, how='left')
            if 'Cluster_y' in df_filtered.columns:
                 df_filtered['Cluster'] = df_filtered['Cluster_y'].fillna(-1).astype(int)

        # 4. Scoring & Selection
        scorer = Scorer(df_filtered)
        if 'Cluster' in df_filtered.columns:
            df_filtered['Sector'] = df_filtered['Cluster'] # Use clusters as sectors
        
        df_selected = scorer.select_diversified_portfolio(df_filtered, n=n_stocks, max_per_sector=2)
        
        # 足りない分を補充
        if len(df_selected) < n_stocks:
             remaining = scorer.calculate_score(df_filtered)
             remaining = remaining[~remaining['Symbol'].isin(df_selected['Symbol'])]
             needed = n_stocks - len(df_selected)
             df_selected = pd.concat([df_selected, remaining.head(needed)])

        history[year] = df_selected['Symbol'].tolist()
        
        # 進捗更新
        progress_bar.progress((i + 1) / total_years)

    status_text.text("Running Backtest Engine...")
    engine = BacktestEngine()
    result = engine.run_dynamic(history, df_prices)
    
    return result, history, last_survival_data

# --- 単純戦略ロジック ---
def run_simple_strategy(df_fund, df_prices, start_year, end_year, n_stocks):
    history = {}
    for year in range(start_year, end_year + 1):
        screener = QualityScreener(df_fund)
        df_filtered = screener.apply_filters(min_yield=0.015, max_payout=1.0, min_growth_10y=0.0)
        df_sorted = df_filtered.sort_values('DividendYield', ascending=False)
        history[year] = df_sorted.head(n_stocks)['Symbol'].tolist()
        
    engine = BacktestEngine()
    return engine.run_dynamic(history, df_prices)


# --- メイン実行ボタン ---
if st.button('🚀 Run Analysis', type="primary"):
    with st.spinner('Simulating... This may take a minute.'):
        
        # 1. AI戦略実行
        result_ai, history_ai, survival_data = run_ai_strategy(df_fund, df_prices, start_year, end_year, target_n)
        
        # 2. 単純戦略実行 (A/Bテスト ONの場合)
        result_simple = None
        if run_ab_test:
            result_simple = run_simple_strategy(df_fund, df_prices, start_year, end_year, target_n)

        # 3. 画像生成 (Plotter利用)
        plotter = Plotter()
        plotter.plot_all(result_ai, pd.DataFrame(), survival_data=survival_data)
        # ポートフォリオ利回りのグラフを作成
        plotter.plot_portfolio_yield(history_ai, df_fund)
        
        if result_simple:
            plotter.plot_ab_test_results(result_ai, result_simple)

    # --- 結果表示エリア ---
    st.divider()
    
    # 1. メトリクス表示
    col1, col2, col3 = st.columns(3)
    metrics = result_ai['metrics']
    
    col1.metric("CAGR (Annual Return)", f"{metrics['CAGR']:.2%}", 
                delta=f"{metrics['CAGR'] - result_simple['metrics']['CAGR']:.2%}" if result_simple else None)
    col2.metric("Sharpe Ratio", f"{metrics['Sharpe']:.2f}",
                delta=f"{metrics['Sharpe'] - result_simple['metrics']['Sharpe']:.2f}" if result_simple else None)
    col3.metric("Max Drawdown", f"{metrics['MaxDrawdown']:.2%}",
                delta=f"{metrics['MaxDrawdown'] - result_simple['metrics']['MaxDrawdown']:.2%}" if result_simple else None,
                delta_color="inverse")

    # 2. A/Bテスト比較タブ
    st.subheader("📊 Analysis Report")
    tab1, tab2, tab3, tab4 = st.tabs(["A/B Comparison", "Survival Analysis", "Risk Analysis", "Portfolio History"])

    def safe_image(path, caption=None):
        if os.path.exists(path):
            st.image(path, caption=caption)
        else:
            st.info(f"Image not generated: {path} (Data unavailable)")

    with tab1:
        if result_simple:
            safe_image("output/figures/0_ab_test_cumulative.png", caption="Cumulative Return Comparison")
            safe_image("output/figures/0_ab_test_drawdown.png", caption="Drawdown Comparison")
        else:
            safe_image("output/figures/1_cumulative_return.png")

    with tab2:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            safe_image("output/figures/9_survival_curve.png", caption="Kaplan-Meier Survival Curve")
        with col_s2:
            safe_image("output/figures/13_cox_coefficients.png", caption="Risk Factors (Cox Model)")

    with tab3:
        safe_image("output/figures/7_rolling_beta.png", caption="Dynamic Risk Exposure (Rolling Beta)")
        safe_image("output/figures/11_risk_return_scatter.png", caption="Risk-Return Trade-off")

    with tab4:
        st.write("### Selected Portfolio by Year")
        
        # リスト表示はループで処理（利回り計算だけここで行う）
        for year, tickers in history_ai.items():
            subset = df_fund[df_fund['Symbol'].isin(tickers)]
            avg_yield = subset['DividendYield'].mean() if not subset.empty else 0
            
            with st.expander(f"Year {year} (Yield: {avg_yield:.2%})"):
                st.info(f"💰 Portfolio Dividend Yield: **{avg_yield:.2%}**")
                st.write(f"**Selected Stocks ({len(tickers)}):**")
                st.code(", ".join(tickers))
                
                available_cols = ['Symbol', 'DividendYield']
                if 'Security' in df_fund.columns: available_cols.append('Security')
                if 'GICS Sector' in df_fund.columns: available_cols.append('GICS Sector')
                elif 'Sector' in df_fund.columns: available_cols.append('Sector')
                
                st.dataframe(subset[available_cols].style.format({'DividendYield': '{:.2%}'}))

        st.divider()
        st.subheader("📈 Portfolio Yield History")
        # Streamlitのグラフではなく、Plotterで作った画像を表示
        safe_image("output/figures/14_portfolio_yield.png", caption="Average Portfolio Yield by Year")