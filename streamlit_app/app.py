import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os


# --- 設定: ページ構成 ---
st.set_page_config(
    page_title="US Dividend Screener",
    page_icon="📊",
    layout="wide"
)

# --- 関数: データの読み込み (キャッシュ化で高速化) ---
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    file_path = os.path.join(data_dir, 'sp500_fundamentals.csv')
    
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path)

# --- メイン画面 ---
st.title("🇺🇸 US Dividend Quality Screener")
st.markdown("財務健全性と過去の暴落耐性に基づく、高配当株分析ツール")

# 1. データのロード
df = load_data()

if df is None:
    st.error("エラー: データファイルが見つかりません。先に `src/fetch_data.py` を実行してください。")
    st.stop()

# --- サイドバー: スクリーニング条件の設定 ---
st.sidebar.header("🔍 Screening Criteria")

# スライダーで値を調整できるようにする
min_yield = st.sidebar.slider("最低 配当利回り (%)", 0.0, 10.0, 3.0, 0.1)
max_payout = st.sidebar.slider("最大 配当性向 (%)", 10, 200, 80, 5)
min_margin = st.sidebar.slider("最低 営業利益率 (%)", -10.0, 50.0, 0.0, 1.0)

# 重み付けの調整
st.sidebar.subheader("⚖️ Scoring Weights")
weight_margin = st.sidebar.slider("稼ぐ力 (利益率) の重み", 0.0, 1.0, 0.6, 0.1)
weight_safety = 1.0 - weight_margin
st.sidebar.text(f"守りの力 (安全性) の重み: {weight_safety:.1f}")

# --- ロジック: フィルタリング & スコアリング ---
# フィルタリング
df_screened = df[
    (df['DividendYield'] >= min_yield / 100) &  # %を小数に変換
    (df['PayoutRatio'] <= max_payout / 100) &
    (df['OperatingMargins'] > min_margin / 100)
].copy()

st.info(f"全 {len(df)} 銘柄中、条件に合致したのは **{len(df_screened)}** 銘柄です。")

if len(df_screened) > 0:
    # スコアリング
    df_screened['Score_Margin'] = df_screened['OperatingMargins'].rank(pct=True) * 100
    df_screened['Score_Safety'] = (1 - df_screened['Beta'].rank(pct=True)) * 100
    df_screened['Total_Score'] = (df_screened['Score_Margin'] * weight_margin) + (df_screened['Score_Safety'] * weight_safety)
    
    # ランキング作成
    df_ranking = df_screened.sort_values('Total_Score', ascending=False)
    
    # --- 結果の表示 (2カラムレイアウト) ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🏆 Top Ranking")
        # 表示したい列だけ選ぶ
        display_cols = ['Symbol', 'Name', 'DividendYield', 'PayoutRatio', 'Total_Score']
        st.dataframe(df_ranking[display_cols].head(10), height=400)
        
        # 選択されたトップ3
        top_tickers = df_ranking['Symbol'].head(3).tolist()
    
    with col2:
        st.subheader("📈 Historical Performance Test")
        if st.button("Top 3銘柄のバックテストを実行"):
            with st.spinner('株価データを取得してグラフを描画中...'):
                bench_ticker = '^GSPC'
                tickers = top_tickers + [bench_ticker]
                
                # 株価取得
                data = yf.download(tickers, start="2020-01-01", end="2024-01-01", auto_adjust=True)
                if 'Close' in data.columns.levels[0]: data = data['Close']
                elif 'Close' in data.columns: data = data['Close']
                
                # リターン計算
                norm_data = (data / data.iloc[0]) * 100 - 100
                
                # グラフ描画 (Matplotlib)
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # S&P500
                ax.plot(norm_data.index, norm_data[bench_ticker], label='S&P 500', color='gray', linestyle='--', alpha=0.7)
                
                # Top銘柄
                colors = ['#FF4B4B', '#1C83E1', '#00C0F2'] # Streamlitっぽい色
                for i, ticker in enumerate(top_tickers):
                    ax.plot(norm_data.index, norm_data[ticker], label=ticker, color=colors[i], linewidth=2)
                
                # コロナショック
                ax.axvspan('2020-02-19', '2020-03-23', color='red', alpha=0.1, label='Covid-19 Crash')
                
                ax.set_title("Performance vs S&P 500 (Covid-19 Stress Test)")
                ax.set_ylabel("Return (%)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Streamlitにグラフを表示
                st.pyplot(fig)
                
                st.success("✅ 分析完了！")

else:
    st.warning("⚠️ 条件に合う銘柄がありません。サイドバーの条件を緩めてください。")