import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os  


# 1. ランキング上位の銘柄を取得
df_ranking = pd.read_csv('data/my_dividend_ranking.csv')
top_tickers = df_ranking['Symbol'].head(3).tolist() 
bench_ticker = '^GSPC' 

print(f"比較対象: {top_tickers} vs S&P500")

# 2. 過去5年の株価データを取得
tickers_to_fetch = top_tickers + [bench_ticker]
data = yf.download(tickers_to_fetch, start="2020-01-01", end="2026-01-01", auto_adjust=True)

# データ形式の調整
if 'Close' in data.columns.levels[0]:
    data = data['Close']
elif 'Close' in data.columns:
    data = data['Close']

# 3. 「リターン（騰落率）」に変換する
normalized_data = (data / data.iloc[0]) * 100 - 100

# 4. グラフ描画
plt.figure(figsize=(12, 6))

# ベンチマーク（S&P500）をグレーで描画
plt.plot(normalized_data.index, normalized_data[bench_ticker], 
         label='S&P 500', color='gray', linestyle='--', alpha=0.7, linewidth=2)

# Top銘柄をカラフルに描画
colors = ['red', 'blue', 'green']
for i, ticker in enumerate(top_tickers):
    plt.plot(normalized_data.index, normalized_data[ticker], 
             label=ticker, color=colors[i], linewidth=2.5)

# 暴落期間（コロナショック）をハイライト
plt.axvspan('2020-02-19', '2020-03-23', color='red', alpha=0.1, label='Covid-19 Crash')

# グラフの装飾
plt.title('Dividend Quality Stocks vs S&P 500 (Historical Performance)', fontsize=15)
plt.ylabel('Return (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')


# 5. 保存先のディレクトリを作成
output_dir = 'output/figures'

# フォルダが存在しない場合、自動的に作成する（エラーを防ぐため）
os.makedirs(output_dir, exist_ok=True)

# 保存するファイルのパスを作成（output/figures/performance_chart.png になる）
save_path = os.path.join(output_dir, 'performance_chart.png')

# 指定したパスに保存
plt.savefig(save_path)
print(f"グラフを '{save_path}' に保存しました。")

# --- 修正ここまで ---

plt.show()