import pandas as pd
import numpy as np
import os 


current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(current_dir), 'data')

input_path = os.path.join(data_dir, 'sp500_fundamentals.csv')
output_path = os.path.join(data_dir, 'my_dividend_ranking.csv')


print(f"データ読み込み元: {input_path}")

if not os.path.exists(input_path):
    print("エラー: 入力ファイルが見つかりません。先に fetch_data.py を実行してください。")
    exit()

df = pd.read_csv(input_path)
print(f"分析開始: 全 {len(df)} 銘柄")

df_screened = df[
    (df['DividendYield'] >= 0.03) & 
    (df['PayoutRatio'] <= 0.8) & 
    (df['OperatingMargins'] > 0)
].copy()

print(f"スクリーニング結果: {len(df_screened)} 銘柄が残りました。")

if len(df_screened) > 0:
    def calculate_score(sub_df):
        res = sub_df.copy()
        res['Score_Margin'] = res['OperatingMargins'].rank(pct=True) * 100
        res['Score_Safety'] = (1 - res['Beta'].rank(pct=True)) * 100
        res['Total_Score'] = (res['Score_Margin'] * 0.6) + (res['Score_Safety'] * 0.4)
        return res

    df_scored = calculate_score(df_screened)
    df_ranking = df_scored.sort_values('Total_Score', ascending=False)

    df_ranking.to_csv(output_path, index=False)
    
    print("\n【優良高配当株ランキング Top 5】")
    cols = ['Symbol', 'Name', 'DividendYield', 'Total_Score']
    display(df_ranking[cols].head())
    
    print(f"\nランキングを保存しました: {output_path}")

else:
    print("\n条件に合う銘柄がありませんでした。")