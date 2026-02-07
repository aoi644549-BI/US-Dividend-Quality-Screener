import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler


class DividendSurvivalAnalysis:
    def __init__(self):
        self.cph = CoxPHFitter(penalizer=0.01)

    def prepare_data(self, df_fund):
        """
        生存時間解析のためのデータセットを作成する
        """
        df = df_fund.copy()
        
        # 1. イベント定義 (減配したら死亡)
        df['E'] = np.where(df['DividendGrowth10Y'] < 0.0, 1, 0)
        
        # 2. 生存期間 (T) の定義
        # ここで配当性向を使っているため、長生き＝配当性向が低い、という正解データを作る
        df['PayoutRatio'] = df['PayoutRatio'].clip(lower=0.01, upper=1.2)
        df['T'] = (1 / df['PayoutRatio']) * 10
        df['T'] = df['T'].clip(upper=50)

        # 3. 説明変数の選択
        # 'PayoutRatio' を入力から外す！
        # 代わりに「結果として健全な配当を支える要因」だけを入れる
        # DividendYield (利回り) を追加して、市場評価も加味する
        cols = ['OperatingMargins', 'ReturnOnEquity', 'DividendYield']
        
        # 欠損値処理
        for c in cols:
            if c in df.columns:
                df[c] = df[c].fillna(df[c].median())
            else:
                df[c] = 0
        
        # 4. データの標準化 (Standardization)
        # これをやらないと、単位が違う変数（%と倍率など）がグラフで比較できない
        df_model = df[['Symbol', 'T', 'E'] + cols].copy()
        
        scaler = StandardScaler()
        # Symbol, T, E 以外の数値データを標準化
        df_model[cols] = scaler.fit_transform(df_model[cols])
        
        return df_model

    def fit_and_predict(self, df_fund, df_prices):
        df_model = self.prepare_data(df_fund)
        train_data = df_model.drop(columns=['Symbol'])
        
        # 分散0の列削除
        train_data = train_data.loc[:, (train_data != train_data.iloc[0]).any()]

        result = df_model[['Symbol']].copy()
        summary_df = None

        try:
            self.cph.fit(train_data, duration_col='T', event_col='E', show_progress=False)
            
            hazards = self.cph.predict_partial_hazard(train_data)
            result['Hazard_Score'] = hazards
            summary_df = self.cph.summary[['coef']].copy()
            
        except Exception as e:
            print(f"   [Note] Cox Model fitting issue: {e}")
            result['Hazard_Score'] = 0
            summary_df = None

        return result, train_data, summary_df