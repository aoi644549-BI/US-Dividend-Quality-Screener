import pandas as pd
import numpy as np
from lifelines import CoxPHFitter


class DividendSurvivalAnalysis:
    def __init__(self):
        # ★修正1: penalizer はここで設定します
        self.cph = CoxPHFitter(penalizer=0.1)

    def prepare_data(self, df_fund):
        """
        生存時間解析のためのデータセットを作成する
        """
        df = df_fund.copy()
        
        # イベント発生フラグ (1: 減配リスクあり, 0: 安定)
        df['E'] = np.where(df['DividendGrowth10Y'] < 0.05, 1, 0)
        
        # 生存期間 (T) - 配当性向の逆数をプロキシとして使用
        df['T'] = (1 / df['PayoutRatio'].clip(lower=0.1)) * 10
        df['T'] = df['T'].clip(upper=50)

        # Coxモデルに使う共変量
        cols = ['PayoutRatio', 'OperatingMargins', 'ReturnOnEquity', 'Beta']
        # データが欠損している列を平均で埋める
        for c in cols:
            df[c] = df[c].fillna(df[c].mean())
            
        df_model = df[['Symbol', 'T', 'E'] + cols].copy()
        return df_model

    def fit_and_predict(self, df_fund, df_prices):
        """
        Cox比例ハザードモデルを学習し、リスクスコアと学習データを返す
        Return: (result_df, train_data)
        """
        print("  ...生存時間解析データの準備中")
        df_model = self.prepare_data(df_fund)
        
        # モデル用のデータ（Symbol以外）
        train_data = df_model.drop(columns=['Symbol'])
        
        # 定数列（分散0）の削除
        train_data = train_data.loc[:, (train_data != train_data.iloc[0]).any()]

        print("  ...Cox比例ハザードモデルの学習中")
        try:
            # ★修正2: ここから penalizer 引数を削除
            self.cph.fit(train_data, duration_col='T', event_col='E', show_progress=False)
            
            # ハザード予測
            hazards = self.cph.predict_partial_hazard(train_data)
            
            result = df_model[['Symbol']].copy()
            result['Hazard_Score'] = hazards
            
            return result, train_data
            
        except Exception as e:
            print(f"  [Warning] Coxモデルの学習に失敗しました: {e}")
            result = df_model[['Symbol']].copy()
            result['Hazard_Score'] = 0
            return result, None