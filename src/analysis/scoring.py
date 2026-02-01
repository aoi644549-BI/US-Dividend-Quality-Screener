import pandas as pd


class Scorer:
    def __init__(self, df):
        self.df = df

    def calculate_score(self, df=None):
        if df is None:
            df = self.df.copy()
        else:
            df = df.copy()
            
        if len(df) == 0:
            return df
            
        # 欠損値処理
        df['OperatingMargins'] = df['OperatingMargins'].fillna(0)
        df['Beta'] = df['Beta'].fillna(1.0)
        
        if 'Hazard_Score' not in df.columns:
            df['Hazard_Score'] = 0

        # --- スコアリング計算 (守備力重視に変更) ---
        
        # 1. 稼ぐ力 (Margin): 
        score_margin = df['OperatingMargins'].rank(pct=True) * 100
        
        # 2. 安定性 (Beta): 
        # Betaが低いほど高得点（市場が暴落しても下がりにくい株）
        score_safety = (1 - df['Beta'].rank(pct=True)) * 100
        
        # 3. 生存確率 (Hazard): 減配リスク。
        score_survival = (1 - df['Hazard_Score'].rank(pct=True)) * 100
        
        #「守り(Beta)」を50%、「生存(Hazard)」を30%、「攻め(Margin)」を20%に
        df['Total_Score'] = (score_safety * 0.5) + (score_survival * 0.3) + (score_margin * 0.2)
        
        return df.sort_values('Total_Score', ascending=False)

    def select_diversified_portfolio(self, df=None, n=10, max_per_sector=2):
        """
        n=10, max_per_sector=2 (10銘柄なら1セクター2銘柄までOKとする)
        """
        if df is None:
            df = self.df
            
        ranked_df = self.calculate_score(df)
        selected = []
        sector_counts = {}
        
        # クラスター（またはセクター）ごとに均等に取るロジック
        # ここではシンプルにスコア順で、セクター制約(max_per_sector)を守りながら埋めていく
        
        for _, row in ranked_df.iterrows():
            if len(selected) >= n:
                break
            
            sec = row['Sector']
            # セクター（クラスター）ごとの上限チェック
            if sector_counts.get(sec, 0) < max_per_sector:
                selected.append(row)
                sector_counts[sec] = sector_counts.get(sec, 0) + 1
        
        return pd.DataFrame(selected)