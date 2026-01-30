import pandas as pd


class Scorer:
    def __init__(self, df):
        self.df = df

    def calculate_score(self, weight_margin=0.6, weight_safety=0.4):
        if len(self.df) == 0:
            return self.df
            
        df = self.df.copy()
        
        # 営業利益率と安定性(Beta)でスコア付け
        df['Score_Margin'] = df['OperatingMargins'].rank(pct=True) * 100
        df['Score_Safety'] = (1 - df['Beta'].rank(pct=True)) * 100
        df['Total_Score'] = (df['Score_Margin'] * weight_margin) + (df['Score_Safety'] * weight_safety)
        
        return df.sort_values('Total_Score', ascending=False)

    def select_diversified_portfolio(self, n=5, max_per_sector=1):

        ranked_df = self.calculate_score()
        selected = []
        sector_counts = {}
        
        for _, row in ranked_df.iterrows():
            if len(selected) >= n:
                break
            
            sec = row['Sector']
            # そのセクターがまだ上限に達していなければ採用
            if sector_counts.get(sec, 0) < max_per_sector:
                selected.append(row)
                sector_counts[sec] = sector_counts.get(sec, 0) + 1
        
        return pd.DataFrame(selected)