import pandas as pd


class QualityScreener:
    def __init__(self, df_fundamentals):
        self.df = df_fundamentals.copy()

    def apply_filters(self, min_yield=0.015, max_payout=1.0, min_growth_10y=0.0):
        """
        シンプルなフィルタリングを適用する
        """
        print(f"  [Screening] 初期候補数: {len(self.df)} 銘柄")
        
        # 1. データの欠損を埋める (NaNで落ちるのを防ぐ)
        # 成長率や配当性向がない場合は、一旦「パス」させるために安全な値で埋める
        self.df['DividendYield'] = self.df['DividendYield'].fillna(0)
        self.df['PayoutRatio'] = self.df['PayoutRatio'].fillna(0.5) # 平均的な値にしておく
        self.df['DividendGrowth10Y'] = self.df['DividendGrowth10Y'].fillna(0)
        self.df['OperatingMargins'] = self.df['OperatingMargins'].fillna(0.1)
        
        # 2. 配当利回りフィルター
        df_filtered = self.df[self.df['DividendYield'] >= min_yield]
        print(f"  [Filter] 利回り {min_yield*100}% 以上: {len(df_filtered)} 銘柄 (脱落: {len(self.df) - len(df_filtered)})")
        
        # 3. 配当性向フィルター
        # 0未満(利益マイナス)や、max_payout超えを除外
        before_len = len(df_filtered)
        df_filtered = df_filtered[
            (df_filtered['PayoutRatio'] > 0) & 
            (df_filtered['PayoutRatio'] <= max_payout)
        ]
        print(f"  [Filter] 配当性向 0~{max_payout*100}%: {len(df_filtered)} 銘柄 (脱落: {before_len - len(df_filtered)})")
        
        # 4. 営業利益率フィルター (赤字企業を除外)
        before_len = len(df_filtered)
        df_filtered = df_filtered[df_filtered['OperatingMargins'] > 0]
        print(f"  [Filter] 黒字企業 (利益率>0): {len(df_filtered)} 銘柄 (脱落: {before_len - len(df_filtered)})")
        
        # 5. 増配率フィルター
        before_len = len(df_filtered)
        df_filtered = df_filtered[df_filtered['DividendGrowth10Y'] >= min_growth_10y]
        print(f"  [Filter] 10年増配率 {min_growth_10y*100}% 以上: {len(df_filtered)} 銘柄 (脱落: {before_len - len(df_filtered)})")

        return df_filtered