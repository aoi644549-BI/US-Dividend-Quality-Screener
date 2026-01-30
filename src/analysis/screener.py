import pandas as pd


class QualityScreener:
    def __init__(self, df):
        self.df = df

    def apply_filters(self, min_yield=3.0, max_payout=0.8, min_growth_10y=0.1, min_roe=0.1, min_mkt_cap=1_000_000):
        """
        min_growth_10y: 10年増配率
        min_roe: 自己資本利益率 (0.1 = 10%)
        min_mkt_cap: 時価総額 (ドル)
        """
        df = self.df.copy()

        df['MarketCapRank'] = df.groupby('Sector')['MarketCap'].rank(ascending=False)

        condition = (
            (df['DividendYield'] >= min_yield) & #配当利回り
            (df['PayoutRatio'] <= max_payout) & 
            (df['DividendGrowth10Y'] >= min_growth_10y) &  # 10年成長
            (df['ReturnOnEquity'] >= min_roe) &             # ROE 10%以上
            (df['MarketCap'] >= min_mkt_cap) &              # 時価総額
            (df['MarketCapRank'] <= 5)
        )
        
        return df[condition].copy()