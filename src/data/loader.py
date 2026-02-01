import pandas as pd
import os


class DataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

    def load_fundamentals(self):
        """財務データを読み込み、単位を統一する"""
        path = os.path.join(self.data_dir, "sp500_fundamentals.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")
        
        df = pd.read_csv(path)
        
        # 1. 配当利回り (DividendYield) の修正
        # データが「1以上（例: 3.5）」ならパーセント表記とみなして100で割る
        # データが「1未満（例: 0.035）」ならそのまま
        if df['DividendYield'].mean() > 1.0:
            print("  [Data Fix] 配当利回りをパーセントから小数に変換します (/100)")
            df['DividendYield'] = df['DividendYield'] / 100
            
        # 2. 配当性向 (PayoutRatio) の修正
        # まれにパーセント表記(50.0)の場合があるので、平均が10(1000%)を超えていたら補正
        if df['PayoutRatio'].mean() > 10.0:
             print("  [Data Fix] 配当性向をパーセントから小数に変換します (/100)")
             df['PayoutRatio'] = df['PayoutRatio'] / 100
             
        # 3. 欠損値の穴埋め (NaN対策)
        # 成長率(DividendGrowth10Y)が空欄の場合は 0 (成長なし) とみなす
        if 'DividendGrowth10Y' in df.columns:
            df['DividendGrowth10Y'] = df['DividendGrowth10Y'].fillna(0.0)
            
        return df

    def load_prices(self):
        """株価データを読み込み、整形して返す"""
        path = os.path.join(self.data_dir, "sp500_stock_prices.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")

        try:
            # マルチインデックスとして読み込み
            df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
        except:
            df = pd.read_csv(path, index_col=0, parse_dates=True)

        # 'Close' または 'Adj Close' の列を抽出
        target_col = None
        for col in ['Close', 'Adj Close']:
            if df.columns.nlevels > 1 and col in df.columns.get_level_values(1):
                target_col = col
                break
        
        if target_col:
            prices = df.xs(target_col, axis=1, level=1, drop_level=True)
        else:
            prices = df

        # 強制的に数値型に変換
        prices = prices.apply(pd.to_numeric, errors='coerce')
        prices = prices.dropna(axis=1, how='all')
        
        return prices