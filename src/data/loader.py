import pandas as pd
import os


class DataLoader:
    def __init__(self, base_dir="data"):
        # 実行場所に関わらずプロジェクトルートのdataフォルダを参照する
        # このファイル(src/data/loader.py)から見て、2つ上がルート
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.data_dir = os.path.join(project_root, base_dir)

    def load_fundamentals(self):
        path = os.path.join(self.data_dir, 'sp500_fundamentals.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(f"データが見つかりません: {path}\n先に fetch_data.py を実行してください。")
        return pd.read_csv(path)

    def load_prices(self):
        path = os.path.join(self.data_dir, 'sp500_stock_prices.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(f"データが見つかりません: {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)