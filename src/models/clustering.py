import pandas as pd
from sklearn.cluster import KMeans


class StockClusterer:
    # random_stateを受け取れるようにしておきます
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit_predict(self, prices_df):
        """
        株価の動きに基づいてクラスタリングを実行する
        """
        # --- 防御策1: データが空、または小さすぎる場合は即リターン ---
        if prices_df.empty or len(prices_df) < 5 or prices_df.shape[1] < 2:
            # 銘柄数が少なすぎる、または日付が短すぎる場合はクラスタリング不能
            # 全員「クラスタ0」として返す
            return pd.Series(0, index=prices_df.columns, name='Cluster')

        # リターン（変化率）に変換
        # fill_method引数の警告を回避するため、先に欠損値を処理する
        returns = prices_df.pct_change().dropna()
        
        # --- 防御策2: 計算後のデータが空になった場合 ---
        if returns.empty:
            return pd.Series(0, index=prices_df.columns, name='Cluster')
            
        # 転置して (銘柄 x 日付) の形にする
        X = returns.T
        
        # --- 防御策3: クラスタ数より銘柄数が少ない場合 ---
        # 例: 5クラスタ作りたいのに、3銘柄しかない場合など
        actual_clusters = min(self.n_clusters, len(X))
        if actual_clusters < 2:
             return pd.Series(0, index=prices_df.columns, name='Cluster')
        
        try:
            # エラーの原因になりやすい scaler は省略し、KMeansで直接計算します
            model = KMeans(n_clusters=actual_clusters, random_state=self.random_state, n_init=10)
            clusters = model.fit_predict(X)
            return pd.Series(clusters, index=X.index, name='Cluster')
            
        except Exception as e:
            print(f"   [Warning] Clustering failed: {e}")
            # エラーが起きても止まらずに、全員「0」として返す
            return pd.Series(0, index=prices_df.columns, name='Cluster')