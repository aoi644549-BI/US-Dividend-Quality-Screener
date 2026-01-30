import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class StockClusterer:
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)

    def fit_predict(self, prices_df):
        """
        株価データ(prices_df)から、値動きの似ている銘柄をクラスタリングする
        return: 銘柄ごとのCluster IDが入ったSeries
        """
        # 1. 日次リターン（変化率）に変換
        # これにより「株価が高い/安い」ではなく「動きの形」だけを見れる
        returns = prices_df.pct_change().dropna()

        # 2. 転置（行:日付、列:銘柄 → 行:銘柄、列:日付）
        X = returns.T

        # 3. 欠損値処理（上場日が浅い銘柄などは0埋め）
        X = X.fillna(0)

        # 4. データの標準化（スケールを揃える）
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 5. K-Means実行
        self.model.fit(X_scaled)
        
        # 6. 結果を整形
        labels = self.model.labels_
        result = pd.Series(labels, index=X.index, name='Cluster')
        
        return result