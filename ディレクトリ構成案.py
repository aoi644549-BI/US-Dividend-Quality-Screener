US-Dividend-Screener/
├── 📁 data/                      # 取得した生データ置き場
│   ├── sp500_stock_prices.csv    # 株価データ
│   └── sp500_fundamentals.csv    # 財務データ
│
├── 📁 output/                    # 出力ファイル置き場
│   └── 📁 figures/               # 生成されたグラフ画像 (.png)
│
├── 📁 src/                       # ソースコード群（ロジックの本体）
│   ├── 📁 data/                  # 【データ責務】
│   │   ├── __init__.py           # (空ファイル: パッケージ認識用)
│   │   ├── fetch_data.py         # ネットからデータをDLするスクリプト
│   │   └── loader.py             # CSVを読み込んでDataFrameを返すクラス
│   │
│   ├── 📁 analysis/              # 【分析責務】
│   │   ├── __init__.py
│   │   ├── screener.py           # フィルタリング ("足切り") ロジック
│   │   └── scoring.py            # スコアリング ("偏差値計算") ロジック
│   │
│   ├── 📁 backtesting/           # 【検証責務】
│   │   ├── __init__.py
│   │   └── engine.py             # 過去データでのシミュレーション機能
│   │
│   ├── 📁 visualization/         # 【可視化責務】
│   │   ├── __init__.py
│   │   └── plotter.py            # グラフを描画して保存するクラス
│   │
│   └── 📁 models/                # 【機械学習責務】(将来拡張用)
│       ├── __init__.py
│       └── clustering.py         # K-Meansなどのモデル定義
│
├── 📁 streamlit_app/             # Webアプリケーション　後で
│   └── app.py                    # ブラウザ表示用のUIコード
│
├── 📄 main.py                    # 全モジュールを統括して実行する
├── 📄 requirements.txt           # 必要なライブラリ一覧
├── 📄 Dockerfile                 # Dockerの設計図 後で
├── 📄 docker-compose.yml         # Dockerの起動設定　後で
├── 📄 .gitignore                 # Gitに上げないファイルを指定　後で
└── 📄 README.md                  # プロジェクトの説明書