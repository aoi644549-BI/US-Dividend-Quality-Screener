US-Dividend-Screener/
├── 📁 data/                  # CSV置き場
├── 📁 output/                # グラフ保存先
├── 📁 src/
│   ├── 📁 data/              # fetch_data.py はここに移動（データ取得責務）
│   │   └── loader.py
│   ├── 📁 analysis/          # analyze.py はここに移動（分析責務）
│   │   ├── screener.py       # フィルタリングロジック
│   │   └── scoring.py        # スコア計算ロジック
│   ├── 📁 backtesting/       # ★Phase 2で追加
│   │   └── engine.py         # 「もし過去に運用していたら」を計算
│   └── 📁 models/            # ★Phase 4で追加
│       └── clustering.py     # 機械学習モデル
├── 📁 streamlit_app/         # ★Phase 1で追加
│   └── app.py                # Web画面のコード
└── requirements.txt